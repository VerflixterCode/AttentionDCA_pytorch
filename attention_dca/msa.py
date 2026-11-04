"""
attention_dca.msa — the multiple sequence alignment as training data.

Reads a FASTA alignment into integer tokens, drops gappy and duplicate sequences, and computes
the sequence weights that counter phylogenetic oversampling. The weights are the one expensive
step (``O(M^2 L)``), so they are cached next to the FASTA file.
"""

from __future__ import annotations

import gzip
import os
import zlib
from string import ascii_lowercase
from typing import Optional

import numpy as np
import torch
from Bio import SeqIO


LETTER_TO_NUM_AA: dict[str, int] = {
    'A': 0,  'B': 20, 'C': 1,  'D': 2,  'E': 3,
    'F': 4,  'G': 5,  'H': 6,  'I': 7,  'J': 20,
    'K': 8,  'L': 9,  'M': 10, 'N': 11, 'O': 20,
    'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16,
    'U': 20, 'V': 17, 'W': 18, 'X': 20, 'Y': 19,
    '-': 20,
}
AMINO_ACIDS: str = "ACDEFGHIKLMNPQRSTVWY"       # in token order
GAP: int = 20
Q_AA: int = 21

_INSERT_STATES: dict[int, None] = str.maketrans("", "", ascii_lowercase + ".")


# =================================================================== #
# The alignment
# =================================================================== #

class MSA:
    r"""
    A protein family alignment: integer tokens ``Z`` and sequence weights ``w``.

    A sequence's weight is the inverse size of its neighbourhood in Hamming space,
    ``w_m = 1 / |{n : d_H(x^m, x^n) < theta L}|``, so a clade of near-identical sequences counts
    about once. ``Meff = sum_m w_m`` is the effective number of independent sequences.

    Parameters
    ----------
    path : FASTA file, optionally gzipped. Lowercase letters and ``.`` are insert states and get
        dropped, so A2M / A3M alignments can be used as they are.
    max_gap : sequences with a larger gap fraction are discarded.
    theta : redundancy radius as a fraction of ``L``. ``None`` picks it from the alignment's
        diversity (:func:`compute_theta`).
    num_seq : keep a random subset of this size after filtering; ``None`` keeps everything.
    seed : drives the subset and the ``theta`` estimate.
    cache : keep the weights in ``<path>.weights.npz`` and reuse them when they match.
    """

    def __init__(
        self,
        path: str,
        max_gap: float = 0.9,
        theta: Optional[float] = None,
        num_seq: Optional[int] = None,
        seed: int = 0,
        cache: bool = True,
    ) -> None:
        self.path = path

        Z = read_fasta(path)
        M_raw = Z.shape[0]
        Z = Z[(Z == GAP).mean(axis=1) <= max_gap]
        Z = Z[np.sort(np.unique(Z, axis=0, return_index=True)[1])]      # keeps the file order
        if num_seq is not None:
            rng = np.random.default_rng(seed)
            Z = Z[np.sort(rng.choice(Z.shape[0], size=num_seq, replace=False))]
        self.Z = Z                                                      # (M, L)

        # Load-or-compute. The checksum ties the cache to these exact tokens, so another filter or
        # subset of the same file recomputes (and overwrites) instead of loading stale weights.
        cache_path = f"{path}.weights.npz"
        checksum   = zlib.crc32(Z)
        self.w     = None
        if cache and os.path.exists(cache_path):
            with np.load(cache_path) as cached:
                if cached["checksum"] == checksum and theta in (None, cached["theta"]):
                    self.theta, self.w = float(cached["theta"]), cached["w"]
        if self.w is None:
            self.theta = compute_theta(Z, seed=seed) if theta is None else theta
            self.w     = compute_weights(Z, self.theta)
            if cache:
                np.savez_compressed(cache_path, w=self.w, theta=self.theta, checksum=checksum)
        self.Meff = float(self.w.sum())

        print(f"MSA loaded    : M={self.M} ({M_raw} before filtering), L={self.L}, "
              f"theta={self.theta:.3f}, Meff={self.Meff:.0f}")

    @property
    def M(self) -> int:
        return self.Z.shape[0]

    @property
    def L(self) -> int:
        return self.Z.shape[1]

    def profile(self) -> np.ndarray:
        r"""Weighted single-site frequencies ``f_i(a)``, shape ``(L, q)``."""
        f = np.zeros((self.L, Q_AA))
        np.add.at(f, (np.arange(self.L)[None, :], self.Z), self.w[:, None])
        return f / self.w.sum()


# =================================================================== #
# Reading and reweighting
# =================================================================== #

def read_fasta(path: str) -> np.ndarray:
    r"""
    Reads an aligned FASTA file into tokens of shape ``(M, L)``.

    Any letter outside the 20 amino acids (ambiguity codes, ``X``, ...) is treated as a gap.
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        seqs = [str(rec.seq).translate(_INSERT_STATES) for rec in SeqIO.parse(fh, "fasta")]

    if len({len(seq) for seq in seqs}) != 1:
        raise ValueError(f"{path}: sequences have different lengths, is this an alignment?")

    byte_to_token = np.full(256, GAP, dtype=np.uint8)
    byte_to_token[[ord(c) for c in LETTER_TO_NUM_AA]] = list(LETTER_TO_NUM_AA.values())
    tokens = byte_to_token[np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8)]
    return tokens.reshape(len(seqs), -1)


def compute_theta(Z: np.ndarray, n_pairs: int = 100_000, seed: int = 0) -> float:
    r"""
    Automatic redundancy radius ``theta = min(0.5, 0.38 * 0.32 / f)``, with ``f`` the mean pairwise
    identity estimated on ``n_pairs`` random pairs. Diverse families get a larger radius.
    """
    i, j = np.random.default_rng(seed).integers(0, Z.shape[0], size=(2, n_pairs))
    f = (Z[i[i != j]] == Z[j[i != j]]).mean()
    return float(min(0.5, 0.38 * 0.32 / f))


def compute_weights(Z: np.ndarray, theta: float, chunk: int = 256,
                    device: Optional[torch.device] = None) -> np.ndarray:
    r"""
    Sequence weights ``w_m``, the inverse neighbour count within ``d_H < theta L`` (self included).

    The number of identical sites of two sequences is the dot product of their one-hot vectors,
    so a single matrix product compares a chunk of sequences against the whole alignment. Time is
    ``O(M^2 L q)`` and memory ``O(M L q)``: about a minute and 0.5 GB for ``M = 50k``, ``L = 111``
    on a laptop CPU.

    Parameters
    ----------
    Z : tokens of shape ``(M, L)``.
    theta : redundancy radius as a fraction of ``L``.
    chunk : sequences compared per matrix product.
    device : computation device; defaults to CUDA if available, else CPU.
    """
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M, L = Z.shape
    thresh = np.floor(theta * L)
    X = torch.eye(Q_AA, device=device)[torch.from_numpy(Z).long().to(device)].flatten(1)    # (M, L q)

    counts = torch.zeros(M, device=device)
    for start in range(0, M, chunk):
        n_identical = X[start:start + chunk] @ X.T                      # (c, M)
        counts[start:start + chunk] = ((L - n_identical) < thresh).sum(dim=1).float()
    return 1.0 / counts.clamp(min=1.0).cpu().numpy().astype(np.float64)
