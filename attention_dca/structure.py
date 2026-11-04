"""
attention_dca.structure — the ground truth: residue distances of a crystal structure on the MSA.

A PDB chain does not share the alignment's numbering, so its sequence is first aligned to the
family profile. Distances are then read through that column-to-residue map, which needs nothing
beyond Biopython (no HMMER, no pre-computed mapping).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBParser, is_aa
from Bio.SeqUtils import seq1
from scipy.spatial.distance import cdist

from .msa import GAP, LETTER_TO_NUM_AA, MSA


def read_chain(pdb_path: str, chain: str = "A") -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Sequence and heavy-atom coordinates of one chain.

    Returns
    -------
    seq : one-letter sequence of the ``n`` resolved residues.
    res_ids : PDB residue numbers, shape ``(n,)``.
    xyz : heavy-atom coordinates, shape ``(n_atoms, 3)``.
    atom_res : residue index of each atom, shape ``(n_atoms,)``.
    """
    structure = PDBParser(QUIET=True).get_structure(pdb_path, pdb_path)
    residues = [res for res in structure[0][chain] if is_aa(res, standard=False)]
    seq = "".join(seq1(res.get_resname(), custom_map={"MSE": "M"}) for res in residues)
    res_ids = np.array([res.id[1] for res in residues])

    atoms = [(r, atom.coord) for r, res in enumerate(residues) for atom in res if atom.element not in ("H", "D")]
    return seq, res_ids, np.array([coord for _, coord in atoms]), np.array([r for r, _ in atoms])


def align_to_profile(seq: str, f: np.ndarray, open_gap: float = -4.0, extend_gap: float = -0.5) -> np.ndarray:
    r"""
    Aligns a sequence to the MSA columns with the position-specific scores
    ``S_i(a) = log( f_i(a) / p(a) )``, in the semi-global sense (overhangs are free).

    ``PairwiseAligner`` accepts integer sequences with an arbitrary substitution matrix, which
    turns it into a profile aligner: column ``i`` becomes token ``i``, amino acid ``a`` becomes
    token ``L + a``, and ``S`` sits in the off-diagonal block of the matrix.

    Parameters
    ----------
    seq : one-letter sequence.
    f : single-site frequencies of the family, shape ``(L, q)`` (:meth:`MSA.profile`).
    open_gap, extend_gap : gap scores in nats, roughly BLOSUM62's -11 / -1.

    Returns
    -------
    np.ndarray — residue index aligned to each column, ``-1`` for a deletion, shape ``(L,)``.
    """
    L = f.shape[0]
    f_aa = f[:, :GAP] + 1e-3                                            # pseudocount, no gap state
    f_aa /= f_aa.sum(axis=1, keepdims=True)

    S = np.zeros((L + GAP, L + GAP))
    S[:L, L:] = np.log(f_aa / f_aa.mean(axis=0))

    aligner = PairwiseAligner(mode="global")
    aligner.open_gap_score, aligner.extend_gap_score = open_gap, extend_gap
    aligner.end_gap_score = 0.0
    aligner.substitution_matrix = S
    tokens = np.array([L + min(LETTER_TO_NUM_AA.get(c, GAP), GAP - 1) for c in seq], dtype=np.int32)
    alignment = aligner.align(np.arange(L, dtype=np.int32), tokens)[0]

    col_to_res = np.full(L, -1)
    for (c0, c1), (r0, r1) in zip(*alignment.aligned):
        col_to_res[c0:c1] = np.arange(r0, r1)
    return col_to_res


def distance_map(pdb_path: str, msa: MSA, chain: str = "A", partner: Optional[str] = None) -> np.ndarray:
    r"""
    Minimum heavy-atom distances of a PDB chain in alignment coordinates, shape ``(L, L)``. Columns
    without an aligned residue are ``NaN`` and are skipped by every evaluation.

    Parameters
    ----------
    pdb_path : PDB file.
    msa : the alignment defining the columns.
    chain : chain identifier.
    partner : a second chain. ``D_ij`` then holds the distance across the interface between column
        ``i`` of one chain and column ``j`` of the other, which is how homo-oligomer contacts look
        from the alignment.
    """
    seq_1, _, xyz_1, atom_res_1 = read_chain(pdb_path, chain)
    seq_2, _, xyz_2, atom_res_2 = read_chain(pdb_path, chain if partner is None else partner)

    D_res = np.full((len(seq_1), len(seq_2)), np.inf)
    np.minimum.at(D_res, (atom_res_1[:, None], atom_res_2[None, :]), cdist(xyz_1, xyz_2))

    f = msa.profile()
    col_to_res_1, col_to_res_2 = align_to_profile(seq_1, f), align_to_profile(seq_2, f)
    aligned_1, aligned_2 = col_to_res_1 >= 0, col_to_res_2 >= 0

    D = np.full((msa.L, msa.L), np.nan)
    D[np.ix_(aligned_1, aligned_2)] = D_res[np.ix_(col_to_res_1[aligned_1], col_to_res_2[aligned_2])]
    print(f"Structure     : {pdb_path} chain {chain}{'' if partner is None else '-' + partner}, "
          f"{aligned_1.sum()}/{msa.L} columns mapped")
    return np.fmin(D, D.T)


def secondary_structure(pdb_path: str, msa: MSA, chain: str = "A") -> list[tuple[str, int, int]]:
    r"""
    Helices and strands from the ``HELIX`` / ``SHEET`` records of a PDB file, on alignment columns.

    Returns
    -------
    list — ``(label, first_column, last_column)`` numbered along the sequence (``"β1"``, ``"α1"``, ...).
    """
    seq, res_ids, _, _ = read_chain(pdb_path, chain)
    col_to_res = align_to_profile(seq, msa.profile())
    col_ids = np.where(col_to_res >= 0, res_ids[col_to_res], -1)        # PDB residue number per column

    elements = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("HELIX") and line[19] == chain:
                elements.append(("α", int(line[21:25]), int(line[33:37])))
            elif line.startswith("SHEET") and line[21] == chain:
                elements.append(("β", int(line[22:26]), int(line[33:37])))

    numbered, counts = [], {"α": 0, "β": 0}
    for kind, first, last in sorted(elements, key=lambda element: element[1]):
        counts[kind] += 1
        columns = np.flatnonzero((col_ids >= first) & (col_ids <= last))
        if len(columns) > 0:
            numbered.append((f"{kind}{counts[kind]}", int(columns[0]), int(columns[-1])))
    return numbered
