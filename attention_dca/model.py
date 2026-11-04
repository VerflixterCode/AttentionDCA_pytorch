"""
attention_dca.model — the ``AttentionDCA`` model: one factored-attention layer as a Potts model.

Holds the parameters ``Q``, ``K``, ``V``, the pseudo-likelihood objective, the training loop, and
per-field I/O (``params.npz`` + ``config.json`` + ``history.csv``).
"""

from __future__ import annotations

import copy
import json
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .msa import MSA, Q_AA


class AttentionDCA(nn.Module):
    r"""
    A Potts model whose couplings are built by a single factored-attention layer,

        ``A^h_ij  = softmax_j( sum_d Q^h_di K^h_dj )``          (attention of head ``h``)
        ``J_ij(a, b) = sum_h A^h_ij V^h(a, b)``,  ``i != j``      (couplings)

    so sites only enter through ``Q, K`` and amino acids only through ``V``. This takes
    ``H (2 d L + q^2)`` parameters against ``L (L-1) q^2 / 2`` for an unconstrained Potts model.
    There are no local fields: the couplings carry the conservation profile as well.

    The model is fit by maximising the pseudo-likelihood of the conditionals
    ``P(x_i = a | x_\i) ∝ exp( sum_{j != i} J_ij(a, x_j) )`` under an L2 penalty on ``J``.

    Parameters
    ----------
    L : alignment length.
    H : number of attention heads.
    d : inner dimension of ``Q`` and ``K``.
    q : alphabet size.
    lambd : L2 penalty strength on ``J``.
    seed : seed of the uniform initialisation (the loss is non-convex, so runs differ).
    device : computation device; defaults to CUDA if available, else CPU.
    dtype : floating-point dtype of the parameters.
    """

    def __init__(
        self,
        L: int,
        H: int = 128,
        d: int = 17,
        q: int = Q_AA,
        lambd: float = 1e-3,
        seed: int = 0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.L, self.H, self.d, self.q = L, H, d, q
        self.lambd  = lambd
        self.seed   = seed
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        generator = torch.Generator().manual_seed(seed)
        self.Q = nn.Parameter(torch.rand(H, d, L, generator=generator, dtype=dtype))
        self.K = nn.Parameter(torch.rand(H, d, L, generator=generator, dtype=dtype))
        self.V = nn.Parameter(torch.rand(H, q, q, generator=generator, dtype=dtype))
        self.register_buffer("off_diag", 1.0 - torch.eye(L, dtype=dtype), persistent=False)     # (L, L)

        self.history: list[dict[str, float]] = []
        self.fit_config: dict[str, float] = {}
        self.to(self.device)

    # ================= Model ================= #

    def attention(self) -> torch.Tensor:
        r"""
        Attention matrices ``A^h_ij``, shape ``(H, L, L)``.

        The softmax runs over all ``j`` and the diagonal is removed afterwards, so each row sums
        to ``1 - A^h_ii <= 1``.
        """
        scores = torch.einsum("hdi,hdj->hij", self.Q, self.K)               # (H, L, L)
        return torch.softmax(scores, dim=2) * self.off_diag

    def couplings(self) -> torch.Tensor:
        r"""Couplings ``J_ij(a, b)`` with index order ``(i, a, j, b)``, shape ``(L, q, L, q)``."""
        return torch.einsum("hij,hab->iajb", self.attention(), self.V)

    def log_conditionals(self, Z: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        r"""
        Per-sequence log pseudo-likelihood ``sum_i log P(x_i | x_\i)``, shape ``(M,)``.

        The energies ``E_i(a | x) = sum_j J_ij(a, x_j)`` are one matrix product between the
        flattened couplings and the one-hot sequences. Materialising ``J`` costs ``L^2 q^2`` memory
        but no longer scales with ``H``, which is much cheaper than looping over the heads
        whenever ``H`` exceeds ``q``.

        Parameters
        ----------
        Z : tokens of shape ``(M, L)``.
        J : couplings from :meth:`couplings`.
        """
        n_states = self.L * self.q
        X = F.one_hot(Z, self.q).to(J.dtype).flatten(1)                     # (M, L q)
        E = (X @ J.reshape(n_states, n_states).T).reshape(-1, self.L, self.q)   # (M, L, q)
        return E.log_softmax(dim=2).gather(2, Z.unsqueeze(2)).sum(dim=(1, 2))

    def loss(self, Z: torch.Tensor, w: torch.Tensor, chunk: Optional[int] = None) -> torch.Tensor:
        r"""
        Regularised negative log pseudo-likelihood,
        ``- sum_m w_m sum_i log P(x_i^m | x_\i^m) / sum_m w_m + lambda sum_{ij,ab} J_ij(a, b)^2``.

        Parameters
        ----------
        Z : tokens of shape ``(M, L)``.
        w : sequence weights of shape ``(M,)``; normalised here, so the penalty keeps the same
            weight relative to the data term at any batch size or alignment depth.
        chunk : evaluate this many sequences at a time to bound memory (validation only).
        """
        J = self.couplings()
        chunk = len(Z) if chunk is None else chunk
        log_P = torch.cat([self.log_conditionals(Z[s:s + chunk], J) for s in range(0, len(Z), chunk)])
        return -(w @ log_P) / w.sum() + self.lambd * J.square().sum()

    # ================= Training ================= #

    def fit(
        self,
        msa: MSA,
        n_epochs: int = 1000,
        batch_size: int = 1000,
        eta: float = 0.005,
        val_frac: float = 0.2,
        patience: int = 10,
        delta: float = 1e-3,
    ) -> "AttentionDCA":
        r"""
        Adam on weighted mini-batches, early-stopped on a held-out part of the alignment.

        An epoch counts as an improvement when the validation objective drops by more than
        ``delta``; training stops after ``patience`` epochs without one and the best parameters
        are restored.

        Parameters
        ----------
        msa : the training alignment.
        n_epochs : upper bound on the number of epochs.
        batch_size : sequences per mini-batch.
        eta : Adam learning rate.
        val_frac : fraction of sequences held out for early stopping.
        patience, delta : early-stopping rule (see above).
        """
        generator = torch.Generator().manual_seed(self.seed)
        Z = torch.from_numpy(msa.Z).long().to(self.device)                  # (M, L)
        w = torch.from_numpy(msa.w).to(self.device, self.Q.dtype)           # (M,)

        order     = torch.randperm(msa.M, generator=generator)
        n_val     = int(val_frac * msa.M)
        val_idx   = order[:n_val]
        train_idx = order[n_val:]

        optimizer = torch.optim.Adam(self.parameters(), lr=eta)
        best_loss, best_epoch, best_state = float("inf"), 0, None
        start = time.time()

        for epoch in range(1, n_epochs + 1):
            shuffled = train_idx[torch.randperm(len(train_idx), generator=generator)]
            train_loss = 0.0
            for batch in shuffled.split(batch_size):
                optimizer.zero_grad()
                l = self.loss(Z[batch], w[batch])
                l.backward()
                optimizer.step()
                train_loss += l.item() * len(batch) / len(train_idx)

            with torch.no_grad():
                val_loss = self.loss(Z[val_idx], w[val_idx], chunk=batch_size).item()
            self.history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                                 "seconds": time.time() - start})
            print(f"Epoch {epoch:4d} : train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"({time.time() - start:.0f}s)", flush=True)

            if val_loss < best_loss - delta:
                best_loss, best_epoch, best_state = val_loss, epoch, copy.deepcopy(self.state_dict())
            elif epoch - best_epoch >= patience:
                print(f"Early stopping: no improvement since epoch {best_epoch}.")
                break

        self.load_state_dict(best_state)
        self.fit_config = {"n_epochs": n_epochs, "batch_size": batch_size, "eta": eta,
                           "val_frac": val_frac, "patience": patience, "delta": delta,
                           "best_epoch": best_epoch, "best_val_loss": best_loss}
        return self

    # ================= I/O ================= #

    def save(self, path: str) -> None:
        r"""Writes ``params.npz``, ``config.json`` and ``history.csv`` to the directory ``path``."""
        os.makedirs(path, exist_ok=True)
        np.savez_compressed(os.path.join(path, "params.npz"),
                            **{name: p.detach().cpu().numpy() for name, p in self.named_parameters()})
        config = {"L": self.L, "H": self.H, "d": self.d, "q": self.q, "lambd": self.lambd,
                  "seed": self.seed, "dtype": str(self.Q.dtype), **self.fit_config}
        with open(os.path.join(path, "config.json"), "w") as fh:
            json.dump(config, fh, indent=2)
        if self.history:
            rows = np.array([list(row.values()) for row in self.history])
            np.savetxt(os.path.join(path, "history.csv"), rows, delimiter=",", comments="",
                       header=",".join(self.history[0]), fmt=["%d", "%.5f", "%.5f", "%.1f"])

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "AttentionDCA":
        r"""Restores a model written by :meth:`save`."""
        with open(os.path.join(path, "config.json")) as fh:
            config = json.load(fh)
        model = cls(L=config["L"], H=config["H"], d=config["d"], q=config["q"], lambd=config["lambd"],
                    seed=config["seed"], device=device, dtype=getattr(torch, config["dtype"].split(".")[1]))
        params = np.load(os.path.join(path, "params.npz"))
        model.load_state_dict({name: torch.from_numpy(params[name]) for name in params.files})
        model.fit_config = {k: config[k] for k in config if k not in ("L", "H", "d", "q", "lambd", "seed", "dtype")}
        history_path = os.path.join(path, "history.csv")
        if os.path.exists(history_path):
            rows = np.genfromtxt(history_path, delimiter=",", names=True)
            model.history = [dict(zip(rows.dtype.names, map(float, row))) for row in np.atleast_1d(rows)]
        return model
