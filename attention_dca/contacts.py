"""
attention_dca.contacts — from a trained model to a predicted contact map.

Two contact scores are available: the classical DCA score on the couplings
(:func:`frobenius_score`) and the attention matrices read directly (:func:`attention_score`).
Both are ranked and evaluated against a crystal structure in the same way. The factored model also
allows a look inside single heads, since ``V^h`` says what a head rewards and ``A^h`` says where
(:func:`salt_bridge_heads`).
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .model import AttentionDCA
from .msa import AMINO_ACIDS, GAP, MSA


COLOR_TRUE: str = "#0072B2"
COLOR_FALSE: str = "#D55E00"
COLOR_ACROSS: str = "#009E73"
COLOR_STRUCTURE: str = "0.83"
COLOR_PARTNER: str = "#BFE3D8"

_CHARGE: dict[str, float] = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0}
_ORDER_BY_CHEMISTRY: str = "DEKRHNQSTGPACVILMFWY"

_HEAT = LinearSegmentedColormap.from_list("heat", [(0.0, 0.447, 0.698, 0.0), (0.0, 0.25, 0.45, 1.0)])
_DIVERGING = LinearSegmentedColormap.from_list("diverging", [COLOR_FALSE, "#F2F2F2", COLOR_TRUE])


# =================================================================== #
# Contact scores
# =================================================================== #

def zero_sum_gauge(J: np.ndarray, axes: tuple[int, int] = (1, 3)) -> np.ndarray:
    r"""
    Moves couplings to the zero-sum gauge, ``J_ij(a, b) - J_ij(a, ·) - J_ij(·, b) + J_ij(·, ·)``,
    with ``axes`` the two amino-acid axes.

    The model has no local fields, so its couplings also carry the conservation profile. This
    gauge moves that part out, which leaves the couplings of smallest norm for the same model.
    """
    a, b = axes
    return J - J.mean(axis=a, keepdims=True) - J.mean(axis=b, keepdims=True) + J.mean(axis=axes, keepdims=True)


def average_product_correction(S: np.ndarray) -> np.ndarray:
    r"""``S_ij - S_i S_j / S_mean``, removing the site-wide background of a score matrix."""
    L = S.shape[0]
    return S - np.outer(S.sum(axis=1), S.sum(axis=0)) / (S.sum() * (1.0 - 1.0 / L))


def frobenius_score(model: AttentionDCA) -> np.ndarray:
    r"""
    The classical DCA score, shape ``(L, L)``: symmetrise the couplings,
    ``J_ij(a, b) <- (J_ij(a, b) + J_ji(b, a)) / 2``, move to zero-sum gauge, take the Frobenius
    norm over the 20 amino acids (gaps excluded), and apply the average product correction.
    """
    with torch.no_grad():
        J = model.couplings().cpu().numpy().astype(np.float64)          # (L, q, L, q)
    J = 0.5 * (J + J.transpose(2, 3, 0, 1))
    F = np.sqrt((zero_sum_gauge(J)[:, :GAP, :, :GAP] ** 2).sum(axis=(1, 3)))    # (L, L)
    return average_product_correction(F)


def attention_score(model: AttentionDCA, k: Optional[int] = None) -> np.ndarray:
    r"""
    Contact score read from the attention matrices alone, shape ``(L, L)``: symmetrise each head,
    aggregate over the heads, and apply the average product correction.

    Parameters
    ----------
    model : trained model.
    k : ``None`` averages the full matrices. Otherwise each head only keeps its ``k`` largest
        entries and the heads are combined by their maximum, which tests how sparse the
        structural information within a head is.
    """
    with torch.no_grad():
        A = model.attention()                                           # (H, L, L)
    A = 0.5 * (A + A.transpose(1, 2))
    if k is None:
        S = A.mean(dim=0)
    else:
        kth_largest = A.flatten(1).topk(k, dim=1).values[:, -1]         # (H,)
        S = (A * (A >= kth_largest[:, None, None])).amax(dim=0)
    return average_product_correction(S.cpu().numpy().astype(np.float64))


# =================================================================== #
# Evaluation
# =================================================================== #

def rank_contacts(S: np.ndarray, min_sep: int = 6) -> np.ndarray:
    r"""Pairs ``(i, j)`` with ``j - i >= min_sep`` by decreasing score, shape ``(n_pairs, 2)``."""
    i, j = np.triu_indices(S.shape[0], k=min_sep)
    order = np.argsort(-S[i, j])
    return np.stack([i[order], j[order]], axis=1)


def ppv_curve(pairs: np.ndarray, D: np.ndarray, cutoff: float = 8.0) -> np.ndarray:
    r"""
    Positive predictive value ``PPV(n) = TP(n) / n`` of the ``n`` best-ranked pairs. Pairs without
    a known distance do not count as predictions.

    Parameters
    ----------
    pairs : ranked pairs from :func:`rank_contacts`.
    D : residue distances of shape ``(L, L)``, ``NaN`` where unknown.
    cutoff : a pair closer than this many Angstrom is a true contact.
    """
    d = D[pairs[:, 0], pairs[:, 1]]
    d = d[~np.isnan(d)]
    return np.cumsum(d < cutoff) / np.arange(1, len(d) + 1)


def far_pairs_across_interface(pairs: np.ndarray, D: np.ndarray, D_partner: np.ndarray, far: float = 12.0,
                               cutoff: float = 8.0, min_sep: int = 6) -> dict[str, float]:
    r"""
    How many predictions that are far apart within the chain are in contact across an interface.

    Parameters
    ----------
    pairs : the predicted pairs to test.
    D, D_partner : distances within the chain and across the interface (:func:`distance_map`).
    far : a pair at least this far apart within the chain is too far for a near-miss.
    cutoff : contact cutoff across the interface.
    min_sep : sequence separation of the background set of pairs.

    Returns
    -------
    dict — ``n_far`` predictions, ``n_across`` of them on the interface, and ``background``, the
    fraction of all far pairs that are on the interface.
    """
    i, j = pairs.T
    is_far = D[i, j] >= far
    i_all, j_all = np.triu_indices(D.shape[0], k=min_sep)
    return {"n_far": int(is_far.sum()),
            "n_across": int((D_partner[i, j][is_far] < cutoff).sum()),
            "background": float((D_partner[i_all, j_all][D[i_all, j_all] >= far] < cutoff).mean())}


def salt_bridge_heads(model: AttentionDCA, msa: MSA, n_pairs: int = 20, min_sep: int = 6) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Tests from two independent angles whether heads have learned electrostatics.

    Parameters
    ----------
    model : trained model.
    msa : the training alignment.
    n_pairs : number of a head's strongest pairs to look at.
    min_sep : minimum sequence separation of those pairs.

    Returns
    -------
    r : what each head rewards, the correlation of its symmetrised, zero-sum ``V^h`` with the
        opposite-charge template ``-c_a c_b``, shape ``(H,)``.
    cov : where each head looks, the covariance of the charges ``Cov(c_i, c_j)`` in the alignment
        averaged over the head's strongest pairs, shape ``(H,)``. Negative means that opposite
        charges co-occur at these positions.
    """
    with torch.no_grad():
        A = model.attention().cpu().numpy().astype(np.float64)          # (H, L, L)
        V = model.V.cpu().numpy().astype(np.float64)[:, :GAP, :GAP]     # (H, 20, 20)
    c = np.array([_CHARGE.get(a, 0.0) for a in AMINO_ACIDS])            # (20,)

    V = zero_sum_gauge(0.5 * (V + V.transpose(0, 2, 1)), axes=(1, 2))
    template = zero_sum_gauge(-np.outer(c, c)[None], axes=(1, 2))
    r = np.corrcoef(np.vstack([template.ravel(), V.reshape(V.shape[0], -1)]))[0, 1:]

    w = msa.w / msa.w.sum()
    c_Z = np.append(c, 0.0)[msa.Z]                                      # (M, L), a gap is neutral
    c_Z = c_Z - w @ c_Z
    C = c_Z.T @ (c_Z * w[:, None])                                      # (L, L)

    i, j = np.triu_indices(msa.L, k=min_sep)
    top = np.argsort(-(A + A.transpose(0, 2, 1))[:, i, j], axis=1)[:, :n_pairs]     # (H, n_pairs)
    return r, C[i[top], j[top]].mean(axis=1)


# =================================================================== #
# Figures
# =================================================================== #

def plot_contact_maps(scores: dict[str, np.ndarray], D: np.ndarray, n_pred: int, D_partner: Optional[np.ndarray] = None,
                      ss: Optional[list] = None, cutoff: float = 8.0, min_sep: int = 6,
                      out_path: Optional[str] = None, figsize=(9, 5.4)):
    r"""
    The ``n_pred`` best pairs of each score on top of the crystal structure, one panel per score.

    Predictions sit in the upper triangle, so the lower triangle keeps the plain structure for
    comparison. With ``D_partner``, the contacts across a homo-oligomer interface are drawn as well
    and false positives lying on them get their own marker.
    """
    fig, axes = plt.subplots(1, len(scores), figsize=figsize, squeeze=False)
    for ax, (name, S) in zip(axes[0], scores.items()):
        pairs = rank_contacts(S, min_sep)
        i, j = pairs[~np.isnan(D[pairs[:, 0], pairs[:, 1]])][:n_pred].T
        hit = D[i, j] < cutoff
        across = ~hit & (D_partner[i, j] < cutoff) if D_partner is not None else np.zeros_like(hit)

        _draw_structure(ax, D, cutoff, ss, D_partner)
        ax.scatter(j[hit], i[hit], s=9, marker="o", color=COLOR_TRUE, linewidths=0)
        ax.scatter(j[across], i[across], s=14, marker="^", color=COLOR_ACROSS, linewidths=0)
        ax.scatter(j[~hit & ~across], i[~hit & ~across], s=11, marker="x", color=COLOR_FALSE, linewidths=0.9)
        ax.set_title(f"{name}, top {n_pred}: PPV = {hit.mean():.2f}", pad=26 if ss else None)

    handles = [Line2D([], [], ls="", marker="o", ms=4, color=COLOR_TRUE, label="predicted, true contact"),
               Line2D([], [], ls="", marker="x", ms=4, color=COLOR_FALSE, label="predicted, no contact"),
               Line2D([], [], ls="", marker="s", ms=4, color=COLOR_STRUCTURE, label=f"structure (< {cutoff:g} Å)")]
    if D_partner is not None:
        handles += [Line2D([], [], ls="", marker="^", ms=4, color=COLOR_ACROSS, label="predicted, across the dimer"),
                    Line2D([], [], ls="", marker="s", ms=4, color=COLOR_PARTNER, label="dimer interface")]
    fig.legend(handles=handles, loc="outside lower center", ncols=3 if D_partner is None else 2)
    if out_path:
        fig.savefig(out_path, dpi=200)
    return fig, axes


def plot_attention_maps(scores: dict[str, np.ndarray], D: np.ndarray, ss: Optional[list] = None, cutoff: float = 8.0,
                        out_path: Optional[str] = None, figsize=(9, 4.9)):
    r"""
    Attention-derived score matrices over the structure: blue on gray means attention on a real
    contact. Each panel is scaled to its own 99.8th percentile, so the colours compare the pattern
    within a panel and not the magnitude between panels.
    """
    fig, axes = plt.subplots(1, len(scores), figsize=figsize, squeeze=False)
    for ax, (name, S) in zip(axes[0], scores.items()):
        _draw_structure(ax, D, cutoff, ss)
        S = np.where(np.eye(S.shape[0], dtype=bool), 0.0, S)
        ax.imshow(S, cmap=_HEAT, vmin=0.0, vmax=np.quantile(S, 0.998), interpolation="nearest")
        ax.set_title(name, pad=26 if ss else None)
    if out_path:
        fig.savefig(out_path, dpi=200)
    return fig, axes


def plot_salt_bridge_head(model: AttentionDCA, msa: MSA, D: np.ndarray, ss: Optional[list] = None, cutoff: float = 8.0,
                          out_path: Optional[str] = None, figsize=(14, 4.6)):
    r"""
    The head most similar to electrostatics: what it rewards (``V^h``), where it looks (``A^h``),
    and how those two relate across all heads (:func:`salt_bridge_heads`).
    """
    r, cov = salt_bridge_heads(model, msa)
    h = int(np.argmax(r))
    with torch.no_grad():
        A = model.attention()[h].cpu().numpy()                          # (L, L)
        V = model.V[h].cpu().numpy()[:GAP, :GAP]                        # (20, 20)

    fig, (ax_V, ax_A, ax_all) = plt.subplots(1, 3, figsize=figsize)

    order = [AMINO_ACIDS.index(a) for a in _ORDER_BY_CHEMISTRY]
    V = zero_sum_gauge(0.5 * (V + V.T)[None], axes=(1, 2))[0][np.ix_(order, order)]
    vmax = np.abs(V).max()
    image = ax_V.imshow(V, cmap=_DIVERGING, vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax_V.set(xticks=range(GAP), xticklabels=_ORDER_BY_CHEMISTRY, yticks=range(GAP), yticklabels=_ORDER_BY_CHEMISTRY)
    ax_V.tick_params(labelsize=8)
    ax_V.grid(False)
    ax_V.minorticks_off()
    ax_V.set_title(f"What head {h} rewards: $V^{{{h}}}$", pad=26 if ss else None)
    fig.colorbar(image, ax=ax_V, shrink=0.8, pad=0.02, label="zero-sum value")

    _draw_structure(ax_A, D, cutoff, ss)
    A = 0.5 * (A + A.T)
    ax_A.imshow(A, cmap=_HEAT, vmin=0.0, vmax=np.quantile(A, 0.998), interpolation="nearest")
    ax_A.set_title(f"Where it looks: $A^{{{h}}}$", pad=26 if ss else None)

    ax_all.scatter(r, cov, s=22, marker="o", color=COLOR_TRUE, edgecolors="white", linewidths=0.5)
    ax_all.annotate(f"head {h}", (r[h], cov[h]), xytext=(-6, 0), textcoords="offset points", ha="right", va="center")
    ax_all.set(xlabel=r"corr$(V^h,\ -c_a c_b)$", ylabel=r"Cov$(c_i, c_j)$ of the head's top pairs")
    ax_all.set_title(f"All {len(r)} heads: r = {np.corrcoef(r, cov)[0, 1]:.2f}", pad=26 if ss else None)
    if out_path:
        fig.savefig(out_path, dpi=200)
    return fig, (ax_V, ax_A, ax_all)


def plot_ppv(scores: dict[str, np.ndarray], D: np.ndarray, cutoff: float = 8.0, min_sep: int = 6,
             out_path: Optional[str] = None, figsize=(5.5, 3.8)):
    r"""
    PPV against the number of predictions, one line per score. The gray line is a perfect predictor,
    which is limited only by how many contacts the structure has.
    """
    i, j = np.triu_indices(D.shape[0], k=min_sep)
    known = ~np.isnan(D[i, j])
    by_distance = np.stack([i[known], j[known]], axis=1)[np.argsort(D[i, j][known])]
    ideal = ppv_curve(by_distance, D, cutoff)

    fig, ax = plt.subplots(figsize=figsize)
    ax.semilogx(np.arange(1, len(ideal) + 1), ideal, color="0.7", lw=1.5, label="perfect predictor")
    for name, S in scores.items():
        ppv = ppv_curve(rank_contacts(S, min_sep), D, cutoff)
        ax.semilogx(np.arange(1, len(ppv) + 1), ppv, lw=2, label=name)
    ax.axvline(D.shape[0], color="0.5", lw=0.8)
    ax.annotate("n = L", (D.shape[0], 0.03), xytext=(4, 0), textcoords="offset points")
    ax.set(xlabel="Number of predictions n", ylabel="PPV(n)", ylim=(0, 1.02), xlim=(1, len(ideal)))
    ax.legend(loc="lower left")
    if out_path:
        fig.savefig(out_path, dpi=200)
    return fig, ax


def plot_training(history: list[dict[str, float]], best_epoch: int, out_path: Optional[str] = None,
                  figsize=(5.5, 3.4)):
    r"""Training and validation objective over the epochs, with the restored epoch marked."""
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(epochs, [row["train_loss"] for row in history], lw=2, label="training")
    ax.plot(epochs, [row["val_loss"] for row in history], lw=2, label="validation")
    ax.axvline(best_epoch, color="0.5", lw=0.8)
    ax.annotate("restored", (best_epoch, 1.0), xycoords=("data", "axes fraction"), xytext=(-4, -14),
                textcoords="offset points", ha="right")
    ax.set(xlabel="Epoch", ylabel="Objective",                          # zoomed past the first steep epochs
           ylim=(0.98 * history[-1]["train_loss"], 1.1 * min(row["val_loss"] for row in history)))
    ax.legend(loc="upper center")
    if out_path:
        fig.savefig(out_path, dpi=200)
    return fig, ax


def _draw_structure(ax, D: np.ndarray, cutoff: float, ss: Optional[list] = None, D_partner: Optional[np.ndarray] = None,
                    tick_step: int = 20) -> None:
    r"""Draws the true contacts as a background, with 1-based column ticks and a secondary structure strip."""
    L = D.shape[0]
    background = (D < cutoff).astype(int)
    colors = ["white", COLOR_STRUCTURE]
    if D_partner is not None:
        background[(D_partner < cutoff) & (background == 0)] = 2
        colors.append(COLOR_PARTNER)
    ax.imshow(background, cmap=ListedColormap(colors), vmin=0, vmax=len(colors) - 1, interpolation="nearest")

    labels = [1, *range(tick_step, L + 1, tick_step)]
    ticks = [label - 1 for label in labels]
    ax.set(xlabel="Alignment column", ylabel="Alignment column", xticks=ticks, xticklabels=labels,
           yticks=ticks, yticklabels=labels)
    ax.grid(False)
    ax.minorticks_off()

    for label, first, last in ss or []:
        height = 0.022 if label.startswith("α") else 0.012
        ax.add_patch(Rectangle((first - 0.5, 1.012), last - first + 1, height, color="0.35", clip_on=False,
                               transform=ax.get_xaxis_transform()))
        ax.text(0.5 * (first + last), 1.04, label, ha="center", va="bottom", fontsize=8,
                transform=ax.get_xaxis_transform())
