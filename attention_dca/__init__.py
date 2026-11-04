"""
attention_dca — a standalone PyTorch implementation of AttentionDCA for protein contact prediction.

Files, one concept each:

* ``msa``       — the **``MSA``**: FASTA reading, filtering and sequence weights.
* ``model``     — **``AttentionDCA``**: the factored-attention Potts model, its objective, the
  training loop and I/O.
* ``structure`` — the ground truth: PDB parsing and the mapping of a chain onto the MSA columns.
* ``contacts``  — contact scores, PPV evaluation, the look inside single heads, and the figures.

Typical usage
-------------
    msa   = MSA("data/RR_50000.fasta.gz")
    model = AttentionDCA(L=msa.L, H=128, d=17).fit(msa)
    S     = frobenius_score(model)
    D     = distance_map("data/3chy.pdb", msa)
    ppv   = ppv_curve(rank_contacts(S), D)
"""

from __future__ import annotations

from .msa import MSA, LETTER_TO_NUM_AA, AMINO_ACIDS, GAP, Q_AA, read_fasta, compute_theta, compute_weights
from .model import AttentionDCA
from .structure import read_chain, align_to_profile, distance_map, secondary_structure
from .contacts import (
    zero_sum_gauge,
    average_product_correction,
    frobenius_score,
    attention_score,
    rank_contacts,
    ppv_curve,
    far_pairs_across_interface,
    salt_bridge_heads,
    plot_contact_maps,
    plot_attention_maps,
    plot_salt_bridge_head,
    plot_ppv,
    plot_training,
)


__all__ = [
    # msa
    "MSA", "LETTER_TO_NUM_AA", "AMINO_ACIDS", "GAP", "Q_AA", "read_fasta", "compute_theta", "compute_weights",
    # model
    "AttentionDCA",
    # structure
    "read_chain", "align_to_profile", "distance_map", "secondary_structure",
    # contacts
    "zero_sum_gauge", "average_product_correction", "frobenius_score", "attention_score",
    "rank_contacts", "ppv_curve", "far_pairs_across_interface", "salt_bridge_heads",
    "plot_contact_maps", "plot_attention_maps", "plot_salt_bridge_head", "plot_ppv", "plot_training",
]
