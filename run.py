"""
run.py — train AttentionDCA on an alignment and evaluate its contact prediction.

Load-or-compute: the sequence weights are cached next to the FASTA file and the model under
``results/<run>/``, so running this again only redoes the evaluation and the figures. Delete a run
directory to retrain it.

Example
-------
    python run.py                                   # the RR alignment and CheY, as in the README
    python run.py --H 200 --d 50 --threads 8
    python run.py --fasta my_family.fasta --pdb my_structure.pdb --chain B --dimer ""
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from attention_dca import (MSA, AttentionDCA, attention_score, distance_map, far_pairs_across_interface,
                           frobenius_score, plot_attention_maps, plot_contact_maps, plot_ppv,
                           plot_salt_bridge_head, plot_training, ppv_curve, rank_contacts,
                           salt_bridge_heads, secondary_structure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fasta", default="data/RR_50000.fasta.gz", help="aligned FASTA file, optionally gzipped")
    parser.add_argument("--pdb", default="data/3chy.pdb", help="structure used as the ground truth")
    parser.add_argument("--chain", default="A", help="chain within --pdb")
    parser.add_argument("--dimer", default="data/1zes.pdb", help="homodimer structure; an empty string skips this check")
    parser.add_argument("--dimer-chains", nargs=2, default=["B", "C"], help="the two chains forming the interface")
    parser.add_argument("--H", type=int, default=128, help="number of attention heads")
    parser.add_argument("--d", type=int, default=17, help="inner dimension of Q and K")
    parser.add_argument("--lambd", type=float, default=1e-3, help="L2 penalty on the couplings")
    parser.add_argument("--eta", type=float, default=0.005, help="Adam learning rate")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--n-epochs", type=int, default=1000, help="upper bound; early stopping usually ends training")
    parser.add_argument("--patience", type=int, default=10, help="epochs without improvement before stopping")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4, help="CPU threads used by PyTorch")
    parser.add_argument("--out", default="results", help="root directory of the run directories")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)

    msa = MSA(args.fasta)

    # ── load-or-train ──
    family  = os.path.basename(args.fasta).split(".")[0]
    run_dir = os.path.join(args.out, f"{family}_H{args.H}_d{args.d}_s{args.seed}")
    if os.path.exists(os.path.join(run_dir, "params.npz")):
        print(f"Loading the trained model from {run_dir}")
        model = AttentionDCA.load(run_dir)
    else:
        model = AttentionDCA(L=msa.L, H=args.H, d=args.d, lambd=args.lambd, seed=args.seed)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Training      : H={args.H}, d={args.d}, {n_params} parameters on {model.device}")
        model.fit(msa, n_epochs=args.n_epochs, batch_size=args.batch_size, eta=args.eta, patience=args.patience)
        model.save(run_dir)

    # ── evaluation against the crystal structure ──
    D  = distance_map(args.pdb, msa, args.chain)
    ss = secondary_structure(args.pdb, msa, args.chain)
    scores = {"couplings": frobenius_score(model), "attention": attention_score(model)}

    metrics = {}
    for name, S in scores.items():
        ppv = ppv_curve(rank_contacts(S), D)
        metrics[name] = {f"PPV@{label}": round(float(ppv[n - 1]), 3)
                         for label, n in [("L/2", msa.L // 2), ("L", msa.L), ("2L", 2 * msa.L), ("3L", 3 * msa.L)]}
        print(f"{name:13s} : " + "  ".join(f"{k}={v:.3f}" for k, v in metrics[name].items()))

    # ── looking at what the "errors" and the single heads are ──
    D_dimer = None
    if args.dimer:
        D_dimer = distance_map(args.dimer, msa, args.dimer_chains[0], partner=args.dimer_chains[1])
        metrics["far pairs in the top 3L"] = far_pairs_across_interface(rank_contacts(scores["couplings"])[:3 * msa.L], D, D_dimer)
        print(f"Dimer         : {metrics['far pairs in the top 3L']}")

    r, cov = salt_bridge_heads(model, msa)
    metrics["salt bridge heads"] = {"best_head": int(r.argmax()), "best_r": round(float(r.max()), 3),
                                    "r_vs_charge_covariance": round(float(np.corrcoef(r, cov)[0, 1]), 3)}
    print(f"Heads         : {metrics['salt bridge heads']}")

    with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    # ── figures ──
    plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), "options.mplstyle"))
    plot_contact_maps(scores, D, n_pred=2 * msa.L, ss=ss, out_path=os.path.join(run_dir, "contact_maps.png"))
    plot_ppv(scores, D, out_path=os.path.join(run_dir, "ppv.png"))
    plot_attention_maps({"all heads, averaged": scores["attention"],
                         f"top {msa.L} entries of each head, maximum": attention_score(model, k=msa.L)},
                        D, ss=ss, out_path=os.path.join(run_dir, "attention_maps.png"))
    plot_salt_bridge_head(model, msa, D, ss=ss, out_path=os.path.join(run_dir, "salt_bridge_head.png"))
    if D_dimer is not None:
        plot_contact_maps({"couplings": scores["couplings"]}, D, n_pred=3 * msa.L, D_partner=D_dimer, ss=ss,
                          figsize=(6.2, 7.7), out_path=os.path.join(run_dir, "dimer_contacts.png"))
    if model.history:
        plot_training(model.history, model.fit_config["best_epoch"], out_path=os.path.join(run_dir, "training.png"))


if __name__ == "__main__":
    main()
