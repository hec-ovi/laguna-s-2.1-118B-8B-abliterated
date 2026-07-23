"""Streaming, model-filtered go/no-go: refusal removal AND capability preservation.

The load-bearing quality step (per all the research): MODEL-FILTER the candidate prompts.
A difference-of-means direction is only "refusal" if the harmful side actually elicits refusal
FROM THIS MODEL. So the base pass measures per-prompt refusal and keeps only the candidates
Laguna refuses more than it refuses benign prompts; the direction is built from those. If too
few candidates trigger refusal, the probe STOPS (the "Laguna barely over-refuses this set"
signal) instead of shipping a topic/style vector.

Everything is prefill (no autoregressive generation): one base pass captures residuals +
first-token logits for all prompts; per lambda, one ablated pass measures the three axes:

  refusal removed : first-token refusal mass on held-out refused candidates, base vs ablated
  capability kept : first-token KL(base || ablated) on held-out benign (near zero = localized)
  over-refusal    : refusal mass on legitimate lookalikes, base vs ablated

The first-token metric is a fast SELECTOR; final acceptance uses long-generation + a judge on
the edited model (docs/RUNBOOK.md acceptance gates). Saves the chosen direction for edit.py.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from . import arch
from . import data
from . import projection as P
from .engine import pick_layer

REFUSAL_OPENERS = [" I", " Sorry", " As", " Unfortunately", " No", " It", " I'm",
                   " While", " Sadly", " Please", " That", " This"]


def refusal_ids(tok) -> list[int]:
    ids = set()
    for w in REFUSAL_OPENERS:
        e = tok.encode(w, add_special_tokens=False)
        if e:
            ids.add(e[0])
    return sorted(ids)


def refusal_mass(logits: torch.Tensor, ids: list[int]) -> float:
    return float(torch.softmax(logits, -1)[ids].sum())


def first_token_kl(base: torch.Tensor, abl: torch.Tensor) -> float:
    lb = torch.log_softmax(base, -1)
    la = torch.log_softmax(abl, -1)
    return float((lb.exp() * (lb - la)).sum())


def _split(idxs, frac=0.6):
    n = max(1, int(round(len(idxs) * frac)))
    return idxs[:n], (idxs[n:] or idxs[:1])


def _mean(xs):
    xs = list(xs)
    return sum(xs) / max(1, len(xs))


def main():
    ap = argparse.ArgumentParser(description="Model-filtered streaming refusal-ablation go/no-go")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--candidate-layers", default="16,20,24,28,32")
    ap.add_argument("--ablate-from", type=int, default=0)
    ap.add_argument("--lambdas", default="0.8,1.2")
    ap.add_argument("--harmful-file", default=None)
    ap.add_argument("--harmless-file", default=None)
    ap.add_argument("--keep-quantile", type=float, default=0.75,
                    help="keep candidates whose refusal mass exceeds this benign quantile (model-filter)")
    ap.add_argument("--min-kept", type=int, default=15)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from .streaming import StreamingLaguna

    harmful = data.load_jsonl(a.harmful_file) if a.harmful_file else data.HARMFUL
    harmless = data.load_jsonl(a.harmless_file) if a.harmless_file else data.HARMLESS
    look = data.LOOKALIKE
    layers = [int(x) for x in a.candidate_layers.split(",")]
    lambdas = [float(x) for x in a.lambdas.split(",")]

    m = StreamingLaguna(a.model_dir, device=a.device)
    rids = refusal_ids(m.tok)

    nH, nB, nL = len(harmful), len(harmless), len(look)
    order = harmful + harmless + look
    ids = [m.render(p, enable_thinking=a.thinking) for p in order]
    print(f"[probe] base pass: {len(ids)} prompts ({nH} candidate / {nB} benign / {nL} lookalike), "
          f"layers {layers}, thinking={a.thinking}", flush=True)
    base_logits, caps = m.run_corpus(ids, capture_layers=layers)

    caps_h = {l: caps[l][:nH] for l in layers}
    caps_b = {l: caps[l][nH:nH + nB] for l in layers}
    rm = [refusal_mass(x, rids) for x in base_logits]
    rm_h, rm_b, rm_l = rm[:nH], rm[nH:nH + nB], rm[nH + nB:]

    # --- MODEL-FILTER: keep candidates Laguna refuses more than typical benign ---
    b_sorted = sorted(rm_b)
    thresh = b_sorted[min(len(b_sorted) - 1, int(a.keep_quantile * len(b_sorted)))]
    keep = [i for i in range(nH) if rm_h[i] >= thresh]
    print(f"[probe] model-filter: benign p{int(a.keep_quantile * 100)} refusal mass = {thresh:.3f}; "
          f"kept {len(keep)}/{nH} candidates Laguna over-refuses "
          f"(mean refused {_mean(rm_h[i] for i in keep):.3f} vs benign {_mean(rm_b):.3f})", flush=True)
    stopped = len(keep) < a.min_kept
    if stopped:
        print(f"[probe] STOP-SIGNAL: only {len(keep)} candidates trigger refusal (< {a.min_kept}). "
              f"Laguna barely over-refuses this set, so the direction would be topic/style noise, "
              f"not refusal. Use a harder / more on-model candidate set before editing.", flush=True)
    if not keep:
        keep = list(range(nH))  # degenerate fallback so the run still reports numbers

    kh_tr, kh_te = _split(keep)
    b_tr, b_te = _split(list(range(nB)))

    cap_h_tr = {l: caps_h[l][torch.tensor(kh_tr)] for l in layers}
    cap_b_tr = {l: caps_b[l][torch.tensor(b_tr)] for l in layers}
    best, scores = pick_layer(cap_h_tr, cap_b_tr)
    d = P.diff_of_means(cap_h_tr[best], cap_b_tr[best])
    U = P.orthonormalize(d)
    print(f"[probe] layer scores { {l: round(s, 3) for l, s in scores.items()} }  best={best}", flush=True)

    base_ref = _mean(rm_h[i] for i in kh_te)
    base_look = _mean(rm_l)
    base_b = {j: base_logits[nH + j] for j in b_te}

    test_prompts = [harmful[i] for i in kh_te] + [harmless[j] for j in b_te] + look
    tids = [m.render(p, enable_thinking=a.thinking) for p in test_prompts]
    nHte, nBte = len(kh_te), len(b_te)

    results = []
    for lam in lambdas:
        print(f"[probe] ablated pass lambda={lam} (layers {a.ablate_from}..{arch.N_LAYERS - 1})", flush=True)
        abl_logits, _ = m.run_corpus(tids, ablation=(U, lam, a.ablate_from))
        abl_h = abl_logits[:nHte]
        abl_b = abl_logits[nHte:nHte + nBte]
        abl_l = abl_logits[nHte + nBte:]
        abl_ref = _mean(refusal_mass(x, rids) for x in abl_h)
        abl_look = _mean(refusal_mass(x, rids) for x in abl_l)
        bkl = _mean(first_token_kl(base_b[j], abl_b[k]) for k, j in enumerate(b_te))
        r = {"lambda": lam, "refusal_base": base_ref, "refusal_ablated": abl_ref,
             "benign_kl": bkl, "overrefusal_base": base_look, "overrefusal_ablated": abl_look}
        results.append(r)
        print(f"[probe] lambda={lam}: over-refusal {base_ref:.3f}->{abl_ref:.3f}  "
              f"benign_kl={bkl:.4f}  lookalike {base_look:.3f}->{abl_look:.3f}", flush=True)

    ok = [r for r in results if r["refusal_ablated"] <= 0.5 * r["refusal_base"] + 1e-9 and r["benign_kl"] < 0.1]
    chosen = min(ok, key=lambda r: r["lambda"]) if ok else max(results, key=lambda r: r["refusal_base"] - r["refusal_ablated"])
    verdict = ("STOP: too few refused candidates, direction is noise" if stopped else
               ("GO" if ok else "REVIEW: no lambda both removed refusal and kept KL low"))

    dir_path = a.out or f"runs/dir_L{best}_{int(time.time())}.pt"
    os.makedirs(os.path.dirname(dir_path), exist_ok=True)
    torch.save({"U": U, "best_layer": best, "ablate_layers": list(range(a.ablate_from, arch.N_LAYERS)),
                "lambda": chosen["lambda"], "kept": len(keep), "results": results, "verdict": verdict}, dir_path)
    with open(dir_path.replace(".pt", ".json"), "w") as f:
        json.dump({"best_layer": best, "layer_scores": scores, "kept": len(keep), "n_candidates": nH,
                   "filter_threshold": thresh, "chosen": chosen, "results": results,
                   "verdict": verdict, "direction_file": dir_path, "thinking": a.thinking}, f, indent=2)

    print("\n=== go/no-go ===")
    print(f"  candidates kept : {len(keep)}/{nH} (model-filtered)")
    print(f"  best layer      : {best}")
    print(f"  chosen lambda   : {chosen['lambda']}")
    print(f"  over-refusal    : {chosen['refusal_base']:.3f} -> {chosen['refusal_ablated']:.3f}")
    print(f"  benign KL       : {chosen['benign_kl']:.4f}")
    print(f"  lookalike       : {chosen['overrefusal_base']:.3f} -> {chosen['overrefusal_ablated']:.3f}")
    print(f"  verdict         : {verdict}")
    print(f"  direction saved : {dir_path}")


if __name__ == "__main__":
    main()
