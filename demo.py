"""DP-RAG minimal end-to-end demo.

Runs the 6-step history-aware DP-RAG pipeline on a hard-coded multi-turn
attack trajectory and prints every intermediate artefact to stdout.

Usage:
    python demo.py --model qwen3-8b --eps 3.0
"""

from func import (
    get_client, load_json,
    init_ledger, snapshot, decay_exposure,
    rewrite_query, bm25_retrieve, assess_risk,
    transform_evidence, privacy_decode, update_ledger,
    budget_exhausted, delta_eps,
)
from args import get_parser


# ---------- pretty printing helpers ----------

C_DIM = "\033[2m"; C_END = "\033[0m"
C_CYAN = "\033[36m"; C_YEL = "\033[33m"
C_GRN = "\033[32m"; C_RED = "\033[31m"; C_MAG = "\033[35m"


def banner(title: str):
    bar = "=" * 64
    print(f"\n{C_CYAN}{bar}\n{title}\n{bar}{C_END}")


def step(tag: str, content: str, color: str = C_YEL):
    print(f"{color}[{tag}]{C_END} {content}")


# ---------- main ----------

def main():
    args = get_parser().parse_args()
    client = get_client(args.base_url, args.api_key)

    docs = load_json(args.docs)
    turns = load_json(args.turns)
    ledger = init_ledger(docs, eps_total=args.eps)

    history = []

    for t, raw_q in enumerate(turns, start=1):
        banner(f"Round {t}")
        print(f"{C_MAG}[User]{C_END} {raw_q}")

        # exposure decays slightly between turns (recent leaks weigh more)
        if t > 1:
            decay_exposure(ledger, args.decay)

        # ---- Step 1: query rewriting -------------------------------
        q = rewrite_query(client, args.model, raw_q, history)
        step("Step1 Rewrite", q)

        # ---- Step 2a: retrieval ------------------------------------
        cands = bm25_retrieve(q, docs, top_k=args.top_k)
        cand_ids = [d["id"] for d in cands]
        step("Step2 Retrieve", f"top-{args.top_k} = {cand_ids}")

        # ---- Step 2b: history-aware risk ---------------------------
        risk, reason = assess_risk(q, cands, ledger,
                                   history_len=len(history),
                                   eps_total=args.eps)
        risk_tag = "HIGH" if risk >= args.risk_high else "LOW"
        step("Step2 Risk", f"{risk:.2f} [{risk_tag}]  {C_DIM}{reason}{C_END}")
        step("Ledger@before", snapshot(ledger), color=C_DIM)

        # ---- Budget / sufficiency gate -----------------------------
        if not cands or budget_exhausted(ledger, cands, args.budget_floor):
            ans = "[Refused] privacy budget exhausted for relevant documents."
            step("Step4 Answer", ans, color=C_RED)
            history.append((raw_q, ans))
            step("Ledger@after", snapshot(ledger), color=C_DIM)
            continue

        # ---- Step 3: evidence transformation -----------------------
        evidence = transform_evidence(cands, risk, args.risk_high)
        for e in evidence:
            tag = "redacted" if e["redacted"] else "kept"
            step(f"Step3 Transform [{e['id']}:{tag}]",
                 e["text"][:160] + ("…" if len(e["text"]) > 160 else ""),
                 color=C_GRN if e["redacted"] else C_DIM)

        # ---- Step 4: privacy-aware decoding ------------------------
        eps_left = min(ledger[d["id"]]["remaining_budget"] for d in cands)
        ans = privacy_decode(client, args.model, q, evidence,
                             eps_remaining=eps_left)
        step("Step4 Answer", ans, color=C_GRN)

        # ---- Step 5: ledger update ---------------------------------
        deltas = update_ledger(ledger, cands, evidence, risk, was_aligned=True)
        step("Step5 Update",
             f"delta_eps={delta_eps(risk)}  per-doc={deltas}", color=C_YEL)
        step("Ledger@after", snapshot(ledger), color=C_DIM)

        history.append((raw_q, ans))

    # ---- Final summary --------------------------------------------
    banner("Final Ledger")
    for did, row in ledger.items():
        print(f"  {did}")
        for k, v in row.items():
            print(f"    {k:>18}: {v}")

    banner("Conversation Trace")
    for i, (q, a) in enumerate(history, start=1):
        print(f"  Round {i}")
        print(f"    Q: {q}")
        print(f"    A: {a}")


if __name__ == "__main__":
    main()
