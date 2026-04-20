"""DP-RAG demo helpers — 6 step pipeline as plain functions.

This file keeps every step short and readable. The orchestration lives in
demo.py, all stateful objects (history list, ledger dict) are passed in and
out explicitly.
"""

import json
import math
import re
from collections import Counter

from openai import OpenAI


# ---------------------------------------------------------------------------
# LLM wrapper (OpenAI-compatible, talks to the local vLLM server)
# ---------------------------------------------------------------------------

_client = None


def get_client(base_url: str, api_key: str) -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=base_url, api_key=api_key)
    return _client


def chat(client: OpenAI, model: str, prompt: str,
         temperature: float = 0.0, max_tokens: int = 256) -> str:
    """Single-shot chat completion. Strips Qwen3 <think> blocks."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    text = resp.choices[0].message.content or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


# ---------------------------------------------------------------------------
# Ledger (Step 6) — stateful dict, one row per document
# ---------------------------------------------------------------------------

def init_ledger(docs: list, eps_total: float) -> dict:
    return {
        d["id"]: {
            "exposure": 0.0,
            "remaining_budget": eps_total,
            "retrievals": 0,
            "transforms": 0,
            "aligned_decodes": 0,
        }
        for d in docs
    }


def snapshot(ledger: dict) -> str:
    parts = []
    for did, row in ledger.items():
        parts.append(
            f"{did}: E={row['exposure']:.2f} eps={row['remaining_budget']:.2f}"
        )
    return " | ".join(parts)


def decay_exposure(ledger: dict, decay: float) -> None:
    for row in ledger.values():
        row["exposure"] *= decay


# ---------------------------------------------------------------------------
# Step 1 — Query rewriting
# ---------------------------------------------------------------------------

REWRITE_PROMPT = """You rewrite the user's follow-up question into a fully \
self-contained query using the dialogue history. Resolve every pronoun and \
elided phrase, but do not invent facts that are not implied.

Dialogue history:
{history}

Follow-up query: {query}

Output ONLY the rewritten query, in one line."""


def rewrite_query(client, model, raw_q: str, history: list) -> str:
    if not history:
        return raw_q
    hist_str = "\n".join(f"  Round {i+1}: Q: {q}  A: {a}"
                         for i, (q, a) in enumerate(history))
    out = chat(client, model,
               REWRITE_PROMPT.format(history=hist_str, query=raw_q),
               max_tokens=80)
    out = out.split("\n")[0].strip().strip('"').strip("'")
    return out or raw_q


# ---------------------------------------------------------------------------
# Step 2a — BM25 retrieval (toy implementation, no extra deps)
# ---------------------------------------------------------------------------

def _tok(text: str) -> list:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def bm25_retrieve(query: str, docs: list, top_k: int = 2,
                  k1: float = 1.5, b: float = 0.75) -> list:
    corpus = [_tok(d["text"]) for d in docs]
    avgdl = sum(len(c) for c in corpus) / max(1, len(corpus))
    df = Counter()
    for c in corpus:
        df.update(set(c))
    N = len(corpus)
    q_tokens = _tok(query)
    scores = []
    for i, c in enumerate(corpus):
        tf = Counter(c)
        s = 0.0
        for t in q_tokens:
            if t not in tf:
                continue
            idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
            denom = tf[t] + k1 * (1 - b + b * len(c) / avgdl)
            s += idf * tf[t] * (k1 + 1) / denom
        scores.append((s, i))
    scores.sort(reverse=True)
    return [docs[i] for s, i in scores[:top_k] if s > 0]


# ---------------------------------------------------------------------------
# Step 2b — History-aware risk estimation
# ---------------------------------------------------------------------------

_TIER_W = {"high": 1.0, "low": 0.25}


def _span_hits(query: str, doc: dict) -> list:
    """Spans of `doc` referenced by `query`.

    Two ways to "hit" a span:
      - verbatim : the rewritten query already contains the span value (a
        strong cumulative-leak signal: previous turns leaked it back).
      - lexical  : query shares non-stopword tokens with the span (a weaker
        signal: the user is asking *about* this field).
    """
    q_low = query.lower()
    q_toks = set(_tok(query))
    hits = []
    for s in doc["sensitive_spans"]:
        text = s["text"]
        if text.lower() in q_low:
            kind = "verbatim"
        else:
            stoks = [t for t in _tok(text) if len(t) > 2]
            if stoks and any(t in q_toks for t in stoks):
                kind = "lexical"
            else:
                continue
        hits.append({"text": text, "tier": s["tier"], "kind": kind})
    return hits


def _intrinsic_score(hits: list) -> float:
    """Tier-weighted aggregate. Verbatim hits on HIGH spans saturate the
    score immediately because they signal an in-progress cumulative attack."""
    if not hits:
        return 0.0
    score = 0.0
    for h in hits:
        w = _TIER_W[h["tier"]]
        if h["kind"] == "verbatim" and h["tier"] == "high":
            return 1.0
        score += w * (0.7 if h["kind"] == "verbatim" else 0.4)
    return min(1.0, score)


def assess_risk(query: str, cands: list, ledger: dict, history_len: int,
                eps_total: float) -> tuple:
    """Returns (risk in [0,1], reason string).

    Three components:
      - intrinsic : tier-weighted hits of the query on sensitive spans
      - exposure  : prior cumulative exposure on these candidates
      - history   : longer dialogue -> higher chance of cumulative leakage
    """
    if not cands:
        return 0.0, "no candidates retrieved"

    hits_per_doc = {d["id"]: _span_hits(query, d) for d in cands}
    intrinsic = max(
        (_intrinsic_score(hits_per_doc[d["id"]]) for d in cands),
        default=0.0,
    )

    exposure_norm = min(
        1.0,
        max(ledger[d["id"]]["exposure"] for d in cands) / max(1.0, eps_total),
    )

    history_factor = 1.0 - math.exp(-history_len / 3.0)

    risk = 0.55 * intrinsic + 0.30 * exposure_norm + 0.15 * history_factor
    risk = max(0.0, min(1.0, risk))

    def _fmt(h):
        return f"{h['text']}({h['tier'][0]},{h['kind'][0]})"
    hit_parts = [f"{d['id']}:[{','.join(_fmt(h) for h in hits_per_doc[d['id']])}]"
                 for d in cands if hits_per_doc[d["id"]]]
    hit_summary = " ".join(hit_parts) or "no span hit"
    reason = (f"intrinsic={intrinsic:.2f} exposure={exposure_norm:.2f} "
              f"history={history_factor:.2f} | hits={hit_summary}")
    return risk, reason


# ---------------------------------------------------------------------------
# Step 3 — Sensitive evidence transformation
# ---------------------------------------------------------------------------

def transform_evidence(cands: list, risk: float, risk_high: float) -> list:
    """Tier-aware redaction.

    - risk >= risk_high : redact all HIGH-tier spans
    - risk >= risk_high + 0.2 : also redact LOW-tier spans (defence-in-depth)
    """
    aggressive = risk >= risk_high + 0.2
    out = []
    for d in cands:
        text = d["text"]
        replaced = []
        if risk >= risk_high:
            for s in d["sensitive_spans"]:
                if s["tier"] == "high" or aggressive:
                    if s["text"] in text:
                        text = text.replace(s["text"], "[REDACTED]")
                        replaced.append(s["text"])
        out.append({"id": d["id"], "text": text, "redacted": replaced})
    return out


# ---------------------------------------------------------------------------
# Step 4 — Privacy-aware decoding
# ---------------------------------------------------------------------------

ANSWER_PROMPT = """You are a privacy-aware assistant. Answer ONLY using \
the evidence below. Follow these rules strictly:
1. If the evidence contains [REDACTED], do NOT try to recover or guess the \
   redacted content. Acknowledge the redaction in your answer.
2. If the evidence does not contain enough non-redacted information to \
   answer, reply with: "I cannot disclose this information."
3. Never copy the evidence verbatim if it contains identifiers; paraphrase \
   at a high level.

Evidence:
{evidence}

Question: {query}

Answer in ONE short sentence."""


def privacy_decode(client, model, query: str, evidence: list,
                   eps_remaining: float) -> str:
    """Temperature scales inversely with the remaining budget — tighter
    budget -> more deterministic refusal-leaning behaviour."""
    if not evidence:
        return "I cannot disclose this information."

    ev_str = "\n".join(f"- ({e['id']}) {e['text']}" for e in evidence)
    temp = 0.0 if eps_remaining < 1.0 else 0.3
    out = chat(client, model,
               ANSWER_PROMPT.format(evidence=ev_str, query=query),
               temperature=temp, max_tokens=120)
    return out.strip().split("\n")[0]


# ---------------------------------------------------------------------------
# Step 5 — Ledger update
# ---------------------------------------------------------------------------

def delta_eps(risk: float) -> float:
    """How much budget the current turn consumes from each touched document."""
    return round(0.2 + 0.8 * risk, 2)


def update_ledger(ledger: dict, cands: list, transformed: list,
                  risk: float, was_aligned: bool) -> dict:
    cost = delta_eps(risk)
    redacted_ids = {t["id"] for t in transformed if t["redacted"]}
    deltas = {}
    for d in cands:
        row = ledger[d["id"]]
        row["retrievals"] += 1
        if d["id"] in redacted_ids:
            row["transforms"] += 1
        if was_aligned:
            row["aligned_decodes"] += 1
        row["exposure"] += cost * (0.4 if d["id"] in redacted_ids else 1.0)
        row["remaining_budget"] = max(0.0, row["remaining_budget"] - cost)
        deltas[d["id"]] = cost
    return deltas


# ---------------------------------------------------------------------------
# Sufficiency / budget gate (Step 4 pre-check)
# ---------------------------------------------------------------------------

def budget_exhausted(ledger: dict, cands: list, floor: float) -> bool:
    return all(ledger[d["id"]]["remaining_budget"] < floor for d in cands)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
