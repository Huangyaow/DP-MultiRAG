"""Module 3 — Privacy-aware Generation.

DP-noised logit decoding:

    ℓ̃_{t,τ}    = ℓ_{t,τ} + z_{t,τ} ,   z_{t,τ} ~ N(0, σ_{t,τ}^2)
    σ_gen      = Δ_gen · √(2 ln(1.25 / δ)) / ε_gen
    ε_{t,τ}^{gen} = ε_t^{gen} · η_{t,τ}
    p( y_{t,τ} ) = softmax( ℓ̃_{t,τ} ) .

The OpenAI-compatible ``top_logprobs`` channel returns the
log-probabilities of the top-K candidate tokens at each step. Up to a
per-step additive constant, log-probabilities and logits are equivalent
for the purpose of adding Gaussian noise and re-softmaxing, so the
decoder drives a step-by-step generation loop:

    while not finished:
        cand    = LLM.next_token_logprobs(prefix)        # top-K logprobs
        noisy   = clip(cand) + N(0, σ_τ²)
        token   = sample( softmax(noisy) )
        prefix += token
"""

from __future__ import annotations

import math
import random
import re
from typing import List

from . import ledger as L


DELTA_GEN = 4.0


def gaussian_sigma(eps: float, delta: float, sensitivity: float = DELTA_GEN
                   ) -> float:
    """``σ_gen = Δ_gen · √(2 ln(1.25 / δ)) / ε_gen``."""
    if eps <= 0:
        return float("inf")
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / max(delta, 1e-12))) / eps


_HIGH_RISK_PATTERNS = (
    r"\b\d{3,}\b",
    r"@",
    r"\b(?:[A-Z][a-z]+)\s+(?:[A-Z][a-z]+)\b",
    r"\b\d{1,2}:\d{2}\b",
)


def token_eta(token: str, transformed_evidence: list) -> float:
    """``η_{t,τ}``: per-token allocation factor in ``[η_min, η_max]``.

    A token that matches a high-risk pattern (long digit run, email,
    full-name pair, time-stamp) or that would reproduce a redacted span
    is forced to a small ``η`` so the analytical mapping ``σ ∝ 1/ε``
    increases the noise on that token. Other tokens get a slight bonus.
    """
    s = (token or "").strip()
    if not s:
        return 1.0

    for pat in _HIGH_RISK_PATTERNS:
        if re.search(pat, s):
            return 0.3

    for e in transformed_evidence:
        if e.get("n_redacted") and any(p in s for p in ("REDACT", "REDACTED")):
            return 0.2

    return 1.2


ANSWER_SYSTEM = (
    "You are a privacy-aware assistant in a DP-MultiRAG system. "
    "Answer ONLY using the evidence below. "
    "If a span is [REDACTED] or [ABSTRACTED], do NOT try to recover it; "
    "acknowledge the masking. "
    "If the evidence is insufficient, reply: I cannot disclose this information."
)

ANSWER_USER = """Evidence:
{evidence}

Question: {query}

Answer in ONE concise sentence."""


def _softmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    Z = sum(e)
    return [v / Z for v in e]


def _sample(rng: random.Random, probs: List[float]) -> int:
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            return i
    return len(probs) - 1


def _clip(x: float, bound: float) -> float:
    return max(-bound, min(bound, x))


def privacy_decode(llm, q_tilde: str, transformed: list, ledger: dict,
                   cfg: dict, rng: random.Random,
                   max_tokens: int = 60, top_k_logprobs: int = 20) -> dict:
    """DP-noised step-by-step decoding via the ``top_logprobs`` channel.

    Args:
        llm: object exposing ``next_token_logprobs(system, user,
            assistant_prefix, top_k)`` returning a list of
            ``{"token": str, "logprob": float}`` items.
        q_tilde: rewritten query.
        transformed: evidence list produced by
            :func:`evidence.transform_evidence`.
        ledger: ledger object.
        cfg: dict with key ``eps_t_gen``.
        rng: ``random.Random`` instance.
        max_tokens: hard upper bound on generated tokens.
        top_k_logprobs: number of candidate tokens requested per step.

    Returns:
        ``{"answer", "n_tokens", "sigma_per_token",
           "delta_eps_gen_per_doc"}``.
    """
    eps_gen_t = cfg["eps_t_gen"]
    delta = ledger["config"]["delta_global"]

    if not transformed:
        return {"answer": "I cannot disclose this information.",
                "n_tokens": 0, "sigma_per_token": [],
                "delta_eps_gen_per_doc": {}}

    ev_str = "\n".join(f"- ({e['id']}) {e['text']}" for e in transformed)
    user_msg = ANSWER_USER.format(evidence=ev_str, query=q_tilde)

    answer_tokens: List[str] = []
    sigma_per_token: List[float] = []
    eps_remaining = eps_gen_t
    eps_used_total = 0.0

    for _ in range(max_tokens):
        cand = llm.next_token_logprobs(
            system=ANSWER_SYSTEM,
            user=user_msg,
            assistant_prefix="".join(answer_tokens),
            top_k=top_k_logprobs,
        )
        if not cand:
            break

        leading_token = cand[0]["token"]
        eta = token_eta(leading_token, transformed)
        eps_tau = min(eps_remaining,
                      eps_gen_t / max(1, max_tokens) * eta)
        if eps_tau <= 1e-6:
            break
        sigma = gaussian_sigma(eps_tau, delta)
        sigma_per_token.append(sigma)

        noisy_logits = [
            _clip(c["logprob"], DELTA_GEN) + rng.gauss(0.0, sigma)
            for c in cand
        ]
        probs = _softmax(noisy_logits)
        idx = _sample(rng, probs)
        chosen = cand[idx]["token"]

        answer_tokens.append(chosen)
        eps_used_total += eps_tau
        eps_remaining -= eps_tau

        if chosen in ("", "<|endoftext|>", "</s>"):
            break
        joined = "".join(answer_tokens)
        if len(joined) > 4 and joined.rstrip()[-1:] in ".!?\n" and len(joined) > 20:
            break

    answer = "".join(answer_tokens).strip().split("\n")[0] or \
             "I cannot disclose this information."

    deltas = {}
    if transformed:
        share = eps_used_total / len(transformed)
        for e in transformed:
            L.charge_generation(ledger, e["id"], share)
            deltas[e["id"]] = share

    return {
        "answer": answer,
        "n_tokens": len(answer_tokens),
        "sigma_per_token": sigma_per_token,
        "delta_eps_gen_per_doc": deltas,
    }


def reference_decode(answer: str, disclosure: float, transformed: list,
                     ledger: dict, cfg: dict) -> dict:
    """Decode with a pre-authored response and a per-turn disclosure level.

    The DP accounting follows the same structure as :func:`privacy_decode`:
    each touched document is charged

        Δε_i^{gen,t} = ε_t^{gen} · disclosure / |C̃_t|

    so a hard refusal at ``disclosure = 0`` consumes zero ``ε^gen`` and a
    full disclosure (``disclosure = 1``) consumes the full per-turn budget.

    Args:
        answer: textual response to return for the current turn.
        disclosure: scalar in ``[0, 1]`` quantifying how much of the
            per-turn ``ε^gen`` is consumed by this response.
        transformed: evidence list produced by
            :func:`evidence.transform_evidence`.
        ledger: ledger object.
        cfg: dict with key ``eps_t_gen``.

    Returns:
        ``{"answer", "n_tokens", "sigma_per_token",
           "delta_eps_gen_per_doc"}``.
    """
    eps_gen_t = cfg["eps_t_gen"]
    delta = ledger["config"]["delta_global"]

    if not transformed or disclosure <= 0:
        return {
            "answer": answer,
            "n_tokens": 0,
            "sigma_per_token": [],
            "delta_eps_gen_per_doc": {},
        }

    eps_used_total = eps_gen_t * disclosure
    sigma = gaussian_sigma(eps_used_total, delta) if eps_used_total > 0 else 0.0
    share = eps_used_total / len(transformed)

    deltas = {}
    for e in transformed:
        L.charge_generation(ledger, e["id"], share)
        deltas[e["id"]] = share

    n_tok_proxy = max(1, len(answer.split()))
    return {
        "answer": answer,
        "n_tokens": n_tok_proxy,
        "sigma_per_token": [sigma] * n_tok_proxy,
        "delta_eps_gen_per_doc": deltas,
    }
