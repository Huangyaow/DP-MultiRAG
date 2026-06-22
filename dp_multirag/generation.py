"""Module 3 — Privacy-aware Generation.

DP-noised logit decoding:

    ℓ̃_{t,τ}    = ℓ_{t,τ} + z_{t,τ} ,   z_{t,τ} ~ N(0, σ_{t,τ}^2)
    σ_{t,τ}    = Δ_gen / √(2ρ_{t,τ}^{gen})       # zCDP form
    ε_{t,τ}^{gen} = ε_t^{gen} · η_{t,τ}          # implementation fallback
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
from typing import Callable, Dict, List, Optional, Sequence

from . import ledger as L


DELTA_GEN = 4.0


def gaussian_sigma(eps: float, delta: float, sensitivity: float = DELTA_GEN
                   ) -> float:
    """``σ_gen = Δ_gen · √(2 ln(1.25 / δ)) / ε_gen``."""
    if eps <= 0:
        return float("inf")
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / max(delta, 1e-12))) / eps


def zcdp_sigma(rho: float, sensitivity: float = DELTA_GEN) -> float:
    """Paper Eq. (15): ``σ = Δ_gen / sqrt(2ρ)`` for Gaussian zCDP."""
    if rho <= 0:
        return float("inf")
    return sensitivity / math.sqrt(2.0 * rho)


def zcdp_to_eps_delta(rho: float, delta: float) -> float:
    """Paper Eq. (18): convert ``ρ``-zCDP to ``(ε, δ)``-DP."""
    if rho <= 0:
        return 0.0
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / max(delta, 1e-12)))


def eps_delta_to_zcdp(eps: float, delta: float) -> float:
    """Conservative inverse of Eq. (18), useful for ε-configured callers."""
    if eps <= 0:
        return 0.0
    log_term = math.sqrt(math.log(1.0 / max(delta, 1e-12)))
    root_rho = max(0.0, math.sqrt(log_term * log_term + eps) - log_term)
    return root_rho * root_rho


ETA_MAX = 1.2


def _detector_score(detector_result) -> Optional[float]:
    if detector_result is None:
        return None
    if isinstance(detector_result, bool):
        return 0.3 if detector_result else None
    if isinstance(detector_result, (int, float)):
        return max(0.05, min(ETA_MAX, float(detector_result)))
    return 0.3 if detector_result else None


def token_eta(token: str, transformed_evidence: list,
              risk_detector: Optional[Callable[[str, list], object]] = None,
              risk_patterns: Optional[Sequence[str]] = None) -> float:
    """``η_{t,τ}``: per-token allocation factor in ``[η_min, η_max]``.

    Callers may provide a domain-specific ``risk_detector`` or
    ``risk_patterns``. Without those hooks, this function only treats attempts
    to reproduce sanitized markers as high risk.
    """
    s = (token or "").strip()
    if not s:
        return 1.0

    if risk_detector is not None:
        score = _detector_score(risk_detector(s, transformed_evidence))
        if score is not None:
            return score

    for pat in risk_patterns or ():
        if re.search(pat, s):
            return 0.3

    for e in transformed_evidence:
        if e.get("n_redacted") and any(p in s for p in ("REDACT", "REDACTED")):
            return 0.2

    return ETA_MAX


def normalise_token_etas(
    tokens: List[str],
    transformed_evidence: list,
    risk_detector: Optional[Callable[[str, list], object]] = None,
    risk_patterns: Optional[Sequence[str]] = None,
) -> List[float]:
    """Normalize per-token ``η`` so ``Σ_τ η_{t,τ} ≤ 1`` as in Eq. (16)."""
    raw = [
        max(0.0, token_eta(tok, transformed_evidence,
                           risk_detector=risk_detector,
                           risk_patterns=risk_patterns))
        for tok in tokens
    ]
    total = sum(raw)
    if total <= 0:
        return [0.0 for _ in raw]
    return [v / total for v in raw]


def online_token_eta(token: str, transformed_evidence: list,
                     max_tokens: int,
                     risk_detector: Optional[Callable[[str, list], object]] = None,
                     risk_patterns: Optional[Sequence[str]] = None) -> float:
    """Streaming-safe ``η_{t,τ}`` with ``Σ_τ η_{t,τ} ≤ 1``."""
    eta = token_eta(token, transformed_evidence,
                    risk_detector=risk_detector,
                    risk_patterns=risk_patterns)
    return eta / max(1.0, ETA_MAX * max_tokens)


def _answer_terms(answer: str) -> set:
    return {
        t for t in re.findall(r"[a-zA-Z0-9]+", answer.lower())
        if len(t) > 2
    }


def document_token_alignment(answer: str, transformed: list,
                             temperature: float = 1.0) -> Dict[str, float]:
    """Approximate Eq. (21)-(23): document contribution weights ``a_i``.

    The paper defines alignment with a public encoder over sanitized evidence
    and output tokens. This module keeps the function local and dependency-free
    by using lexical overlap against the sanitized evidence text. The returned
    weights sum to ``1`` over selected evidence documents.
    """
    if not transformed:
        return {}

    answer_terms = _answer_terms(answer)
    raw_scores = []
    for e in transformed:
        ev_terms = _answer_terms(e.get("text", ""))
        overlap = len(answer_terms & ev_terms) / max(1, len(answer_terms))
        mask_signal = 0.25 * bool(e.get("n_redacted"))
        raw_scores.append(overlap + mask_signal + 1e-6)

    m = max(raw_scores)
    exps = [
        math.exp((s - m) / max(temperature, 1e-6))
        for s in raw_scores
    ]
    total = sum(exps)
    return {
        e["id"]: exps[i] / total
        for i, e in enumerate(transformed)
    }


def charge_generation_by_alignment(answer: str, transformed: list,
                                   ledger: dict, eps_used_total: float,
                                   exposure_mass: float = 1.0) -> Dict[str, float]:
    """Charge ``Δε_i^{gen,t}`` according to output-evidence alignment.

    This is the implementation hook for Eq. (23)-(25): generated text is
    aligned with sanitized evidence, then both privacy cost and alignment
    exposure are distributed to document-level ledger rows.
    """
    weights = document_token_alignment(answer, transformed)
    if not weights or eps_used_total <= 0:
        return {}

    deltas = {}
    exposure_mass = max(0.0, min(1.0, exposure_mass))
    for e in transformed:
        weight = weights.get(e["id"], 0.0)
        delta = eps_used_total * weight
        charged = L.charge_generation(ledger, e["id"], delta,
                                      align_weight=exposure_mass * weight)
        deltas[e["id"]] = charged
    return deltas


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
    rho_gen_t = cfg.get("rho_t_gen", eps_delta_to_zcdp(eps_gen_t, delta))
    risk_detector = cfg.get("risk_detector")
    risk_patterns = cfg.get("risk_patterns")

    if not transformed:
        return {"answer": "I cannot disclose this information.",
                "n_tokens": 0, "sigma_per_token": [],
                "delta_eps_gen_per_doc": {},
                "alignment_per_doc": {}}

    ev_str = "\n".join(f"- ({e['id']}) {e['text']}" for e in transformed)
    user_msg = ANSWER_USER.format(evidence=ev_str, query=q_tilde)

    answer_tokens: List[str] = []
    sigma_per_token: List[float] = []
    rho_remaining = rho_gen_t
    rho_used_total = 0.0

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
        eta = online_token_eta(
            leading_token, transformed, max_tokens,
            risk_detector=risk_detector,
            risk_patterns=risk_patterns,
        )
        rho_tau = min(rho_remaining, rho_gen_t * eta)
        if rho_tau <= 1e-12:
            break
        sigma = zcdp_sigma(rho_tau)
        sigma_per_token.append(sigma)

        noisy_logits = [
            _clip(c["logprob"], DELTA_GEN) + rng.gauss(0.0, sigma)
            for c in cand
        ]
        probs = _softmax(noisy_logits)
        idx = _sample(rng, probs)
        chosen = cand[idx]["token"]

        answer_tokens.append(chosen)
        rho_used_total += rho_tau
        rho_remaining -= rho_tau

        if chosen in ("", "<|endoftext|>", "</s>"):
            break
        joined = "".join(answer_tokens)
        if len(joined) > 4 and joined.rstrip()[-1:] in ".!?\n" and len(joined) > 20:
            break

    answer = "".join(answer_tokens).strip().split("\n")[0] or \
             "I cannot disclose this information."

    eps_used_total = min(eps_gen_t, zcdp_to_eps_delta(rho_used_total, delta))
    deltas = charge_generation_by_alignment(
        answer, transformed, ledger, eps_used_total,
        exposure_mass=1.0 if eps_used_total > 0 else 0.0,
    )

    return {
        "answer": answer,
        "n_tokens": len(answer_tokens),
        "sigma_per_token": sigma_per_token,
        "delta_eps_gen_per_doc": deltas,
        "alignment_per_doc": document_token_alignment(answer, transformed),
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
            "alignment_per_doc": {},
        }

    eps_used_total = eps_gen_t * disclosure
    sigma = gaussian_sigma(eps_used_total, delta) if eps_used_total > 0 else 0.0
    deltas = charge_generation_by_alignment(
        answer, transformed, ledger, eps_used_total,
        exposure_mass=disclosure,
    )

    n_tok_proxy = max(1, len(answer.split()))
    return {
        "answer": answer,
        "n_tokens": n_tok_proxy,
        "sigma_per_token": [sigma] * n_tok_proxy,
        "delta_eps_gen_per_doc": deltas,
        "alignment_per_doc": document_token_alignment(answer, transformed),
    }
