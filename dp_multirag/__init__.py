"""DP-MultiRAG: History-Aware Differential Privacy for Multi-Turn RAG.

The package exposes the four modules of the DP-MultiRAG paper and a
high-level pipeline for one interaction turn:

    Module 1 — Exposure-aware Retrieval Control     (`retrieval.py`)
    Module 2 — Controlled Evidence Utilization      (`evidence.py`)
    Module 3 — Privacy-aware Generation             (`generation.py`)
    Module 4 — Document-level Privacy Ledger        (`ledger.py`)
"""

from . import ledger, retrieval, evidence, generation
from .llm import LLMClient
from .pipeline import run_turn, TurnResult

__all__ = [
    "ledger",
    "retrieval",
    "evidence",
    "generation",
    "LLMClient",
    "run_turn",
    "TurnResult",
]

__version__ = "0.1.0"
