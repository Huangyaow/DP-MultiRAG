import argparse


def get_parser():
    p = argparse.ArgumentParser(description="DP-RAG minimal demo")
    p.add_argument("--model", type=str, default="qwen3-8b",
                   help="vLLM served model name")
    p.add_argument("--base-url", type=str, default="http://localhost:8003/v1",
                   help="OpenAI-compatible endpoint")
    p.add_argument("--api-key", type=str, default="EMPTY",
                   help="placeholder for vLLM (no auth)")
    p.add_argument("--eps", type=float, default=3.0,
                   help="global per-document privacy budget")
    p.add_argument("--risk-high", type=float, default=0.65,
                   help="risk threshold above which evidence is transformed")
    p.add_argument("--budget-floor", type=float, default=0.3,
                   help="if remaining budget < floor, refuse to answer")
    p.add_argument("--decay", type=float, default=0.85,
                   help="exposure decay factor per turn")
    p.add_argument("--top-k", type=int, default=2,
                   help="number of candidate documents to retrieve")
    p.add_argument("--docs", type=str, default="data/docs.json")
    p.add_argument("--turns", type=str, default="data/turns.json")
    return p
