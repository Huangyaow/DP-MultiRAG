"""OpenAI / vLLM client wrapper.

Two methods are exposed:

    * :meth:`LLMClient.chat` — one-shot chat completion. Used by Module 1
      for query rewriting.
    * :meth:`LLMClient.next_token_logprobs` — request the top-K logprobs
      for the next token given a ``(system, user, assistant_prefix)``
      triple. Used by Module 3 to drive its DP decoding loop.
"""

from __future__ import annotations

import re
from typing import List, Optional

from openai import OpenAI


class LLMClient:
    """Thin wrapper over :class:`openai.OpenAI` with chat + logprobs."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def chat(self, prompt: str, *, system: Optional[str] = None,
             temperature: float = 0.0, max_tokens: int = 256) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        text = resp.choices[0].message.content or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text

    def next_token_logprobs(self, *, system: str, user: str,
                            assistant_prefix: str = "",
                            top_k: int = 20) -> List[dict]:
        """Return the top-K logprobs for the next token.

        Returns a list of ``{"token": str, "logprob": float}`` items, or
        an empty list if the served model does not expose logprobs.
        """
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if assistant_prefix:
            msgs.append({"role": "assistant", "content": assistant_prefix})

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=msgs,
                max_tokens=1,
                temperature=1.0,
                logprobs=True,
                top_logprobs=top_k,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "add_generation_prompt": False if assistant_prefix else True,
                    "continue_final_message": bool(assistant_prefix),
                },
            )
        except Exception:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    max_tokens=1,
                    temperature=1.0,
                    logprobs=True,
                    top_logprobs=top_k,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
            except Exception:
                return []

        choice = resp.choices[0]
        logprobs = getattr(choice, "logprobs", None)
        if not logprobs or not getattr(logprobs, "content", None):
            text = (choice.message.content or "")
            if not text:
                return []
            return [{"token": text, "logprob": 0.0}]

        first = logprobs.content[0]
        out = []
        for tl in (first.top_logprobs or []):
            out.append({"token": tl.token, "logprob": float(tl.logprob)})
        if not out:
            out.append({"token": first.token, "logprob": float(first.logprob)})
        return out
