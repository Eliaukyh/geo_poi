"""LLM client with exact per-request accounting (calls, tokens, latency, cost)."""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field

from openai import OpenAI

from . import config


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0

    def add(self, other: "Usage") -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.latency_s += other.latency_s
        self.cost_usd += other.cost_usd

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_s": round(self.latency_s, 3),
            "cost_usd": round(self.cost_usd, 6),
        }


def _price(model: str) -> tuple[float, float]:
    for name, price in config.PRICES.items():
        if model == name or model.startswith(name):
            return price
    return (0.0, 0.0)


class LLMClient:
    def __init__(self, cfg: config.AgentConfig | None = None):
        self.cfg = cfg or config.AGENT
        kwargs = config.client_kwargs()
        self.model = self.cfg.model
        self.client = OpenAI(api_key=kwargs["api_key"], base_url=kwargs["base_url"], timeout=kwargs["timeout"])
        self.usage = Usage()
        self._lock = threading.Lock()  # usage accumulation is shared across workers

    def chat_json(self, system: str, user: str) -> tuple[dict | None, Usage]:
        """One JSON-mode completion with retries. Returns (parsed, usage_of_this_call)."""
        if not config.client_kwargs()["api_key"]:
            raise RuntimeError("DMX_API_KEY is not set (see .tools/.env)")

        last_err = None
        for attempt in range(self.cfg.max_retries):
            u = Usage(calls=1)
            t0 = time.time()
            try:
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        temperature=self.cfg.temperature,
                        max_tokens=self.cfg.max_tokens,
                        response_format={"type": "json_object"},
                    )
                except Exception as fmt_err:  # provider may reject json_object
                    if "response_format" not in str(fmt_err) and "json" not in str(fmt_err).lower():
                        raise
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        temperature=self.cfg.temperature,
                        max_tokens=self.cfg.max_tokens,
                    )
                u.latency_s = time.time() - t0
                text = resp.choices[0].message.content or ""
                if resp.usage:
                    u.prompt_tokens = resp.usage.prompt_tokens or 0
                    u.completion_tokens = resp.usage.completion_tokens or 0
                pin, pout = _price(self.model)
                u.cost_usd = (u.prompt_tokens * pin + u.completion_tokens * pout) / 1_000_000
                with self._lock:
                    self.usage.add(u)
                return _parse_json(text), u
            except Exception as exc:  # noqa: BLE001 - retry any transport error
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {self.cfg.max_retries} attempts: {last_err}")


def _parse_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    # last resort: pull integers out in order
    ids = [int(x) for x in re.findall(r"\b\d+\b", text)]
    return {"ranking": ids, "reason": ""} if ids else None
