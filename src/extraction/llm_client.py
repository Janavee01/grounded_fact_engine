"""
Thin wrapper around a local Ollama model.

Both the extractor and the comparator send prompts through here, so
retries, <think>-tag stripping (qwen3 is a reasoning model and can emit
these even in JSON mode), and JSON repair are handled in exactly one
place.

Requires `ollama serve` running locally with the model pulled, e.g.:
    ollama pull qwen3:4b
"""

import json
import os
import re
from typing import Any

import requests


class LLMClient:
    def __init__(
        self,
        model: str = None,
        host: str = None,
        timeout: int = 120,
    ):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> Any:
        """
        Calls the model and returns parsed JSON (dict or list).

        Raises:
            RuntimeError: if Ollama can't be reached.
            ValueError: if the model's output can't be parsed as JSON
                even after cleanup. Callers should catch this and skip
                the current chunk/pair rather than crash the pipeline —
                a single bad chunk shouldn't take down the whole run.
        """
        raw = self._call(system_prompt, user_prompt, temperature)
        return self._parse_json(raw)

    def _call(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature},
        }

        try:
            resp = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. "
                f"Make sure 'ollama serve' is running and the model is pulled "
                f"(ollama pull {self.model})."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"Ollama timed out after {self.timeout}s calling '{self.model}'. "
                f"Large chunks on small/CPU-bound models can be slow — "
                f"consider lowering MAX_CHUNK_CHARS in pdf_extractor.py."
            ) from exc

        data = resp.json()
        return data.get("message", {}).get("content", "")

    def _parse_json(self, raw: str) -> Any:
        # qwen3 can emit reasoning inside <think>...</think> even when
        # asked for JSON-only output. Strip it before parsing.
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # Strip markdown code fences if the model wrapped its answer anyway.
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", cleaned.strip(), flags=re.MULTILINE
        ).strip()

        if not cleaned:
            raise ValueError(f"Model returned empty output. Raw response:\n{raw[:500]}")

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Last resort: grab the first {...} or [...] block in the text.
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from model output:\n{raw[:500]}")
