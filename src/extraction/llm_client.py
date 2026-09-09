"""
Thin wrapper around Ollama or OpenRouter chat models.

Both the extractor and the comparator send prompts through here, so
retries, <think>-tag stripping (qwen3 is a reasoning model and can emit
these even in JSON mode), and JSON repair are handled in exactly one
place.

Set ``LLM_PROVIDER=openrouter`` and ``OPENROUTER_API_KEY`` to use a
hosted OpenRouter model.  The default remains local Ollama.
"""

import json
import os
import re
import time
from typing import Any

import requests


class LLMClient:
    def __init__(
        self,
        model: str = None,
        host: str = None,
        timeout: int = 300,
        provider: str = None,
        max_retries: int = 5,
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
        if self.provider not in {"ollama", "openrouter"}:
            raise ValueError("LLM_PROVIDER must be 'ollama' or 'openrouter'.")

        default_model = (
            "google/gemma-4-26b-a4b-it:free"
            if self.provider == "openrouter"
            # Keep the runtime default aligned with the documented local
            # setup, which pulls this model.
            else "qwen3:8b"
        )
        model_env = "OPENROUTER_MODEL" if self.provider == "openrouter" else "OLLAMA_MODEL"
        self.model = model or os.getenv(model_env, default_model)
        default_host = (
            "https://openrouter.ai/api/v1"
            if self.provider == "openrouter"
            else "http://localhost:11434"
        )
        host_env = "OPENROUTER_HOST" if self.provider == "openrouter" else "OLLAMA_HOST"
        self.host = (host or os.getenv(host_env, default_host)).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> Any:
        """
        Calls the model and returns parsed JSON (dict or list).

        Raises:
            RuntimeError: if the selected provider can't be reached.
            ValueError: if the model's output can't be parsed as JSON
                even after cleanup. Callers should catch this and skip
                the current chunk/pair rather than crash the pipeline —
                a single bad chunk shouldn't take down the whole run.
        """
        raw = self._call(system_prompt, user_prompt, temperature)
        return self._parse_json(raw)

    def _call(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        if self.provider == "openrouter":
            return self._call_openrouter(
                system_prompt,
                user_prompt,
                temperature,
            )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature},
            "think": False,
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
                f"The current chunk was skipped so extraction can continue."
            ) from exc

        data = resp.json()
        return data.get("message", {}).get("content", "")

    def _call_openrouter(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> str:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter."
            )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        retries = 0
        while True:
            try:
                resp = requests.post(
                    f"{self.host}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                break
            except requests.exceptions.ConnectionError as exc:
                raise RuntimeError(
                    f"Could not reach OpenRouter at {self.host}."
                ) from exc
            except requests.exceptions.Timeout as exc:
                raise RuntimeError(
                    f"OpenRouter timed out after {self.timeout}s calling "
                    f"'{self.model}'."
                ) from exc
            except requests.exceptions.HTTPError as exc:
                status = resp.status_code

                retryable = (
                    status == 429
                    or status >= 500
                )

                if retryable and retries < self.max_retries:
                    retries += 1
                    delay = min(2 ** retries, 30)
                    # Respect the server's Retry-After header when present.
                    retry_after = resp.headers.get("Retry-After")
                    try:
                        if retry_after is not None:
                            delay = max(
                                delay,
                                float(retry_after),
                            )
                    except (TypeError, ValueError):
                        pass
                    print(
                        f"[llm] '{self.model}' {status} "
                        f"(attempt {retries}/{self.max_retries}), "
                        f"backing off {delay:.0f}s",
                        flush=True,
                    )
                    time.sleep(delay)
                    continue

                detail = resp.text[:500]
                raise RuntimeError(
                    f"OpenRouter rejected '{self.model}': {detail}"
                ) from exc

        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "")

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
