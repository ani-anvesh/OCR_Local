from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential_jitter

from ..config import LLMConfig
from ..utils.logging import get_logger

logger = get_logger("extraction.llm_client")


class LLMExtractionError(RuntimeError):
    pass


@dataclass
class LLMClient:
    config: LLMConfig

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _payload(self, prompt: str) -> Dict[str, Any]:
        if self.config.request_mode == "chat-completions":
            if not self.config.model:
                raise LLMExtractionError(
                    "LLM model name is required for chat-completions mode."
                )
            messages = []
            if self.config.system_prompt:
                messages.append({"role": "system", "content": self.config.system_prompt})
            messages.append({"role": "user", "content": prompt})
            payload: Dict[str, Any] = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
            }
            if self.config.enforce_json:
                payload["response_format"] = {"type": "json_object"}
            return payload

        payload = {
            "prompt": prompt,
            "temperature": self.config.temperature,
        }
        if self.config.enforce_json:
            payload["response_format"] = {"type": "json_object"}
        if self.config.model:
            payload["model"] = self.config.model
        return payload

    def _post(self, payload: Dict[str, Any]) -> requests.Response:
        logger.debug("LLM request payload: %s", payload)
        response = requests.post(
            self.config.endpoint,
            headers=self._headers(),
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        if response.status_code >= 400:
            raise LLMExtractionError(
                f"LLM call failed with status {response.status_code}: {response.text}"
            )
        logger.debug("LLM raw response text: %s", response.text)
        return response

    def extract(self, prompt: str) -> Dict[str, Any]:
        payload = self._payload(prompt)
        try:
            response = self._retryable_post(payload)
        except RetryError as exc:
            raise LLMExtractionError(f"LLM extraction failed: {exc}") from exc
        try:
            data = response.json()
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise LLMExtractionError("LLM response was not valid JSON") from exc
        return self._normalize_response(data)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=6),
        reraise=True,
    )
    def _retryable_post(self, payload: Dict[str, Any]) -> requests.Response:
        logger.debug("Posting extraction request to %s", self.config.endpoint)
        return self._post(payload)

    def _normalize_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize the JSON payload returned by different LLM providers.
        """
        if not isinstance(data, dict):
            raise LLMExtractionError("LLM response payload must be a JSON object.")

        if "choices" in data:
            try:
                choice = data["choices"][0]
            except (KeyError, IndexError) as exc:
                raise LLMExtractionError("LLM response missing choices content.") from exc
            content = (
                choice.get("message", {}).get("content")
                or choice.get("text")
            )
            if not isinstance(content, str):
                raise LLMExtractionError("LLM response choices did not include text content.")
            parsed = self._extract_json_payload(content)
            if parsed is None:
                raise LLMExtractionError("LLM choices content was not valid JSON.")
            return parsed

        if "response" in data and isinstance(data["response"], str):
            parsed = self._extract_json_payload(data["response"])
            if parsed is None:
                raise LLMExtractionError("LLM string response was not valid JSON.")
            return parsed

        return data

    @staticmethod
    def _extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
        """Extract the last JSON object embedded in text (handles fenced code blocks)."""

        if not text:
            return None

        candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if not candidates:
            simple_matches = re.findall(r"(\{[^{}]*\})", text)
            candidates.extend(simple_matches)
            if not candidates:
                # manual brace matching to capture larger JSON blocks
                stack = []
                segments = []
                for idx, char in enumerate(text):
                    if char == "{":
                        stack.append(idx)
                    elif char == "}" and stack:
                        start = stack.pop()
                        if not stack:
                            segments.append(text[start : idx + 1])
                candidates.extend(segments)

        for candidate in reversed(candidates):  # prefer the last JSON block (likely the data)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return None
