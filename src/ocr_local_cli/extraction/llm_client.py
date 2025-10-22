from __future__ import annotations

import json
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
            try:
                return json.loads(content)
            except json.JSONDecodeError as exc:
                raise LLMExtractionError("LLM choices content was not valid JSON.") from exc

        if "response" in data and isinstance(data["response"], str):
            try:
                return json.loads(data["response"])
            except json.JSONDecodeError as exc:
                raise LLMExtractionError("LLM string response was not valid JSON.") from exc

        return data
