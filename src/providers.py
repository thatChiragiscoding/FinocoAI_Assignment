"""
providers.py — LLM provider abstraction.

Both providers expose the same interface:
    provider.complete(prompt: str) -> dict

The dict is the parsed JSON answer object.
Switching providers is a one-flag change at the CLI level.
"""

import os
import json
import time
from abc import ABC, abstractmethod
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class BaseProvider(ABC):
    name: str = "base"

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> dict:
        """Call the LLM and return parsed JSON dict."""
        ...

    def _parse_json(self, raw: str) -> dict:
        """Safely parse JSON, stripping markdown fences if present."""
        raw = raw.strip()
        if raw.startswith("```"):
            # Strip ```json ... ``` fences
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"[{self.name}] Failed to parse JSON: {e}\nRaw: {raw[:300]}")


# ---------------------------------------------------------------------------
# Gemini provider (google-genai SDK)
# ---------------------------------------------------------------------------
class GeminiProvider(BaseProvider):
    name = "gemini"
    MODEL = "gemini-2.5-flash-lite"

    def __init__(self):
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set in .env")
        self.client = genai.Client(api_key=api_key)

    def complete(self, system_prompt: str, user_prompt: str) -> dict:
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        user_prompt = user_prompt.encode("ascii", "ignore").decode("ascii")
        t0 = time.time()
        response = self.client.models.generate_content(
            model=self.MODEL,
            contents=full_prompt,
            config={"response_mime_type": "application/json"},
        )
        latency_ms = int((time.time() - t0) * 1000)
        result = self._parse_json(response.text)
        result["latency_ms"] = latency_ms
        result["provider"] = self.name
        return result


# ---------------------------------------------------------------------------
# Groq provider (groq SDK — llama-3.3-70b-versatile)
# ---------------------------------------------------------------------------
class GroqProvider(BaseProvider):
    name = "groq"
    MODEL = "llama-3.3-70b-versatile"

    def __init__(self):
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set in .env")
        self.client = Groq(api_key=api_key)

    def complete(self, system_prompt: str, user_prompt: str) -> dict:
        t0 = time.time()
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        latency_ms = int((time.time() - t0) * 1000)
        raw = response.choices[0].message.content
        result = self._parse_json(raw)
        result["latency_ms"] = latency_ms
        result["provider"] = self.name
        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
PROVIDERS = {
    "gemini": GeminiProvider,
    "groq": GroqProvider,
}


def get_provider(name: str) -> BaseProvider:
    name = name.lower()
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider '{name}'. Choose from: {list(PROVIDERS.keys())}")
    return PROVIDERS[name]()
