import warnings

warnings.filterwarnings("ignore")

import json
import os
from typing import Any

from google import genai
from google.genai import types as genai_types

from ttsizer.asr_backends.base import ASRBackend, ASRResult
from ttsizer.utils.logger import get_logger

logger = get_logger("asr_gemini")


class GeminiASRBackend(ASRBackend):
    """
    ASR backend using Gemini audio understanding.

    Note: This returns text only (no reliable word timestamps). The ASRProcessor will
    skip timestamp-based flagging when timestamps are absent.
    """

    name = "gemini"

    def __init__(self, *, model_name: str, api_key: str | None = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing GEMINI_API_KEY for Gemini ASR backend.")
        self.client = genai.Client(api_key=self.api_key)

    def transcribe_batch(self, paths: list[str]) -> list[ASRResult]:
        out: list[ASRResult] = []
        for path in paths:
            uploaded = self.client.files.upload(file=path)
            prompt = (
                "Transcribe the provided audio accurately.\n"
                "Return ONLY valid JSON: {\"text\": \"...\"}.\n"
                "Do not include timestamps.\n"
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    genai_types.Content(
                        role="user",
                        parts=[
                            genai_types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type),
                            genai_types.Part.from_text(text=prompt),
                        ],
                    )
                ],
                config=genai_types.GenerateContentConfig(
                    temperature=0.0,
                    top_p=0.8,
                    response_mime_type="application/json",
                ),
            )
            text = ""
            try:
                raw = response.candidates[0].content.parts[0].text or ""
                data = json.loads(raw)
                if isinstance(data, dict):
                    text = str(data.get("text", "") or "")
            except Exception as e:
                logger.warning(f"Gemini ASR parse failed for {path}: {type(e).__name__}: {e}")
            out.append(ASRResult(text=text, times=None, raw=response))
        return out

