"""Thin HTTP client the GUI uses to talk to the SafeRoute Logic Tier (FastAPI).

The GUI never imports model code: every prediction goes through the API, which keeps the
three tiers separate (Presentation -> Logic -> Data Access).
"""
import os

import httpx

DEFAULT_URL = os.getenv("SAFEROUTE_API", "http://127.0.0.1:8000")


class ApiError(RuntimeError):
    pass


class ApiClient:
    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(base_url=self.base_url, timeout=timeout)

    def _check(self, r: httpx.Response):
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise ApiError(f"HTTP {r.status_code}: {detail}")
        return r.json()

    def health(self, timeout: float = 1.5) -> dict:
        return self._check(self.http.get("/health", timeout=timeout))

    def detect(self, image_bytes: bytes, session_id: int | None = None,
               filename: str = "frame.jpg", content_type: str = "image/jpeg") -> dict:
        params = {"session_id": session_id} if session_id is not None else None
        r = self.http.post("/detect/image", params=params,
                           files={"file": (filename, image_bytes, content_type)})
        return self._check(r)

    def predict(self, image_bytes: bytes, filename: str = "crop.png",
                content_type: str = "image/png") -> dict:
        return self._check(self.http.post("/predict/image",
                                          files={"file": (filename, image_bytes, content_type)}))

    def start_session(self, model_version: str) -> int:
        return self._check(self.http.post("/sessions", json={"model_version": model_version}))["session_id"]

    def close_session(self, session_id: int) -> dict:
        return self._check(self.http.patch(f"/sessions/{session_id}/close"))

    def post_telemetry(self, session_id: int, fps: float, inference_ms: float, end_to_end_ms: float):
        return self._check(self.http.post(f"/sessions/{session_id}/telemetry",
                                          json={"fps": fps, "inference_ms": inference_ms,
                                                "end_to_end_ms": end_to_end_ms}))
