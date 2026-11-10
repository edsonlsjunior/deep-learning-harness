"""Credenciais BYOK efêmeras, isoladas por sessão anônima.

Nada neste módulo escreve em disco, banco de dados ou logs. As chaves vivem
somente na memória do processo que atende a aplicação e somem ao expirar,
serem removidas pelo usuário ou o processo ser reiniciado.
"""
from __future__ import annotations

import os
import secrets
import threading
import time
from dataclasses import dataclass


COOKIE_NAME = "dl_lab_session"
TTL_SECONDS = max(300, min(int(os.environ.get("DL_LAB_SESSION_KEY_TTL", "1800")), 86_400))
SUPPORTED_PROVIDERS = frozenset({"openai"})


@dataclass
class _Credential:
    value: str
    expires_at: float


class SessionCredentialStore:
    """Pequeno cofre em RAM; a API nunca oferece uma operação de leitura da chave."""

    def __init__(self, *, ttl_seconds: int = TTL_SECONDS, clock=time.monotonic):
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._values: dict[tuple[str, str], _Credential] = {}
        self._lock = threading.Lock()

    @staticmethod
    def new_session_id() -> str:
        # 256 bits de entropia, guardados somente no cookie HttpOnly do navegador.
        return secrets.token_urlsafe(32)

    def _purge(self, now: float) -> None:
        expired = [key for key, item in self._values.items() if item.expires_at <= now]
        for key in expired:
            del self._values[key]

    def set(self, session_id: str, provider: str, value: str) -> None:
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError("Este provedor ainda não aceita chave por sessão.")
        clean = value.strip()
        if not clean or len(clean) > 1024:
            raise ValueError("Informe uma chave de API válida.")
        with self._lock:
            now = self._clock()
            self._purge(now)
            self._values[(session_id, provider)] = _Credential(clean, now + self.ttl_seconds)

    def get(self, session_id: str | None, provider: str) -> str | None:
        if not session_id:
            return None
        with self._lock:
            now = self._clock()
            self._purge(now)
            item = self._values.get((session_id, provider))
            return item.value if item else None

    def configured(self, session_id: str | None, provider: str) -> bool:
        return self.get(session_id, provider) is not None

    def expires_in(self, session_id: str | None, provider: str) -> int:
        if not session_id:
            return 0
        with self._lock:
            now = self._clock()
            self._purge(now)
            item = self._values.get((session_id, provider))
            return max(0, int(item.expires_at - now)) if item else 0

    def remove(self, session_id: str | None, provider: str) -> None:
        if not session_id:
            return
        with self._lock:
            self._values.pop((session_id, provider), None)


store = SessionCredentialStore()
