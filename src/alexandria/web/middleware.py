"""Path-aware auth middleware for the combined MCP+web ASGI app.

- ``/mcp`` and ``/mcp/*``       → require ``Authorization: Bearer <mcp-token>``
                                  (MCP spec compliance — no cookie fallback)
- ``/api/*``                     → accept EITHER bearer OR valid session cookie
- ``/files/<doc_id>``            → accept bearer, session cookie, OR a valid
                                   ``?exp=...&sig=...`` HMAC-signed URL. Signed
                                   URLs let the get_original MCP tool hand a
                                   remote agent a self-contained fetchable link.
- ``/login`` / ``/logout`` /     → anonymous OK (login handler does its own
  ``/static/*``                    password check)
- everything else (page routes)  → require session cookie (bearer also honored
                                   for curl scripting); 303 → ``/login`` on
                                   missing/invalid auth

``no_auth=True`` bypasses all checks (Tailscale/WireGuard/LAN deploys).
"""
from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, RedirectResponse

from alexandria.web.auth import COOKIE_NAME, verify_session
from alexandria.web.signed_url import verify as verify_signed_url

_OPEN_PATHS = {"/login", "/logout"}


class SmartAuthMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        *,
        mcp_token: str | None,
        session_secret: bytes | None,
        no_auth: bool,
    ) -> None:
        super().__init__(app)
        self._mcp_token = mcp_token
        self._session_secret = session_secret
        self._no_auth = no_auth

    async def dispatch(self, request, call_next):
        if self._no_auth:
            return await call_next(request)

        path = request.url.path
        if path == "/mcp" or path.startswith("/mcp/"):
            return await self._require_bearer(request, call_next)
        if path.startswith("/api/"):
            return await self._require_bearer_or_cookie(request, call_next)
        if path.startswith("/files/"):
            return await self._require_bearer_cookie_or_signed(request, call_next, path)
        if path in _OPEN_PATHS or path.startswith("/static/"):
            return await call_next(request)
        return await self._require_session_or_redirect(request, call_next)

    def _bearer_ok(self, request) -> bool | None:
        """True if a valid bearer is present, False if wrong bearer, None if absent."""
        hdr = request.headers.get("authorization", "")
        if not hdr.startswith("Bearer "):
            return None
        return hdr[7:].strip() == self._mcp_token

    async def _require_bearer(self, request, call_next):
        ok = self._bearer_ok(request)
        if ok is None:
            return JSONResponse({"error": "missing bearer token"}, status_code=401)
        if not ok:
            return JSONResponse({"error": "invalid token"}, status_code=401)
        return await call_next(request)

    def _cookie_ok(self, request) -> bool:
        cookie = request.cookies.get(COOKIE_NAME)
        return bool(
            cookie
            and self._session_secret
            and verify_session(self._session_secret, cookie)
        )

    async def _require_bearer_or_cookie(self, request, call_next):
        ok = self._bearer_ok(request)
        if ok is True:
            return await call_next(request)
        if ok is False:
            return JSONResponse({"error": "invalid token"}, status_code=401)
        if self._cookie_ok(request):
            return await call_next(request)
        return JSONResponse(
            {"error": "authentication required"}, status_code=401
        )

    async def _require_session_or_redirect(self, request, call_next):
        # Bearer still honored so curl scripts can hit page routes.
        ok = self._bearer_ok(request)
        if ok is True or (ok is None and self._cookie_ok(request)):
            return await call_next(request)
        return RedirectResponse(url="/login", status_code=303)

    def _signed_url_ok(self, request, path: str) -> bool:
        """Verify a ``/files/<doc_id>?exp=...&sig=...`` signed URL.

        Missing exp/sig → not a signed-URL attempt (return False, let the
        caller fall back to bearer/cookie). Present-but-invalid → False.
        """
        exp = request.query_params.get("exp")
        sig = request.query_params.get("sig")
        if not exp or not sig or not self._session_secret:
            return False
        # path == "/files/<doc_id>" — anything past a second slash isn't ours.
        rest = path[len("/files/"):]
        if not rest or "/" in rest:
            return False
        return verify_signed_url(self._session_secret, rest, exp, sig)

    async def _require_bearer_cookie_or_signed(self, request, call_next, path: str):
        if self._signed_url_ok(request, path):
            return await call_next(request)
        ok = self._bearer_ok(request)
        if ok is True:
            return await call_next(request)
        if ok is False:
            return JSONResponse({"error": "invalid token"}, status_code=401)
        if self._cookie_ok(request):
            return await call_next(request)
        return JSONResponse(
            {"error": "authentication required"}, status_code=401
        )
