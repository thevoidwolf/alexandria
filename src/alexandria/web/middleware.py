"""Path-aware auth middleware for the combined MCP+web ASGI app.

- ``/mcp`` and ``/mcp/*``       → require ``Authorization: Bearer <mcp-token>``
                                  (MCP spec compliance — no cookie fallback)
- ``/api/*``                     → accept EITHER bearer OR valid session cookie
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
