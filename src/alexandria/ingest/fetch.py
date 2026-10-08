from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from alexandria.config import HttpConfig


@dataclass(frozen=True)
class Fetched:
    data: bytes
    source_kind: str      # "file" | "url"
    source_uri: str       # absolute path or URL (final URL after redirects)
    content_type_hint: str | None = None  # from HTTP or None for files


class RobotsDisallowed(Exception):
    """robots.txt disallows fetching the requested URL."""


class DownloadTooLarge(Exception):
    """Response body exceeded the configured size cap."""


# Google's crawler stops reading robots.txt at 500 KiB; do the same.
_ROBOTS_MAX_BYTES = 500 * 1024


def _get_capped(
    client: httpx.Client, url: str, max_bytes: int
) -> tuple[httpx.Response, bytes]:
    """GET ``url`` but stop reading once the body passes ``max_bytes``.

    Counts decoded bytes, so a small gzip body that inflates past the cap
    is caught too. The returned response's body is already consumed; use
    the returned bytes. Error responses (>= 400) come back with an empty
    body so callers see the HTTP status, not a size error.
    """
    with client.stream("GET", url) as resp:
        if resp.status_code >= 400:
            return resp, b""
        declared = resp.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > max_bytes:
            raise DownloadTooLarge(f"{url}: {declared} bytes > cap of {max_bytes}")
        buf = bytearray()
        for chunk in resp.iter_bytes():
            buf.extend(chunk)
            if len(buf) > max_bytes:
                raise DownloadTooLarge(f"{url}: body exceeds cap of {max_bytes} bytes")
    return resp, bytes(buf)


_ROBOTS_CACHE: dict[str, RobotFileParser | None] = {}


def fetch_file(path: Path) -> Fetched:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return Fetched(
        data=path.read_bytes(),
        source_kind="file",
        source_uri=str(path),
    )


def _robots_parser(url: str, client: httpx.Client) -> RobotFileParser | None:
    """Return a RobotFileParser for the URL's origin, or None if unavailable."""
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if origin in _ROBOTS_CACHE:
        return _ROBOTS_CACHE[origin]

    rp = RobotFileParser()
    robots_url = f"{origin}/robots.txt"
    try:
        resp, body = _get_capped(client, robots_url, _ROBOTS_MAX_BYTES)
    except (httpx.HTTPError, DownloadTooLarge):
        _ROBOTS_CACHE[origin] = None
        return None

    if resp.status_code >= 400:
        _ROBOTS_CACHE[origin] = None
        return None

    rp.parse(body.decode(resp.encoding or "utf-8", errors="replace").splitlines())
    _ROBOTS_CACHE[origin] = rp
    return rp


def fetch_url(url: str, http_cfg: HttpConfig) -> Fetched:
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"URL must be http(s): {url}")

    headers = {"User-Agent": http_cfg.user_agent}
    client = httpx.Client(
        headers=headers,
        follow_redirects=True,
        max_redirects=http_cfg.max_redirects,
        timeout=http_cfg.timeout_seconds,
    )
    try:
        if http_cfg.respect_robots_txt:
            rp = _robots_parser(url, client)
            if rp is not None and not rp.can_fetch(http_cfg.user_agent, url):
                raise RobotsDisallowed(url)

        resp, data = _get_capped(
            client, url, http_cfg.max_download_mb * 1024 * 1024
        )
        resp.raise_for_status()
        content_type = resp.headers.get("content-type")
        final_url = str(resp.url)
    finally:
        client.close()

    return Fetched(
        data=data,
        source_kind="url",
        source_uri=final_url,
        content_type_hint=content_type,
    )
