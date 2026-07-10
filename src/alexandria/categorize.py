from __future__ import annotations

import fnmatch
from urllib.parse import urlparse

from alexandria.config import Config


def categorize_url(
    url: str,
    cfg: Config,
    supplied_category: str | None = None,
    supplied_tags: list[str] | None = None,
) -> tuple[str | None, list[str]]:
    """Resolve final (category, tags) for a URL.

    User-supplied values always win. If category is None, walk config.category_rules
    for the first host_glob that matches the URL's host and adopt its category and tags.
    Rule tags are merged with (not replaced by) supplied_tags.
    """
    tags = list(supplied_tags or [])
    if supplied_category is not None:
        return supplied_category, tags

    host = (urlparse(url).hostname or "").lower()
    if not host:
        return None, tags

    for rule in cfg.category_rules:
        if fnmatch.fnmatch(host, rule.host_glob.lower()):
            merged = list(dict.fromkeys([*tags, *rule.tags]))  # preserve order, dedupe
            return rule.category, merged

    return None, tags
