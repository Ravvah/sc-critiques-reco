from __future__ import annotations

import html
import re

_ws = re.compile(r"\s+")
_tags = re.compile(r"<[^>]+>")


def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = html.unescape(s)
    s = _tags.sub(" ", s)
    s = s.replace("\u00a0", " ")
    s = _ws.sub(" ", s).strip()
    return s

