"""Text cleaning and path-text extraction."""
import re

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>(.*)", re.DOTALL | re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{")


def extract_think(text: str) -> str:
    """Extract the reasoning inside <think>...</think>.

    - With closing tag: take the content between the tags.
    - Only <think>, no </think> (truncated): take everything after <think>.
    - No <think> at all: return an empty string (the caller decides whether to fall back to full).
    """
    if not text:
        return ""
    m = _THINK_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _THINK_OPEN_RE.search(text)
    if m:
        return m.group(1).strip()
    return ""


def strip_think_tags(text: str) -> str:
    """Remove the <think></think> tags themselves, keeping all body text (light cleaning for the full view)."""
    if not text:
        return ""
    return text.replace("<think>", " ").replace("</think>", " ").replace(
        "<THINK>", " ").replace("</THINK>", " ").strip()


def get_path_text(response: str, mode: str) -> str:
    """Return the path text used for diversity analysis according to mode.

    mode="think": only the reasoning; falls back to full if there is no <think>.
    mode="full" : the whole response (tags stripped).
    """
    if mode == "think":
        t = extract_think(response)
        if t:
            return t
        return strip_think_tags(response)
    elif mode == "full":
        return strip_think_tags(response)
    else:
        raise ValueError(f"unknown path_text mode: {mode}")


def truncate(text: str, max_chars: int) -> str:
    if max_chars and len(text) > max_chars:
        return text[:max_chars]
    return text
