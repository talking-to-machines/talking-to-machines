"""Multimodal media utilities for extracting and processing embedded media.

This module provides a unified API for detecting, extracting, and
converting image, video, and audio content embedded in prompt text into
provider-compatible multimodal content blocks.

Three embedding formats are recognised (in priority order):

    1. **Bracket tags** (preferred)::

           [IMAGE: https://example.com/photo.jpg]
           [VIDEO: https://example.com/clip.mp4]
           [AUDIO: https://example.com/sound.mp3]

    2. **Markdown image syntax**::

           ![alt text](https://example.com/photo.jpg)

    3. **Raw URLs** (auto-detected by file extension)::

           https://example.com/photo.png

URL normalisation is applied automatically — for example, Dropbox share
links (``dl=0``) are converted to direct-download links (``dl=1``) so
that LLM providers can fetch the raw bytes.

Typical usage::

    media_items = extract_all_media(prompt_text)
    clean_text  = strip_media_tags(prompt_text)
    content     = build_multimodal_content(clean_text, media_items)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

# ---------------------------------------------------------------------------
# Extension sets
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".bmp",
        ".tiff",
        ".svg",
    }
)
VIDEO_EXTENSIONS = frozenset(
    {
        ".mp4",
        ".mov",
        ".avi",
        ".webm",
        ".mkv",
        ".m4v",
        ".wmv",
        ".flv",
    }
)
AUDIO_EXTENSIONS = frozenset(
    {
        ".mp3",
        ".wav",
        ".ogg",
        ".flac",
        ".aac",
        ".m4a",
        ".opus",
        ".wma",
    }
)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# [IMAGE: url] / [VIDEO: url] / [AUDIO: url]
_TAG_PATTERN = re.compile(
    r"\[(IMAGE|VIDEO|AUDIO):\s*(https?://[^\]\s]+)\]",
    re.IGNORECASE,
)

# Markdown: ![alt](url)
_MD_IMAGE_PATTERN = re.compile(
    r"!\[([^\]]*)\]\((https?://[^\)]+)\)",
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class MediaContent:
    """A single media item extracted from prompt text.

    Attributes:
        url: The normalised URL pointing to the media resource.
        media_type: One of ``"image"``, ``"video"``, or ``"audio"``.
        caption: Optional caption or alt text (from Markdown syntax).
    """

    url: str
    media_type: str
    caption: str = ""


# ---------------------------------------------------------------------------
# URL normalisation
# ---------------------------------------------------------------------------


def normalize_url(url: str) -> str:
    """Normalise a media URL for direct-download access where possible.

    Applies provider-specific transformations so that the resulting URL
    returns raw file bytes rather than an HTML preview page.

    Currently handles:
        - **Dropbox**: Converts ``?dl=0`` to ``?dl=1``.

    Args:
        url: The original media URL to normalise.

    Returns:
        The normalised URL string. If no transformation applies, the
        original URL is returned unchanged.
    """
    if "dropbox.com" in url:
        parsed = urlparse(url)
        params = dict(parse_qsl(parsed.query))
        params["dl"] = "1"
        new_query = urlencode(params)
        return urlunparse(parsed._replace(query=new_query))
    return url


def _media_type_from_extension(url: str) -> Optional[str]:
    """Infer media type from a URL's file extension.

    Args:
        url: The URL whose path is inspected for a known media
            extension.

    Returns:
        ``"image"``, ``"video"``, or ``"audio"`` if the extension is
        recognised, otherwise ``None``.
    """
    path = urlparse(url).path.lower()
    # Strip query string from extension detection
    ext = "." + path.rsplit(".", 1)[-1] if "." in path else ""
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    return None


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_all_media(content: str) -> list[MediaContent]:
    """Extract all media items from *content* in document order.

    Scans the text using three strategies in priority order: bracket
    tags, Markdown image syntax, and raw URLs with known media
    extensions. Duplicate URLs are deduplicated (first occurrence wins).

    Args:
        content: The prompt text to scan for embedded media references.

    Returns:
        A list of ``MediaContent`` instances in the order they appear.
        Returns an empty list if no media is found.
    """
    seen: set[str] = set()
    items: list[MediaContent] = []

    def _add(url: str, media_type: str, caption: str = "") -> None:
        """Normalise *url* and append a MediaContent item if not yet seen."""
        url = normalize_url(url)
        if url not in seen:
            seen.add(url)
            items.append(MediaContent(url=url, media_type=media_type, caption=caption))

    # 1. Bracket tags: [IMAGE: ...] / [VIDEO: ...] / [AUDIO: ...]
    for m in _TAG_PATTERN.finditer(content):
        tag_type = m.group(1).lower()
        url = m.group(2).strip()
        _add(url, tag_type)

    # 2. Markdown images: ![alt](url)
    for m in _MD_IMAGE_PATTERN.finditer(content):
        caption = m.group(1)
        url = m.group(2).strip()
        _add(url, "image", caption)

    # 3. Raw URLs — only those with unambiguous media extensions
    raw_url_pattern = re.compile(r"https?://\S+", re.IGNORECASE)
    for m in raw_url_pattern.finditer(content):
        url = m.group(0).rstrip(".,;:)")
        if url in seen:
            continue
        detected = _media_type_from_extension(url)
        if detected:
            _add(url, detected)

    return items


def extract_media_url(content: str, media_type: str = "image") -> Optional[str]:
    """Return the first media URL of the given type found in *content*.

    Args:
        content: Prompt text to search for embedded media references.
        media_type: The kind of media to look for — one of
            ``"image"``, ``"video"``, or ``"audio"``.

    Returns:
        The normalised URL string of the first matching media item, or
        ``None`` if no match is found.
    """
    for item in extract_all_media(content):
        if item.media_type == media_type:
            return item.url
    return None


def strip_media_tags(content: str) -> str:
    """Remove bracket media tags from *content*, leaving surrounding text intact.

    Runs of extra whitespace introduced by tag removal are collapsed to
    a single space, and leading/trailing whitespace is stripped.

    Args:
        content: The prompt text containing bracket media tags.

    Returns:
        The cleaned text with all ``[IMAGE: ...]``, ``[VIDEO: ...]``,
        and ``[AUDIO: ...]`` tags removed.

    Example::

        >>> strip_media_tags("Here is a photo. [IMAGE: https://x.com/a.jpg] What do you see?")
        'Here is a photo. What do you see?'
    """
    text = _TAG_PATTERN.sub("", content)
    # Collapse runs of whitespace introduced by removal
    text = re.sub(r" {2,}", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Multimodal content block builders
# ---------------------------------------------------------------------------


def build_multimodal_content(text: str, media_items: list[MediaContent]) -> list[dict]:
    """Build an OpenAI-compatible multimodal content array.

    The text portion always comes first, followed by one block per media
    item. Image items use the standard ``image_url`` block format. Video
    and audio items use ``video_url`` and ``input_audio`` blocks
    respectively (provider-specific handling may be needed downstream).

    Args:
        text: The rendered prompt text with media tags already stripped.
        media_items: Ordered list of ``MediaContent`` items to embed
            as additional content blocks.

    Returns:
        A list of content-block dictionaries suitable for the
        ``content`` field of an ``{"role": "user", ...}`` message.
        The first element is always a ``{"type": "text", ...}`` block.
    """
    blocks: list[dict] = [{"type": "text", "text": text}]

    for item in media_items:
        if item.media_type == "image":
            block: dict = {
                "type": "image_url",
                "image_url": {"url": item.url},
            }
            if item.caption:
                # Some providers honour an alt/caption field; harmless otherwise
                block["image_url"]["alt"] = item.caption
            blocks.append(block)

        elif item.media_type == "video":
            # OpenAI Responses API supports video via a `video_url` block (preview).
            # Fall back to text for providers that do not support it.
            blocks.append(
                {
                    "type": "video_url",
                    "video_url": {"url": item.url},
                }
            )

        elif item.media_type == "audio":
            # Audio support varies widely; use an input_audio block (OpenAI format).
            blocks.append(
                {
                    "type": "input_audio",
                    "input_audio": {"url": item.url},
                }
            )

    return blocks


def has_media(content: str) -> bool:
    """Check whether *content* contains at least one extractable media item.

    Args:
        content: The prompt text to scan.

    Returns:
        ``True`` if any image, video, or audio reference is found,
        ``False`` otherwise.
    """
    return bool(extract_all_media(content))


def has_video(content: str) -> bool:
    """Check whether *content* contains at least one video item.

    Args:
        content: The prompt text to scan.

    Returns:
        ``True`` if a video reference is found, ``False`` otherwise.
    """
    return any(m.media_type == "video" for m in extract_all_media(content))


def has_audio(content: str) -> bool:
    """Check whether *content* contains at least one audio item.

    Args:
        content: The prompt text to scan.

    Returns:
        ``True`` if an audio reference is found, ``False`` otherwise.
    """
    return any(m.media_type == "audio" for m in extract_all_media(content))
