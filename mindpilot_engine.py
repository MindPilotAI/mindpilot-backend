# mindpilot_engine.py
import re
from typing import List

import logging

import httpx
import io
from mindpilot_analyze import (
    fetch_transcript_text,
    # ... other things ...
    TranscriptUnavailableError,
)

class ContentBlockedError(RuntimeError):
    """
    Raised when an HTTP fetch is clearly blocked by the remote site
    (e.g., 401/403/429 or 5xx). Carries the URL + status code for logging.
    """
    def __init__(self, message: str, source_url: str | None = None, status_code: int | None = None):
        super().__init__(message)
        self.source_url = source_url
        self.status_code = status_code

from datetime import datetime
from urllib.parse import urlparse
from bs4 import BeautifulSoup
# Optional doc parsers (PDF, Word). These are optional dependencies.
try:
    import pypdf
except ImportError:
    pypdf = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None
def _slugify(text: str) -> str:
    """
    Simple slug helper:
    - lowercases
    - replaces non-alphanumeric with dashes
    - trims leading/trailing dashes
    """
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or "report"


def generate_report_id(source_label: str = "", source_url: str | None = None) -> str:
    """
    Generate a MindPilot-controlled report_id that does NOT depend on YouTube IDs.

    Example:
      20251201-nytimes-com-ai-will-replace-knowledge-workers
    """
    date_part = datetime.utcnow().strftime("%Y%m%d")

    # Try to pull a nice host from the URL (e.g., nytimes.com)
    host_part = ""
    if source_url:
        try:
            parsed = urlparse(source_url)
            host = (parsed.netloc or "").lower()
            if host.startswith("www."):
                host = host[4:]
            host_part = _slugify(host)
        except Exception:
            host_part = ""

    label_part = _slugify(source_label) if source_label else ""

    pieces = [date_part]
    if host_part:
        pieces.append(host_part)
    if label_part:
        pieces.append(label_part)

    report_id = "-".join(pieces)
    return report_id or f"report-{date_part}"

from mindpilot_analyze import (
    extract_video_id,
    fetch_transcript_text,
    chunk_text,
    build_chunk_prompt,
    build_global_summary_prompt,
    build_html_report,  # from your current HTML builder
    MAX_CHARS_PER_CHUNK,
)

from mindpilot_llm_client import run_mindpilot_analysis, run_grok_enrichment

def build_quick_global_prompt(transcript_text: str) -> str:
    """
    Quick mode: single-pass reasoning scan over the entire content.
    No chunk-by-chunk analysis, just a compact global profile,
    but using the SAME top-level headings as the full report
    so the HTML builder can parse it.
    """
    return f"""
You are MindPilot, a neutral reasoning-analysis copilot.

You will analyze the following content in ONE global pass (quick mode).
Focus on how the reasoning is structured, not on political or ideological alignment.

CONTENT BEGIN
----------------
{transcript_text}
----------------
CONTENT END

Return a concise report in Markdown with EXACTLY these numbered headings
and in this order. KEEP EACH SECTION SHORTER THAN THE FULL VERSION:

# 1. Full-Lesson Reasoning Summary
- 2–4 short paragraphs explaining the main argument(s) and overall reasoning quality.
- Mention the most important reasoning strengths and weaknesses only.

# 2. Master Fallacy & Bias Map
- A compact list of the most notable logical fallacies (name + 1–2 line explanation).
- A compact list of the most notable cognitive biases.
- A compact list of the most notable rhetorical / persuasion tactics.
- Keep each item to a single bullet per pattern.

# 3. Rationality Profile for the Entire Segment
- 1 short paragraph on strengths.
- 1 short paragraph on weaknesses.
- Then a short list of 4–6 dimensions
  (e.g., Evidence use, Causal reasoning, Emotional framing, Fairness/balance)
  with 1–5 ratings.
- At the very end of this section, add a standalone line in this exact format:
  "Overall reasoning score: NN/100"

# 4. Condensed Investor-Facing Summary
- 2–4 short paragraphs describing:
  - What the content is about (1–2 sentences).
  - What MindPilot found (main fallacies/biases/persuasion patterns,
    overall rationality level).
  - Why this demonstrates the value of MindPilot as a product.

# 5. Critical Thinking Questions to Ask Yourself
- 4–8 neutral, practical questions a reader could ask
  to think more clearly about this content.
""".strip()

def run_analysis_from_transcript(
    transcript_text: str,
    source_label: str = "",
    youtube_url: str | None = None,
    video_id: str | None = None,
) -> str:
    """
    Core engine: given transcript text, run chunk-level and global analysis,
    and return the final HTML report as a string.
    """
    # 1) Chunk
    chunks = chunk_text(transcript_text, MAX_CHARS_PER_CHUNK)
    total_chunks = len(chunks)

    # 2) Per-chunk analysis
    chunk_analyses: List[str] = []
    for idx, chunk in enumerate(chunks):
        chunk_prompt = build_chunk_prompt(chunk, idx, total_chunks)
        analysis = run_mindpilot_analysis(chunk_prompt)
        chunk_analyses.append(analysis)

    # 3) Global summary
    global_prompt = build_global_summary_prompt(chunk_analyses)
    global_report = run_mindpilot_analysis(global_prompt)

    # 3b) Optional Grok enrichment
    grok_label = source_label or (youtube_url or "Pasted content")
    try:
        grok_insights = run_grok_enrichment(grok_label, global_report)
    except Exception as e:
        logging.warning(f"Grok enrichment failed: {e}")
        grok_insights = ""

    # 4) Build HTML
    report_id = generate_report_id(
        source_label=source_label or (youtube_url or "Public source"),
        source_url=youtube_url,
    )

    final_html = build_html_report(
        source_url=youtube_url or source_label or "",
        report_id=report_id,
        total_chunks=total_chunks,
        chunk_analyses=chunk_analyses,
        global_report=global_report,
        grok_insights=grok_insights,
        depth="full",
    )

    return final_html


def run_full_analysis_from_youtube(youtube_url: str) -> str:
    """
    End-to-end pipeline for a YouTube URL.
    Returns the HTML report as a string.
    """
    try:
        video_id = extract_video_id(youtube_url)
    except Exception as e:
        logging.exception(
            "[MindPilot] extract_video_id error for URL %s", youtube_url
        )
        raise RuntimeError(
            "I couldn't read that YouTube link. Please double-check the URL, "
            "or copy the transcript from YouTube and paste the text into MindPilot instead."
        ) from e

    try:
        transcript_text = fetch_transcript_text(video_id)
    except TranscriptUnavailableError as e:
        # Let /analyze see the specific "no transcript" condition
        logging.warning(
            "[MindPilot] Transcript unavailable for video_id=%s: %s",
            video_id,
            e,
        )
        raise
    except Exception as e:
        logging.exception(
            "[MindPilot] fetch_transcript_text error for video_id=%s", video_id
        )
        raise RuntimeError(
            "I couldn't fetch a transcript for that video due to a technical error. "
            "If captions are visible to you, copy the transcript text and paste it into MindPilot."
        ) from e

    # Clean the transcript before chunking (remove sponsor / housekeeping lines)
    transcript_text = clean_transcript_text(transcript_text)

    return run_analysis_from_transcript(
        transcript_text=transcript_text,
        source_label=youtube_url,
        youtube_url=youtube_url,
        video_id=video_id,
    )

def run_full_analysis_from_text(raw_text: str, source_label: str = "Pasted text") -> str:
    """
    End-to-end pipeline for arbitrary pasted text.
    (No YouTube transcript step needed.)
    """
    transcript_text = raw_text.strip()
    if not transcript_text:
        raise ValueError("No text provided for analysis.")

    return run_analysis_from_transcript(
        transcript_text=transcript_text,
        source_label=source_label,
    )

def fetch_article_text(url: str) -> str:
    """
    Fetch a web page and extract main text content in a naive but useful way.
    For now, we:
      - GET the page with httpx
      - Parse HTML with BeautifulSoup
      - Collect reasonably-long <p> blocks

    If the site clearly blocks automated access (401/403/429 or 5xx),
    raise ContentBlockedError so the API can give a friendly fallback message.
    """
    try:
        resp = httpx.get(
            url,
            timeout=15.0,
            follow_redirects=True,
            headers={
                # Normal, non-sneaky browser-ish UA; this is "HTTP hygiene", not bypassing paywalls.
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"
                )
            },
        )
    except Exception as e:
        # Network-level issues (DNS, timeout, etc.)
        raise RuntimeError(f"Network error fetching article URL: {e}")

    status = resp.status_code

    # Respect sites that clearly don't want automated readers
    if status in (401, 403, 429) or status >= 500:
        raise ContentBlockedError(
            f"Blocked fetching article URL (HTTP {status}). "
            "Some sites don't allow automated readers or require you to be logged in. "
            "If you can read it in your browser, copy the text or save as PDF and upload it to MindPilot.",
            source_url=url,
            status_code=status,
        )

    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        # Other non-2xx cases that aren't explicit "blocked" patterns
        raise RuntimeError(f"Error fetching article URL (HTTP {status}): {e}") from e

    soup = BeautifulSoup(resp.text, "html.parser")

    # ... keep your existing <p> extraction logic below ...

    paragraphs: list[str] = []
    for p in soup.find_all("p"):
        text = p.get_text(" ", strip=True)
        if not text:
            continue
        # Filter out super-short boilerplate-ish lines
        if len(text) < 40:
            continue
        paragraphs.append(text)

    if not paragraphs:
        raise RuntimeError("Could not extract readable text from the article.")

    return "\n\n".join(paragraphs)
def extract_text_from_document_bytes(
    raw_bytes: bytes,
    filename: str = "",
    content_type: str | None = None,
) -> str:
    """
    Best-effort extractor for uploaded documents.

    - PDF  -> pypdf (if available)
    - DOCX -> python-docx (if available)
    - TXT  -> UTF-8 decode
    - Fallback: UTF-8 decode with errors='ignore'
    """
    name_lower = (filename or "").lower()
    ext = ""
    if "." in name_lower:
        ext = name_lower.rsplit(".", 1)[-1]

    # --- PDF ---
    if ext == "pdf" or (content_type or "").lower() == "application/pdf":
        if not pypdf:
            raise RuntimeError(
                "PDF support is not installed on the server. "
                "Ask the operator to install `pypdf`."
            )
        reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
        pages = []
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if text:
                pages.append(text)
        full = "\n\n".join(pages).strip()
        if not full:
            raise RuntimeError("Could not extract any text from the PDF.")
        return full

    # --- DOCX ---
    if ext == "docx" or (content_type or "").lower() in {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }:
        if not DocxDocument:
            raise RuntimeError(
                "DOCX support is not installed on the server. "
                "Ask the operator to install `python-docx`."
            )
        doc = DocxDocument(io.BytesIO(raw_bytes))
        paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        full = "\n\n".join(paras).strip()
        if not full:
            raise RuntimeError("Could not extract any text from the DOCX file.")
        return full

    # --- Plain text or fallback ---
    try:
        text = raw_bytes.decode("utf-8", errors="ignore")
    except Exception:
        raise RuntimeError("Unable to decode the uploaded file as text.")

    text = text.strip()
    if not text:
        raise RuntimeError("Uploaded file appears to be empty or non-text.")
    return text
def run_full_analysis_from_document(file_bytes: bytes, filename: str = "") -> str:
    """
    Full-mode pipeline for uploaded documents (PDF / DOCX / TXT).
    We:
    - extract readable text
    - route into the existing full text pipeline
    """
    text = extract_text_from_document_bytes(file_bytes, filename=filename)
    return run_full_analysis_from_text(
        raw_text=text,
        source_label=filename or "Uploaded document",
    )


def run_quick_analysis_from_document(
    file_bytes: bytes,
    filename: str = "",
    include_grok: bool = False,
) -> str:
    """
    Quick-mode pipeline for uploaded documents.
    - single-pass global profile
    - no chunk cards
    """
    text = extract_text_from_document_bytes(file_bytes, filename=filename)
    return run_quick_analysis_from_text(
        raw_text=text,
        source_label=filename or "Uploaded document",
        include_grok=include_grok,
    )


def run_full_analysis_from_article(article_url: str) -> str:
    """
    Fetch article, extract text, and run the existing MindPilot reasoning pipeline.
    """
    article_text = fetch_article_text(article_url)
    return run_full_analysis_from_text(
        raw_text=article_text,
        source_label=article_url,
    )
def run_quick_analysis_from_text(
    raw_text: str,
    source_label: str = "Pasted text",
    include_grok: bool = False,
) -> str:
    """
    Quick mode for arbitrary text:
    - one global scan
    - no chunk-level deep dive
    - Grok disabled by default (1 OpenAI call only)
    """
    transcript_text = raw_text.strip()
    if not transcript_text:
        raise ValueError("No text provided for analysis (quick mode).")

    quick_prompt = build_quick_global_prompt(transcript_text)
    quick_global_report = run_mindpilot_analysis(quick_prompt)

    # Optional Grok enrichment (default off for cost)
    grok_insights = ""
    if include_grok:
        label = source_label or "Pasted text"
        try:
            grok_insights = run_grok_enrichment(label, quick_global_report)
        except Exception as e:
            logging.warning(f"[Quick] Grok enrichment failed: {e}")
            grok_insights = ""

    # Use the same HTML template; no chunk cards, just global overview.
    report_id = generate_report_id(
        source_label=source_label or "Pasted text",
        source_url=None,
    )

    html = build_html_report(
        source_url=source_label or "Pasted text",
        report_id=report_id,
        total_chunks=0,
        chunk_analyses=[],
        global_report=quick_global_report,
        grok_insights=grok_insights,
    )
    return html


def run_quick_analysis_from_youtube(
    youtube_url: str,
    include_grok: bool = False,
) -> str:
    """
    Quick mode for YouTube:
    - fetch + clean transcript
    - one global scan
    - no chunk cards, no Grok by default
    """
    video_id = extract_video_id(youtube_url)
    transcript_text = fetch_transcript_text(video_id)
    transcript_text = clean_transcript_text(transcript_text)

    return run_quick_analysis_from_text(
        raw_text=transcript_text,
        source_label=youtube_url,
        include_grok=include_grok,
    )


def run_quick_analysis_from_article(
    article_url: str,
    include_grok: bool = False,
) -> str:
    """
    Quick mode for article URLs.
    """
    article_text = fetch_article_text(article_url)
    return run_quick_analysis_from_text(
        raw_text=article_text,
        source_label=article_url,
        include_grok=include_grok,
    )

AD_PHRASES = [
    "this video is sponsored by",
    "our sponsor today",
    "thanks to our sponsor",
    "use code",
    "link in the description",
    "smash that like button",
    "hit that like button",
    "hit the subscribe button",
    "remember to subscribe",
    "click the bell",
    "follow me on",
    "check out my merch",
    "patreon.com",
    "before we get started",
    "quick word from our sponsor",
    "and now back to the video",
]

def clean_transcript_text(raw: str) -> str:
    """
    Roughly remove obvious sponsor / housekeeping lines from YouTube transcripts.
    If everything gets filtered (edge case), fall back to the original.
    """
    # Split on sentence-ish boundaries
    sentences = re.split(r"(?<=[.!?])\s+", raw)
    kept: list[str] = []

    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        lower = s_clean.lower()
        if any(phrase in lower for phrase in AD_PHRASES):
            # Drop obvious ad / housekeeping lines
            continue
        kept.append(s_clean)

    # If we went too aggressive, don't break the pipeline
    if not kept:
        return raw
    return " ".join(kept)
