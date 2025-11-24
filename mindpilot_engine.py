# mindpilot_engine.py

from typing import List

import logging
import re

import httpx
from bs4 import BeautifulSoup

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
    final_html = build_html_report(
        youtube_url or source_label or "",  # source_url
        video_id or source_label or "N/A",  # video_id
        total_chunks,                       # total_chunks
        chunk_analyses,                     # chunk_analyses
        global_report,                      # global_report
        grok_insights,                      # NEW: Grok enrichment
    )

    return final_html



def run_full_analysis_from_youtube(youtube_url: str) -> str:
    """
    End-to-end pipeline for a YouTube URL.
    Returns the HTML report as a string.
    """
    video_id = extract_video_id(youtube_url)
    transcript_text = fetch_transcript_text(video_id)

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
    """
    try:
        resp = httpx.get(url, timeout=15.0)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Error fetching article URL: {e}")

    soup = BeautifulSoup(resp.text, "html.parser")

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


def run_full_analysis_from_article(article_url: str) -> str:
    """
    Fetch article, extract text, and run the existing MindPilot reasoning pipeline.
    """
    article_text = fetch_article_text(article_url)
    return run_full_analysis_from_text(
        raw_text=article_text,
        source_label=article_url,
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
