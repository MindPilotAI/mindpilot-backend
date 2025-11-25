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
    html = build_html_report(
        source_url=source_label or "Pasted text",
        video_id=source_label or "N/A",
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
