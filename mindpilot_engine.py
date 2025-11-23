# mindpilot_engine.py

from typing import List, Tuple

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from mindpilot_analyze_old import (
    extract_video_id,
    fetch_transcript_text,
    chunk_text,
    build_chunk_prompt,
    build_global_summary_prompt,
    build_html_report,  # from your current HTML builder
    MAX_CHARS_PER_CHUNK,
)
from mindpilot_llm_client import run_mindpilot_analysis


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

    # 4) Build HTML
    final_html = build_html_report(
        source_url=youtube_url or source_label or "",
        video_id=video_id or source_label or "N/A",
        total_chunks=total_chunks,
        chunk_analyses=chunk_analyses,
        global_report=global_report,
    )

    return final_html


def run_full_analysis_from_youtube(youtube_url: str) -> str:
    """
    End-to-end pipeline for a YouTube URL.
    Returns the HTML report as a string.
    """
    video_id = extract_video_id(youtube_url)
    transcript_text = fetch_transcript_text(video_id)

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
