import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

MAX_CHARS_PER_CHUNK = 5000


@dataclass
class ChunkAnalysis:
    chunk_index: int
    chunk_text: str
    analysis_text: str


def extract_video_id(youtube_url: str) -> Optional[str]:
    """
    Extract the video ID from a YouTube URL.

    Supports the common formats:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://www.youtube.com/embed/VIDEO_ID

    Returns the video ID as a string, or None if not found.
    """
    if not youtube_url:
        return None

    # 1) Short link format: https://youtu.be/VIDEO_ID
    match = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", youtube_url)
    if match:
        return match.group(1)

    # 2) Standard watch URL: https://www.youtube.com/watch?v=VIDEO_ID
    match = re.search(r"v=([A-Za-z0-9_-]{11})", youtube_url)
    if match:
        return match.group(1)

    # 3) Embed format: https://www.youtube.com/embed/VIDEO_ID
    match = re.search(r"/embed/([A-Za-z0-9_-]{11})", youtube_url)
    if match:
        return match.group(1)

    return None


def fetch_transcript_text(video_id: str) -> str:
    """
    Placeholder function for fetching transcript text.
    The real implementation is on the API side (fastapi + youtube_transcript_api).
    """
    raise NotImplementedError("Transcript fetching is handled in the backend.")


def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    """
    Chunk the given text into pieces of at most max_chars characters, trying
    to break on sentence boundaries where possible.
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks = []
    current_chunk = []

    current_length = 0
    for sentence in re.split(r"([.!?])\s+", text):
        if not sentence:
            continue

        # Add punctuation back, if it's a single character punctuation from split
        if sentence in [".", "!", "?"]:
            if current_chunk:
                current_chunk[-1] += sentence
            else:
                current_chunk.append(sentence)
        else:
            sentence_length = len(sentence) + 1  # Rough space or punctuation
            if current_length + sentence_length > max_chars and current_chunk:
                chunks.append(" ".join(current_chunk).strip())
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks


def build_chunk_prompt(
    chunk_text: str, chunk_index: int, total_chunks: int
) -> str:
    """
    Build the prompt for chunk-level analysis for MindPilot.
    """
    return f"""
You are MindPilot, a neutral reasoning-analysis copilot.

You are analyzing chunk {chunk_index + 1} of {total_chunks} of a larger transcript.

This chunk is a portion of a spoken or written argument. Your job is to:
1. Summarize the main claims and reasoning in this chunk.
2. Identify any fallacies, biases, or rhetorical tactics present.
3. Note any key evidence, data, or sources mentioned.
4. Highlight any places where the reasoning structure is incomplete, vague, or manipulative.

CRITICAL:
- Be neutral and non-partisan.
- Do not inject your own opinions or fact-checking.
- Focus only on the structure and style of reasoning.

Return your output in a concise markdown format with:
- "Chunk Summary"
- "Reasoning Structure"
- "Notable Fallacies / Biases / Tactics"
- "Evidence and Support"
- "Questions for the Larger Context"

Here is the transcript chunk:

\"\"\"{chunk_text}\"\"\"
""".strip()


def build_global_summary_prompt(chunk_analyses: List[str]) -> str:
    """
    Build the prompt for the global summary that synthesizes all chunk-level analyses.
    """
    joined_analyses = "\n\n---\n\n".join(chunk_analyses)

    return f"""
You are MindPilot, a neutral reasoning-analysis copilot.

You have been given a series of chunk-level analyses of a single piece of content
(speech, podcast, debate, or other longform argument). Each chunk-level analysis
was already produced by you or a similar system following the MindPilot
reasoning-analysis framework.

Your job now is to produce a GLOBAL Cognitive Flight Report that synthesizes all
the chunk-level analyses.

CRITICAL:
- Do not repeat the chunk analyses verbatim.
- Summarize the overall reasoning patterns, not just each piece.
- Stay neutral and avoid partisan or ideological framing.
- Do not claim to have fact-checked anything.

Your output MUST follow this EXACT STRUCTURE in Markdown:

[GLOBAL_COGNITIVE_FLIGHT_REPORT_START]

## Cognitive Flight Summary
- 1–3 short bullets summarizing the main themes and argumentative approach.
- Focus on how the argument is being made, not who is right.

## Master Fallacy & Bias Map
- List the major fallacy and bias "clusters" that show up repeatedly across chunks.
- For each cluster, briefly describe:
  - How it shows up (style, phrasing, patterns).
  - The possible impact on clarity or fairness.
- Use a bullet list or short table-style formatting where helpful.

## Rationality Profile
- Provide a reasoned description (not a score yet) of:
  - Evidence handling (strong, mixed, weak, absent).
  - Treatment of counterarguments (fair, partial, absent, distorted).
  - Emotional tone and rhetorical pressure (calm, persuasive, loaded, aggressive, etc.).
  - Consistency vs. contradiction across the content.
- Aim for 2–4 concise paragraphs.

## Investor / Decision-Maker Summary (Optional)
- Pretend you are advising a careful decision-maker (e.g., an investor, policy analyst, or senior leader)
  who has to make a choice influenced by this content.
- Summarize:
  - The reliability of the reasoning.
  - The key "red flags" to keep in mind.
  - Where more information or due diligence would be required.
- Keep it neutral, not promotional or dismissive.

## Critical Thinking Questions Card
- Provide 5–10 short, pointed questions that a thoughtful listener/reader could ask
  themselves after engaging with this content. For example:
  - "What specific evidence was offered for X?"
  - "What important perspectives or stakeholders were not mentioned?"
  - "How would the argument change if Y turned out to be false?"
- Each question should be on its own line with a leading bullet.

[GLOBAL_COGNITIVE_FLIGHT_REPORT_END]

IMPORTANT:
- Return ONLY the content inside [GLOBAL_COGNITIVE_FLIGHT_REPORT_START] and
  [GLOBAL_COGNITIVE_FLIGHT_REPORT_END], including those tags.
- Do not add commentary before or after.
- Do not summarize the instructions.

Here are the chunk-level analyses:

\"\"\"{joined_analyses}\"\"\"
""".strip()


def parse_global_report(global_report: str) -> Tuple[str, str, str, str, str]:
    """
    Parse the global report into:
      - full_summary
      - master_map
      - rationality_profile
      - investor_summary
      - questions_block

    We expect the global_report to be wrapped in:
      [GLOBAL_COGNITIVE_FLIGHT_REPORT_START]
      ...
      [GLOBAL_COGNITIVE_FLIGHT_REPORT_END]
    """
    # First, isolate the content between the markers if present.
    start_tag = "[GLOBAL_COGNITIVE_FLIGHT_REPORT_START]"
    end_tag = "[GLOBAL_COGNITIVE_FLIGHT_REPORT_END]"

    if start_tag in global_report and end_tag in global_report:
        inner = global_report.split(start_tag, 1)[1].split(end_tag, 1)[0].strip()
    else:
        # If tags are missing, treat entire text as inner.
        inner = global_report.strip()

    # Now split by top-level headings.
    # We expect headings like "## Cognitive Flight Summary", etc.
    # We'll capture them using a regex that finds '## ' headings.
    sections = re.split(r"(?m)^##\s+", inner)
    # sections[0] should be empty or preamble, we ignore it.
    section_map = {}
    current_header = None
    for sec in sections[1:]:
        # sec looks like "Heading\nRest of content..."
        lines = sec.splitlines()
        if not lines:
            continue
        header = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        section_map[header] = body

    full_summary = section_map.get("Cognitive Flight Summary", "")
    master_map = section_map.get("Master Fallacy & Bias Map", "")
    rationality_profile = section_map.get("Rationality Profile", "")
    investor_summary = section_map.get("Investor / Decision-Maker Summary (Optional)", "")
    questions_block = section_map.get("Critical Thinking Questions Card", "")

    return full_summary, master_map, rationality_profile, investor_summary, questions_block


def escape_html(text: str) -> str:
    """
    Minimal HTML escaping for pre blocks.
    """
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def build_html_report(
    source_url,
    video_id,
    total_chunks,
    chunk_analyses,
    global_report,
    grok_insights: str | None = None,
):
    """
    Build an HTML report string from the MindPilot analyses.
    """
    # Parse the global report into sub-sections
    (
        full_summary,
        master_map,
        rationality_profile,
        investor_summary,
        questions_block,
    ) = parse_global_report(global_report)

    # Escape global sections
    esc_full = escape_html(full_summary) if full_summary else ""
    esc_map = escape_html(master_map) if master_map else ""
    esc_profile = escape_html(rationality_profile) if rationality_profile else ""
    esc_investor = escape_html(investor_summary) if investor_summary else ""
    esc_questions = escape_html(questions_block) if questions_block else ""
    esc_global_fallback = escape_html(global_report) if (global_report and not esc_full) else ""
    esc_grok = escape_html(grok_insights) if grok_insights else ""

    # Prepare chunk analyses as HTML
    esc_chunks = [escape_html(a) for a in chunk_analyses]

    # Shareable URL note (if we want to root the domain)
    source_url_html = (
        f'<a href="{source_url}" target="_blank" rel="noopener noreferrer">{source_url}</a>'
        if source_url
        else "(pasted text or unknown source)"
    )

    # For educational or label, we define a short source label:
    if source_url:
        if "youtube.com" in source_url or "youtu.be" in source_url:
            source_label = f"YouTube Video ({video_id})"
        else:
            source_label = f"Source content: {source_url}"
    else:
        source_label = "Pasted text content"

    # Build HTML document
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>MindPilot Cognitive Flight Report</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <style>
    :root {{
      --bg: #0f172a;              /* deep navy background */
      --bg-secondary: #020617;    /* near-black */
      --card-bg: #020617;         /* card background */
      --card-border: #1e293b;     /* subtle slate border */
      --accent: #38bdf8;          /* bright sky blue for highlights */
      --accent-soft: rgba(56, 189, 248, 0.18);
      --accent-soft-strong: rgba(56, 189, 248, 0.35);
      --accent-muted: #7dd3fc;    /* lighter accent for secondary text */
      --accent-alt: #a855f7;      /* violet accent for contrast badges */
      --text-main: #e5e7eb;       /* main foreground text (light gray) */
      --text-soft: #9ca3af;       /* softer text */
      --text-subtle: #6b7280;     /* very subtle descriptions */
      --chip-bg: #020617;         /* pill backgrounds */
      --chip-border: #1f2937;     /* pill border */
      --footer-bg: #000000;       /* deepest footer background */
      --danger: #f97373;          /* warm red for warnings */
      --warning: #facc15;         /* amber warning */
      --good: #4ade80;            /* success green */
      --muted-line: #111827;      /* for separators */
      --shadow-soft: 0 18px 45px rgba(15, 23, 42, 0.9);
    }}

    * {{
      box-sizing: border-box;
    }}

    html, body {{
      margin: 0;
      padding: 0;
      background-color: var(--bg);
      color: var(--text-main);
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text",
                   "Segoe UI", sans-serif;
      -webkit-font-smoothing: antialiased;
      text-rendering: optimizeLegibility;
    }}

    body {{
      padding: 1.75rem;
      display: flex;
      justify-content: center;
    }}

    .page {{
      width: 100%;
      max-width: 1120px;
      background: radial-gradient(circle at top, #1e293b 0%, #020617 52%, #000000 100%);
      border-radius: 24px;
      padding: 1.75rem 1.75rem 1.5rem;
      box-shadow: var(--shadow-soft);
      border: 1px solid rgba(148, 163, 184, 0.25);
      position: relative;
      overflow: hidden;
    }}

    .page::before {{
      content: "";
      position: absolute;
      inset: 0;
      background: radial-gradient(circle at 10% -10%, rgba(56,189,248,0.22), transparent 55%),
                  radial-gradient(circle at 80% -10%, rgba(168,85,247,0.18), transparent 60%);
      opacity: 0.95;
      pointer-events: none;
    }}

    main {{
      position: relative;
      z-index: 1;
    }}

    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 0.75rem;
      margin-bottom: 1.75rem;
    }}

    .logo-stack {{
      display: flex;
      align-items: center;
      gap: 0.9rem;
    }}

    .badge-pill {{
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      padding: 0.35rem 0.7rem;
      border-radius: 999px;
      border: 1px solid rgba(148,163,184,0.45);
      background: linear-gradient(120deg, rgba(15,23,42,0.96), rgba(15,23,42,0.8));
      box-shadow: 0 0 0 1px rgba(15,23,42,0.9);
    }}

    .badge-dot {{
      width: 7px;
      height: 7px;
      border-radius: 999px;
      background: radial-gradient(circle, var(--accent), #0ea5e9);
      box-shadow: 0 0 0 3px rgba(56,189,248,0.35);
    }}

    .badge-pill span {{
      font-size: 0.74rem;
      letter-spacing: 0.13em;
      text-transform: uppercase;
      color: var(--accent-muted);
    }}

    .badge-pill .badge-sub {{
      color: var(--text-subtle);
      font-weight: 400;
      text-transform: none;
      letter-spacing: 0.05em;
      font-size: 0.73rem;
    }}

    .title-block h1 {{
      margin: 0.1rem 0 0.25rem 0;
      font-size: 1.5rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: #e5e7eb;
    }}

    .title-block .subtitle {{
      margin: 0;
      font-size: 0.9rem;
      color: var(--text-soft);
    }}

    .meta-chip-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.45rem;
      margin-top: 0.6rem;
    }}

    .meta-chip {{
      border-radius: 999px;
      border: 1px solid var(--chip-border);
      padding: 0.2rem 0.55rem;
      background: radial-gradient(circle at top, #111827, #020617);
      font-size: 0.75rem;
      color: var(--text-subtle);
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
    }}

    .meta-label {{
      text-transform: uppercase;
      letter-spacing: 0.18em;
      font-size: 0.7rem;
      color: var(--text-subtle);
    }}

    .meta-value {{
      color: #e5e7eb;
      font-weight: 500;
      font-size: 0.78rem;
    }}

    .header-right {{
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      gap: 0.35rem;
    }}

    .pill-kpi {{
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      padding: 0.4rem 0.8rem;
      border-radius: 999px;
      background: radial-gradient(circle at top, rgba(15,23,42,1), rgba(15,23,42,0.9));
      border: 1px solid rgba(148,163,184,0.5);
    }}

    .pill-kpi-label {{
      font-size: 0.75rem;
      color: var(--text-subtle);
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }}

    .pill-kpi-value {{
      font-size: 0.9rem;
      color: var(--accent-muted);
      font-weight: 600;
    }}

    .nav-row {{
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }}

    .nav-link {{
      font-size: 0.78rem;
      color: var(--text-soft);
      text-decoration: none;
      opacity: 0.9;
    }}

    .nav-link:hover {{
      color: var(--accent-muted);
    }}

    .nav-dot {{
      width: 4px;
      height: 4px;
      border-radius: 999px;
      background: var(--text-subtle);
      opacity: 0.6;
    }}

    .main-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) minmax(0, 1.1fr);
      gap: 1.4rem;
      align-items: flex-start;
      margin-bottom: 1.35rem;
    }}

    @media (max-width: 920px) {{
      .header {{
        flex-direction: column;
        align-items: flex-start;
      }}
      .header-right {{
        align-items: flex-start;
      }}
      .main-grid {{
        grid-template-columns: minmax(0, 1fr);
      }}
    }}

    .card {{
      position: relative;
      border-radius: 18px;
      background: radial-gradient(circle at top left, rgba(15,23,42,1), rgba(2,6,23,0.98));
      border: 1px solid rgba(51,65,85,0.9);
      padding: 1.2rem 1.1rem 1rem;
      box-shadow: 0 16px 40px rgba(15,23,42,0.88);
      overflow: hidden;
    }}

    .card::before {{
      content: "";
      position: absolute;
      inset: 0;
      background: radial-gradient(circle at 120% -10%, rgba(56,189,248,0.18), transparent 65%);
      opacity: 0.9;
      pointer-events: none;
    }}

    .card-inner {{
      position: relative;
      z-index: 1;
    }}

    .card-title {{
      font-size: 1.05rem;
      font-weight: 600;
      margin-bottom: 0.3rem;
      color: #f9fafb;
      letter-spacing: 0.03em;
    }}

    .card-tagline {{
      font-size: 0.82rem;
      color: var(--text-soft);
      margin-bottom: 0.85rem;
    }}

    .badge-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.4rem;
      margin-bottom: 0.8rem;
    }}

    .badge {{
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      border-radius: 999px;
      border: 1px solid rgba(148,163,184,0.6);
      padding: 0.2rem 0.55rem;
      color: rgba(209,213,219,0.95);
      background: rgba(15,23,42,0.95);
    }}

    .badge-accent {{
      background: radial-gradient(circle at top right, rgba(56,189,248,0.35), rgba(15,23,42,1));
      border-color: rgba(56,189,248,0.7);
      color: #ecfeff;
    }}

    .whisper {{
      font-size: 0.75rem;
      color: var(--text-subtle);
      margin-top: 0.6rem;
    }}

    .two-column-body {{
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr);
      gap: 0.7rem;
      margin-top: 0.3rem;
      align-items: flex-start;
    }}

    @media (max-width: 800px) {{
      .two-column-body {{
        grid-template-columns: minmax(0, 1fr);
      }}
    }}

    .summary-text {{
      font-size: 0.87rem;
      color: var(--text-main);
      line-height: 1.5;
    }}

    .summary-text ul,
    .summary-text ol {{
      padding-left: 1.1rem;
      margin: 0.5rem 0;
    }}

    .summary-text li {{
      margin-bottom: 0.25rem;
    }}

    .meta-panel {{
      background: radial-gradient(circle at top, rgba(15,23,42,1), rgba(2,6,23,0.96));
      border-radius: 14px;
      padding: 0.6rem 0.7rem;
      border: 1px solid rgba(55,65,81,0.9);
      font-size: 0.78rem;
      color: var(--text-soft);
    }}

    .meta-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 0.6rem;
      margin-bottom: 0.3rem;
    }}

    .meta-row-label {{
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 0.7rem;
      color: var(--text-subtle);
    }}

    .meta-row-value {{
      color: var(--text-main);
      font-weight: 500;
      font-size: 0.8rem;
    }}

    .meta-row-value a {{
      color: var(--accent-muted);
      text-decoration: none;
    }}

    .meta-row-value a:hover {{
      text-decoration: underline;
    }}

    .meta-hint {{
      font-size: 0.75rem;
      color: var(--text-subtle);
      margin-top: 0.2rem;
    }}

    .rationality-pills {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.4rem;
      margin-top: 0.4rem;
    }}

    .rationality-pill {{
      display: inline-flex;
      align-items: center;
      gap: 0.25rem;
      font-size: 0.75rem;
      padding: 0.25rem 0.5rem;
      border-radius: 999px;
      border: 1px solid rgba(55,65,81,1);
      background: radial-gradient(circle at top, #020617, #020617);
    }}

    .rationality-pill span:first-child {{
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: linear-gradient(135deg, #22c55e, #4ade80);
    }}

    .rationality-pill--warn span:first-child {{
      background: linear-gradient(135deg, #facc15, #f97316);
    }}

    .rationality-pill--danger span:first-child {{
      background: linear-gradient(135deg, #f97373, #dc2626);
    }}

    .section-divider {{
      border-top: 1px solid rgba(15,23,42,1);
      margin: 1.1rem 0 0.9rem;
    }}
    details-list {{
      margin-top: 1rem;
    }}

    details {{
      margin-bottom: 0.9rem;
      border-radius: 14px;
      background: radial-gradient(circle at top left, rgba(15,23,42,1), rgba(2,6,23,0.98));
      border: 1px solid rgba(55,65,81,0.9);
      padding: 0.9rem 1rem 0.2rem;
      box-shadow: 0 12px 28px rgba(15,23,42,0.8);
    }}

    summary {{
      cursor: pointer;
      font-size: 0.9rem;
      font-weight: 600;
      color: #f3f4f6;
      letter-spacing: 0.02em;
      margin-bottom: 0.6rem;
      outline: none;
      list-style: none;
    }}

    summary::-webkit-details-marker {{
      display: none;
    }}

    summary::after {{
      content: "›";
      float: right;
      font-size: 1.1rem;
      color: var(--accent-muted);
      transform: rotate(0deg);
      transition: transform 0.25s ease;
      margin-left: 0.5rem;
    }}

    details[open] summary::after {{
      transform: rotate(90deg);
    }}

    .chunk-block pre {{
      margin-top: 0.4rem;
      padding: 0.85rem;
      background: #0b1120;
      border-radius: 12px;
      font-size: 0.78rem;
      line-height: 1.45;
      white-space: pre-wrap;
      color: #d1d5db;
      border: 1px solid rgba(56,189,248,0.2);
    }}

    .global-section strong {{
      color: #e2e8f0;
    }}

    .footer {{
      margin-top: 2rem;
      padding-top: 1.4rem;
      border-top: 1px solid rgba(55,65,81,0.7);
      text-align: center;
      font-size: 0.8rem;
      color: var(--text-subtle);
    }}

    .footer a {{
      color: var(--accent-muted);
      text-decoration: none;
    }}

    .footer a:hover {{
      text-decoration: underline;
    }}

    .grok-card {{
      margin-top: 1rem;
      border-radius: 14px;
      background: radial-gradient(circle at top left, rgba(25,25,55,1), rgba(10,10,25,1));
      border: 1px solid rgba(100,100,160,0.5);
      padding: 1rem;
      box-shadow: 0 10px 24px rgba(15,23,42,0.7);
    }}

    .grok-title {{
      font-size: 1rem;
      font-weight: 600;
      margin-bottom: 0.6rem;
      color: #e0e7ff;
      letter-spacing: 0.03em;
    }}

    .grok-section-title {{
      font-size: 0.85rem;
      font-weight: 600;
      margin-top: 0.8rem;
      margin-bottom: 0.35rem;
      color: #c7d2fe;
    }}

    .grok-body {{
      font-size: 0.85rem;
      line-height: 1.5;
      color: #dbeafe;
    }}

  </style>
</head>

<body>
<div class="page">
<main>
<div class="header">

  <div class="logo-stack">
    <div class="badge-pill">
      <div class="badge-dot"></div>
      <span>MindPilot</span>
      <span class="badge-sub">Cognitive Flight Report</span>
    </div>
  </div>

  <div class="header-right">
    <div class="nav-row">
      <a class="nav-link" href="/">Home</a>
      <div class="nav-dot"></div>
      <a class="nav-link" href="/legal/terms.html">Terms</a>
      <div class="nav-dot"></div>
      <a class="nav-link" href="/legal/privacy.html">Privacy</a>
      <div class="nav-dot"></div>
      <a class="nav-link" href="/legal/accessibility.html">Accessibility</a>
    </div>
  </div>

</div>

<div class="main-grid">

  <div>

    <div class="meta-panel">
      <div class="meta-row">
        <span class="meta-row-label">Source</span>
        <span class="meta-row-value">{source_label}</span>
      </div>

      <div class="meta-row">
        <span class="meta-row-label">Chunks</span>
        <span class="meta-row-value">{total_chunks}</span>
      </div>

      <div class="meta-row">
        <span class="meta-row-label">URL</span>
        <span class="meta-row-value">{source_url_html}</span>
      </div>

      <div class="meta-hint">
        Report auto-generated by MindPilot Engine
      </div>
    </div>

    <div class="section-divider"></div>

    <div class="card global-section">
      <div class="card-inner">
        <div class="card-title">Cognitive Flight Summary</div>
        <div class="summary-text">{esc_full}</div>
      </div>
    </div>

    <div class="card global-section" style="margin-top: 1rem;">
      <div class="card-inner">
        <div class="card-title">Master Fallacy &amp; Bias Map</div>
        <div class="summary-text">{esc_map}</div>
      </div>
    </div>

    <div class="card global-section" style="margin-top: 1rem;">
      <div class="card-inner">
        <div class="card-title">Rationality Profile</div>
        <div class="summary-text">{esc_profile}</div>
      </div>
    </div>

    <div class="card global-section" style="margin-top: 1rem;">
      <div class="card-inner">
        <div class="card-title">Investor / Decision-Maker Summary</div>
        <div class="summary-text">{esc_investor}</div>
      </div>
    </div>

    <div class="card global-section" style="margin-top: 1rem;">
      <div class="card-inner">
        <div class="card-title">Critical Thinking Questions</div>
        <div class="summary-text">{esc_questions}</div>
      </div>
    </div>

    <!-- GROK CARD (optional) -->
    {""
    if not grok_insights
    else f'''
    <div class="grok-card">
      <div class="grok-title">MindPilot × Grok Live Context & Creative Debrief</div>
      <div class="grok-body">{esc_grok}</div>
    </div>
    '''}

  </div>

  <div>
    <h2 style="font-size: 1.1rem; margin-top: 0;">Chunk-Level Analyses</h2>
    <p style="color: var(--text-soft); font-size: 0.85rem; margin-bottom: 0.8rem;">
      Expand any chunk below to see detailed reasoning analysis for that section.
    </p>

    <div class="details-list">
"""

      <!-- LOOP THROUGH CHUNKS -->
      {''.join(
        f"""
        <details class="chunk-block">
          <summary>Chunk {i+1} of {total_chunks}</summary>
          <pre>{esc_chunks[i]}</pre>
        </details>
        """
        for i in range(total_chunks)
      )}
    </div>

  </div>

</div>  <!-- end .main-grid -->

<div class="footer">
  © 2025 Insight Dynamics, LLC • All Rights Reserved •
  <a href="/">mind-pilot.ai</a>
</div>

</main>
</div>

</body>
</html>
""".strip()

    return html
