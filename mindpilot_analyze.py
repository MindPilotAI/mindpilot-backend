import os
import re
import textwrap

def clean_question_line(line: str) -> str:
    """
    Normalize a line that might be a bullet/numbered item into a clean question.
    Keeps the logic centralized so it's easy to tune without regex warnings.
    """
    if not line:
        return ""

    s = line.strip()

    # Strip common markdown-ish prefixes and list markers
    s = re.sub(r"^[#>\s]*", "", s)          # headings/quotes/leading whitespace
    s = re.sub(r"^[-*•]\s+", "", s)         # bullets like -, *, •
    s = re.sub(r"^\d+[\).]\s+", "", s)      # numbering like "1. " or "1) "

    s = s.strip()

    # Must contain a question mark to be treated as a question
    if "?" not in s:
        return ""

    if not s.endswith("?"):
        s += "?"

    return s


class TranscriptUnavailableError(RuntimeError):
    """Raised when YouTube transcript is disabled or not found."""
    pass

from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
def escape_html(text: str) -> str:
    if text is None:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

# from mindpilot_llm_client import run_mindpilot_analysis, classify_content


# ---------- CONFIG ----------

TRANSCRIPT_FILE = "mindpilot_transcript_output.txt"
PROMPT_PACK_FILE = "mindpilot_prompt_pack.md"
MAX_CHARS_PER_CHUNK = 2000  # tweak if you want bigger/smaller chunks


# ---------- YOUTUBE HELPERS ----------

def extract_video_id(youtube_url: str) -> str:
    parsed = urlparse(youtube_url)

    if parsed.hostname in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        # watch?v=...
        query = parse_qs(parsed.query)
        if "v" in query:
            return query["v"][0]

        # shorts/VIDEOID or live/VIDEOID style paths
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) >= 2 and path_parts[0] in {"shorts", "live"}:
            return path_parts[1]

    if parsed.hostname == "youtu.be":
        return parsed.path.lstrip("/")

    match = re.search(r"v=([a-zA-Z0-9_-]{11})", youtube_url)
    if match:
        return match.group(1)

    raise ValueError(f"Could not extract video ID from URL: {youtube_url}")



def fetch_transcript_text(video_id: str) -> str:
    """
    Fetches the transcript for a given video ID and returns it as one combined string.
    Uses the modern YouTubeTranscriptApi().fetch(...) interface.
    """
    api = YouTubeTranscriptApi()

    try:
        fetched = api.fetch(video_id, languages=["en"])
        raw_chunks = fetched.to_raw_data()
    except TranscriptsDisabled:
        raise TranscriptUnavailableError("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise TranscriptUnavailableError("No transcript found for this video.")
    except Exception as e:
        raise RuntimeError(f"Unexpected error fetching transcript: {e}")

    full_text = " ".join(chunk.get("text", "") for chunk in raw_chunks)
    full_text = re.sub(r"\s+", " ", full_text).strip()
    return full_text

def save_text_to_file(text: str, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------- CHUNKING & PROMPT GENERATION ----------

def load_transcript(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Transcript file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        raise ValueError("Transcript file is empty.")
    return text


def chunk_text(text: str, max_chars: int):
    """
    Naive but effective: split on sentence-ish boundaries while trying
    to keep each chunk under max_chars.
    """
    sentences = []
    # Very rough sentence split
    for piece in text.replace("?", "?.").replace("!", "!.").split("."):
        piece = piece.strip()
        if not piece:
            continue
        sentences.append(piece + ".")

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def build_chunk_prompt(chunk_text: str, chunk_index: int, total_chunks: int) -> str:
    """
    Build a MindPilot-style reasoning analysis prompt for a single chunk.
    This version uses a compressed taxonomy for stability and consistency.
    """
    header = f"## Chunk {chunk_index + 1} of {total_chunks}\n"

    wrapped_chunk = textwrap.fill(chunk_text, width=100)

    prompt = """
You are MindPilot, a neutral reasoning-analysis copilot.

Your job is to analyze the following segment of a transcript for:
- Logical fallacies (structural reasoning errors)
- Cognitive biases (internal distortions and heuristics)
- Rhetorical manipulation and persuasion tactics

Focus on HOW the argument thinks, not WHAT it believes.
Do NOT fact-check external claims; assess internal reasoning only.

Here is the transcript segment:

---
%s
---

------------------------------
SECTION A — Local Reasoning Summary
In 1–3 short paragraphs, describe:
- What the speaker is trying to argue in this segment.
- How clearly the reasoning is structured.
- Any major strengths or weaknesses in the logic.

------------------------------
SECTION B — Logical Fallacy Scan
Use this compact grid:
- Ad Hominem
- Straw Man
- Appeal to Emotion
- False Dichotomy
- Slippery Slope
- Circular Reasoning
- Hasty Generalization
- Appeal to Authority
- False Cause / Post Hoc
- False Equivalence
- Equivocation
- Loaded Question

You are NOT limited to these. If another fallacy appears, name it and define it briefly.

For each fallacy:
- Name it
- Quote or summarize the location
- Explain why it qualifies

If none appear, say so clearly.

------------------------------
SECTION C — Cognitive Bias Scan
Use this compact grid:
- Confirmation Bias
- Anchoring
- Availability Heuristic
- Overconfidence
- Hindsight Bias
- Sunk Cost
- Framing Effect
- In-Group / Out-Group
- Survivorship Bias
- Dunning–Kruger
- Projection
- Illusion of Explanatory Depth
- Fundamental Attribution Error

You are NOT limited to these.

For each bias:
- Name it
- Show how it influences reasoning
- Explain its distortion

If none appear, say so clearly.

------------------------------
SECTION D — Rhetorical Manipulation & Persuasion
Use this compact grid:
- Cherry-Picking
- Loaded Language
- Whataboutism
- Moving the Goalposts
- Gish Gallop
- Appeal to Ridicule
- False Balance
- Motte-and-Bailey
- Tone Policing
- Flooding the Zone
- False Certainty

For each tactic:
- Name it
- Give a short example from the text
- Explain its effect

------------------------------
SECTION E — Clarity & Evidence Quality
Briefly assess:
- Clarity of communication
- Whether premises support conclusions
- Whether key terms are defined
- Whether evidence is vague, selective, or missing

Do NOT bring external facts; focus on internal coherence.

------------------------------
SECTION F — Reflective Questions
Provide 2–4 neutral questions the reader can ask themselves to think more clearly.

------------------------------
FORMAT TO RETURN
Use exactly these headings:

### A. Local Reasoning Summary
### B. Logical Fallacies
### C. Cognitive Biases
### D. Rhetorical Manipulation & Persuasion
### E. Clarity & Evidence
### F. Reflective Questions
""" % wrapped_chunk

    return header + "\n" + prompt.strip() + "\n\n---\n\n"



def build_global_prompts() -> str:
    """
    Global prompts for full-lesson reasoning analysis + condensed report.
    These are what you'll paste AFTER you have chunk-level outputs.
    """
    prompt = """
# Global MindPilot Reasoning Prompts for This Lesson

Use these AFTER you have generated chunk-level analyses.

---

## 1. Full-Lesson Reasoning Summary

**Prompt:**

"Using all of the chunk-level MindPilot analyses (argument maps, fallacies, biases,
persuasion tactics, manipulation patterns, rationality ratings), write a coherent
full-lesson reasoning summary (6–10 paragraphs). Focus on:

- The overall narrative being presented
- How causal claims are made (strong vs weak)
- How evidence is (or is not) used
- How fallacies, biases, and persuasion tactics show up across the whole segment
- How the reasoning evolves from start to finish."

---

## 2. Master Fallacy & Bias Map

**Prompt:**

"From all chunk-level analyses, build a master map of:

- Logical fallacies (F domain)
- Cognitive biases (B domain)
- Rhetorical/persuasion tactics (R domain)
- Manipulative/conditioning patterns (M domain)

For each category:
- List the specific types detected
- Describe how often they appear (Low/Medium/High)
- Summarize the overall impact on reasoning quality."

---

## 3. Rationality Profile for the Entire Segment

**Prompt:**

"Combine all chunk-level 'Rationality Flight Reports' into a single
lesson-level rationality profile. Include:

- A 1–2 paragraph overview of reasoning strengths
- A 1–2 paragraph overview of reasoning weaknesses
- A table or structured list of key reasoning dimensions
  (Evidence use, Causal reasoning, Emotional framing, Fairness/balance,
   Motive attribution, etc.) with 1–5 ratings
- A final overall reasoning score on a 0–100 scale.
- At the very end of this section, add a standalone line in this exact format:
  "Overall reasoning score: NN/100"
  where NN is an integer from 0 to 100."


---

## 4. Condensed Investor-Facing Summary

**Prompt:**

"Imagine you are MindPilot generating a concise, investor-facing summary
of this analysis.

In 3–6 short paragraphs, explain:

1. What the content is about (one or two sentences).
2. What MindPilot detected in terms of reasoning:
   - Main fallacy/bias patterns
   - Degree of emotional/persuasive framing
   - Overall rationality score.
3. Why this demonstrates the value of MindPilot as a product:
   - Ability to turn raw media into structured reasoning diagnostics
   - Potential use cases (media literacy, education, compliance, etc.)

Keep it clear, punchy, and non-technical. Focus on showcasing MindPilot's capabilities."
"""
    return prompt.strip() + "\n"

def build_global_summary_prompt(chunk_analyses):
    """
    Build a prompt that summarizes all chunk-level MindPilot analyses into:
    - a full-lesson reasoning summary,
    - a master fallacy & bias map,
    - a rationality profile,
    - an investor-facing summary, and
    - a set of critical-thinking questions based on detected errors.
    """
    joined_analyses = "\n\n---\n\n".join(
        f"Chunk {i+1} Analysis:\n{text}"
        for i, text in enumerate(chunk_analyses)
    )

    prompt = f"""
You are MindPilot, a neutral reasoning-analysis copilot.

You have already analyzed several chunks of a single piece of content.
Below are your own chunk-level analyses, including argument maps,
fallacies, biases, rhetorical tactics, manipulation patterns, and
rationality scores.

Using ONLY those analyses (do not invent new content), produce a single
global report in clean Markdown with the following EXACT numbered
headings:

# 1. Full-Lesson Reasoning Summary
- In 6–10 paragraphs, explain the overall narrative of the content.
- Describe how causal claims are made (well-supported vs. speculative).
- Note how evidence is used or not used.
- Summarize how fallacies, biases, and persuasion tactics appear across
  the whole segment.
- Keep the tone neutral and descriptive, not partisan or emotional.

# 2. Master Fallacy & Bias Map
- List the main logical fallacies (F domain) detected across all chunks.
- List the main cognitive biases (B domain).
- List key rhetorical/persuasion tactics (R domain).
- List any notable manipulative/conditioning patterns (M domain).
- For each, briefly describe:
  - how it shows up in the content, and
  - how often it appears (Low / Medium / High).
Format this section in EXACTLY this markdown pattern:

- **Logical Fallacies**
  - **[Name]**: description (High/Medium/Low)
  - **[Name]**: description (High/Medium/Low)

- **Cognitive Biases**
  - **[Name]**: description (High/Medium/Low)

- **Rhetorical / Persuasion Tactics**
  - **[Name]**: description (High/Medium/Low)

- **Manipulative / Conditioning Patterns**
  - **[Name]**: description (High/Medium/Low)  

# 3. Rationality Profile for the Entire Segment
- Create a short overview of reasoning strengths.
- Create a short overview of reasoning weaknesses.
- Provide a list of reasoning dimensions with 1–5 ratings in this *exact* line format:
  - Evidence use: 3/5
  - Causal reasoning: 2/5
  - Emotional framing: 4/5
  - Fairness/balance: 3/5
  - Motive attribution: 3/5
- At the very end of this section, add a standalone line in this exact format:
  Overall reasoning score: NN/100
- Keep this grounded in your own chunk-level findings.


# 4. Condensed Investor-Facing Summary
- In 3–6 short paragraphs, describe:
  - What the content is about (1–2 sentences).
  - What MindPilot found (main fallacies/biases/persuasion patterns,
    overall rationality level).
  - Why this demonstrates the value of MindPilot as a product (media
    literacy, education, compliance, etc.).
- Keep it punchy and non-technical, suitable for an investor demo.

# 5. Critical Thinking Questions to Ask Yourself
- Based strictly on the most important fallacies, biases, rhetorical
  tactics, and manipulation patterns you have already identified.
- Write 6–12 specific, neutral questions that a thoughtful reader could
  ask or research to counteract these errors in judgment.
- Group questions under short subheadings that correspond to the main
  patterns (e.g., "Bandwagon & Social Pressure", "Confirmation Bias &
  Selective Evidence", "Appeals to Fear", etc.).
- Make the questions actionable and reflective, for example:
  - "What evidence would I need to see to change my mind about X?"
  - "Whose perspectives are missing from this narrative?"
  - "Am I accepting this claim mainly because it feels familiar or
     aligns with my group identity?"
- DO NOT tell the reader what to believe; focus on how they can think
  more clearly and investigate further.

Here are your chunk-level analyses to base this on:

----------------- BEGIN CHUNK ANALYSES -----------------

{joined_analyses}

----------------- END CHUNK ANALYSES -------------------
"""
    return prompt.strip()

# Standalone social snippet page, reusing build_social_card_html so it matches
# the card rendered at the top of the full report.


def build_social_card_html(
    *,
    source_type: str,
    overall_score_100: int | None,
    score_label: str,
    fallacy_snippet: str,
    questions_snippet: str,
    grok_line: str | None,
    report_url: str | None,
    source_url: str | None,
    escape_html,
    creator_checklist_html: str = "",
) -> str:
    """
    Build the reusable social card HTML block used both:
    - at the top of the full report
    - on the standalone /social page.
    """

    # --- Header line: MindPilot + snapshot title (link to source when possible) ---
    brandline = "MindPilot · Your Co-Pilot for Critical Thinking"
    snapshot_label = "Reasoning Snapshot for this Article / Media"

    if source_url:
        # Prefer linking directly to the original article / media
        header_title_html = (
            f'<a href="{escape_html(source_url)}" '
            'class="social-source-link" '
            'style="color: inherit; text-decoration: underline;">'
            f"{escape_html(snapshot_label)}</a>"
        )
    elif report_url:
        # Fallback: link to the MindPilot report if we don't have the original URL
        header_title_html = (
            f'<a href="{escape_html(report_url)}" '
            'class="social-source-link" '
            'style="color: inherit; text-decoration: underline;">'
            f"{escape_html(snapshot_label)}</a>"
        )
    else:
        # Last resort: plain text
        header_title_html = snapshot_label

    header_title_html = header_title_html.strip()

    # --- Score display + bar width ---
    if overall_score_100 is None:
        score_display = "–"
        bar_width = 60  # neutral mid-bar when we don't have a numeric score
        bar_caption = score_label or "Reasoning profile overview"
    else:
        score_display = f"{overall_score_100}/100"
        # keep bar visually shorter; clamp between 10% and 90%
        bar_width = max(10, min(overall_score_100, 90))
        bar_caption = score_label

    # --- Normalize raw snippet text ---
    fallacy_text = (
        fallacy_snippet.strip()
        if fallacy_snippet
        else "Key fallacy and bias signals highlighted in the full report."
    )
    questions_text = (
        questions_snippet.strip()
        if questions_snippet
        else "See the full report for critical questions to stress-test this piece."
    )

    # --- Questions block: normalize and de-duplicate ---
    question_lines: list[str] = []

    if questions_text:
        # Split on line breaks
        raw_parts = [ln.strip() for ln in questions_text.splitlines() if ln.strip()]

        normalized = []
        seen = set()

        for part in raw_parts:

            cleaned = clean_question_line(part)
            if not cleaned:
                continue

            # Must end in a "?" to count as a real question
            if "?" not in cleaned:
                continue
            if not cleaned.endswith("?"):
                cleaned += "?"

            # Normalize for dedup
            key = re.sub(r"\W+", "", cleaned).lower()

            if key not in seen:
                seen.add(key)
                normalized.append(cleaned)

        # Take at most 2
        question_lines = normalized[:2]
    # --- Render questions block HTML ---
    if question_lines:
        questions_block_html = (
            '<ul class="social-questions social-questions-list">\n' +
            "\n".join(
                f"  <li>{escape_html(q)}</li>"
                for q in question_lines
            ) +
            "\n</ul>"
        )
    else:
        # Fallback: show the raw snippet as a paragraph
        questions_block_html = f'<p class="social-text">{escape_html(questions_text)}</p>'

    def slice_questions_by_tier(questions: list[str], tier: str, depth: str) -> list[str]:
        if tier in ("free", "anonymous"):
            return questions[:4]

        if depth == "quick":
            return questions[:8] if len(questions) >= 6 else questions[:6]

        # full
        return questions[:12] if len(questions) >= 8 else questions[:8]

    # --- Grok enrichment: collapse to a single punchy line ---
    grok_text_raw = (grok_line or "").strip()
    grok_display = ""
    if grok_text_raw:
        # Split into non-empty lines
        lines = [ln.strip() for ln in grok_text_raw.splitlines() if ln.strip()]

        if lines:
            # Heuristic: drop a heading-like first line (often a title)
            if len(lines) >= 2:
                first = lines[0]
                first_lower = first.lower()
                if (
                        first.startswith("#")
                        or first_lower.startswith("grok enrichment")
                        or (len(first) < 140 and "." not in first)
                ):
                    lines = lines[1:]

            # Collapse remaining lines and take the first sentence-ish chunk
            collapsed = " ".join(lines).strip()
            if collapsed:
                first_sentence = collapsed.split(". ")[0].strip()
                # Strip leading markdown noise
                first_sentence = re.sub(r"^[#*\s]+", "", first_sentence)
                if first_sentence and not first_sentence.endswith("."):
                    first_sentence += "."
                grok_display = first_sentence

    # --- Footer link: short visible text, full URL behind link ---
    if report_url:
        footer_left = (
            'Full Cognitive Flight Report → '
            f'<a href="{escape_html(report_url)}" '
            'style="color: inherit; text-decoration: underline;">'
            'View detailed analysis</a>'
        )
    else:
        footer_left = "Full Cognitive Flight Report available in the MindPilot app."

    # --- Tiny fallacy table: Pattern | Severity (top 3) ---
    # Fixed Cockpit rule: ALWAYS render the table and ALWAYS render the Severity column.
    # If severity is unavailable (gated or missing), show a locked badge instead of blank.
    fallacy_table_html = ""
    rows: list[tuple[str, str]] = []

    if fallacy_text:
        raw_items = [item.strip() for item in fallacy_text.split(";") if item.strip()]
        for item in raw_items[:3]:
            pattern = item
            severity = ""
            if "(" in item and item.endswith(")"):
                base, _, tail = item.rpartition("(")
                pattern = base.strip(" ,")
                severity = tail[:-1].strip()  # drop trailing ')'
            rows.append((pattern, severity))

    # If we got no rows at all, still render a stable 3-row placeholder
    if not rows:
        rows = [
            ("Fallacy / bias signal (sample)", ""),
            ("Framing / persuasion cue (sample)", ""),
            ("Evidence / inference weakness (sample)", ""),
        ]

    # If NONE of the rows have severity, treat severity as locked
    severity_locked = not any(sev.strip() for _, sev in rows)

    row_html = "\n".join(
        (
            f"<tr><td>{escape_html(pattern)}</td>"
            f'<td class="severity-cell">'
            f'{escape_html(severity) if severity.strip() else locked_badge_html("Locked")}'
            f"</td></tr>"
        )
        for pattern, severity in rows
    )

    locked_class = " locked" if severity_locked else ""
    fallacy_table_html = f"""
      <table class="social-fallacy-table{locked_class}">
        <thead>
          <tr><th>Pattern</th><th>Severity</th></tr>
        </thead>
        <tbody>
        {row_html}
        </tbody>
      </table>
    """.rstrip()

    fallacy_block_html = fallacy_table_html

    # --- Final card HTML ---
    return f"""
            <section class="card-sub social-card" id="mp-social-card">
        <div class="social-header">
          <div class="social-title-block">
            <div class="social-brandline">{escape_html(brandline)}</div>
            <div class="social-title">{header_title_html}</div>
          </div>

          <div class="social-score-row">
            <div class="social-score-number">{escape_html(score_display)}</div>
            <div class="score-bar-track">
              <div class="score-bar-fill" style="width: {bar_width}%;"></div>
            </div>
            <div class="score-bar-label">{escape_html(bar_caption)}</div>
          </div>

          <div class="social-logo">
            <img
              src="https://mind-pilot.ai/assets/mindpilot-symbol.png"
              alt="MindPilot symbol"
            />
          </div>
        </div>

        <div class="social-grid">
          <div>
            <div class="social-label">Fallacies &amp; bias signals</div>
            {fallacy_block_html}
          </div>
          <div>
            <div class="social-label">Questions to ask</div>
            {questions_block_html}
          </div>
        </div>
    """ + (
        f"""
        <div class="social-grok">
          <span class="social-label-inline">Grok Enrichment</span>
          <span class="social-grok-text">{escape_html(grok_display)}</span>
        </div>
    """ if grok_display else ""
    ) + f"""
        <div class="social-footer">
          <span>{footer_left}</span>
          <span class="social-watermark">MindPilot · Cognitive Flight Report</span>
        </div>
        {creator_checklist_html}
      </section>
    
    """

def build_social_page_html(
    *,
    source_type: str,
    overall_score_100: int | None,
    score_label: str,
    fallacy_snippet: str,
    questions_snippet: str,
    grok_line: str | None,
    report_url: str | None,
) -> str:
    """
    Return a standalone HTML page that just contains the social card.

    We'll use this for:
      - quickly grabbing a PNG via browser screenshot / html2canvas
      - sharing a clean linkable snippet page per report
    """

    def _escape_html(text: str) -> str:
        if text is None:
            return ""
        return (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
        )

    # Reuse the existing card builder (which already supports the Grok one-liner)
    card_html = build_social_card_html(
        source_type=source_type,
        overall_score_100=overall_score_100,
        score_label=score_label,
        fallacy_snippet=fallacy_snippet,
        questions_snippet=questions_snippet,
        grok_line=grok_line,        # ⬅️ Grok one-liner flows straight into the card
        report_url=report_url,
        source_url=None,     # no direct source link on teh standalone snippet page (for now)
        escape_html=_escape_html,
        creator_checklist_html="",
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>MindPilot Reasoning Snapshot</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {{
      --dark-navy: #0B1B33;
      --sky-blue: #4FD1C5;
      --soft-bg: #020617;
      --card-bg: #020617;
      --border-subtle: rgba(148, 163, 184, 0.6);
      --text-main: #E2E8F0;
      --text-muted: #94A3B8;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top left, #0B1B33, #020617 55%, #000000 100%);
      color: var(--text-main);
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      padding: 1.5rem 0.75rem;
    }}
    .page {{
      width: 100%;
      max-width: 600px;
    }}
    .brand-top {{
      text-align: center;
      margin-bottom: 0.75rem;
      font-size: 0.82rem;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--text-muted);
    }}
    .brand-top strong {{
      color: #E2E8F0;
    }}
    .instruction-block {{
      border-left: 4px solid var(--sky-blue);
      background: rgba(79, 209, 197, 0.04); /* very light tint */
      padding: 12px 16px;
      margin: 16px 0;
      border-radius: 8px;
    }}
    .instruction-block h3,
    .instruction-block h4 {{
      margin-top: 0;
    }}

    /* Card styling (matches your social card, but tuned for standalone use) */
    .social-card {{
      border-radius: 1.1rem;
      border: 1px solid var(--border-subtle);
      padding: 1rem 1.1rem;
      background: radial-gradient(circle at top left, #0B1B33, #020617);
      color: var(--text-main);
      box-shadow: 0 22px 45px rgba(15, 23, 42, 0.75);
    }}
    .social-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.9rem;
      margin-bottom: 0.65rem;
      flex-wrap: wrap; /* lets things wrap nicely on narrow screens */
    }}
    .social-title-block {{
      display: flex;
      flex-direction: column;
      gap: 0.15rem;
      flex: 1 1 180px; /* title takes the flexible left side */
    }}
    .social-brandline {{
      font-size: 0.85rem;
      font-weight: 600;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: #A0AEC0;
    }}
    .social-title {{
      font-size: 0.9rem;
      font-weight: 600;
      line-height: 1.35;
    }}
    .social-logo img {{
      width: 72px;
      height: 72px;
      display: block;
    }}
    .social-score-row {{
      display: flex;
      align-items: center;
      gap: 0.6rem;
      flex: 0 0 260px; /* lives as a center column on typical desktop width */
    }}
    .social-questions {{
      margin: 0;
      padding-left: 1.1rem;
      font-size: 0.83rem;
      color: #E2E8F0;
    }}
    .social-questions li {{
      margin-bottom: 0.15rem;
    }}
    .score-bar-wrapper {{
      flex: 0 0 260px;   /* HARD width */
    }}
    .social-score-number {{
      font-size: 0.95rem;
      font-weight: 600;
      padding: 0.15rem 0.6rem;
      border-radius: 999px;
      border: 1px solid rgba(226, 232, 240, 0.85);
      background: rgba(15, 23, 42, 0.85);
      text-align: center;
      min-width: 3.2rem;
    }}
    .score-bar-track {{
      height: 0.5rem;
      border-radius: 999px;
      background: rgba(148, 163, 184, 0.45);
      overflow: hidden;
      max-width: 260px;  /* roughly half the card width */
    }}
    .score-bar-fill {{
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, #4FD1C5, #3182CE);
    }}
    .score-bar-label {{
      grid-column: 1 / -1;
      font-size: 0.78rem;
      color: #CBD5F5;
    }}
    .social-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 0.55rem 1rem;
      margin-bottom: 0.65rem;
    }}
    .social-label {{
      font-size: 0.72rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--text-muted);
      margin-bottom: 0.18rem;
    }}
    .social-text {{
      margin: 0;
      font-size: 0.83rem;
      line-height: 1.45;
      color: #E2E8F0;
    }}
        .social-text {{
      margin: 0;
      font-size: 0.83rem;
      line-height: 1.45;
      color: #E2E8F0;
    }}
    .social-fallacy-table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 0.2rem;
        font-size: 0.78rem;
        color: #E2E8F0;
    }}
    .social-fallacy-table th,
    .social-fallacy-table td {{
      padding: 0.2rem 0.4rem;
      vertical-align: top;
    }}

    .social-fallacy-table thead th {{
      font-size: 0.65rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: #A0AEC0;
      border-bottom: 1px solid rgba(148, 163, 184, 0.5);
    }}

    .social-fallacy-table tbody tr {{
      border-bottom: 1px solid rgba(148, 163, 184, 0.15);
    }}

    .social-fallacy-table tbody tr:last-child {{
      border-bottom: none;
    }}
    .social-fallacy-table td:first-child {{
    width: 90%;      /* fill more of its half */
    }}
    .social-fallacy-table td:last-child {{
      white-space: nowrap;
      text-align: right;
    }}
    .social-fallacy-table .severity-cell {{
      text-align: right;
      white-space: nowrap;
      font-weight: 600;
      color: #FBD38D; /* gentle attention */
    }}
    .social-grok {{
      margin-top: 0.4rem;
      margin-bottom: 0.55rem;
      font-size: 0.8rem;
      color: #E2E8F0;
    }}
    .social-label-inline {{
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 0.7rem;
      margin-right: 0.35rem;
      color: #A0AEC0;
    }}
    .social-footer {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 0.75rem;
      font-size: 0.75rem;
      color: var(--text-muted);
    }}
    .social-watermark {{
      font-weight: 500;
      color: #81E6D9;
      white-space: nowrap;
    }}

    #download-btn {{
      margin-top: 0.8rem;
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      padding: 0.4rem 0.8rem;
      border-radius: 999px;
      border: 1px solid rgba(148, 163, 184, 0.9);
      background: rgba(15, 23, 42, 0.9);
      color: #E2E8F0;
      font-size: 0.8rem;
      cursor: pointer;
    }}
  </style>
  <script src="https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js"></script>
</head>
<body>
  <div class="page">
    <div class="brand-top">
      <strong>MindPilot</strong> · Your co-pilot for critical thinking
    </div>
    {card_html}
    <button id="download-btn" type="button">
      Download snippet as image
    </button>
  </div>
  <script>
    (function() {{
      const btn = document.getElementById('download-btn');
      const card = document.getElementById('mp-social-card');
      if (!btn || !card) return;
      btn.addEventListener('click', function() {{
        html2canvas(card, {{
          useCORS: true,
          backgroundColor: null,
          scale: 2
        }}).then(function(canvas) {{
          const link = document.createElement('a');
          link.download = 'mindpilot-reasoning-snapshot.png';
          link.href = canvas.toDataURL('image/png');
          link.click();
        }});
      }});
    }})();
  </script>
</body>
</html>
"""

# Social card: this is the same design used on /social/{report_id}
# We render it at the top of the full report for continuity from social → report.

def parse_master_map_items(raw_map: str) -> list[dict]:
    """
    Parse the Master Fallacy & Bias Map into structured items:
      {category, name, severity, description}

    Expected input resembles:
      - **Logical Fallacies**
        - **Straw Man**: ... (High)
    """
    if not raw_map:
        return []

    items: list[dict] = []
    current_category = ""

    for line in raw_map.splitlines():
        s = line.strip()
        if not s:
            continue

        # Category headers like: - **Logical Fallacies**
        if s.startswith("- **") and s.endswith("**") and ":" not in s:
            current_category = s.replace("- **", "").replace("**", "").strip()
            continue

        # Item lines like: - **Name**: description (High)
        m = re.match(r"^-?\s*\*\*(.+?)\*\*:\s*(.+)$", s.replace("- ", ""), flags=re.IGNORECASE)
        if not m:
            continue

        name = m.group(1).strip()
        tail = m.group(2).strip()

        severity = ""
        desc = tail

        sev_match = re.search(r"\((High|Medium|Med|Low)\)\s*$", tail, flags=re.IGNORECASE)
        if sev_match:
            sev = sev_match.group(1).lower()
            severity = "Medium" if sev == "med" else sev.capitalize()
            desc = re.sub(r"\((High|Medium|Med|Low)\)\s*$", "", tail, flags=re.IGNORECASE).strip()

        items.append(
            {
                "category": current_category or "Pattern group",
                "name": name,
                "severity": severity or "Medium",
                "description": desc,
            }
        )

    return items


def extract_overall_score_100(rationality_profile: str) -> int | None:
    if not rationality_profile:
        return None
    m = re.search(r"Overall reasoning score:\s*([0-9]{1,3})\s*/\s*100", rationality_profile, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        val = int(m.group(1))
        return max(0, min(100, val))
    except Exception:
        return None


def build_pro_quick_creator_checklist_html(
    *,
    master_map: str,
    rationality_profile: str,
    questions_block: str,
) -> str:
    """
    Deterministic, rule-based checklist. No extra LLM calls.
    Triggers are driven by patterns present in master_map + rationality_profile.
    """
    items = parse_master_map_items(master_map)
    score100 = extract_overall_score_100(rationality_profile)

    # Normalize names for matching
    names = [it["name"].lower() for it in items]
    severities = {it["name"].lower(): it.get("severity", "Medium") for it in items}

    def has(pattern: str) -> bool:
        p = pattern.lower()
        return any(p in n for n in names)

    def severity_is_high(pattern: str) -> bool:
        p = pattern.lower()
        for n in names:
            if p in n:
                return severities.get(n, "").lower() == "high"
        return False

    checklist: list[tuple[str, str]] = []

    # --- Baseline “creator hygiene” (always show) ---
    checklist.append(("Clarity", "State your central claim in one sentence (then restate it in plain language)."))
    checklist.append(("Receipts", "Add at least 1 concrete example, statistic, quote, or observable fact for each major claim."))

    # --- Evidence + reasoning (score-triggered) ---
    if score100 is not None and score100 < 65:
        checklist.append(("Evidence", "Add one paragraph explicitly separating: facts, assumptions, and interpretation."))
    if score100 is not None and score100 < 55:
        checklist.append(("Structure", "Rewrite the intro so the argument steps are numbered (Premise → Premise → Conclusion)."))

    # --- Fallacy/bias-triggered items (mapping rules) ---
    if has("straw man") or has("motte") or has("false dichotomy"):
        checklist.append(("Opposition", "Steelman the best opposing view in 2–4 sentences before you rebut it."))

    if has("confirmation bias") or has("cherry-picking") or has("availability"):
        checklist.append(("Balance", "Add one disconfirming datapoint (or limitation) and explain why it doesn’t overturn your conclusion."))

    if has("appeal to emotion") or has("loaded language") or has("fear"):
        checklist.append(("Tone", "Replace 3–5 emotionally loaded phrases with neutral equivalents (keep the meaning, lower the heat)."))

    if has("gish gallop") or has("flooding") or has("whataboutism"):
        checklist.append(("Focus", "Cut to your top 3 claims; move the rest to a ‘notes’ section or a follow-up post."))

    if has("appeal to authority"):
        checklist.append(("Authority", "Cite the underlying evidence (study/data) — not just the credential/title."))

    if has("false cause") or has("post hoc") or has("causal"):
        checklist.append(("Causality", "Add one sentence stating: correlation vs causation, and what would falsify your causal claim."))

    if has("in-group") or has("out-group") or has("attribution"):
        checklist.append(("Attribution", "Replace motive claims with behavior claims; label uncertainty where you can’t infer intent."))

    # --- Severity escalation (if any HIGH patterns, force 2 extra actions) ---
    if any(it.get("severity", "").lower() == "high" for it in items):
        checklist.append(("High-risk", "Add a ‘fair-minded disclaimer’ sentence: what your argument is NOT claiming."))
        checklist.append(("High-risk", "Add a ‘scope boundary’ sentence: where your claim stops / doesn’t generalize."))

    # --- Questions block (if present, remind creator to use it as QA) ---
    if questions_block and len(questions_block.strip()) > 40:
        checklist.append(("QA", "Answer the top 2 ‘Critical Thinking Questions’ in your draft before publishing."))

    # Render HTML (compact)
    rows = "\n".join(
        f'<li><strong>{escape_html(tag)}:</strong> {escape_html(text)}</li>'
        for tag, text in checklist[:12]  # hard cap for quick mode
    )

    return f"""
      <section class="card">
        <div class="section-label">Pro Quick</div>
        <h2>Creator pre-publish checklist</h2>
        <p class="muted">
          Automatically generated from the patterns detected in this report. (No extra AI calls.)
        </p>
        <ul class="checklist">
          {rows}
        </ul>
      </section>
    """.strip()
# single source of truth for lock badging
def locked_badge_html(label: str = "Locked") -> str:
    return f'<span class="locked-badge">{label}</span>'


def build_html_report(
    source_url,
    report_id,
    total_chunks,
    chunk_analyses,
    global_report,
    grok_insights: str | None = None,
    depth: str = "full",  # "quick" or "full"
    creator_checklist_mode: str = "none",  # "none" | "pro_quick"
):
    SOCIAL_HANDLES = {
        "twitter": "@mindpilotai360",  # update once you decide
        "tiktok": "@mindpilotai",
        "linkedin": "MindPilot · Cognitive Flight Report",
    }
    # ------------------------------------------------------------------
    # Temporary default: report_url will be wired properly later.
    # For now, always define it so we never hit an UnboundLocalError.
    # ------------------------------------------------------------------
    report_url = None
    """
    Build the MindPilot Cognitive Flight Report HTML.

    - Splits the global_report into:
        1) Full-Lesson Reasoning Summary
        2) Master Fallacy & Bias Map
        3) Rationality Profile
        4) Investor-Facing Summary
        5) Critical Thinking Questions
    - Derives an overall 0–100 score and per-dimension 1–5 bars.
    - Includes Grok context if provided.
    - Generates social text snippets for X / TikTok / LinkedIn.
    """

    # ---------- helpers ----------

    def infer_source_type(url: str) -> str:
        if not url:
            return "Pasted text / transcript"
        lower = url.lower()
        if "youtube.com" in lower or "youtu.be" in lower:
            return "YouTube video"
        if lower.startswith("http://") or lower.startswith("https://"):
            return "Article / web page"
        return "Pasted text / transcript"

    def depth_label(depth_value: str) -> str:
        dv = (depth_value or "").strip().lower()
        if dv == "quick":
            return "Quick scan (single-pass global profile)"
        return "Full analysis (section-level scans + global summary)"

    def extract_summary_body(md_section: str) -> str:
        """
        Take a markdown section, drop the heading line if present,
        flatten to a single paragraph.
        """
        if not md_section:
            return ""
        lines = md_section.splitlines()
        if lines and lines[0].lstrip().startswith("#"):
            lines = lines[1:]
        clean_lines = [ln.strip() for ln in lines if ln.strip()]
        text = " ".join(clean_lines)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ---------- split global_report into sections ----------

    full_summary = ""
    master_map = ""
    rationality_profile = ""
    investor_summary = ""
    questions_block = ""


    if global_report:
        raw = global_report
        headings = [
            ("full", "# 1. Full-Lesson Reasoning Summary"),
            ("map", "# 2. Master Fallacy & Bias Map"),
            ("profile", "# 3. Rationality Profile for the Entire Segment"),
            ("investor", "# 4. Condensed Investor-Facing Summary"),
            ("questions", "# 5. Critical Thinking Questions to Ask Yourself"),
        ]
        found = []
        for key, heading in headings:
            idx = raw.find(heading)
            if idx != -1:
                found.append((key, idx, heading))

        if found:
            found.sort(key=lambda x: x[1])
            section_blocks = {}
            for i, (key, idx, heading) in enumerate(found):
                start = idx
                end = found[i + 1][1] if i + 1 < len(found) else len(raw)
                section_blocks[key] = raw[start:end].strip()

            full_summary = section_blocks.get("full", "")
            master_map = section_blocks.get("map", "")
            rationality_profile = section_blocks.get("profile", "")
            investor_summary = section_blocks.get("investor", "")
            questions_block = section_blocks.get("questions", "")

    # ---------- escape chunk analyses ----------

    def _strip_internal_subheadings(block: str) -> str:
        """
        Remove the internal A–F sub-headings we used for the initial PoC
        (Local Reasoning Summary, Logical Fallacies, Cognitive Biases,
        Emotional Framing/Manipulation/Persuasion, Clarity & Evidence,
        Reflective Questions), plus the duplicate global heading.
        """
        if not block:
            return ""
        patterns = [
            # Old A–F PoC section subheadings
            r"^###?\s*[A-F1-6][\.)]\s*Local Reasoning Summary.*$",
            r"^###?\s*[A-F1-6][\.)]\s*Logical Fallacies.*$",
            r"^###?\s*[A-F1-6][\.)]\s*Cognitive Biases.*$",
            r"^###?\s*[A-F1-6][\.)]\s*Emotional.*?(Framing|Manipulation|Persuasion).*$",
            r"^###?\s*[A-F1-6][\.)]\s*Clarity\s*&\s*Evidence.*$",
            r"^###?\s*[A-F1-6][\.)]\s*Reflective Questions.*$",

            # Old numbered PoC headings we no longer want in the body text
            r"^#+\s*1\.\s*Full-Lesson Reasoning Summary.*$",
            r"^#+\s*2\.\s*Master Fallacy\s*&\s*Bias Map.*$",
            r"^#+\s*3\.\s*Rationality Profile for the Entire Segment.*$",
            r"^#+\s*4\.\s*Condensed Investor-Facing Summary.*$",
            r"^#+\s*5\.\s*Critical Thinking Questions to Ask Yourself.*$",
        ]

        for pat in patterns:
            block = re.sub(pat, "", block, flags=re.MULTILINE)
        # collapse excess blank lines
        block = re.sub(r"\n{3,}", "\n\n", block)
        return block.strip()

    def build_fallacy_table(raw_map: str, depth: str = "full") -> str:
        """
        Parse the markdown-style Master Fallacy & Bias Map into an HTML table.

        Expected input pattern (what the global prompt asks for):

        - **Logical Fallacies**
          - **False Cause / Post Hoc**: description ... (High)
          - **Hasty Generalization**: description ... (Medium)

        - **Cognitive Biases**
          - **Confirmation Bias**: description ... (High)

        - **Rhetorical / Persuasion Tactics**
          - **Appeal to Emotion**: description ... (Medium)

        - **Manipulative / Conditioning Patterns**
          - **Bandwagon Conditioning**: description ... (Low)

        We turn that into rows:
            Category | Name | Description (no severity text) | Severity
        """
        if not raw_map:
            return ""

        rows: list[tuple[str, str, str, str]] = []

        # Default bucket so we don't drop items when categories are missing
        current_category: str | None = "Pattern group"

        for line in raw_map.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            # ---------- CATEGORY LINES ----------
            # Normalize out bold markers and trailing colons
            normalized = stripped.replace("**", "").rstrip(":").strip()

            # Bullet-style categories: "- Logical Fallacies"
            m_cat_bullet = re.match(r"^[-*]\s*(.+)$", normalized)
            # Heading-style categories: "### Logical Fallacies"
            m_cat_heading = re.match(r"^#{1,6}\s*(.+)$", normalized)

            m_cat = m_cat_bullet or m_cat_heading
            is_category_line = False
            if m_cat:
                maybe_cat = m_cat.group(1).strip()

                # Only treat as a category if it doesn't look like an item
                if ":" not in maybe_cat:
                    current_category = maybe_cat
                    is_category_line = True

            if is_category_line:
                # Skip further parsing for this line
                continue

            # ---------- ITEM LINES ----------
            # Most common pattern:
            #   - **Name**: description... (High)
            m_item = re.match(
                r"^[-*]\s*\*\*(.+?)\*\*\s*:\s*(.+)$",
                stripped,
            )

            # Fallback:
            #   - Name: description... (High)
            if not m_item:
                m_item = re.match(r"^[-*]\s*([^:]+?):\s*(.+)$", stripped)

            if not m_item:
                continue

            name = m_item.group(1).strip()
            desc = m_item.group(2).strip()

            # Pull severity "(High|Medium|Low)" out of the tail if present
            # Allow an optional period after the closing parenthesis, e.g. "(High)."
            severity = ""
            m_sev = re.search(r"\((High|Medium|Med|Low)\)\.?\s*$", desc)
            if m_sev:
                sev_raw = (m_sev.group(1) or "").strip()
                severity = "Medium" if sev_raw.lower() == "med" else sev_raw
                # Trim the "(High)" off the description
                desc = desc[: m_sev.start()].rstrip(" .-")

            rows.append(
                (
                    current_category or "Pattern group",
                    name,
                    desc,
                    severity,
                )
            )

        if not rows:
            return ""

        # ---------- Build HTML ----------
        body_rows: list[str] = []
        display_rows = rows
        if depth == "quick":
            display_rows = rows[:5]

        for category, name, desc, severity in display_rows:
            sev_class = (
                f"severity-{severity.lower()}" if severity else "severity-none"
            )
            body_rows.append(
                f"""
              <tr>
                <td class="fallacy-type col-type">{escape_html(category)}</td>
                <td class="fallacy-name col-name">{escape_html(name)}</td>
                <td class="fallacy-desc col-desc">{escape_html(desc)}</td>
                <td class="fallacy-severity col-sev">
                  <span class="fallacy-tag {severity.lower() if severity else ''}">
                    {escape_html(severity)}
                  </span>
                </td>
              </tr>
                """
            )

        return (
                """
                <div class="fallacy-table-wrapper">
                  <table class="fallacy-table">
                    <thead>
                      <tr>
                        <th class="col-type">Pattern type</th>
                        <th class="col-name">Fallacy / bias</th>
                        <th class="col-desc">How it shows up in this piece</th>
                        <th class="col-sev">Severity</th>
                      </tr>
                    </thead>
                    <tbody>
                """
                + "".join(body_rows)
                + """
                </tbody>
              </table>
            </div>
            """
        )

    def summarize_fallacies_for_social(master_map: str) -> str:
        """
        Build a compact "Name (Severity)" list from the Master Fallacy & Bias Map.

        Example output:
            "False Cause / Post Hoc (High); Hasty Generalization (High); Cherry-Picking (Medium)"
        """
        if not master_map:
            return ""

        items: list[str] = []

        for line in master_map.splitlines():
            stripped = line.strip()
            if not stripped.startswith("-"):
                continue

            # Most common pattern:
            #   - **False Cause / Post Hoc**: description ... (High).
            m = re.match(
                r"^-\s*\*\*(.+?)\*\*\s*:\s*(.+?)(?:\s*\((High|Medium|Low)\)\.?\s*)?$",
                stripped,
            )
            # Fallback:
            #   - False Cause / Post Hoc: description ... (High).
            if not m:
                m = re.match(
                    r"^-\s*([^:]+?):\s*(.+?)(?:\s*\((High|Medium|Low)\)\.?\s*)?$",
                    stripped,
                )

            if not m:
                continue

            name = m.group(1).strip()
            severity = (m.group(3) or "").strip()

            if severity:
                items.append(f"{name} ({severity})")
            else:
                items.append(name)

        # De-dupe but preserve order
        seen: set[str] = set()
        unique: list[str] = []
        for it in items:
            if it not in seen:
                seen.add(it)
                unique.append(it)

        # At most 6 items for social
        return "; ".join(unique[:6])

    def summarize_questions_for_social(raw_questions: str) -> str:
        """
        Take the global 'Critical Thinking Questions' block and distill
        1–2 short questions for the social card.
        """
        if not raw_questions:
            return ""

        # Prefer markdown bullet lines
        bullets = []
        for line in raw_questions.splitlines():
            stripped = line.strip()
            if stripped.startswith(("- ", "* ")):
                q = stripped[2:].strip()
                if q:
                    bullets.append(q)

        if not bullets:
            # Fallback: break on sentences
            sentences = re.split(r"(?<=[.?])\s+", raw_questions)
            bullets = [s.strip() for s in sentences if s.strip()]

        if not bullets:
            return ""

        # Take up to 2, trim length
        selected = []
        for q in bullets[:2]:
            if len(q) > 140:
                q = q[:137].rstrip() + "…"
            selected.append(q)

        return "\n".join(selected)

    # Clean global sections
    full_summary = _strip_internal_subheadings(full_summary)
    master_map = _strip_internal_subheadings(master_map)
    rationality_profile = _strip_internal_subheadings(rationality_profile)
    investor_summary = _strip_internal_subheadings(investor_summary)
    questions_block = _strip_internal_subheadings(questions_block)
    # --- Fixed Cockpit: tier-based question slicing (Quick + Full) ---
    # Remove the old behavior that collapsed quick questions to 1–2 items.

    # Determine effective tier locally (mindpilot_analyze.py only):
    # - quick + no pro checklist => anonymous behavior
    # - quick + pro checklist enabled => pro_quick behavior
    # - full => pro_full behavior
    effective_tier = "pro_full"
    if depth == "quick":
        effective_tier = "pro_quick" if creator_checklist_mode == "pro_quick" else "anonymous"

    def _extract_question_lines(block: str) -> list[str]:
        if not block:
            return []
        out: list[str] = []
        seen: set[str] = set()

        for raw in block.splitlines():
            s = raw.strip()
            if not s:
                continue

            # Strip bullets / numbering like "- ", "* ", "1. ", "1) "
            s = re.sub(r"^[#>\s\-*•\d\.\)\(]+", "", s).strip()

            # Only keep things that look like real questions
            if "?" not in s:
                continue
            if not s.endswith("?"):
                s += "?"

            key = re.sub(r"\W+", "", s).lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)

        return out

    def _defaults() -> list[str]:
        # Neutral evergreen “stress-test” questions (used only if the model returns too few)
        return [
            "What would change my mind about the main claim?",
            "What key evidence is missing or under-specified?",
            "What alternative explanation fits the same facts?",
            "Which assumptions does the argument rely on most?",
            "What would a strong critic say is the weakest link?",
            "Are there incentives shaping what’s being emphasized or omitted?",
            "What parts are evidence versus interpretation?",
            "What uncertainty is being presented as certainty?",
            "What are the strongest counterexamples to the conclusion?",
            "What testable prediction follows from this argument?",
            "What definitions or terms are being used loosely?",
            "What would this look like if the opposite were true?",
        ]

    def slice_questions_by_tier(questions: list[str], tier: str, depth_: str) -> list[str]:
        # Anonymous: exactly 4
        if tier in ("free", "anonymous"):
            sliced = questions[:4]
            if len(sliced) < 4:
                for q in _defaults():
                    if len(sliced) >= 4:
                        break
                    if q not in sliced:
                        sliced.append(q)
            return sliced[:4]

        # Pro Quick: 6–8
        if depth_ == "quick":
            sliced = questions[:8]
            if len(sliced) < 6:
                for q in _defaults():
                    if len(sliced) >= 6:
                        break
                    if q not in sliced:
                        sliced.append(q)
            return sliced[:8]

        # Full: 8–12
        sliced = questions[:12]
        if len(sliced) < 8:
            for q in _defaults():
                if len(sliced) >= 8:
                    break
                if q not in sliced:
                    sliced.append(q)
        return sliced[:12]

    extracted_questions = _extract_question_lines(questions_block)
    sliced_questions = slice_questions_by_tier(extracted_questions, effective_tier, depth)

    # Rebuild questions_block as markdown bullets so existing rendering stays stable
    questions_block = "\n".join(f"- {q}" for q in sliced_questions) if sliced_questions else ""

    global_report_clean = _strip_internal_subheadings(global_report or "")
    creator_checklist_html = ""

    # ✅ Unlock checklist for paid plans in BOTH quick and full
    # We reuse the same checklist builder for now (stable UI > perfection).
    if creator_checklist_mode in ("pro_quick", "pro_full"):
        creator_checklist_html = build_pro_quick_creator_checklist_html(
            master_map=master_map,
            rationality_profile=rationality_profile,
            questions_block=questions_block,
        )

    # Clean per-chunk analyses
    cleaned_chunk_analyses = [_strip_internal_subheadings(a) for a in chunk_analyses]

    escaped_chunks = [escape_html(a) for a in cleaned_chunk_analyses]

    esc_full = escape_html(full_summary) if full_summary else ""
    esc_map = escape_html(master_map) if master_map else ""
    esc_profile = escape_html(rationality_profile) if rationality_profile else ""
    esc_investor = escape_html(investor_summary) if investor_summary else ""
    esc_questions = escape_html(questions_block) if questions_block else ""
    esc_global_fallback = escape_html(global_report_clean) if (global_report and not esc_full) else ""
    esc_grok = escape_html(grok_insights) if grok_insights else ""
    # Build a richer HTML table view of the Master Map if possible
    fallacy_table_html = build_fallacy_table(master_map)
    # Social-card text snippets (raw)
    fallacy_snippet = summarize_fallacies_for_social(master_map)
    questions_snippet = summarize_questions_for_social(questions_block)

    # Optional one-line Grok summary for the card
    grok_line = ""
    if grok_insights:
        for line in grok_insights.splitlines():
            candidate = line.strip()
            if not candidate:
                continue

            lower = candidate.lower()

            # Skip banner / title-ish lines
            if re.search(r"mindpilot\s*×\s*grok", lower):
                continue
            if lower.startswith("grok enrichment"):
                continue
            if "current context snapshot" in lower:
                continue
            # Very short, no-period line → likely a heading, skip
            if len(candidate) < 80 and "." not in candidate:
                continue

            grok_line = candidate
            # Keep it short for the card
            if len(grok_line) > 160:
                grok_line = grok_line[:157].rstrip() + "…"
            break

    # ---------- dimension bars from Rationality Profile ----------

    dimension_bars_html = ""
    if rationality_profile:
        dimension_scores = []
        for match in re.finditer(
                r"^\s*[-•]?\s*([A-Za-z][A-Za-z\s/]+?):\s*([1-5](?:\.\d+)?)\s*/\s*5\b",
                rationality_profile,
                flags=re.MULTILINE,
        ):
            dim_name = match.group(1).strip()
            try:
                dim_score = float(match.group(2))

            except ValueError:
                continue
            if 1 <= dim_score <= 5:
                dimension_scores.append((dim_name, dim_score))

        if dimension_scores:
            rows = []
            for dim_name, dim_score in dimension_scores:
                width_pct = int(round(dim_score / 5 * 100))
                rows.append(
                    f"""
          <div class="dimension-row">
            <div class="dimension-label">{escape_html(dim_name)}</div>
            <div class="dimension-track">
              <div class="dimension-fill" style="width: {width_pct}%"></div>
            </div>
            <div class="dimension-score">{dim_score}/5</div>
          </div>
            """
                )
            dimension_bars_html = (
                """
        <div class="dimension-bars">
        """
                + "".join(rows)
                + """
        </div>
        """
            )

    # ---------- overall 0–100 score ----------

    overall_score_100: int | None = None
    label_for_chip = ""

    if rationality_profile:
        score_match = re.search(
            r"(?i)overall reasoning score\s*:\s*([0-9]{1,3})\s*/\s*100",
            rationality_profile,
        )
        if not score_match:
            score_match = re.search(
                r"(?i)score[^0-9]{0,80}([0-9]{1,3})\s*/\s*100",
                rationality_profile,
            )
        if score_match:
            try:
                candidate = int(score_match.group(1))
                if 0 <= candidate <= 100:
                    overall_score_100 = candidate
            except ValueError:
                overall_score_100 = None

    score_chip_html = ""
    score_bar_html = ""

    if overall_score_100 is not None:
        band = overall_score_100
        if band < 25:
            label_for_chip = "Very low reasoning quality"
        elif band < 45:
            label_for_chip = "Low / fragile reasoning"
        elif band < 65:
            label_for_chip = "Mixed / uneven reasoning"
        elif band < 85:
            label_for_chip = "Generally strong reasoning"
        else:
            label_for_chip = "Very strong reasoning"

        score_chip_html = f"""
        <div class="pill-row">
          <div class="pill"><strong>Overall reasoning score:</strong> {overall_score_100}/100 · {label_for_chip}</div>
        </div>
        """
        score_bar_html = f"""
        <div class="score-bar-container">
          <div class="score-bar-wrapper">
            <div class="score-bar-track">
            </div>
            <div class="score-bar-fill" style="width: {overall_score_100}%"></div>
          </div>
          <div class="score-bar-label">{overall_score_100}/100 overall reasoning quality</div>
        </div>
        """
    depth_text = depth_label(depth)

    summary_body = extract_summary_body(full_summary)
    preview = summary_body
    if len(preview) > 220:
        preview = preview[:220].rstrip() + "…"

    readable_score = (
        f"{overall_score_100}/100" if overall_score_100 is not None else "reasoning risk detected"
    )

    # ---------- meta + social text snippets ----------

    # 1) Infer source type for wording
    source_type = infer_source_type(source_url or "")

    # 2) Build a simple per-report slug and URL
    #    This ties each snippet/card to a specific report path:
    #    https://mind-pilot.ai/reports/{slug}
    safe_id = (report_id or "report").lower()
    safe_id = re.sub(r"[^a-z0-9]+", "-", safe_id).strip("-")
    if not safe_id:
        safe_id = "report"
    report_url = f"https://mind-pilot.ai/reports/{safe_id}"

    # 3) Build social-card HTML (only if we have at least some signal)
    social_card_html = ""
    if (overall_score_100 is not None) or fallacy_snippet or questions_snippet:
        social_card_html = build_social_card_html(
            source_type=source_type,
            overall_score_100=overall_score_100,
            score_label=label_for_chip or "Reasoning profile overview",
            fallacy_snippet=fallacy_snippet,
            questions_snippet=questions_snippet,
            grok_line=grok_line,
            report_url=report_url,
            source_url=source_url,  # <-- NEW: link snapshot text to the source
            escape_html=escape_html,
        )

    # 4) Canonical, platform-agnostic social snippet (single source of truth)
    #
    # This mirrors the "Canonical MindPilot Social Snippet Template" from your doc.
    # We reuse existing data instead of inventing new fields.
    # - Source label: domain from URL (e.g., "nytimes.com")
    # - Content line: uses the truncated summary preview as a stand-in
    # - Format: derived from source_type
    #
    from urllib.parse import urlparse

    # Source label from URL host
    source_label = "Public source"
    try:
        if source_url:
            parsed = urlparse(source_url)
            host = parsed.netloc or ""
            host = host.lower()
            if host.startswith("www."):
                host = host[4:]
            if host:
                source_label = host
    except Exception:
        pass

    # Content "title" approximation from the preview text
    content_title = preview
    if len(content_title) > 140:
        content_title = content_title[:137].rstrip() + "…"

    # Format bucket from source_type
    lower_type = source_type.lower()
    if "youtube" in lower_type or "video" in lower_type:
        content_format = "Video / Commentary"
    elif "article" in lower_type or "web page" in lower_type:
        content_format = "News / Analysis"
    else:
        content_format = "Analysis / Commentary"

    # Score line for the diagnostic section
    if overall_score_100 is not None:
        score_line = f"{overall_score_100}/100"
    else:
        score_line = "Reasoning risk detected (no numerical score)"

    # Fallacy & bias signals line(s)
    fallacy_text = (fallacy_snippet or "").strip()
    if not fallacy_text:
        fallacy_text = "Reasoning patterns and bias signals surfaced in this piece."

    # Creator-level question
    question_text = (questions_snippet or "").strip()
    if not question_text:
        question_text = "What does this imply for the decisions you’re about to make?"

    # Final link line – tied to this report URL
    link_line = f"View the full Cognitive Flight Report → {report_url}"

    canonical_snippet = textwrap.dedent(
        f"""\
        MindPilot — your co-pilot for critical thinking
        Reasoning Snapshot · Public Media Analysis

        Analyzed: {source_label}
        Content: “{content_title}”
        Format: {content_format}

        Reasoning Risk Detected
        {score_line}
        (Early-warning signal for reasoning quality — not a fact-check)

        Fallacies & Bias Signals Identified
        {fallacy_text}
        (Signals reflect structure and persuasion patterns, not truth claims.)

        A Question This Analysis Raises
        “{question_text}”

        Cross-validated across multiple reasoning models
        (Designed to surface non-obvious weaknesses in arguments.)

        {link_line}
        This same reasoning diagnostic can be run on your articles, research, or drafts.
        """
    ).strip()

    esc_canonical_snippet = escape_html(canonical_snippet)

    # Grok panel — Fixed Cockpit: ALWAYS render
    grok_locked = not bool(esc_grok)
    grok_wrapper_class = "locked" if grok_locked else ""

    if esc_grok:
        grok_body_html = f'<pre class="pre-block">{esc_grok}</pre>'
    else:
        grok_body_html = f"""
          <p class="muted" style="margin:0;">
            {locked_badge_html("Locked")} Grok context is available on Pro tiers.
          </p>
          <p class="muted" style="margin:0.5rem 0 0;">
            Upgrade to unlock: live context, creative debrief, and alternate framings.
          </p>
        """

    grok_card_html = f"""
      <section class="card-sub {grok_wrapper_class}">
        <div class="collapsible-header" onclick="toggleSection('grok-card')">
          <span>Grok Context</span>
          <span class="collapsible-toggle" id="toggle-grok-card">{"Hide" if not grok_locked else "Show"}</span>
        </div>
        <div class="collapsible-body {"open" if not grok_locked else ""}" id="section-grok-card">
          {grok_body_html}
        </div>
      </section>
    """
    # ---------- Debug strip (env-controlled) ----------
    debug_strip_html = ""
    try:
        import os  # safe even if already imported

        if os.getenv("MP_DEBUG_UI", "0") == "1":
            # Use locals().get(...) so missing vars never crash rendering
            _depth = locals().get("depth", "")
            _effective_tier = locals().get("effective_tier", "")
            _creator_mode = locals().get("creator_checklist_mode", "")
            _esc_grok = locals().get("esc_grok", "")
            _questions_block = locals().get("questions_block", "")

            # questions_count: prefer sliced_questions if it exists, else count bullet lines
            _sliced = locals().get("sliced_questions", None)
            if isinstance(_sliced, list):
                q_count = len(_sliced)
            else:
                q_count = sum(
                    1 for ln in str(_questions_block).splitlines()
                    if ln.strip().startswith(("-", "*"))
                )

            debug_items = [
                ("depth", str(_depth)),
                ("effective_tier", str(_effective_tier)),
                ("creator_checklist_mode", str(_creator_mode)),
                ("grok_present", "yes" if bool(_esc_grok) else "no"),
                ("questions_count", str(q_count)),
            ]
            debug_kv = " · ".join(f"{k}={v}" for k, v in debug_items)

            debug_strip_html = f"""
              <div class="debug-strip">
                <strong>DEBUG</strong> — {escape_html(debug_kv)}
              </div>
            """
    except Exception:
        # Never let debug UI break report rendering
        debug_strip_html = ""

    # ---------- HTML skeleton ----------

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>MindPilot – Cognitive Flight Report</title>
  <link rel="icon" href="https://mind-pilot.ai/assets/mindpilot-symbol.png" type="image/png" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {{
      --dark-navy: #0B1B33;
      --sky-blue: #4FD1C5;
      --soft-bg: #F7FAFC;
      --card-bg: #FFFFFF;
      --border-subtle: #E2E8F0;
      --text-main: #1A202C;
      --text-muted: #4A5568;
      --accent-warn: #E53E3E;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top left, #EBF8FF, #F7FAFC 45%, #EDFDFD 100%);
      color: var(--text-main);
    }}
    .page {{
      max-width: 960px;
      margin: 0 auto;
      padding: 1.5rem 1.25rem 2.5rem;
    }}
    header {{
      margin-bottom: 1.5rem;
    }}
    .debug-strip {{
      margin: 0.75rem 0 0.75rem;
      padding: 0.5rem 0.75rem;
      border: 1px dashed #f59e0b;
      background: #fff7ed;
      color: #7c2d12;
      border-radius: 0.6rem;
      font-size: 0.78rem;
    }}
    .logo-title {{
      font-size: 1.8rem;
      font-weight: 700;
      color: var(--dark-navy);
      letter-spacing: 0.02em;
      margin-bottom: 0.25rem;
    }}
    .tagline {{
      font-size: 0.9rem;
      color: var(--text-muted);
    }}
    .header-meta {{
      margin-top: 0.35rem;
      font-size: 0.78rem;
      color: var(--text-muted);
      display: flex;
      flex-wrap: wrap;
      gap: 0.35rem;
    }}

    .section-heading {{
      font-size: 1.05rem;
      font-weight: 600;
      margin: 1.3rem 0 0.4rem;
      color: var(--dark-navy);
    }}
    .card {{
      background: var(--card-bg);
      border-radius: 1.25rem;
      padding: 1rem 1.2rem;
      box-shadow: 0 18px 40px rgba(0, 0, 0, 0.08);
      border-top: 4px solid var(--sky-blue);
      margin-bottom: 1rem;
    }}
    .card-sub {{
      border-radius: 1rem;
      border: 1px solid var(--border-subtle);
      background: var(--card-bg);
      padding: 0.9rem 1rem;
      margin-bottom: 0.8rem;
    }}
    .card-title {{
      font-weight: 600;
      font-size: 0.98rem;
      margin-bottom: 0.35rem;
    }}
    .card-body {{
      font-size: 0.9rem;
      color: var(--text-muted);
    }}
    .card-body-text {{
      margin: 0 0 0.6rem 0;
      font-size: 0.9rem;
      color: var(--text-muted);
      line-height: 1.5;
    }}
    .card-instruction {{
    border-left: 4px solid #CBD5F5;
    background: #F8FAFF;
    }}
    .fallacy-table-wrapper {{
      margin-top: 0.5rem;
      overflow-x: auto;
    }}
    .fallacy-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.82rem;
        table-layout: fixed;
    }}
        /* Column width control */
    .fallacy-table .col-type {{
      width: 16%;
    }}
    
    .fallacy-table .col-name {{
      width: 20%;
    }}
    
    .fallacy-table .col-desc {{
      width: 56%;
      white-space: normal;
      word-wrap: break-word;
      word-break: break-word;
    }}
    
    .fallacy-table .col-sev {{
      width: 8%;
      text-align: right;
      white-space: nowrap;
    }}

    .fallacy-table th,
    .fallacy-table td {{
      padding: 0.35rem 0.4rem;
      border-bottom: 1px solid var(--border-subtle);
      vertical-align: top;
    }}
    .fallacy-table th {{
      text-align: left;
      font-weight: 600;
      color: var(--dark-navy);
      background: #F7FAFC;
    }}
    .fallacy-type {{
        font-weight: 500;
        color: var(--text-muted);
        width: 18%;
        white-space: normal;
    }}

    .fallacy-name {{
      font-weight: 500;
      color: var(--dark-navy);
    }}
    .fallacy-desc {{
      color: var(--text-muted);
    }}
    .fallacy-severity {{
      text-align: right;
    }}
    .fallacy-tag {{
      display: inline-block;
      padding: 0.1rem 0.55rem;
      border-radius: 999px;
      font-size: 0.72rem;
      border: 1px solid var(--border-subtle);
    }}
    .fallacy-tag {{
        min-width: 3.2rem;   /* just wider than "Medium" */
        text-align: center;
    }}

    .fallacy-tag.high {{
      background: rgba(229, 62, 62, 0.08);
      color: #C53030;
      border-color: rgba(229, 62, 62, 0.5);
    }}
    .fallacy-tag.medium {{
      background: rgba(236, 201, 75, 0.08);
      color: #B7791F;
      border-color: rgba(236, 201, 75, 0.6);
    }}
    .fallacy-tag.low {{
      background: rgba(72, 187, 120, 0.08);
      color: #2F855A;
      border-color: rgba(72, 187, 120, 0.5);
    }}

    .pre-block {{
      font-family: Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 0.84rem;
      white-space: pre-wrap;
      word-wrap: break-word;
      overflow-x: auto;
      margin: 0;
      color: var(--text-muted);
    }}
    .chunk-card {{
      border-radius: 1rem;
      border: 1px solid var(--border-subtle);
      background: var(--card-bg);
      margin-bottom: 0.9rem;
      overflow: hidden;
    }}
    .chunk-header {{
      padding: 0.7rem 0.9rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      cursor: pointer;
      background: linear-gradient(90deg, #EDF2F7, #E6FFFA);
      font-size: 0.9rem;
      font-weight: 600;
      color: var(--dark-navy);
    }}
    .chunk-toggle {{
       font-size: 0.78rem;
       color: var(--text-muted);
       padding: 0.08rem 0.55rem;
       border-radius: 999px;
       border: 1px solid rgba(148, 163, 184, 0.55);
       background: rgba(255, 255, 255, 0.7);
     }}

    .chunk-body {{
      padding: 0.75rem 0.9rem 0.9rem;
      border-top: 1px solid var(--border-subtle);
      display: none;
    }}
    .chunk-body.open {{
      display: block;
    }}
    .footer {{
      margin-top: 2rem;
      font-size: 0.8rem;
      color: #A0AEC0;
      text-align: center;
    }}
    .footer-meta {{
      margin-bottom: 0.6rem;
      font-size: 0.82rem;
      color: var(--text-muted);
      line-height: 1.4;
    }}
    .footer-meta a {{
      color: var(--sky-blue);
      text-decoration: none;
    }}
    .footer-meta a:hover {{
      text-decoration: underline;
    }}
    .pill-row {{
      display: inline-flex;
      flex-wrap: wrap;
      gap: 0.4rem;
      justify-content: center;
      margin-top: 0.5rem;
    }}
    .pill {{
      font-size: 0.78rem;
      padding: 0.2rem 0.5rem;
      border-radius: 999px;
      border: 1px solid var(--border-subtle);
      background: rgba(79, 209, 197, 0.06);
      color: var(--text-muted);
    }}
    .collapsible-header {{
      font-size: 0.9rem;
      font-weight: 600;
      display: flex;
      justify-content: space-between;
      align-items: center;
      cursor: pointer;
      color: var(--dark-navy);
      margin-bottom: 0.35rem;
    }}
    .collapsible-body {{
      display: none;
    }}
    .collapsible-body.open {{
      display: block;
    }}
    .collapsible-toggle {{
      font-size: 0.8rem;
      color: var(--text-muted);
    }}
    .quick-questions {{
      font-size: 0.85rem;
      opacity: 0.9;
    }}
    .score-bar-container {{
      margin: 0.4rem 0;
    }}
    .score-bar-track {{
      width: 100%;
      height: 0.5rem;
      border-radius: 999px;
      background: #E2E8F0;
      overflow: hidden;
    }}
    .score-bar-fill {{
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, #4FD1C5, #3182CE);
      transition: width 0.4s ease-out;
    }}
    .score-bar-label {{
      margin-top: 0.25rem;
      font-size: 0.78rem;
      color: var(--text-muted);
    }}
    .dimension-bars {{
      margin-top: 0.4rem;
    }}
    .dimension-row {{
      display: grid;
      grid-template-columns: minmax(0, 1.8fr) 3fr auto;
      align-items: center;
      gap: 0.5rem;
      margin-bottom: 0.25rem;
    }}
    .dimension-label {{
      font-size: 0.8rem;
      color: var(--text-muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .dimension-track {{
      width: 100%;
      height: 0.4rem;
      border-radius: 999px;
      background: #E2E8F0;
      overflow: hidden;
    }}
    .dimension-fill {{
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, #9F7AEA, #ED64A6);
      transition: width 0.3s ease-out;
    }}
    .dimension-score {{
      font-size: 0.78rem;
      color: var(--text-muted);
      min-width: 2.5rem;
      text-align: right;
    }}
    .instruction-block{{
      border-left: 6px solid var(--accent-warn);
      background: rgba(229, 62, 62, 0.06);
      padding: 0.85rem 1rem;
      border-radius: 0.9rem;
    }}
    .instruction-block .card-title{{
      color: var(--dark-navy);
    }}
    .instruction-block .card-body{{
      color: var(--text-main);
    }}
    .social-snippet {{
      margin-top: 0.6rem;
      margin-bottom: 0.6rem;
    }}
    .social-snapshot-link {{
      font-size: 0.82rem;
      color: #E2E8F0;
      text-decoration: underline;
    }}
    .social-snapshot-link:hover {{
      text-decoration: none;
      opacity: 0.9;
    }}
    .social-label {{
      font-size: 0.78rem;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 0.2rem;
    }}
    .social-card {{
        margin-top: 0.9rem;
        border-radius: 1rem;
        border: 1px solid rgba(148, 163, 184, 0.55);
        padding: 0.9rem 0.95rem;
        background: radial-gradient(circle at top left, #0B1B33, #1A365D);
        color: #E2E8F0;
      }}
      .social-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        margin-bottom: 0.6rem;
      }}
      .social-title {{
        font-size: 0.88rem;
        font-weight: 600;
        line-height: 1.3;
      }}

      /* ⬇️ Make the MindPilot symbol much larger */
      .social-logo img {{
        width: 72px;   /* was ~36–40px */
        height: 72px;
        display: block;
      }}
      .social-brandline {{
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #A0AEC0;
      }}
      .social-fallacy-table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 0.2rem;
        font-size: 0.78rem;
        color: #E2E8F0;
      }}
      .social-fallacy-table th,
      .social-fallacy-table td {{
        padding: 0.2rem 0.4rem;
        vertical-align: top;
      }}
      .social-fallacy-table thead th {{
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #A0AEC0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.5);
      }}
      .social-fallacy-table tbody tr {{
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
      }}
      .social-fallacy-table tbody tr:last-child {{
        border-bottom: none;
      }}
      .social-fallacy-table td:first-child {{
        width: 90%;      /* fill most of the left half */
      }}
      .social-fallacy-table td:last-child {{
        white-space: nowrap;
        text-align: right;
      }}
      .social-fallacy-table .severity-cell {{
        text-align: right;
        white-space: nowrap;
        font-weight: 600;
        color: #FBD38D; /* gentle highlight for High/Med/Low */
      }}
      .social-score-row {{
        display: grid;
        grid-template-columns: auto 1fr;
        gap: 0.4rem 0.5rem;
        align-items: center;
        margin-bottom: 0.7rem;
      }}
      /* Short, compact pill */
      .social-score-number {{
        font-size: 0.9rem;
        font-weight: 600;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        border: 1px solid rgba(226, 232, 240, 0.9);
        background: rgba(15, 23, 42, 0.7);
        text-align: center;
        min-width: 3.5rem;
      }}
      .score-bar-track {{
        height: 0.5rem;
        background: rgba(148, 163, 184, 0.45);
      }}
      .social-score-row .score-bar-label {{
        grid-column: 1 / -1;
        font-size: 0.78rem;
        color: #CBD5F5;
      }}
      .social-grid {{
        display: grid;
        grid-template-columns: minmax(0, 1fr);
        gap: 0.5rem 1rem;
        margin-bottom: 0.7rem;
      }}
      .social-text {{
        margin: 0;
        font-size: 0.82rem;
        line-height: 1.4;
      }}
      @media (min-width: 640px) {{
       .social-grid {{
         grid-template-columns: minmax(0, 1.0fr) minmax(0, 1fr);
       }}
    .social-label {{
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: #A0AEC0;
      margin-bottom: 0.25rem;
    }}
    .social-text {{
      margin: 0;
      color: #E2E8F0;
      line-height: 1.4;
    }}
    .social-grok {{
      font-size: 0.75rem;
      color: #C4F1F9;
      margin-bottom: 0.5rem;
    }}
    .social-footer {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.7rem;
      color: #A0AEC0;
      border-top: 1px solid rgba(226, 232, 240, 0.2);
      padding-top: 0.45rem;
    }}
    .social-watermark {{
      font-weight: 500;
      color: #81E6D9;
    }}
    .social-card {{
       margin-top: 0.9rem;
       margin-bottom: 0.3rem;
       background: #0B1B33;
       color: #E2E8F0;
       border-radius: 1rem;
       padding: 0.95rem 1rem 1rem;
       box-shadow: 0 18px 40px rgba(11, 27, 51, 0.55);
     }}
     .social-header {{
       display: flex;
       align-items: center;
       justify-content: space-between;
       gap: 0.75rem;
       margin-bottom: 0.6rem;
     }}
     .social-title {{
       font-size: 0.9rem;
       font-weight: 600;
       letter-spacing: 0.01em;
     }}
     .social-logo img {{
       width: 60px;
       height: 60px;
       display: block;
       border-radius: 999px;
     }}
     .social-score-row {{
       display: grid;
       grid-template-columns: auto 1fr;
       align-items: center;
       gap: 0.55rem 0.75rem;
       margin-bottom: 0.7rem;
     }}
     .social-score-number {{
       font-size: 1.1rem;
       font-weight: 700;
       white-space: nowrap;
     }}
     .social-score-row .score-bar-track {{
        position: relative;
        height: 0.5rem;
        background: rgba(148, 163, 184, 0.45);
        border-radius: 999px;
        overflow: hidden;
        max-width: 260px;  /* keep the bar to ~half the card width */
      }}
     .social-score-row .score-bar-label {{
       grid-column: 1 / -1;
       font-size: 0.78rem;
       color: #CBD5F5;
     }}
     .social-grid {{
       display: grid;
       grid-template-columns: minmax(0, 1fr);
       gap: 0.5rem 1rem;
       margin-bottom: 0.7rem;
     }}
     .social-text {{
       margin: 0;
       font-size: 0.82rem;
       line-height: 1.4;
     }}
     .social-grok {{
       margin-top: 0.4rem;
       padding-top: 0.4rem;
       border-top: 1px dashed rgba(148, 163, 184, 0.6);
       font-size: 0.8rem;
     }}
     .social-footer {{
       margin-top: 0.6rem;
       display: flex;
       justify-content: space-between;
       gap: 0.75rem;
       font-size: 0.75rem;
       color: #CBD5F5;
       opacity: 0.9;
     }}
     .social-watermark {{
       font-weight: 500;
       white-space: nowrap;
     }}
     @media (min-width: 640px) {{
       .social-grid {{
         grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
       }}
     }}
    /* ===============================
   Locked / Gated UI (Cockpit)
   =============================== */

    .locked {{
      opacity: 0.45;
      filter: grayscale(0.3);
      pointer-events: none;
    }}
    
    .locked-badge {{
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      font-size: 0.72rem;
      padding: 0.2rem 0.55rem;
      border-radius: 999px;
      background: #EDF2F7;
      border: 1px dashed var(--border-subtle);
      color: var(--text-muted);
      margin-left: 0.5rem;
    }}
    
    .locked-badge::before {{
      content: "🔒";
      font-size: 0.8rem;
    }}

    </style>
</head>
<body>
    <div class="page">
    <header>
      <div class="logo-title">MindPilot Cognitive Flight Report</div>
      <div class="tagline">Your critical thinking Co-Pilot’s readback of this content.</div>
      <div class="header-meta">
        <span>{escape_html(source_type)}</span>
        <span>· {escape_html(depth_text)}</span>
      </div>
    </header>
    {debug_strip_html}
        <section class="card-sub">
      <div class="card-title">Source & analysis mode</div>
      <div class="card-body">
        <div style="font-size:0.88rem;margin-bottom:0.25rem;">
          <strong>Content type:</strong> {escape_html(source_type)}
        </div>
        {""
        if not source_url
        else f'<div style="font-size:0.85rem;word-wrap:break-word;"><strong>Source:</strong> <a href="{escape_html(source_url)}" target="_blank" rel="noopener noreferrer">{escape_html(source_url)}</a></div>'
        }
        <div class="subtext" style="margin-top:0.4rem;">
          {escape_html(depth_text)}
        </div>
      </div>
    </section>

"""
    # ---------- global presence flag ----------
    # Fixed Cockpit: Quick report must always render the global container,
    # even if some sections are empty/locked.
    has_any_global = any(
        [
            esc_profile,
            esc_full,
            esc_map,
            esc_questions,
            esc_investor,
        ]
    )

    if depth == "quick":
        has_any_global = True

    # ---------- Global card + subcards ----------
    if has_any_global:
        html += f"""
    <section class="card">
      <div class="card-title">Your Critical Thinking Report</div>
      {social_card_html}
      """

        # 1) Rationality Profile – always first if present (open by default)
        if esc_profile:
            html += f"""
      <section class="card-sub">
        <div class="collapsible-header" onclick="toggleSection('rationality-profile')">
          <span>Rationality Profile for the Entire Segment</span>
          <span class="collapsible-toggle" id="toggle-rationality-profile">Hide</span>
        </div>
        <div class="collapsible-body open" id="section-rationality-profile">
          {dimension_bars_html}
          <pre class="pre-block">{esc_profile}</pre>
        </div>
      </section>
      """
            # 2) Critical Thinking Questions — Fixed Cockpit: ALWAYS render
        if esc_questions:
            questions_body_html = f'<pre class="pre-block">{esc_questions}</pre>'
            questions_locked_class = ""
            questions_badge_html = ""
        else:
            # Should be rare now that slicing fills defaults, but keep cockpit stable
            questions_body_html = """
                <p class="muted" style="margin:0;">
                  Questions will appear here when available.
                </p>
              """
            questions_locked_class = "locked"
            questions_badge_html = locked_badge_html("Locked")

        html += f"""
        <section class="card-sub {questions_locked_class}">
          <div class="collapsible-header" onclick="toggleSection('critical-questions')">
            <span>Critical Thinking Questions to Ask Yourself</span>
            <span class="collapsible-toggle" id="toggle-critical-questions">Hide</span>
          </div>
          <div class="collapsible-body open" id="section-critical-questions">
            {questions_badge_html}
            {questions_body_html}
          </div>
        </section>
        """

        # 3) MindPilot × Grok (if present) – we render this card next
        html += grok_card_html

        # 3b) Global fallback (when we have a global report but no structured full summary)
        if esc_global_fallback and not esc_full:
            html += f"""
          <section class="card-sub">
            <div class="collapsible-header" onclick="toggleSection('global-fallback')">
              <span>Global MindPilot Reasoning Overview</span>
              <span class="collapsible-toggle" id="toggle-global-fallback">Show</span>
            </div>
            <div class="collapsible-body" id="section-global-fallback">
              <pre class="pre-block">{esc_global_fallback}</pre>
            </div>
          </section>
            """

        # 4) Full Reasoning Scan + Master Map (full mode only)
        if depth == "full":
            # 4) Full Reasoning Scan – Global Summary (Full cockpit: ALWAYS render)
            if esc_full:
                full_body = f'<pre class="pre-block">{esc_full}</pre>'
                full_class = ""
                full_badge = ""
            else:
                full_body = f"""
                  <p class="muted" style="margin:0;">
                    {locked_badge_html("Locked")} Full Reasoning Scan is available on Pro Full tiers.
                  </p>
                """
                full_class = "locked"
                full_badge = ""

            html += f"""
              <section class="card-sub {full_class}">
                <div class="collapsible-header" onclick="toggleSection('global-summary')">
                  <span>Full Reasoning Scan – Global Summary</span>
                  <span class="collapsible-toggle" id="toggle-global-summary">{"Show" if full_class else "Hide"}</span>
                </div>
                <div class="collapsible-body {"open" if not full_class else ""}" id="section-global-summary">
                  {full_badge}
                  {full_body}
                </div>
              </section>
            """

            # 5) Master Fallacy & Bias Map (Full cockpit: ALWAYS render)
            if esc_map or fallacy_table_html:
                map_body = (fallacy_table_html or f'<pre class="pre-block">{esc_map}</pre>')
                map_class = ""
                map_badge = ""
            else:
                map_body = f"""
                  <p class="muted" style="margin:0;">
                    {locked_badge_html("Locked")} Master Map is available on Pro Full tiers.
                  </p>
                """
                map_class = "locked"
                map_badge = ""

            html += f"""
              <section class="card-sub {map_class}">
                <div class="collapsible-header" onclick="toggleSection('master-map')">
                  <span>Master Fallacy &amp; Bias Map</span>
                  <span class="collapsible-toggle" id="toggle-master-map">{"Show" if map_class else "Hide"}</span>
                </div>
                <div class="collapsible-body {"open" if not map_class else ""}" id="section-master-map">
                  {map_badge}
                  {map_body}
                </div>
              </section>
            """

        # 6) Condensed Executive Summary — Full cockpit: ALWAYS render (locked if gated/empty)
        exec_locked = not bool(esc_investor)
        exec_wrapper_class = "locked" if exec_locked else ""

        if esc_investor:
            exec_body_html = f'<pre class="pre-block">{esc_investor}</pre>'
        else:
            exec_body_html = f"""
              <p class="muted" style="margin:0;">
                {locked_badge_html("Locked")} Condensed Executive Summary is available on Pro Full tiers.
              </p>
              <p class="muted" style="margin:0.5rem 0 0;">
                Upgrade to unlock the executive summary view.
              </p>
            """

        html += f"""
          <section class="card-sub {exec_wrapper_class}">
            <div class="collapsible-header" onclick="toggleSection('investor-summary')">
              <span>Condensed Executive Summary</span>
              <span class="collapsible-toggle" id="toggle-investor-summary">Show</span>
            </div>
            <div class="collapsible-body" id="section-investor-summary">
              {exec_body_html}
            </div>
          </section>
        """

        # 7) Social snippet – single canonical template for all platforms
        html += f"""
        <section class="card-sub">
          <div class="collapsible-header" onclick="toggleSection('social-snippets')">
            <span>Copy-Ready Social Snippet</span>
            <span class="collapsible-toggle" id="toggle-social-snippets">Show</span>
          </div>
          <div class="collapsible-body" id="section-social-snippets">
            <p class="card-body-text">
              This single snippet is designed to be reused across X, LinkedIn, and other platforms.
              Copy it, make minor edits if needed for tone or length, and always include your link
              back to this specific Cognitive Flight Report.
            </p>

            <div class="social-snippet">
              <pre class="pre-block">{esc_canonical_snippet}</pre>
            </div>
          </div>
        </section>
        """

    # ---------- Disclaimer + Creator checklist ----------

    # 1) Instruction card (always shown)
    html += """
      <section class="card-sub">
        <div class="instruction-block">
          <div class="card-title">How to read this report</div>
          <div class="card-body">
            MindPilot is your critical thinking co-pilot. This analysis highlights patterns in
            reasoning—such as logical fallacies, cognitive biases, and attempts to persuade—
            so you can reflect more deliberately on what you’re hearing or reading.
            <br/><br/>
            <strong>Important:</strong> MindPilot does <em>not</em> act as a fact checker and does
            <em>not</em> independently verify whether specific claims or statements are true.
            It focuses on <strong>how</strong> arguments are made, not on adjudicating the
            real-world accuracy of every assertion.
            <br/><br/>
            <strong>Note on Quick vs. Full Report:</strong> MindPilot runs a single global scan in
            quick mode. In full mode, it also runs section-level diagnostics, so scores may shift
            slightly within a small tolerance as the system "thinks harder" about the reasoning.
          </div>
        </div>
      </section>
    """

    # --- Creator Pre-Publish Checklist (always render) ---
    # Locked unless we actually have the Pro Quick checklist HTML (Fixed Cockpit)
    checklist_locked = not bool(creator_checklist_html)

    if creator_checklist_html:
        checklist_body = creator_checklist_html
    else:
        checklist_body = """
        <ul>
          <li>Are claims clearly distinguished from opinions?</li>
          <li>Are key assumptions stated explicitly?</li>
          <li>Would a skeptical reader understand your reasoning?</li>
        </ul>
        """

    checklist_wrapper_class = "locked" if checklist_locked else ""

    creator_checklist_section = f"""
    <section class="card {checklist_wrapper_class}">
      <h3>
        Creator Pre-Publish Checklist
        {locked_badge_html("Locked") if checklist_locked else ""}
      </h3>
      {checklist_body}
    </section>
    """.strip()

    html += creator_checklist_section

    # ---------- Section-level deep dive ----------

    if depth == "full" and escaped_chunks:

        html += """

            <section>

              <div class="section-heading">Section-Level Deep Dive</div>

        """

        for i, text in enumerate(escaped_chunks):
            html += f"""

              <article class="chunk-card">

                <div class="chunk-header" onclick="toggleChunk({i})">

                  <span>Section {i + 1} – Reasoning Scan</span>

                  <span class="chunk-toggle" id="toggle-label-{i}">Show</span>

                </div>

                <div class="chunk-body" id="chunk-body-{i}">

                  <pre class="pre-block">{text}</pre>

                </div>

              </article>

        """

    # ---------- Footer ----------

    html += f"""
    </section>

    <div class="footer">
      <div class="footer-meta">
        <div><strong>Source:</strong> {escape_html(source_url or "Pasted text")}</div>
      </div>
      <div class="pill-row">
        <div class="pill">MindPilot · Cognitive Flight Report</div>
        <div class="pill">Reasoning Scorecard</div>
        <div class="pill">F/B/R/M diagnostic engine</div>
      </div>
    </div>
  </div>

  <script>
    function toggleChunk(index) {{
      const body = document.getElementById('chunk-body-' + index);
      const label = document.getElementById('toggle-label-' + index);
      const isOpen = body && body.classList.contains('open');
      if (isOpen) {{
        body.classList.remove('open');
        label.textContent = 'Show';
      }} else if (body) {{
        body.classList.add('open');
        label.textContent = 'Hide';
      }}
    }}

    function toggleSection(key) {{
      const body = document.getElementById('section-' + key);
      const label = document.getElementById('toggle-' + key);
      const isOpen = body && body.classList.contains('open');
      if (isOpen) {{
        body.classList.remove('open');
        label.textContent = 'Show';
      }} else if (body) {{
        body.classList.add('open');
        label.textContent = 'Hide';
      }}
    }}
  </script>

</body>
</html>
"""
    return html

# # ---------- MAIN PIPELINE   (legacy CLI, currently unused)
#
# def main():
#     print("=== MindPilot One-Step Reasoning Analysis ===")
#     youtube_url = input("Enter YouTube URL: ").strip()
#
#     # 1) Extract video ID
#     try:
#         video_id = extract_video_id(youtube_url)
#         print(f"[1/4] Extracted video ID: {video_id}")
#     except ValueError as e:
#         print(f"Error: {e}")
#         return
#
#     # 2) Fetch transcript
#     try:
#         transcript_text = fetch_transcript_text(video_id)
#     except RuntimeError as e:
#         print(f"Error fetching transcript: {e}")
#         return
#
#     # 2b) Safety classification on the transcript (forbidden-content pre-filter)
#     attach_disclaimer = False
#     try:
#         classification = classify_content(transcript_text)
#     except Exception as e:
#         print(f"[safety] Classifier failed ({e}); defaulting to allow.")
#         classification = {
#             "classification": "allow",
#             "reason": "classifier error",
#             "allowed_scope": "",
#         }
#
#     cls = (classification.get("classification") or "allow").lower()
#     reason = classification.get("reason", "")
#
#     if cls == "block":
#         print("\n=== MindPilot Safety Gate ===")
#         print("MindPilot cannot analyze this content as written.")
#         if reason:
#             print(f"Reason: {reason}")
#         print(
#             "\nYou may instead:\n"
#             "- Analyze public news coverage about this topic\n"
#             "- Submit a non-instructional excerpt\n"
#             "- Frame your request around rhetoric or reasoning patterns"
#         )
#         return
#
#     if cls == "restricted":
#         attach_disclaimer = True
#         print("\n[Note] This content is sensitive.")
#         if reason:
#             print(f"Reason: {reason}")
#         print("MindPilot will analyze reasoning patterns only (no advice or endorsement).")
#
#     # 3) Save raw transcript
#     save_text_to_file(transcript_text, TRANSCRIPT_FILE)
#     print(f"[2/4] Transcript saved to: {TRANSCRIPT_FILE}")
#     print(f"      Transcript length (characters): {len(transcript_text)}")
#
#     # 4) Chunk text
#     chunks = chunk_text(transcript_text, MAX_CHARS_PER_CHUNK)
#     total_chunks = len(chunks)
#     print(f"[3/4] Number of chunks created: {total_chunks}")
#
#     # 5) Build prompt pack (for manual ChatGPT use)
#     lines = []
#     lines.append("# MindPilot Reasoning Analysis Prompt Pack\n")
#     lines.append(f"_Source URL_: {youtube_url}\n")
#     lines.append(f"_Video ID_: `{video_id}`\n")
#     lines.append(f"_Chunks_: {total_chunks}\n")
#     lines.append("\n---\n\n")
#
#     for idx, chunk in enumerate(chunks):
#         chunk_prompt = build_chunk_prompt(chunk, idx, total_chunks)
#         lines.append(chunk_prompt)
#
#     # Add global prompts (manual, copy/paste style)
#     lines.append(build_global_prompts())
#
#     with open(PROMPT_PACK_FILE, "w", encoding="utf-8") as f:
#         f.write("\n".join(lines))
#
#     print(f"[4/4] Prompt pack written to: {PROMPT_PACK_FILE}")
#
#     # ---------- AUTOMATIC ANALYSIS SECTION ----------
#
#     print("\n=== Running automatic MindPilot analysis via OpenAI API ===")
#     chunk_analyses = []  # store each chunk's analysis for global summary
#
#     report_lines = []
#     report_lines.append("# MindPilot Reasoning Analysis Report\n")
#     report_lines.append(f"_Source URL_: {youtube_url}\n")
#     report_lines.append(f"_Video ID_: `{video_id}`\n")
#     report_lines.append(f"_Chunks_: {total_chunks}\n")
#
#     if attach_disclaimer:
#         report_lines.append(
#             "\n> Safety note: This content was classified as sensitive. "
#             "MindPilot is analyzing reasoning patterns only and is not "
#             "providing advice, instructions, or endorsement.\n"
#         )
#
#     report_lines.append("\n---\n\n")
#
#     # Per-chunk analysis
#     for idx, chunk in enumerate(chunks):
#         print(f"  -> Analyzing chunk {idx + 1}/{total_chunks}...")
#         chunk_prompt = build_chunk_prompt(chunk, idx, total_chunks)
#         analysis = run_mindpilot_analysis(chunk_prompt)
#
#         chunk_analyses.append(analysis)
#
#         report_lines.append(f"## Chunk {idx + 1} of {total_chunks} — MindPilot Analysis\n\n")
#         report_lines.append(analysis)
#         report_lines.append("\n\n---\n\n")
#
#     report_path = "mindpilot_analysis_report.md"
#     with open(report_path, "w", encoding="utf-8") as rf:
#         rf.write("\n".join(report_lines))
#
#     print(f"\nChunk-level analysis report written to: {report_path}")
#
#     # ----- AUTOMATIC GLOBAL SUMMARY (no prompts) -----
#
#     global_report = ""
#
#     print("\n  -> Automatically generating global summary...")
#     global_prompt = build_global_summary_prompt(chunk_analyses)
#     global_report = run_mindpilot_analysis(global_prompt)
#
#     if attach_disclaimer:
#         global_report = (
#             "Safety note: This content was classified as sensitive; "
#             "MindPilot is analyzing reasoning patterns only and is not providing "
#             "advice, instructions, or endorsement.\n\n"
#             + global_report
#         )
#
#     with open(report_path, "a", encoding="utf-8") as rf:
#         rf.write("\n\n# Global MindPilot Reasoning Summary\n\n")
#         rf.write(global_report)
#         rf.write("\n")
#
#     print(f"Global summary automatically appended to: {report_path}")
#
#     # ----- Build HTML report -----
#     html_report = build_html_report(
#         source_url=youtube_url,
#         video_id=video_id,
#         total_chunks=total_chunks,
#         chunk_analyses=chunk_analyses,
#         global_report=global_report,
#         depth="full",
#     )
#     html_path = "mindpilot_report.html"
#     with open(html_path, "w", encoding="utf-8") as hf:
#         hf.write(html_report)
#
#     print(f"\nHTML report written to: {html_path}")
#     print("\nAll done! Open the markdown OR HTML report file to review the analysis.\n")
#
#
# if __name__ == "__main__":
#     main()

#    main()

