import os
import re
import textwrap
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from mindpilot_llm_client import run_mindpilot_analysis



# ---------- CONFIG ----------

TRANSCRIPT_FILE = "mindpilot_transcript_output.txt"
PROMPT_PACK_FILE = "mindpilot_prompt_pack.md"
MAX_CHARS_PER_CHUNK = 1200  # tweak if you want bigger/smaller chunks


# ---------- YOUTUBE HELPERS ----------

def extract_video_id(youtube_url: str) -> str:
    """
    Extracts the YouTube video ID from various URL formats.
    Examples:
    - https://www.youtube.com/watch?v=VIDEOID
    - https://youtu.be/VIDEOID
    """
    parsed = urlparse(youtube_url)

    # Case 1: standard watch URL
    if parsed.hostname in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        query = parse_qs(parsed.query)
        if "v" in query:
            return query["v"][0]

    # Case 2: shortened youtu.be URL
    if parsed.hostname == "youtu.be":
        return parsed.path.lstrip("/")

    # Fallback: try to pull a video-like ID via regex
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
        fetched = api.fetch(video_id, languages=['en'])
        raw_chunks = fetched.to_raw_data()
    except TranscriptsDisabled:
        raise RuntimeError("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise RuntimeError("No transcript found for this video.")
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
    Focus: fallacies, biases, persuasion, manipulation (F/B/R/M domains).
    """
    header = f"## Chunk {chunk_index + 1} of {total_chunks}\n"

    wrapped_chunk = textwrap.fill(chunk_text, width=100)

    prompt = f"""
You are **MindPilot**, a neutral reasoning-analysis copilot.

Your job is to analyze this transcript chunk for:
- Logical fallacies (F domain)
- Cognitive biases (B domain)
- Rhetorical / persuasion tactics (R domain)
- Manipulative or psychological conditioning patterns (M domain)

Use the following approach:

**Transcript chunk (verbatim):**
\"\"\"TEXT_START
{wrapped_chunk}
TEXT_END\"\"\"

Now produce the following sections in clean Markdown:

1. **Argument Map**
   - Bullet list of the main claims and the key reasons/premises in your own words.
   - Note any important assumptions that are taken for granted.

2. **Logical Fallacies (F)**
   - For each fallacy you detect, list:
     - **Name** (e.g., Straw Man, Ad Hominem, Slippery Slope, False Cause, False Dilemma)
     - **Why it applies** (1–3 sentences, referencing the text)
     - **Severity**: Low / Medium / High
   - If none are clear, say: "No clear fallacies detected with high confidence."

3. **Cognitive Biases (B)**
   - For each bias you detect (e.g., Confirmation Bias, Anchoring, Availability, Dunning–Kruger, Negativity Bias):
     - Name
     - Short explanation of how it shows up in this chunk
     - Severity: Low / Medium / High
   - If none are clear, say so explicitly.

4. **Rhetorical / Persuasion Tactics (R)**
   - Identify any persuasion or framing tactics, such as:
     - Emotional priming, bandwagon, scapegoating, authority appeal, nostalgia, fear appeal, virtue signaling, simplification/slogans, etc.
   - For each:
     - Name the tactic
     - Explain briefly how the language in this chunk uses it.

5. **Manipulative / Conditioning Patterns (M)**
   - Only include when you see stronger patterns like:
     - Gaslighting, belief deconstruction, information laundering, astroturfing, weaponized uncertainty, or propaganda-style narrative control.
   - Be cautious and neutral; if present, describe the pattern and why you suspect it.

6. **Rationality Flight Report**
   - In 2–4 sentences, summarize how clear, fair, and evidence-based this chunk is.
   - Provide a simple rating from 1–5 for overall reasoning quality, where:
     - 5 = very sound, well-supported reasoning
     - 3 = mixed: some valid reasoning with notable issues
     - 1 = heavily distorted or manipulative reasoning

Avoid political or ideological judgment. Focus strictly on the structure of arguments, evidence, and language.
"""
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
- A final overall reasoning score from 1–5."

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
    Build a prompt that summarizes all chunk-level MindPilot analyses into
    a full-lesson reasoning summary, master map, rationality profile, and
    condensed investor-style summary.
    """
    joined_analyses = "\n\n---\n\n".join(
        f"Chunk {i+1} Analysis:\n{text}"
        for i, text in enumerate(chunk_analyses)
    )

    prompt = f"""
You are MindPilot, a neutral reasoning-analysis copilot.

You have already analyzed several chunks of a single piece of content.
Below are your chunk-level analyses, including argument maps, fallacies,
biases, persuasion tactics, manipulation patterns, and rationality ratings.

Use them to build a GLOBAL reasoning report.

Chunk-level analyses:
\"\"\"TEXT_START
{joined_analyses}
TEXT_END\"\"\"

Now produce the following sections in clean Markdown:

1. **Full-Lesson Reasoning Summary (6–10 paragraphs)**
   - Explain the overall narrative of the content.
   - Describe how causal claims are made (well-supported vs speculative).
   - Note how evidence is used or not used.
   - Summarize how fallacies, biases, and persuasion tactics appear across the whole segment.

2. **Master Fallacy & Bias Map**
   - List the main logical fallacies (F domain) detected across all chunks.
   - List the main cognitive biases (B domain).
   - List key rhetorical/persuasion tactics (R domain).
   - List any notable manipulative/conditioning patterns (M domain).
   - For each, briefly describe its role and how often it appears (Low/Medium/High).

3. **Rationality Profile for the Entire Segment**
   - Create a short overview of reasoning strengths.
   - Create a short overview of reasoning weaknesses.
   - Provide a structured list or table of reasoning dimensions
     (e.g., Evidence use, Causal reasoning, Emotional framing, Fairness/balance,
     Motive attribution) with ratings from 1–5.
   - Provide a final overall reasoning score from 1–5.

4. **Condensed Investor-Facing Summary**
   - In 3–6 short paragraphs, describe:
     - What the content is about.
     - What MindPilot found (fallacies/biases/persuasion patterns, rationality level).
     - Why this demonstrates the value of MindPilot as a product
       (media literacy, education, compliance, etc.).
   - Keep it punchy and non-technical, suitable for an investor demo.
"""
    return prompt.strip()


def build_html_report(
        source_url_or_label: str,
        rationality_profile_md: str,
        master_fallacy_bias_map_md: str,
        full_lesson_summary_md: str,
        condensed_investor_summary_md: str,
        chunk_analyses: list,
):
    """Generate a fully-branded MindPilot Cognitive Flight Report (HTML). Matches the static report exactly."""

    # Number of dynamic sections
    num_sections = len(chunk_analyses)

    # Build dynamic section cards
    sections_html = ""
    for idx, analysis_md in enumerate(chunk_analyses):
        sections_html += f"""
        <article class="chunk-card">
          <div class="chunk-header" onclick="toggleChunk({idx})">
            <span>Section {idx + 1} of {num_sections} – Reasoning Scan</span>
            <span class="chunk-toggle" id="toggle-label-{idx}">Show</span>
          </div>
          <div class="chunk-body" id="chunk-body-{idx}">
            <pre class="pre-block">
{analysis_md}
            </pre>
          </div>
        </article>
        """

    # Build full HTML report (self-contained)
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>MindPilot – Cognitive Flight Report &amp; Reasoning Scorecard</title>
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
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: 'Montserrat', system-ui, sans-serif;
      background: radial-gradient(circle at top left, #EBF8FF, #F7FAFC 45%, #EDFDFD 100%);
      color: var(--text-main);
    }}
    .page {{
      max-width: 960px;
      margin: 0 auto;
      padding: 1.5rem 1.25rem 2.5rem;
    }}

    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1.5rem;
      flex-wrap: wrap;
      gap: 1rem;
    }}
    .logo-left {{
      display: flex;
      flex-direction: column;
    }}
    .logo-title {{
      font-size: 1.35rem;
      font-weight: 700;
      color: var(--dark-navy);
    }}
    .logo-sub {{
      font-size: 0.75rem;
      text-transform: uppercase;
      color: var(--text-muted);
      letter-spacing: 0.16em;
    }}
    .header-pill {{
      border: 1px solid var(--border-subtle);
      background: #fff;
      padding: 0.35rem 0.75rem;
      border-radius: 999px;
      font-size: 0.75rem;
      color: var(--text-muted);
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
    }}
    .header-pill-dot {{
      width: 7px;
      height: 7px;
      background: #38A169;
      border-radius: 999px;
      box-shadow: 0 0 10px rgba(56,161,105,0.8);
    }}

    .card {{
      background: var(--card-bg);
      border-radius: 1.25rem;
      padding: 1rem 1.2rem;
      margin-bottom: 1rem;
      border-top: 4px solid var(--sky-blue);
      box-shadow: 0 18px 40px rgba(0,0,0,0.08);
    }}
    .card-sub {{
      border: 1px solid var(--border-subtle);
      background: var(--card-bg);
      padding: 1rem;
      border-radius: 1rem;
      margin-bottom: 1rem;
    }}
    .card-title {{
      font-size: 1rem;
      font-weight: 600;
      margin-bottom: 0.4rem;
      color: var(--dark-navy);
    }}
    .card-body {{
      font-size: 0.9rem;
      color: var(--text-muted);
    }}

    .subtext {{
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-bottom: 0.5rem;
    }}
    .pre-block {{
      white-space: pre-wrap;
      font-size: 0.85rem;
      color: var(--text-muted);
      font-family: Menlo, Consolas, monospace;
    }}

    .collapsible-header {{
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--dark-navy);
    }}
    .collapsible-body {{
      display: none;
      margin-top: 0.5rem;
    }}
    .collapsible-body.open {{
      display: block;
    }}
    .collapsible-toggle {{
      font-size: 0.8rem;
      color: var(--text-muted);
    }}

    .chunk-card {{
      border: 1px solid var(--border-subtle);
      border-radius: 1rem;
      background: var(--card-bg);
      margin-bottom: 1rem;
    }}
    .chunk-header {{
      padding: 0.75rem;
      display: flex;
      justify-content: space-between;
      background: linear-gradient(90deg,#EDF2F7,#E6FFFA);
      cursor: pointer;
      font-size: 0.9rem;
      font-weight: 600;
      color: var(--dark-navy);
    }}
    .chunk-body {{
      display: none;
      padding: 1rem 1rem;
    }}
    .chunk-body.open {{
      display: block;
    }}

    .footer {{
      margin-top: 2rem;
      text-align: center;
      font-size: 0.8rem;
      color: var(--text-muted);
    }}
    .pill-row {{
      display: flex;
      gap: 0.4rem;
      justify-content: center;
      flex-wrap: wrap;
      margin-top: 0.5rem;
    }}
    .pill {{
      border: 1px solid var(--border-subtle);
      padding: 0.25rem 0.55rem;
      border-radius: 999px;
      background: rgba(79,209,197,0.1);
      font-size: 0.75rem;
      color: var(--text-muted);
    }}
  </style>
</head>
<body>
  <div class="page">

    <header>
      <div class="logo-left">
        <div class="logo-title">MindPilot Cognitive Flight Report</div>
        <div class="logo-sub">Reasoning Scorecard · F/B/R/M Map</div>
      </div>
      <div class="header-pill">
        <span class="header-pill-dot"></span>
        <span>Explainable, competitor-proof reasoning engine</span>
      </div>
    </header>

    <section class="card">
      <div class="card-title">Reasoning Scorecard (Rationality Profile)</div>
      <pre class="pre-block">{rationality_profile_md}</pre>
    </section>

    <section class="card-sub">
      <div class="card-title">Master Fallacy & Bias Map</div>
      <pre class="pre-block">{master_fallacy_bias_map_md}</pre>
    </section>

    <section class="card-sub">
      <div class="collapsible-header" onclick="toggleSection('full-summary')">
        <span>Full-Lesson Reasoning Summary</span>
        <span id="toggle-full-summary" class="collapsible-toggle">Show</span>
      </div>
      <div class="collapsible-body" id="section-full-summary">
        <pre class="pre-block">{full_lesson_summary_md}</pre>
      </div>
    </section>

    <section class="card-sub">
      <div class="collapsible-header" onclick="toggleSection('investor-summary')">
        <span>Condensed Executive / Investor Summary</span>
        <span id="toggle-investor-summary" class="collapsible-toggle">Show</span>
      </div>
      <div class="collapsible-body" id="section-investor-summary">
        <pre class="pre-block">{condensed_investor_summary_md}</pre>
      </div>
    </section>

    <section>
      <div class="card-title" style="margin-top:1.5rem;">Section-Level Deep Dive</div>
      {sections_html}
    </section>

    <div class="footer">
      <div><strong>Source:</strong> {source_url_or_label}</div>
      <div><strong>Sections analyzed:</strong> {num_sections}</div>

      <div class="pill-row">
        <div class="pill">Cognitive Flight Report</div>
        <div class="pill">Reasoning Scorecard</div>
        <div class="pill">F/B/R/M Diagnostic Engine</div>
      </div>
    </div>
  </div>

  <script>
    function toggleChunk(index) {{
      const body = document.getElementById('chunk-body-' + index);
      const label = document.getElementById('toggle-label-' + index);
      const isOpen = body.classList.contains('open');
      if (isOpen) {{
        body.classList.remove('open');
        label.textContent = 'Show';
      }} else {{
        body.classList.add('open');
        label.textContent = 'Hide';
      }}
    }}

    function toggleSection(key) {{
      const body = document.getElementById('section-' + key);
      const label = document.getElementById('toggle-' + key);
      const isOpen = body.classList.contains('open');
      if (isOpen) {{
        body.classList.remove('open');
        label.textContent = 'Show';
      }} else {{
        body.classList.add('open');
        label.textContent = 'Hide';
      }}
    }}
  </script>
</body>
</html>
"""

    return html


# ---------- MAIN PIPELINE ----------

# ---------- MAIN PIPELINE ----------

def main():
    print("=== MindPilot One-Step Reasoning Analysis ===")
    youtube_url = input("Enter YouTube URL: ").strip()

    # 1) Extract video ID
    try:
        video_id = extract_video_id(youtube_url)
        print(f"[1/4] Extracted video ID: {video_id}")
    except ValueError as e:
        print(f"Error: {e}")
        return

    # 2) Fetch transcript
    try:
        transcript_text = fetch_transcript_text(video_id)
    except RuntimeError as e:
        print(f"Error fetching transcript: {e}")
        return

    # 3) Save raw transcript
    save_text_to_file(transcript_text, TRANSCRIPT_FILE)
    print(f"[2/4] Transcript saved to: {TRANSCRIPT_FILE}")
    print(f"      Transcript length (characters): {len(transcript_text)}")

    # 4) Chunk text
    chunks = chunk_text(transcript_text, MAX_CHARS_PER_CHUNK)
    total_chunks = len(chunks)
    print(f"[3/4] Number of chunks created: {total_chunks}")

    # 5) Build prompt pack (for manual ChatGPT use)
    lines = []
    lines.append("# MindPilot Reasoning Analysis Prompt Pack\n")
    lines.append(f"_Source URL_: {youtube_url}\n")
    lines.append(f"_Video ID_: `{video_id}`\n")
    lines.append(f"_Chunks_: {total_chunks}\n")
    lines.append("\n---\n\n")

    for idx, chunk in enumerate(chunks):
        chunk_prompt = build_chunk_prompt(chunk, idx, total_chunks)
        lines.append(chunk_prompt)

    # Add global prompts (manual, copy/paste style)
    lines.append(build_global_prompts())

    with open(PROMPT_PACK_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[4/4] Prompt pack written to: {PROMPT_PACK_FILE}")

    # ---------- AUTOMATIC ANALYSIS SECTION ----------

    # Previous interactive version (kept for later if you want it back):
    # run_auto = input(
    #     "\nRun automatic MindPilot analysis via OpenAI API now? (y/n): "
    # ).strip().lower()
    #
    # if run_auto != "y":
    #     print("\nSkipping automatic analysis. You can still use the prompt pack manually.\n")
    #     return

    print("\n=== Running automatic MindPilot analysis via OpenAI API ===")
    chunk_analyses = []  # store each chunk's analysis for global summary

    report_lines = []
    report_lines.append("# MindPilot Reasoning Analysis Report\n")
    report_lines.append(f"_Source URL_: {youtube_url}\n")
    report_lines.append(f"_Video ID_: `{video_id}`\n")
    report_lines.append(f"_Chunks_: {total_chunks}\n")
    report_lines.append("\n---\n\n")

    # Per-chunk analysis
    for idx, chunk in enumerate(chunks):
        print(f"  -> Analyzing chunk {idx + 1}/{total_chunks}...")
        chunk_prompt = build_chunk_prompt(chunk, idx, total_chunks)
        analysis = run_mindpilot_analysis(chunk_prompt)

        chunk_analyses.append(analysis)

        report_lines.append(f"## Chunk {idx + 1} of {total_chunks} — MindPilot Analysis\n\n")
        report_lines.append(analysis)
        report_lines.append("\n\n---\n\n")

    report_path = "mindpilot_analysis_report.md"
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write("\n".join(report_lines))

    print(f"\nChunk-level analysis report written to: {report_path}")

    # Optional: global summary + master map + rationality profile + investor summary
    # ----- AUTOMATIC GLOBAL SUMMARY (no prompts) -----

    global_report = ""

    # Original interactive code preserved for later:
    #
    # build_global = input(
    #     "Also generate global summary & investor-style overview? (y/n): "
    # ).strip().lower()
    #
    # if build_global == "y":

    print("\n  -> Automatically generating global summary...")
    global_prompt = build_global_summary_prompt(chunk_analyses)
    global_report = run_mindpilot_analysis(global_prompt)

    with open(report_path, "a", encoding="utf-8") as rf:
        rf.write("\n\n# Global MindPilot Reasoning Summary\n\n")
        rf.write(global_report)
        rf.write("\n")

    print(f"Global summary automatically appended to: {report_path}")

    # ----- Build HTML report -----
    html_report = build_html_report(
        source_url=youtube_url,
        video_id=video_id,
        total_chunks=total_chunks,
        chunk_analyses=chunk_analyses,
        global_report=global_report,
    )
    html_path = "mindpilot_report.html"
    with open(html_path, "w", encoding="utf-8") as hf:
        hf.write(html_report)

    print(f"\nHTML report written to: {html_path}")
    print("\nAll done! Open the markdown OR HTML report file to review the analysis.\n")


if __name__ == "__main__":

    main()

