import os
import re
import textwrap

from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from mindpilot_llm_client import run_mindpilot_analysis



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



def build_social_card_html(
    *,
    source_type: str,
    overall_score_100: int | None,
    score_label: str,
    fallacy_snippet: str,
    questions_snippet: str,
    grok_line: str | None,
    report_url: str | None,
    escape_html,
) -> str:
    """
    Build a compact 'social card' HTML block that can be screenshotted
    for posts on X/Twitter, LinkedIn, TikTok, etc.

    - source_type: 'YouTube video', 'Article / web page', etc.
    - overall_score_100: 0–100, or None if we didn't parse a score
    - score_label: human-readable label (e.g. 'Mixed / uneven reasoning')
    - fallacy_snippet: short text (1–2 items) summarizing fallacies/biases
    - questions_snippet: short text summarizing questions to ask
    - grok_line: optional one-line 'Grok enrichment' summary
    - report_url: public URL for the full report (can be None for now)
    - escape_html: function to escape arbitrary text for HTML
    """

    # Title: keep it soft, not cocky
    header_title = f"Reasoning snapshot for this {source_type.lower()}"

    if overall_score_100 is None:
        score_display = "Reasoning snapshot"
        bar_width = 100
        bar_caption = score_label or "Reasoning profile overview"
    else:
        score_display = f"{overall_score_100}/100"
        bar_width = overall_score_100
        bar_caption = f"{overall_score_100}/100 · {score_label}"

    fallacy_text = fallacy_snippet.strip() if fallacy_snippet else "Key fallacy and bias signals highlighted in the full report."
    questions_text = questions_snippet.strip() if questions_snippet else "See the full report for critical questions to stress-test this piece."
    grok_text = (grok_line or "").strip()

    if report_url:
        footer_left = f"Full Cognitive Flight Report → {escape_html(report_url)}"
    else:
        footer_left = "Full Cognitive Flight Report available in the MindPilot app."

    return f"""
      <section class="card-sub social-card">
        <div class="social-header">
          <div class="social-title">{escape_html(header_title)}</div>
          <div class="social-logo">
            <!-- Adjust path if needed for your deployed assets -->
            <img src="/assets/mindpilot-symbol.png" alt="MindPilot symbol" />
          </div>
        </div>

        <div class="social-score-row">
          <div class="social-score-number">{escape_html(score_display)}</div>
          <div class="score-bar-track">
            <div class="score-bar-fill" style="width: {bar_width}%;"></div>
          </div>
          <div class="score-bar-label">{escape_html(bar_caption)}</div>
        </div>

        <div class="social-grid">
          <div>
            <div class="social-label">Fallacies &amp; bias signals</div>
            <p class="social-text">{escape_html(fallacy_text)}</p>
          </div>
          <div>
            <div class="social-label">Questions to ask</div>
            <p class="social-text">{escape_html(questions_text)}</p>
          </div>
        </div>
    """ + (
        f"""
        <div class="social-grok">
          Grok enrichment: {escape_html(grok_text)}
        </div>
    """ if grok_text else ""
    ) + f"""
        <div class="social-footer">
          <span>{footer_left}</span>
          <span class="social-watermark">MindPilot · Cognitive Flight Report</span>
        </div>
      </section>
    """


def build_html_report(
    source_url,
    video_id,
    total_chunks,
    chunk_analyses,
    global_report,
    grok_insights: str | None = None,
    depth: str = "full",  # "quick" or "full"
):
    SOCIAL_HANDLES = {
        "twitter": "@YourHandleHere",  # update once you decide
        "tiktok": "@YourHandleHere",
        "linkedin": "MindPilot · Cognitive Flight Report",
    }

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

    def escape_html(text: str) -> str:
        if text is None:
            return ""
        return (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
        )

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
            sections = {}
            for i, (key, idx, heading) in enumerate(found):
                start = idx
                end = found[i + 1][1] if i + 1 < len(found) else len(raw)
                sections[key] = raw[start:end].strip()

            full_summary = sections.get("full", "")
            master_map = sections.get("map", "")
            rationality_profile = sections.get("profile", "")
            investor_summary = sections.get("investor", "")
            questions_block = sections.get("questions", "")

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
    def build_fallacy_table(raw_map: str) -> str:
        """
        Parse the markdown-style Master Fallacy & Bias Map into an HTML table.

        Expected pattern (your current output):

        - **Logical Fallacies**
          - **False Cause / Post Hoc**: Implies direct causation... (High)

        - **Cognitive Biases**
          - **Availability Heuristic**: ... (Medium)
        """
        if not raw_map:
            return ""

        rows: list[tuple[str, str, str, str]] = []
        current_category: str | None = None

        for line in raw_map.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            # Category lines: "- **Logical Fallacies**"
            m_cat = re.match(r"^-\s*\*\*(.+?)\*\*\s*$", stripped)
            if m_cat:
                current_category = m_cat.group(1).strip()
                continue

            # Item lines: "- **Name**: description (High)"
            m_item = re.match(
                r"^-\s*\*\*(.+?)\*\*\s*:\s*(.+?)(?:\s*\((High|Medium|Low)\))?\s*$",
                stripped,
            )
            if m_item and current_category:
                name = m_item.group(1).strip()
                desc = m_item.group(2).strip()
                severity = (m_item.group(3) or "").strip()
                rows.append((current_category, name, desc, severity))

        if not rows:
            return ""

        severity_rank = {"High": 0, "Medium": 1, "Low": 2, "": 3, None: 3}

        # Sort: High → Medium → Low, then by category/name
        rows.sort(
            key=lambda r: (
                severity_rank.get(r[3], 3),
                r[0].lower(),
                r[1].lower(),
            )
        )

        body_rows = []
        for category, name, desc, severity in rows:
            sev_label = severity or "—"
            sev_class = severity.lower() if severity else "none"
            body_rows.append(
                f"""
          <tr>
            <td class="fallacy-type">{escape_html(category)}</td>
            <td class="fallacy-name">{escape_html(name)}</td>
            <td class="fallacy-desc">{escape_html(desc)}</td>
            <td class="fallacy-severity">
              <span class="fallacy-tag {sev_class}">{escape_html(sev_label)}</span>
            </td>
          </tr>
                """
            )

        table_html = (
            """
        <div class="fallacy-table-wrapper">
          <table class="fallacy-table">
            <thead>
              <tr>
                <th>Pattern type</th>
                <th>Fallacy / bias</th>
                <th>How it shows up in this piece</th>
                <th>Severity</th>
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
        return table_html
    def summarize_fallacies_for_social(raw_map: str) -> str:
        """
        Build a short, 1–2 item summary like:
        'False Cause / Post Hoc (High); Appeal to Emotion (High)'
        for use in the social card.
        """
        if not raw_map:
            return ""

        items: list[tuple[str, str, str]] = []
        for line in raw_map.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            m_item = re.match(
                r"^-\s*\*\*(.+?)\*\*\s*:\s*(.+?)(?:\s*\((High|Medium|Low)\))?\s*$",
                stripped,
            )
            if not m_item:
                continue
            name = m_item.group(1).strip()
            severity = (m_item.group(3) or "").strip()
            items.append((name, severity))

        if not items:
            return ""

        # Prefer High -> Medium -> Low
        rank = {"High": 0, "Medium": 1, "Low": 2, "": 3, None: 3}
        items.sort(key=lambda x: rank.get(x[1], 3))

        top = items[:2]
        parts = []
        for name, sev in top:
            if sev:
                parts.append(f"{name} ({sev})")
            else:
                parts.append(name)
        return "; ".join(parts)

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

        return " · ".join(selected)

    # Clean global sections
    full_summary = _strip_internal_subheadings(full_summary)
    master_map = _strip_internal_subheadings(master_map)
    rationality_profile = _strip_internal_subheadings(rationality_profile)
    investor_summary = _strip_internal_subheadings(investor_summary)
    questions_block = _strip_internal_subheadings(questions_block)
    global_report_clean = _strip_internal_subheadings(global_report or "")

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
            if line.strip():
                grok_line = line.strip()
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
          <div class="score-bar-track">
            <div class="score-bar-fill" style="width: {overall_score_100}%"></div>
          </div>
          <div class="score-bar-label">{overall_score_100}/100 overall reasoning quality</div>
        </div>
        """

    # ---------- meta + social text snippets ----------

    source_type = infer_source_type(source_url or "")
    # Build social-card HTML (only if we have at least some signal)
    social_card_html = ""
    if (overall_score_100 is not None) or fallacy_snippet or questions_snippet:
        social_card_html = build_social_card_html(
            source_type=source_type,
            overall_score_100=overall_score_100,
            score_label=label_for_chip or "Reasoning profile overview",
            fallacy_snippet=fallacy_snippet,
            questions_snippet=questions_snippet,
            grok_line=grok_line,
            report_url=None,  # you can swap in a real public URL later
            escape_html=escape_html,
        )
    depth_text = depth_label(depth)

    summary_body = extract_summary_body(full_summary)
    preview = summary_body
    if len(preview) > 220:
        preview = preview[:220].rstrip() + "…"

    readable_score = (
        f"{overall_score_100}/100" if overall_score_100 is not None else "a mixed reasoning profile"
    )
    base_line = f"MindPilot Cognitive Flight Report on this {source_type}"

    if preview:
        core_takeaway = preview
    else:
        core_takeaway = (
            "MindPilot flagged key reasoning patterns, fallacies, and bias signals in this piece."
        )

    source_link = source_url or ""
    tw_handle = SOCIAL_HANDLES.get("twitter", "")
    tt_handle = SOCIAL_HANDLES.get("tiktok", "")
    li_handle = SOCIAL_HANDLES.get("linkedin", "")

    twitter_snippet = (
        f"{base_line}: reasoning score {readable_score}. "
        f"{core_takeaway} "
        f"Source: {source_link} "
        f"{tw_handle} #MindPilot #CriticalThinking"
    )

    tiktok_snippet = (
        f"This {source_type.lower()} just got a MindPilot Cognitive Flight Report.\n"
        f"Reasoning score: {readable_score}.\n"
        f"Source: {source_link}\n"
        "MindPilot mapped fallacies, bias patterns, and rhetorical pressure so you don't have to.\n"
        f"{tt_handle} · Full Cognitive Flight Report at mind-pilot.ai"
    )

    linkedin_snippet = (
        f"Today's MindPilot Cognitive Flight Report evaluates the reasoning quality of a "
        f"{source_type.lower()}.\n"
        f"Reasoning score: {readable_score}.\n"
        f"{core_takeaway}\n"
        f"Source: {source_link}\n"
        "MindPilot surfaces argument structure, fallacies, bias patterns, and critical questions "
        "to help professionals think more clearly.\n"
        f"{li_handle}"
    )

    esc_twitter_snippet = escape_html(twitter_snippet)
    esc_tiktok_snippet = escape_html(tiktok_snippet)
    esc_linkedin_snippet = escape_html(linkedin_snippet)

    # ---------- global presence flag & Grok card ----------

    has_any_global = any(
        [
            esc_profile,
            esc_full,
            esc_map,
            esc_questions,
            esc_investor,
            bool(esc_grok),
        ]
    )

    if esc_grok:
        grok_card_html = f"""
          <section class="card-sub">
            <div class="collapsible-header" onclick="toggleSection('grok-card')">
              <span>MindPilot × Grok Live Context &amp; Creative Debrief</span>
              <span class="collapsible-toggle" id="toggle-grok-card">Hide</span>
            </div>
            <div class="collapsible-body open" id="section-grok-card">
              <pre class="pre-block">{esc_grok}</pre>
            </div>
          </section>
    """

    else:
        grok_card_html = ""

    # ---------- HTML skeleton ----------

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>MindPilot – Cognitive Flight Report</title>
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
        .fallacy-table-wrapper {{
      margin-top: 0.5rem;
      overflow-x: auto;
    }}
    .fallacy-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.82rem;
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
      white-space: nowrap;
      font-weight: 500;
      color: var(--text-muted);
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
    .score-bar-container {{
      margin: 0.4rem 0;
    }}
    .score-bar-track {{
      width: 100%;
      height: 0.55rem;
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
    .social-snippet {{
      margin-top: 0.6rem;
      margin-bottom: 0.6rem;
    }}
    .social-label {{
      font-size: 0.78rem;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 0.2rem;
    }}
        .social-card {{
      margin-top: 0.8rem;
      border-radius: 1rem;
      border: 1px solid var(--border-subtle);
      background: linear-gradient(135deg, #0B1B33, #1A365D);
      color: #E2E8F0;
      padding: 0.9rem 1rem;
      box-shadow: 0 18px 40px rgba(0, 0, 0, 0.35);
    }}
    .social-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.6rem;
    }}
    .social-title {{
      font-size: 0.9rem;
      font-weight: 600;
      color: #EDF2F7;
    }}
    .social-logo img {{
      width: 28px;
      height: 28px;
      border-radius: 999px;
      display: block;
    }}
    .social-score-row {{
      margin-bottom: 0.7rem;
      font-size: 0.78rem;
    }}
    .social-score-number {{
      font-size: 1.6rem;
      font-weight: 700;
      color: #FBD38D;
      margin-bottom: 0.25rem;
    }}
    .social-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.7rem;
      margin-bottom: 0.6rem;
      font-size: 0.78rem;
    }}
    @media (max-width: 640px) {{
      .social-grid {{
        grid-template-columns: minmax(0, 1fr);
      }}
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
       width: 28px;
       height: 28px;
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

  </style>
</head>
<body>
    <div class="page">
    <header>
      <div class="logo-title">MindPilot Cognitive Flight Report</div>
      <div class="tagline">Your critical thinking copilot’s readback of this content.</div>
      <div class="header-meta">
        <span>{escape_html(source_type)}</span>
        <span>· {total_chunks} section(s) analyzed</span>
        <span>· {escape_html(depth_text)}</span>
      </div>
    </header>

"""

    # ---------- Global card + subcards ----------

    if has_any_global:
        html += f"""
        <section class="card">
          <div class="card-title">Global MindPilot Reasoning Overview</div>
          <div class="card-body">
            <p class="card-body-text">
              Quick read of the whole piece – overall reasoning quality, dominant fallacy &amp; bias patterns,
              and how emotionally loaded the framing is.
              In quick mode, MindPilot runs a single global scan. In full mode, it also
              runs section-level diagnostics, so scores may shift slightly within a
              small tolerance as the system “thinks harder” about the reasoning.
            </p>
            {score_chip_html}
            {score_bar_html}
            <ul class="card-body-text" style="margin-top:0.4rem;padding-left:1.1rem;">
              <li><strong>Source type:</strong> {escape_html(source_type)}</li>
            </ul>
          </div>
          {social_card_html}
    """

        # 1) Rationality Profile – always first if present
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

        # 2) Critical Thinking Questions
        if esc_questions:
            html += f"""
          <section class="card-sub">
            <div class="collapsible-header" onclick="toggleSection('critical-questions')">
              <span>Critical Thinking Questions to Ask Yourself</span>
              <span class="collapsible-toggle" id="toggle-critical-questions">Hide</span>
            </div>
            <div class="collapsible-body" id="section-critical-questions">
              <pre class="pre-block">{esc_questions}</pre>
            </div>
          </section>
    """

        # 3) MindPilot × Grok Live Context (if present)
        html += grok_card_html

        # 4) Full Reasoning Scan pieces
        if depth == "full":
            if esc_full:
                html += f"""
              <section class="card-sub">
                <div class="collapsible-header" onclick="toggleSection('global-summary')">
                  <span>Full Reasoning Scan – Global Summary</span>
                  <span class="collapsible-toggle" id="toggle-global-summary">Show</span>
                </div>
                <div class="collapsible-body" id="section-global-summary">
                  <pre class="pre-block">{esc_full}</pre>
                </div>
              </section>
        """

            if esc_map:
                html += f"""
              <section class="card-sub">
                <div class="collapsible-header" onclick="toggleSection('master-map')">
                  <span>Master Fallacy &amp; Bias Map</span>
                  <span class="collapsible-toggle" id="toggle-master-map">Hide</span>
                </div>
                <div class="collapsible-body open" id="section-master-map">
                  {fallacy_table_html or f'<pre class="pre-block">{esc_map}</pre>'}
                </div>
              </section>
        """

        if esc_investor:
                html += f"""
              <section class="card-sub">
                <div class="collapsible-header" onclick="toggleSection('investor-summary')">
                  <span>Condensed Executive Summary</span>
                  <span class="collapsible-toggle" id="toggle-investor-summary">Show</span>
                </div>
                <div class="collapsible-body" id="section-investor-summary">
                  <pre class="pre-block">{esc_investor}</pre>
                </div>
              </section>
        """

        else:
            # Quick reports: slimmed-down executive summary only
            if esc_investor:
                html += f"""
          <section class="card-sub">
            <div class="collapsible-header" onclick="toggleSection('investor-summary')">
              <span>Condensed Executive Summary</span>
              <span class="collapsible-toggle" id="toggle-investor-summary">Show</span>
            </div>
            <div class="collapsible-body" id="section-investor-summary">
              <pre class="pre-block">{esc_investor}</pre>
            </div>
          </section>
    """

        # 5) Social snippet drafts – kept for your marketing runs
        html += f"""
          <section class="card-sub">
            <div class="collapsible-header" onclick="toggleSection('social-snippets')">
              <span>Social Snippet Drafts</span>
              <span class="collapsible-toggle" id="toggle-social-snippets">Show</span>
            </div>
            <div class="collapsible-body" id="section-social-snippets">
              <p class="card-body-text">
                Copy and lightly edit these for your MindPilot social posts. They’re auto-tuned
                to this Cognitive Flight Report and meant for your own accounts
                (X/Twitter, TikTok/Shorts, LinkedIn).
              </p>

              <div class="social-snippet">
                <div class="social-label">X / Twitter post</div>
                <pre class="pre-block">{esc_twitter_snippet}</pre>
              </div>

              <div class="social-snippet">
                <div class="social-label">TikTok / Shorts caption</div>
                <pre class="pre-block">{esc_tiktok_snippet}</pre>
              </div>

              <div class="social-snippet">
                <div class="social-label">LinkedIn post intro</div>
                <pre class="pre-block">{esc_linkedin_snippet}</pre>
              </div>
            </div>
          </section>
        </section>
    """

    elif esc_global_fallback:

        html += f"""

        <section class="card">

          <div class="card-title">Global MindPilot Reasoning Overview</div>

          <div class="card-body">

            <pre class="pre-block">{esc_global_fallback}</pre>

          </div>

        </section>

    """

    # ---------- Disclaimer + Creator checklist (always shown) ----------

    html += """

        <section class="card-sub">

          <div class="card-title">How to read this report</div>

          <div class="card-body">

            MindPilot is your critical thinking copilot. This analysis highlights patterns in

            reasoning—such as logical fallacies, cognitive biases, and attempts to persuade—

            so you can reflect more deliberately on what you’re hearing or reading.

            <br/><br/>

            <strong>Important:</strong> MindPilot does <em>not</em> act as a fact checker and does

            <em>not</em> independently verify whether specific claims or statements are true.

            It focuses on <strong>how</strong> arguments are made, not on adjudicating the

            real-world accuracy of every assertion.

          </div>

        </section>


        <section class="card-sub">

          <div class="card-title">Creator Pre-Publish Checklist</div>

          <div class="card-body">

            <p class="card-body-text">

              If you are the one publishing this piece (article, video, newsletter, or post),

              use this checklist to tighten your draft before it goes live.

            </p>


            <ul class="card-body-text" style="margin-top:0.4rem;padding-left:1.1rem;">

              <li>

                <strong>Headline &amp; opener:</strong>

                Does your title or hook accurately reflect the substance of the piece,

                or is it leaning on exaggeration, fear, or outrage just to get clicks?

              </li>

              <li>

                <strong>Claims vs. evidence:</strong>

                For your 2–3 core claims, have you clearly shown what evidence supports them?

                Would a skeptical reader understand <em>why</em> you believe each claim is true?

              </li>

              <li>

                <strong>Counter-arguments:</strong>

                Have you acknowledged the strongest reasonable objections or alternative views,

                and either addressed them or clearly scoped what you’re <em>not</em> claiming?

              </li>

              <li>

                <strong>Language intensity:</strong>

                Are you using loaded or absolute language ("always", "never", "everyone")

                where more precise wording would tell the truth without inflaming emotions?

              </li>

              <li>

                <strong>Audience autonomy:</strong>

                Are you giving your audience enough context, nuance, and uncertainty

                to make up their own mind, or are you steering them toward a single permitted conclusion?

              </li>

            </ul>


            <p class="card-body-text" style="margin-top:0.6rem;">

              You don’t need to remove all emotion or persuasion to publish responsibly.

              The goal is to make your reasoning <strong>transparent</strong> so a thoughtful

              reader can see what you’re doing and decide whether they agree.

            </p>

          </div>

        </section>

    """

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
        depth="full",
    )
    html_path = "mindpilot_report.html"
    with open(html_path, "w", encoding="utf-8") as hf:
        hf.write(html_report)

    print(f"\nHTML report written to: {html_path}")
    print("\nAll done! Open the markdown OR HTML report file to review the analysis.\n")


if __name__ == "__main__":

    main()

