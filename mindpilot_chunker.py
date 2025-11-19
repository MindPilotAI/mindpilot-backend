import os
import textwrap


TRANSCRIPT_FILE = "mindpilot_transcript_output.txt"
OUTPUT_FILE = "mindpilot_lesson_pack.md"

# Target size per chunk (in characters). You can tweak this.
MAX_CHARS_PER_CHUNK = 1200


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
    Naive but effective: split on sentence boundaries while trying
    to keep each chunk under max_chars.
    """
    # Rough sentence split
    sentences = []
    current = []

    # Split on periods, question marks, exclamation marks
    for piece in text.replace("?", "?.").replace("!", "!.").split("."):
        piece = piece.strip()
        if not piece:
            continue
        sentences.append(piece + ".")

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding this sentence would exceed the limit, start a new chunk
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

    # Wrap the text so it's easier to read in the .md file
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
   - If none are clear, say: “No clear fallacies detected with high confidence.”

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
    Prompts that operate on the whole video/lesson (you'll paste these after
    you've seen all the chunk outputs).
    """
    prompt = """
# Global MindPilot Prompts for This Lesson

After you have generated chunk-level outputs, you can use the following prompts
(with access to ALL chunk summaries / key concepts) to build higher-level artifacts.

---

## 1. Full-Lesson Summary

**Prompt:**

"Using all of the chunk summaries and key concepts we've created so far, write a
coherent full-lesson summary (6–12 paragraphs). Make it suitable for a motivated
adult learner who wants a clear understanding of the main ideas."

---

## 2. Master Key-Concept Map

**Prompt:**

"From all chunks combined, build a master list of 10–20 key concepts. Group them
into 3–6 logical categories. For each concept, give:
- A 1–2 sentence explanation
- (Optional) a short memory hook or analogy."

---

## 3. Lesson Quiz

**Prompt:**

"Using all of the chunk-level comprehension questions as raw material, build a
mixed-format quiz for this lesson:
- 5–10 multiple-choice questions
- 5–10 short-answer questions
- 3–5 'explain in your own words' questions.

Ensure coverage across all major concepts."

---

## 4. Teacher / Facilitator Guide

**Prompt:**

"Create a teacher/facilitator guide for this entire lesson. Include:
- Learning objectives
- Suggested order of presentation
- Key talking points
- Where to use each visual/diagram
- Where to ask which questions
- Tips for checking understanding and avoiding common misconceptions."
"""
    return prompt.strip() + "\n"


def main():
    print("Reading transcript file:", TRANSCRIPT_FILE)
    transcript_text = load_transcript(TRANSCRIPT_FILE)

    print("Transcript length (characters):", len(transcript_text))
    chunks = chunk_text(transcript_text, MAX_CHARS_PER_CHUNK)
    total_chunks = len(chunks)
    print("Number of chunks created:", total_chunks)

    lines = []
    lines.append("# MindPilot Lesson Prompt Pack\n")
    lines.append(f"_Source: {TRANSCRIPT_FILE}_\n")
    lines.append(f"_Chunks: {total_chunks}_\n")
    lines.append("\n---\n\n")

    for idx, chunk in enumerate(chunks):
        chunk_prompt = build_chunk_prompt(chunk, idx, total_chunks)
        lines.append(chunk_prompt)

    # Add global prompts at the end
    lines.append(build_global_prompts())

    # Write to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nLesson prompt pack written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
