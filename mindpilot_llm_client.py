import os
from openai import OpenAI
import httpx  # NEW: for Grok/xAI HTTP calls

_client = None


def get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        _client = OpenAI(api_key=api_key)
    return _client


def run_mindpilot_analysis(prompt: str, model: str = "gpt-4o-mini") -> str:
    """
    Core MindPilot LLM call.

    Default model:
    - gpt-4o-mini: widely available, cheap, good reasoning
    - If you later confirm access to gpt-4.1-mini, you can switch back.
    """

    client = get_openai_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are MindPilot, a neutral reasoning-analysis copilot.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


# ================================
# Grok / xAI integration
# ================================

XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_BASE_URL = "https://api.x.ai/v1"


def run_grok_enrichment(content_label: str, global_summary_md: str) -> str:
    """
    Optional enrichment call to Grok (xAI).
    Returns an empty string if Grok is not configured or if the call fails.

    content_label: short human-readable label (video title, 'Pasted text', etc.)
    global_summary_md: MindPilot's global reasoning summary in Markdown.
    """
    if not XAI_API_KEY:
        # Grok not configured: silently return empty enrichment
        return ""

    prompt = f"""You are Grok, assisting MindPilot, a neutral reasoning-analysis copilot.

MindPilot has ALREADY completed a structured diagnostic of a piece of content
using its own system (Cognitive Flight Report, Rationality Profile, etc.).
You are NOT repeating that analysis. Instead, you are adding a short, engaging,
responsible “Live Context & Critical Thinking Debrief” that builds ON TOP OF
MindPilot’s output.

Your job is to:
- Comment on the overall reasoning patterns that MindPilot identified.
- Add real-time, reality-based social context (how similar claims or topics are
  being discussed across news and social media RIGHT NOW).
- Suggest practical critical-thinking moves the user can make next.

STRICT REQUIREMENTS:
- Absolutely no profanity, vulgarity, or edgy personal attacks.
- Be politically neutral. Do not advocate for or against any person, party, or policy.
- If the content touches public policy or ideology, describe multiple perspectives calmly.
- Do not use partisan language, slogans, or culture-war framing.
- Do not contradict MindPilot’s diagnostic structure. If something seems off,
  frame it gently as “a possible additional angle” rather than a correction.
- Do not give fact-checking verdicts. Focus on reasoning, context, and questions.

Tone:
- Warm, lightly witty, curious.
- Never mean-spirited or mocking.
- Playfully creative, but always responsible and educational.

Your output will appear as a special enrichment card in the MindPilot report.

---------------------------------------
INPUTS YOU WILL RECEIVE:
1. A short label describing the content (e.g., video title or “Pasted text”)
2. The MindPilot GLOBAL SUMMARY (Markdown) that already describes:
   - main claims and conclusions
   - key reasoning patterns
   - fallacies, biases, or rhetorical tactics

---------------------------------------
TASK:
Using real-time knowledge (Grok search allowed), produce EXACTLY:

## MindPilot × Grok Live Context & Creative Debrief

### Current Context Snapshot
- 2–4 calm, reality-based bullets about how this topic, style of argument,
  or set of claims is currently showing up across news and social media.
- If the topic is niche or not widely discussed, say so and focus on closely
  related themes instead.

### How This Relates to the MindPilot Analysis
- 2–3 short bullets that explicitly reference MindPilot's findings
  (e.g., fallacy clusters, bias patterns, argumentative style)
  and place them in a broader real-world pattern:
  “This kind of framing often shows up when…”
  “This combination of emotional appeal + selective evidence is common in…”

### Creative Analogy
- 1 short, vivid, PG-rated metaphor that helps the user see the reasoning
  pattern more clearly (without being cruel to any real people or groups).

### Reflective Questions
- 3 neutral, metacognitive questions that encourage better thinking, such as:
  - questions about what evidence would actually change someone’s mind
  - questions about missing perspectives or incentives
  - questions about how the user might respond more thoughtfully

Keep everything compact, friendly, helpful, safe, and non-partisan.
Do NOT tell the user what to believe. Help them think more clearly instead.

---------------------------------------
CONTENT LABEL:
{content_label}

---------------------------------------
MINDPILOT GLOBAL SUMMARY:
{global_summary_md}
"""

    try:
        response = httpx.post(
            f"{XAI_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {XAI_API_KEY}"},
            json={
                "model": "grok-4",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "temperature": 0.4,
                "max_tokens": 700,
                "extra_body": {
                    "real_time_data": True,
                    "search_enabled": True,
                },
            },
            timeout=45.0,
        )

        data = response.json()
        if not isinstance(data, dict) or "choices" not in data:
            return ""

        content = data["choices"][0]["message"]["content"]
        return content.strip() if content else ""
    except Exception:
        # Fail safe: Grok is optional, so quietly ignore errors
        return ""
