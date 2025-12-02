# mindpilot_api.py

import logging
import re
from datetime import datetime

from fastapi import FastAPI, Form, UploadFile, File

from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from mindpilot_engine import (
    run_full_analysis_from_youtube,
    run_full_analysis_from_text,
    run_full_analysis_from_article,
    run_quick_analysis_from_youtube,
    run_quick_analysis_from_text,
    run_quick_analysis_from_article,
    run_full_analysis_from_document,   # 🔹 NEW-ish
    run_quick_analysis_from_document,  # 🔹 NEW-ish
)

# Simple in-memory store: report_id -> HTML
REPORT_STORE: dict[str, str] = {}


def generate_report_id(source_label: str = "", mode: str = "", depth: str = "full") -> str:
    """
    Generate a MindPilot-controlled report_id.

    Example:
      20251201-article-nyt-com-ai-will-replace-knowledge-workers
    """
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    base = (source_label or mode or "report").lower()
    base = re.sub(r"[^a-z0-9]+", "-", base).strip("-")
    if not base:
        base = "report"

    return f"{ts}-{base}"




# ---------------------------------------------------------
# Logging config (critical for debugging Railway crashes)
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

app = FastAPI(title="MindPilot API")

# ---------------------------------------------------------
# CORS (works with Netlify; later restrict origins)
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # TODO: later restrict → ["https://mind-pilot.ai"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------
# Main Analysis Endpoint
# ---------------------------------------------------------
@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    mode: str = Form("text"),        # default so file-only posts don't 422
    input_value: str = Form(""),     # allow empty when using file
    depth: str = Form("full"),       # "quick" or "full"
    file: UploadFile | None = File(None),
):
    """
    Primary MindPilot analysis endpoint.
    Form-based to match Netlify frontend.

    - mode = "youtube": input_value is a YouTube URL
    - mode = "text":    input_value is a block of text to analyze
    - mode = "article": input_value is a news/article URL
    - depth = "quick" or "full"
    """
    logging.info(f"[MindPilot] /analyze received mode={mode}, depth={depth}")

    depth = (depth or "full").lower().strip()
    if depth not in ("quick", "full"):
        depth = "full"

    # We'll use this to generate a stable-ish report_id for storage/routes
    source_label_for_id = ""
    html_report: str | None = None

    try:
        # 1) If a file is present, treat this as a document analysis
        if file is not None and file.filename:
            filename = file.filename
            source_label_for_id = filename  # use the filename in the slug

            logging.info(
                f"[MindPilot] Running {depth} document analysis for upload: {filename}"
            )

            raw_bytes = await file.read()

            if depth == "quick":
                html_report = run_quick_analysis_from_document(
                    file_bytes=raw_bytes,
                    filename=filename,
                    include_grok=False,  # keep quick cheap for now
                )
            else:
                html_report = run_full_analysis_from_document(
                    file_bytes=raw_bytes,
                    filename=filename,
                )

        # 2) Otherwise, fall back to the existing mode-based logic
        elif mode.lower() == "youtube":
            youtube_url = input_value.strip()
            source_label_for_id = youtube_url  # use the URL in the slug

            logging.info(f"[MindPilot] Running {depth} YouTube analysis: {youtube_url}")

            if depth == "quick":
                html_report = run_quick_analysis_from_youtube(
                    youtube_url,
                    include_grok=False,  # keep quick = cheap; can revisit later
                )
            else:
                html_report = run_full_analysis_from_youtube(youtube_url)

        elif mode.lower() == "text":
            source_label_for_id = "Pasted text"

            logging.info(f"[MindPilot] Running {depth} TEXT analysis (pasted).")

            if depth == "quick":
                html_report = run_quick_analysis_from_text(
                    raw_text=input_value,
                    source_label="Pasted text",
                    include_grok=False,
                )
            else:
                html_report = run_full_analysis_from_text(
                    raw_text=input_value,
                    source_label="Pasted text",
                )

        elif mode.lower() == "article":
            article_url = input_value.strip()
            source_label_for_id = article_url  # use the URL

            logging.info(f"[MindPilot] Running {depth} article analysis: {article_url}")

            if depth == "quick":
                html_report = run_quick_analysis_from_article(
                    article_url,
                    include_grok=False,
                )
            else:
                html_report = run_full_analysis_from_article(article_url)

        else:
            # Unsupported mode
            return PlainTextResponse(f"Unsupported mode: {mode}", status_code=400)

        # ---------- Store the report and return HTML ----------

        if not html_report:
            return PlainTextResponse("No report was generated.", status_code=500)

        # Generate a MindPilot report_id and stash the HTML in memory
        report_id = generate_report_id(
            source_label=source_label_for_id,
            mode=mode,
            depth=depth,
        )
        REPORT_STORE[report_id] = html_report

        # For dev_index.html, we still return raw HTML,
        # but we also include the report_id in a header for debugging.
        response = HTMLResponse(content=html_report, status_code=200)
        response.headers["X-MindPilot-Report-ID"] = report_id
        return response

    except Exception as e:
        logging.error("Error in /analyze endpoint", exc_info=True)
        # During development, return the actual error text so we can see what failed
        return PlainTextResponse(
            f"MindPilot backend ERROR:\n{str(e)}",
            status_code=500,
        )


from mindpilot_llm_client import run_mindpilot_analysis  # add near top with other imports


@app.get("/test_openai")
async def test_openai():
    """
    Simple sanity check endpoint for the core MindPilot LLM client.
    Calls run_mindpilot_analysis() with a tiny prompt and returns either
    a preview of the output or a detailed error.
    """
    import traceback

    try:
        result = run_mindpilot_analysis(
            "Short sanity check: reply with one sentence confirming MindPilot is online."
        )
        return {
            "status": "ok",
            "preview": result[:400],
        }
    except Exception as e:
        return {
            "status": "error",
            "detail": str(e),
            "trace": traceback.format_exc(),
        }

@app.get("/test_grok")
async def test_grok():
    from mindpilot_llm_client import run_grok_enrichment

    try:
        result = run_grok_enrichment("Test Label", "This is a test global summary.")
        return {"status": "ok", "result": result}
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "trace": traceback.format_exc(),
        }
@app.get("/reports/{report_id}", response_class=HTMLResponse)
async def get_report(report_id: str):
    """
    Serve a previously generated Cognitive Flight Report by its report_id.

    NOTE: This is using an in-memory store (REPORT_STORE),
    so reports are lost if the server restarts. Good enough for dev
    and short-term testing; later we can swap this for a database or S3.
    """
    html = REPORT_STORE.get(report_id)
    if html is None:
        return PlainTextResponse("Report not found", status_code=404)
    return HTMLResponse(content=html, status_code=200)

