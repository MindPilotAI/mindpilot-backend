# mindpilot_api.py

import logging

from fastapi import FastAPI, Form, UploadFile, File

from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from mindpilot_engine import (
    run_full_analysis_from_youtube,
    run_full_analysis_from_text,
    run_full_analysis_from_article,
    run_quick_analysis_from_youtube,
    run_quick_analysis_from_text,
    run_quick_analysis_from_article,
    run_full_analysis_from_document,   # 🔹 NEW
    run_quick_analysis_from_document,  # 🔹 NEW
)



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
    mode: str = Form("text"),        # 👈 give it a default so file-only posts don't 422
    input_value: str = Form(""),     # allow empty when using file
    depth: str = Form("full"),       # "quick" or "full"
    file: UploadFile | None = File(None),
):

    """
    Primary MindPilot analysis endpoint.
    Form-based to match Netlify frontend.

    - mode = "YouTube": input_value is a YouTube URL
    - mode = "text":    input_value is a block of text to analyze
    - mode = "article": input_value is a news/article URL
    - depth = "quick" or "full"
    """
    logging.info(f"[MindPilot] /analyze received mode={mode}, depth={depth}")

    depth = (depth or "full").lower().strip()
    if depth not in ("quick", "full"):
        depth = "full"

    try:
        # 🔹 1) If a file is present, treat this as a document analysis
        if file is not None and file.filename:
            filename = file.filename
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

            return HTMLResponse(content=html_report, status_code=200)

        # 🔹 2) Otherwise, fall back to the existing mode-based logic
        if mode == "youtube":
            youtube_url = input_value.strip()
            logging.info(f"[MindPilot] Running {depth} YouTube analysis: {youtube_url}")

            if depth == "quick":
                html_report = run_quick_analysis_from_youtube(
                    youtube_url,
                    include_grok=False,  # keep quick = cheap; can revisit later
                )
            else:
                html_report = run_full_analysis_from_youtube(youtube_url)

        elif mode == "text":
            logging.info(f"[MindPilot] Running {depth} text analysis")

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

        elif mode == "article":
            article_url = input_value.strip()
            logging.info(f"[MindPilot] Running {depth} article analysis: {article_url}")

            if depth == "quick":
                html_report = run_quick_analysis_from_article(
                    article_url,
                    include_grok=False,
                )
            else:
                html_report = run_full_analysis_from_article(article_url)

        else:
            return PlainTextResponse(f"Unsupported mode: {mode}", status_code=400)

        return HTMLResponse(content=html_report, status_code=200)

    except Exception as e:
        logging.error("Error in /analyze endpoint", exc_info=True)
        #During development, return the actual error text so we can see what failed
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
