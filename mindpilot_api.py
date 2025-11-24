# mindpilot_api.py

import logging

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from mindpilot_engine import (
    run_full_analysis_from_youtube,
    run_full_analysis_from_text,
    run_full_analysis_from_article,
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
async def analyze(mode: str = Form(...), input_value: str = Form(...)):
    """
    Primary MindPilot analysis endpoint.
    Form-based to match Netlify frontend.

    - mode = "youtube": input_value is a YouTube URL
    - mode = "text":    input_value is a block of text to analyze
    - mode = "article": input_value is a news/article URL
    """
    logging.info(f"[MindPilot] /analyze received mode={mode}")

    try:
        if mode == "youtube":
            youtube_url = input_value.strip()
            logging.info(f"[MindPilot] Running YouTube analysis: {youtube_url}")
            html_report = run_full_analysis_from_youtube(youtube_url)

        elif mode == "text":
            html_report = run_full_analysis_from_text(
                raw_text=input_value,
                source_label="Pasted text",
            )

        elif mode == "article":
            article_url = input_value.strip()
            logging.info(f"[MindPilot] Running article analysis: {article_url}")
            html_report = run_full_analysis_from_article(article_url)

        else:
            return PlainTextResponse(f"Unsupported mode: {mode}", status_code=400)

        return HTMLResponse(content=html_report, status_code=200)

    except Exception as e:
        logging.error("Error in /analyze endpoint", exc_info=True)
        return JSONResponse(
            {
                "status": "error",
                "message": "MindPilot backend crashed during analysis.",
                "detail": str(e),
            },
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
