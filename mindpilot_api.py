# mindpilot_api.py

import logging

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from mindpilot_engine import (
    run_full_analysis_from_youtube,
    run_full_analysis_from_text,
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
    mode: str = Form(...),        # "YouTube" or "text"
    input_value: str = Form(...),
):
    """
    Primary MindPilot analysis endpoint.
    Form-based to match Netlify frontend.

    - mode = "YouTube": input_value is a YouTube URL
    - mode = "text":    input_value is a block of text to analyze
    """

    logging.info(f"[MindPilot] /analyze received mode={mode}")

    try:
        if mode == "youtube":
            youtube_url = input_value.strip()
            logging.info(f"[MindPilot] Running YouTube analysis: {youtube_url}")
            html_report = run_full_analysis_from_youtube(youtube_url)

        elif mode == "text":
            logging.info(
                f"[MindPilot] Running text analysis. Length={len(input_value)} chars"
            )
            html_report = run_full_analysis_from_text(
                raw_text=input_value,
                source_label="Pasted text",
            )

        else:
            logging.error(f"Unsupported mode: {mode}")
            return PlainTextResponse(f"Unsupported mode: {mode}", status_code=400)

        return HTMLResponse(content=html_report, status_code=200)

    except Exception as e:
        # Logs full stack trace to Railway for debugging
        logging.error("Error in /analyze endpoint", exc_info=True)

        return JSONResponse(
            {
                "status": "error",
                "message": "MindPilot backend crashed during analysis.",
                "detail": str(e),
                # You can uncomment this if you want full trace in JSON (dev only):
                # "trace": traceback.format_exc(),
            },

            status_code=500,
        )
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
