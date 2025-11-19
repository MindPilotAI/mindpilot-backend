# mindpilot_api.py

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from mindpilot_engine import run_full_analysis_from_youtube, run_full_analysis_from_text

app = FastAPI(title="MindPilot API")

# Allow your Netlify frontend to call this backend (adjust origin when deployed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev; later restrict to https://mind-pilot.ai
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    mode: str = Form(...),    # "youtube" or "text"
    input_value: str = Form(...),
):
    """
    Simple form-based endpoint.

    - mode = "youtube": input_value is a YouTube URL
    - mode = "text":    input_value is a block of text to analyze
    """
    try:
        if mode == "youtube":
            html_report = run_full_analysis_from_youtube(input_value.strip())
        elif mode == "text":
            html_report = run_full_analysis_from_text(input_value, source_label="Pasted text")
        else:
            return PlainTextResponse(f"Unsupported mode: {mode}", status_code=400)

        return HTMLResponse(content=html_report, status_code=200)

    except Exception as e:
        # Lightweight error handling for now
        return PlainTextResponse(f"Error during analysis: {e}", status_code=500)
