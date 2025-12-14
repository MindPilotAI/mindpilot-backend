# mindpilot_api.py  — CLEAN VERSION
import base64
import hmac
import secrets
import hashlib
import logging
import os
import re
import jwt  # PyJWT
import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import (
    FastAPI,
    Form,
    UploadFile,
    File,
    Depends,
    HTTPException,
    status,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.security import OAuth2PasswordBearer

from pydantic import BaseModel
from passlib.context import CryptContext

from mindpilot_engine import (
    fetch_youtube_transcript,
    fetch_article_text,
    run_analysis_from_transcript,
    run_quick_analysis_from_youtube,
    run_quick_analysis_from_text,
    run_quick_analysis_from_article,
    run_full_analysis_from_document,
    run_quick_analysis_from_document,
    ContentBlockedError,
)

from mindpilot_analyze import TranscriptUnavailableError
import html as html_lib  # for safe escaping in helper HTML

# -------------------------------------------------------------------
# GLOBALS / CONFIG
# -------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)

REPORT_STORE: dict[str, str] = {}

DATABASE_URL = os.getenv("DATABASE_URL")

SECRET_KEY = os.getenv("MP_JWT_SECRET", "change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

SUPERUSER_EMAIL = os.getenv("MP_SUPERUSER_EMAIL")  # optional

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

##  this gives one knob-board for cost parity and makes later pricing changes trivial.

def resolve_tier_settings(current_user: Optional[dict]) -> dict:
    plan = (current_user.get("plan") if current_user else "free") or "free"
    plan = plan.lower().strip()

    if plan in ("admin", "superuser"):
        return dict(
            plan=plan,
            include_grok_quick=True,
            include_grok_full=True,
            allow_full=True,
            allow_section_deep_dive=True,
            max_chunks_full=999,
            openai_model_quick="gpt-4o-mini",
            openai_model_full="gpt-4o-mini",
        )

    if plan in ("pro", "creator", "pro_creator"):
        return dict(
            plan=plan,
            include_grok_quick=True,     # key differentiator
            include_grok_full=True,
            allow_full=True,
            allow_section_deep_dive=False,   # default OFF (cost), can be Pro+ later
            max_chunks_full=12,              # or whatever you decide
            openai_model_quick="gpt-4o-mini",
            openai_model_full="gpt-4o-mini",
        )

    # free / anon
    return dict(
        plan="free",
        include_grok_quick=False,
        include_grok_full=False,
        allow_full=False,
        allow_section_deep_dive=False,
        max_chunks_full=0,
        openai_model_quick="gpt-4o-mini",
        openai_model_full="gpt-4o-mini",
    )


# -------------------------------------------------------------------
# DATABASE CONNECTION
# -------------------------------------------------------------------

def get_db_connection():
    """Return a new psycopg2 connection, or None on failure."""
    if not DATABASE_URL:
        return None

    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=DictCursor)
        return conn
    except Exception:
        logging.exception("Failed to connect to Postgres")
        return None


# -------------------------------------------------------------------
# AUTH HELPERS (PBKDF2 + PyJWT)
# -------------------------------------------------------------------

# ---------- Password helpers (PBKDF2-HMAC-SHA256, no 72-byte limit) ----------

def hash_password(plain_password: str) -> str:
    """
    Derive a hash using PBKDF2-HMAC-SHA256.
    Stored format: "pbkdf2_sha256$iterations$salt$base64(digest)".
    """
    if not isinstance(plain_password, str):
        plain_password = str(plain_password)

    iterations = 100_000
    salt = secrets.token_hex(16)

    dk = hashlib.pbkdf2_hmac(
        "sha256",
        plain_password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    )
    digest_b64 = base64.b64encode(dk).decode("ascii")
    return f"pbkdf2_sha256${iterations}${salt}${digest_b64}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a PBKDF2-HMAC-SHA256 password hash.
    If the stored format doesn't match, return False (treat as invalid).
    """
    try:
        if not isinstance(plain_password, str):
            plain_password = str(plain_password)

        scheme, iter_str, salt, digest_b64 = hashed_password.split("$", 3)
        if scheme != "pbkdf2_sha256":
            # Unknown / old scheme — treat as mismatch.
            return False

        iterations = int(iter_str)
        expected = base64.b64decode(digest_b64.encode("ascii"))

        dk = hashlib.pbkdf2_hmac(
            "sha256",
            plain_password.encode("utf-8"),
            salt.encode("utf-8"),
            iterations,
        )
        return hmac.compare_digest(expected, dk)
    except Exception:
        return False


def create_access_token(
    data: Dict[str, Any],
    expires_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES,
) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    if conn is None:
        return None

    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, email, password_hash, plan, is_active
                FROM users
                WHERE lower(email) = lower(%s)
                """,
                (email,),
            )
            row = cur.fetchone()
            if not row:
                return None

            return {
                "id": row["id"],
                "email": row["email"],
                "password_hash": row["password_hash"],
                "plan": row["plan"] or "free",
                "is_active": row["is_active"],
            }
    except Exception:
        logging.exception("Error fetching user by email")
        return None
    finally:
        conn.close()


def create_user(email: str, password: str) -> Dict[str, Any]:
    """
    Insert a new user into the `users` table as defined:

    id text PRIMARY KEY
    email text
    password_hash text
    plan text DEFAULT 'free'
    is_active boolean DEFAULT TRUE
    stripe_customer_id text
    created_at timestamptz DEFAULT now()
    updated_at timestamptz DEFAULT now()
    last_login_at timestamptz
    meta text
    """
    conn = get_db_connection()
    if conn is None:
        raise RuntimeError("Database unavailable")

    user_id = str(hashlib.sha256(f"{email}{datetime.utcnow()}".encode("utf-8")).hexdigest())
    hashed = hash_password(password)

    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (id, email, password_hash, plan, is_active, created_at, updated_at)
                VALUES (%s, %s, %s, 'free', TRUE, now(), now())
                """,
                (user_id, email.lower(), hashed),
            )
        return {
            "id": user_id,
            "email": email.lower(),
            "plan": "free",
            "is_active": True,
        }
    except Exception:
        logging.exception("Error creating user")
        raise
    finally:
        conn.close()


class SignupRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    username: str  # matches frontend field name
    password: str


def get_current_user_optional(request: Request) -> Optional[Dict[str, Any]]:
    """
    Return current user dict if Authorization header holds a valid JWT,
    otherwise return None. Does not raise on invalid/expired token.
    """
    auth_header = request.headers.get("Authorization") or request.headers.get("authorization")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        return None

    token = auth_header.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None

    email = payload.get("email")
    if not email:
        return None

    user = get_user_by_email(email)
    if not user or not user.get("is_active"):
        return None

    return user


# -------------------------------------------------------------------
# REPORT ID + SOCIAL HTML HELPERS
# -------------------------------------------------------------------

def generate_report_id(source_label: str = "", mode: str = "", depth: str = "full") -> str:
    """
    Generate a MindPilot-controlled report_id.
    Example: 20251201-article-nyt-com-ai-will-replace-knowledge-workers
    """
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    base = (source_label or mode or "report").lower()
    base = re.sub(r"[^a-z0-9]+", "-", base).strip("-")
    if not base:
        base = "report"
    return f"{ts}-{base}"


def build_social_share_page(cfr_html: str, report_id: str) -> str:
    """
    Build a simple HTML page suitable for /social/{report_id}.
    This wraps the CFR snippet and a CTA.
    """
    escaped_title = "MindPilot · Cognitive Flight Report"
    escaped_report_id = html_lib.escape(report_id)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{escaped_title}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
  <main>
    <h1>MindPilot Cognitive Flight Report</h1>
    <p>Report ID: {escaped_report_id}</p>
    <section>
      {cfr_html}
    </section>
    <hr />
    <p>Generated by <strong>MindPilot</strong> — your co-pilot for critical thinking.</p>
  </main>
</body>
</html>
"""


def strip_copy_ready_snippet_section(html: str) -> str:
    """
    Remove any embedded 'copy-ready snippet' block from the CFR HTML.
    We’ll build a separate social card page instead.
    """
    return re.sub(
        r"<!-- COPY-READY-SNIPPET-START -->.*?<!-- COPY-READY-SNIPPET-END -->",
        "",
        html,
        flags=re.DOTALL,
    )


def insert_marketing_cta(html: str, report_id: str) -> str:
    """
    Inject a lightweight CTA near the bottom of the CFR HTML,
    linking back to your main site.
    """
    cta = f"""
    <section class="mindpilot-cta">
      <hr />
      <p><strong>MindPilot</strong> helps you see the reasoning patterns behind modern media.</p>
      <p><a href="https://mind-pilot.ai" target="_blank" rel="noopener noreferrer">
         Learn more about MindPilot</a> · Report ID {html_lib.escape(report_id)}</p>
    </section>
    """
    if "</body>" in html:
        return html.replace("</body>", cta + "</body>")
    return html + cta


# -------------------------------------------------------------------
# DB HELPERS FOR REPORTS / LOGGING
# -------------------------------------------------------------------

def save_report_to_db(
    report_id: str,
    mode: str,
    depth: str,
    source_url: Optional[str],
    source_label: Optional[str],
    cfr_html: str,
    social_html: Optional[str],
) -> None:
    """
    Best-effort insert into `reports` table.
    Schema may differ slightly; any failure is logged and swallowed.
    """
    conn = get_db_connection()
    if conn is None:
        return

    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO reports (report_id, mode, depth, source_url, source_label, cfr_html, social_html, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, now())
                """,
                (report_id, mode, depth, source_url, source_label, cfr_html, social_html),
            )
    except Exception:
        logging.exception("Failed to save report to Postgres")
    finally:
        conn.close()


def load_report_from_db(report_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    if conn is None:
        return None

    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT report_id, mode, depth, source_url, source_label, cfr_html, social_html
                FROM reports
                WHERE report_id = %s
                """,
                (report_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "report_id": row["report_id"],
                "mode": row["mode"],
                "depth": row["depth"],
                "source_url": row["source_url"],
                "source_label": row["source_label"],
                "cfr_html": row["cfr_html"],
                "social_html": row["social_html"],
            }
    except Exception:
        logging.exception("Failed to load report from Postgres")
        return None
    finally:
        conn.close()


def log_blockage_to_db(
    mode: str,
    source_url: Optional[str],
    source_label: Optional[str],
    error_category: str,
    error_detail: str,
    http_status: Optional[int],
    user_id: Optional[str],
) -> None:
    conn = get_db_connection()
    if conn is None:
        return

    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO source_blockages
                    (mode, source_url, source_label, error_category, error_detail, http_status, user_id, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, now())
                """,
                (
                    mode,
                    source_url,
                    source_label,
                    error_category,
                    error_detail,
                    http_status,
                    user_id,
                ),
            )
    except Exception:
        logging.exception("Failed to log blockage")
    finally:
        conn.close()


def log_usage(
    user_id: Optional[str],
    ip_hash: Optional[str],
    source_type: Optional[str],
    depth: str,
    mode: str,
    report_id: str,
    success: bool,
    tokens_used: Optional[int] = None,
    error_category: Optional[str] = None,
    error_detail: Optional[str] = None,
) -> None:
    conn = get_db_connection()
    if conn is None:
        return

    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO usage_logs
                    (user_id, ip_hash, source_type, depth, mode, report_id,
                     tokens_used, success, error_category, error_detail, created_at)
                VALUES (%s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, now())
                """,
                (
                    user_id,
                    ip_hash,
                    source_type,
                    depth,
                    mode,
                    report_id,
                    tokens_used,
                    success,
                    error_category,
                    error_detail,
                ),
            )
    except Exception:
        logging.exception("Failed to log usage")
    finally:
        conn.close()


# -------------------------------------------------------------------
# FASTAPI APP + CORS
# -------------------------------------------------------------------

app = FastAPI(title="MindPilot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can narrow later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------
# AUTH ROUTES
# -------------------------------------------------------------------

@app.post("/signup")
async def signup(payload: SignupRequest):
    # 1) Normalize inputs
    email = (payload.email or "").strip()
    raw_password = payload.password or ""

    # 2) Enforce bcrypt's 72-byte limit *before* we ever call hash_password
    pw_bytes = raw_password.encode("utf-8")
    if len(pw_bytes) > 72:
        pw_bytes = pw_bytes[:72]
    safe_password = pw_bytes.decode("utf-8", errors="ignore")

    # 3) Check for existing user
    existing = get_user_by_email(email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    # 4) Create user with the truncated password
    try:
        user = create_user(email, safe_password)
    except Exception as e:
        logging.exception("Signup failed")
        # TEMPORARY: surface the error so we can still debug if something else goes wrong
        raise HTTPException(
            status_code=500,
            detail=f"DB error during signup: {e}",
        )

    # 5) Issue JWT
    access_token = create_access_token(
        {"sub": user["id"], "email": user["email"], "plan": user["plan"]}
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "email": user["email"],
        "plan": user["plan"],
    }

@app.post("/login")
async def login(payload: LoginRequest):
    user = get_user_by_email(payload.username)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect email or password")

    if not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=400, detail="Incorrect email or password")

    if not user["is_active"]:
        raise HTTPException(status_code=403, detail="Account is inactive")

    access_token = create_access_token(
        {"sub": user["id"], "email": user["email"], "plan": user["plan"]}
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "email": user["email"],
        "plan": user["plan"],
    }


@app.get("/me")
async def me(current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return current_user


# -------------------------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}

# -------------------------------------------------------------------
# GROK / DIAGNOSTIC HELPER
# -------------------------------------------------------------------

@app.get("/test_grok")
async def test_grok():
    """
    Lightweight diagnostic endpoint.
    Right now it just:
      - Confirms the API is up
      - Confirms we can reach Postgres (if DATABASE_URL is set)

    Later, we can extend this to actually call Grok/OpenAI via
    mindpilot_llm_client to verify LLM routing.
    """
    db_ok = False
    if DATABASE_URL:
        conn = get_db_connection()
        if conn is not None:
            db_ok = True
            try:
                conn.close()
            except Exception:
                pass

    return {
        "ok": True,
        "message": "MindPilot backend is running.",
        "database_ok": db_ok,
    }


# -------------------------------------------------------------------
# MAIN ANALYSIS ENDPOINT
# -------------------------------------------------------------------

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    mode: str = Form("text"),
    input_value: str = Form(""),
    depth: str = Form("full"),
    file: UploadFile | None = File(None),
    include_marketing_cta: str = Form("0"),
    article_title: str = Form(""),
    article_url: str = Form(""),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
    request: Request = None,
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

    user_id_for_logging = current_user["id"] if current_user else None

    client_ip = request.client.host if request and request.client else None
    ip_hash = (
        hashlib.sha256(client_ip.encode("utf-8")).hexdigest()[:64] if client_ip else None
    )
    settings = resolve_tier_settings(current_user)

    # Normalise depth
    depth = (depth or "full").lower().strip()
    if depth not in ("quick", "full"):
        depth = "full"



    include_marketing_cta_flag = str(include_marketing_cta or "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    article_title = (article_title or "").strip()
    article_url = (article_url or "").strip() or None

    source_label_for_id: str = ""
    html_report: Optional[str] = None

    try:
        # 1) File upload → document mode
        if file is not None and file.filename:
            filename = file.filename
            source_label_for_id = filename
            source_type_for_logs = "document"

            logging.info(
                f"[MindPilot] Running {depth} document analysis for upload: {filename}"
            )

            raw_bytes = await file.read()
            checklist_mode = "pro_quick" if settings["plan"] in ("pro", "creator", "pro_creator", "admin",
                                                                 "superuser") else "none"
            if depth == "quick":

                html_report = run_quick_analysis_from_document(
                    file_bytes=raw_bytes,
                    filename=filename,
                    include_grok=settings["include_grok_quick"],
                    creator_checklist_mode=checklist_mode,
                )
            else:
                html_report = run_full_analysis_from_document(
                    file_bytes=raw_bytes,
                    filename=filename,
                )

            source_url_for_db = None
            source_label_for_db = filename

        # 2) YouTube mode
        elif mode.lower() == "youtube":
            youtube_url = input_value.strip()
            source_label_for_id = youtube_url
            source_type_for_logs = "youtube"

            logging.info(f"[MindPilot] Running {depth} YouTube analysis: {youtube_url}")
            checklist_mode = "pro_quick" if settings["plan"] in ("pro", "creator", "pro_creator", "admin",
                                                                 "superuser") else "none"
            if depth == "quick":
                html_report = run_quick_analysis_from_youtube(
                    youtube_url,
                    include_grok=settings["include_grok_quick"],
                    creator_checklist_mode=checklist_mode,
                )
            else:
                if not settings["allow_full"]:
                    return PlainTextResponse(
                        "Full Creator Reports are available on the Pro plan. Run a Quick Scan, or upgrade to unlock full reports.",
                        status_code=402,
                    )

                transcript_text, video_id = fetch_youtube_transcript(youtube_url)

                html_report = run_analysis_from_transcript(
                    transcript_text=transcript_text,
                    source_label=youtube_url,
                    youtube_url=youtube_url,
                    video_id=video_id,
                    include_grok=settings["include_grok_full"],
                    allow_section_deep_dive=settings["allow_section_deep_dive"],
                    max_chunks=settings["max_chunks_full"] if settings["max_chunks_full"] else None,
                )

            source_url_for_db = youtube_url
            source_label_for_db = youtube_url

        # 3) Plain text mode
        elif mode.lower() == "text":
            if article_title:
                source_label_for_id = article_title
            else:
                snippet = (input_value or "").strip().splitlines()[0] if input_value else ""
                snippet = snippet.strip()
                if snippet:
                    if len(snippet) > 80:
                        snippet = snippet[:77] + "…"
                    source_label_for_id = f"Text: {snippet}"
                else:
                    source_label_for_id = "Pasted text"
            source_type_for_logs = "text"

            logging.info(
                f"[MindPilot] Running {depth} TEXT analysis (label={source_label_for_id!r})."
            )
            checklist_mode = "pro_quick" if settings["plan"] in ("pro", "creator", "pro_creator", "admin",
                                                                 "superuser") else "none"
            if depth == "quick":

                html_report = run_quick_analysis_from_text(
                    raw_text=input_value,
                    source_label=source_label_for_id,
                    include_grok=settings["include_grok_quick"],
                    creator_checklist_mode=checklist_mode,
                )
            else:
                if not settings["allow_full"]:
                    return PlainTextResponse(
                        "Full Creator Reports are available on the Pro plan. Run a Quick Scan, or upgrade to unlock full reports.",
                        status_code=402,
                    )
                text_value = (input_value or "").strip()
                if not text_value:
                    return PlainTextResponse("Please paste text to analyze.", status_code=400)
                html_report = run_analysis_from_transcript(
                    transcript_text=text_value,
                    source_label=source_label_for_id,
                    include_grok=settings["include_grok_full"],
                    allow_section_deep_dive=settings["allow_section_deep_dive"],
                    max_chunks=settings["max_chunks_full"] if settings["max_chunks_full"] else None,
                )

            source_url_for_db = article_url
            source_label_for_db = source_label_for_id

        # 4) Article URL mode
        elif mode.lower() == "article":
            url_from_input = input_value.strip()
            effective_url = article_url or url_from_input or None
            article_url = effective_url

            source_label_for_id = article_title or (effective_url or "Article")
            source_type_for_logs = "article"

            logging.info(
                f"[MindPilot] Running {depth} ARTICLE analysis: {effective_url}"
            )

            if not effective_url:
                return PlainTextResponse(
                    "No article URL was provided.", status_code=400
                )
            checklist_mode = "pro_quick" if settings["plan"] in ("pro", "creator", "pro_creator", "admin",
                                                                 "superuser") else "none"
            if depth == "quick":
                html_report = run_quick_analysis_from_article(
                    effective_url,
                    include_grok=settings["include_grok_quick"],
                    creator_checklist_mode=checklist_mode,
                )
            else:
                # Gate full reports by plan
                if not settings["allow_full"]:
                    return PlainTextResponse(
                        "Full Creator Reports are available on the Pro plan. "
                        "Run a Quick Scan, or upgrade to unlock full reports.",
                        status_code=402,
                    )

                # Fetch article text (engine helper)
                article_text = fetch_article_text(effective_url)

                # Run new-plan full creator report (single-pass by default)
                html_report = run_analysis_from_transcript(
                    transcript_text=article_text,
                    source_label=source_label_for_id,
                    youtube_url=None,
                    include_grok=settings["include_grok_full"],
                    allow_section_deep_dive=settings["allow_section_deep_dive"],
                    max_chunks=settings["max_chunks_full"] if settings["max_chunks_full"] else None,
                )

            source_url_for_db = effective_url
            source_label_for_db = source_label_for_id

        else:
            return PlainTextResponse(f"Unsupported mode: {mode}", status_code=400)

        if not html_report:
            return PlainTextResponse("No report was generated.", status_code=500)

        report_id = generate_report_id(
            source_label=source_label_for_id,
            mode=mode,
            depth=depth,
        )

        social_html: Optional[str] = None
        try:
            if include_marketing_cta_flag:
                social_html = build_social_share_page(html_report, report_id)

            html_report = strip_copy_ready_snippet_section(html_report)

            if include_marketing_cta_flag:
                html_report = insert_marketing_cta(html_report, report_id)
        except Exception:
            logging.exception("Failed to build social/CTA HTML")

        REPORT_STORE[report_id] = html_report
        if social_html:
            REPORT_STORE[f"{report_id}-social"] = social_html

        try:
            save_report_to_db(
                report_id=report_id,
                mode=mode.lower(),
                depth=depth,
                source_url=source_url_for_db,
                source_label=source_label_for_db,
                cfr_html=html_report,
                social_html=social_html,
            )
        except Exception:
            logging.exception("Failed to save report to Postgres")

        try:
            log_usage(
                user_id=user_id_for_logging,
                ip_hash=ip_hash,
                source_type=source_type_for_logs if "source_type_for_logs" in locals() else None,
                depth=depth,
                mode=mode.lower(),
                report_id=report_id,
                success=True,
                tokens_used=None,
            )
        except Exception:
            logging.exception("Failed to log usage")

        response = HTMLResponse(content=html_report, status_code=200)
        response.headers["X-MindPilot-Report-ID"] = report_id
        return response

    except ContentBlockedError as e:
        logging.warning("ContentBlockedError in /analyze: %s", e, exc_info=True)

        if mode.lower() == "article":
            source_url_for_log = article_url or input_value
        elif mode.lower() == "youtube":
            source_url_for_log = input_value
        else:
            source_url_for_log = None

        log_blockage_to_db(
            mode=mode.lower(),
            source_url=source_url_for_log,
            source_label=source_label_for_id or None,
            error_category="http_blocked",
            error_detail=str(e),
            http_status=getattr(e, "status_code", None),
            user_id=user_id_for_logging,
        )

        msg = (
            "MindPilot couldn’t fetch that article automatically.\n\n"
            "This usually means the site doesn’t allow automated readers or requires you to be logged in.\n\n"
            "Plan B:\n"
            "  1. Open the article in your browser.\n"
            "  2. Copy the text you can see OR use 'Print to PDF' and upload the file.\n"
            "  3. Optionally paste the original article URL back into MindPilot so it’s saved in the header."
        )
        return PlainTextResponse(msg, status_code=422)

    except TranscriptUnavailableError as e:
        logging.warning("TranscriptUnavailableError in /analyze: %s", e, exc_info=True)

        log_blockage_to_db(
            mode="youtube",
            source_url=input_value,
            source_label=source_label_for_id or None,
            error_category="transcript_unavailable",
            error_detail=str(e),
            http_status=None,
            user_id=user_id_for_logging,
        )

        msg = (
            "MindPilot couldn’t access the YouTube transcript automatically.\n\n"
            f"{e}\n\n"
            "Plan B:\n"
            "  1. On YouTube, click the three dots under the video and choose 'Show transcript'.\n"
            "  2. Copy the transcript text.\n"
            "  3. Paste it into MindPilot as plain text (we’ll still keep the original video URL as the source)."
        )
        return PlainTextResponse(msg, status_code=422)

    except Exception as e:
        logging.error("Error in /analyze endpoint", exc_info=True)

        source_url_for_log = None
        if mode.lower() == "youtube":
            source_url_for_log = input_value
        elif mode.lower() == "article":
            source_url_for_log = article_url or input_value

        log_blockage_to_db(
            mode=mode.lower(),
            source_url=source_url_for_log,
            source_label=source_label_for_id or None,
            error_category="other",
            error_detail=str(e),
            http_status=None,
            user_id=user_id_for_logging,
        )

        return PlainTextResponse(
            "MindPilot backend ERROR:\n"
            + str(e)
            + "\n\nIf this keeps happening, try copying the text or saving to PDF and uploading it.",
            status_code=500,
        )


# -------------------------------------------------------------------
# REPORT RETRIEVAL ROUTES
# -------------------------------------------------------------------

@app.get("/reports/{report_id}", response_class=HTMLResponse)
async def get_report(report_id: str):
    if report_id in REPORT_STORE:
        return HTMLResponse(REPORT_STORE[report_id])

    db_row = load_report_from_db(report_id)
    if db_row and db_row.get("cfr_html"):
        return HTMLResponse(db_row["cfr_html"])

    raise HTTPException(status_code=404, detail="Report not found")


@app.get("/social/{report_id}", response_class=HTMLResponse)
async def get_social(report_id: str):
    key = f"{report_id}-social"
    if key in REPORT_STORE:
        return HTMLResponse(REPORT_STORE[key])

    db_row = load_report_from_db(report_id)
    if db_row and db_row.get("social_html"):
        return HTMLResponse(db_row["social_html"])

    raise HTTPException(status_code=404, detail="Social snapshot not found")
