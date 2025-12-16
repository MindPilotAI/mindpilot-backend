# mindpilot_api.py  — CLEAN VERSION
import base64
import hmac
import uuid
import secrets
import hashlib
import logging
import os
import re
import jwt  # PyJWT
import psycopg2
import stripe
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

from pydantic import BaseModel, Field
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
    """
    Central tier policy: feature flags + usage caps.

    Notes:
    - "preview" = Tier 0 (no login required). We will assign this when current_user is None.
    - "free" = logged-in free user (if/when you enable signup on public site).
    """

    # Tier 0: unauthenticated preview
    if not current_user:
        plan = "preview"
    else:
        plan = (current_user.get("plan") or "free").lower().strip()

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
            # caps
            max_quick_per_24h=999999,
            max_full_per_30d=999999,
            max_requests_per_15m=999999,
        )

    if plan in ("pro_plus", "pro+", "plus", "proplus"):
        return dict(
            plan="pro_plus",
            include_grok_quick=True,
            include_grok_full=True,
            allow_full=True,
            allow_section_deep_dive=True,   # Pro+ unlock
            max_chunks_full=18,
            openai_model_quick="gpt-4o-mini",
            openai_model_full="gpt-4o-mini",
            # caps
            max_quick_per_24h=75,
            max_full_per_30d=40,
            max_requests_per_15m=60,
        )

    if plan in ("academic", "edu", "education", "student"):
        return dict(
            plan="academic",
            include_grok_quick=False,       # keep costs sane; you can flip later
            include_grok_full=False,
            allow_full=True,                # academic can do full (but capped)
            allow_section_deep_dive=False,
            max_chunks_full=10,
            openai_model_quick="gpt-4o-mini",
            openai_model_full="gpt-4o-mini",
            # caps
            max_quick_per_24h=15,
            max_full_per_30d=10,
            max_requests_per_15m=30,
        )

    if plan in ("pro", "creator", "pro_creator"):
        return dict(
            plan=plan,
            include_grok_quick=True,     # key differentiator
            include_grok_full=True,
            allow_full=True,
            allow_section_deep_dive=False,
            max_chunks_full=12,
            openai_model_quick="gpt-4o-mini",
            openai_model_full="gpt-4o-mini",
            # caps
            max_quick_per_24h=25,
            max_full_per_30d=15,
            max_requests_per_15m=40,
        )

    # Tier 0 preview
    if plan == "preview":
        return dict(
            plan="preview",
            include_grok_quick=False,
            include_grok_full=False,
            allow_full=False,
            allow_section_deep_dive=False,
            max_chunks_full=0,
            openai_model_quick="gpt-4o-mini",
            openai_model_full="gpt-4o-mini",
            # caps
            max_quick_per_24h=1,
            max_full_per_30d=0,
            max_requests_per_15m=10,
        )

    # Logged-in free (Tier 1)
    return dict(
        plan="free",
        include_grok_quick=False,
        include_grok_full=False,
        allow_full=False,
        allow_section_deep_dive=False,
        max_chunks_full=0,
        openai_model_quick="gpt-4o-mini",
        openai_model_full="gpt-4o-mini",
        # caps
        max_quick_per_24h=3,
        max_full_per_30d=0,
        max_requests_per_15m=15,
    )
# -----------------------
# STRIPE (v1 minimal)
# -----------------------
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
MP_APP_BASE_URL = os.getenv("MP_APP_BASE_URL", "https://mind-pilot.ai").rstrip("/")

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

def _stripe_price_for_plan(plan: str) -> str:
    p = (plan or "").lower().strip()
    if p == "pro":
        return os.getenv("STRIPE_PRICE_PRO", "")
    if p == "pro_plus":
        return os.getenv("STRIPE_PRICE_PRO_PLUS", "")
    if p == "academic":
        return os.getenv("STRIPE_PRICE_ACADEMIC", "")
    return ""
def _set_user_plan_from_stripe(
    user_id: str,
    plan: str,
    stripe_customer_id: Optional[str] = None,
    stripe_subscription_id: Optional[str] = None,
    stripe_status: Optional[str] = None,
) -> None:
    conn = get_db_connection()
    if conn is None:
        return
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE users
                SET plan = %s,
                    stripe_customer_id = COALESCE(%s, stripe_customer_id),
                    stripe_subscription_id = COALESCE(%s, stripe_subscription_id),
                    stripe_status = COALESCE(%s, stripe_status),
                    plan_source = 'stripe',
                    updated_at = now()
                WHERE id = %s
                """,
                (plan, stripe_customer_id, stripe_subscription_id, stripe_status, user_id),
            )
    except Exception:
        logging.exception("Stripe: failed to update user plan")
    finally:
        conn.close()

class StripeCheckoutRequest(BaseModel):
    plan: str  # "pro" | "pro_plus" | "academic"

logging.info("Stripe config check:")
logging.info(f"  STRIPE_SECRET_KEY set: {bool(STRIPE_SECRET_KEY)}")
logging.info(f"  STRIPE_PRICE_PRO: {os.getenv('STRIPE_PRICE_PRO')}")
logging.info(f"  STRIPE_PRICE_PRO_PLUS: {os.getenv('STRIPE_PRICE_PRO_PLUS')}")
logging.info(f"  STRIPE_PRICE_ACADEMIC: {os.getenv('STRIPE_PRICE_ACADEMIC')}")


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
    email: str
    password: str


    class Config:
        allow_population_by_field_name = True



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
from datetime import date  # add near the other datetime imports if not present

def save_report_to_db(
    report_id: str,
    mode: str,
    depth: str,
    source_url: Optional[str],
    source_label: Optional[str],
    cfr_html: str,
    social_html: Optional[str],
    include_grok: bool = False,
    cache_key: str = "",
    expires_on: Optional[date] = None,
) -> None:
    """
    Best-effort insert into `reports` table.
    Matches your current schema:
      reports: id, mode, depth, source_url, source_label, created_at, cfr_html, social_html, include_grok, cache_key, expires_at/expires_on
    """
    conn = get_db_connection()
    if conn is None:
        return

    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO reports
                  (id, mode, depth, source_url, source_label, created_at,
                   cfr_html, social_html, include_grok, cache_key, expires_on)
                VALUES
                  (%s, %s, %s, %s, %s, now(),
                   %s, %s, %s, %s, %s)
                """,
                (
                    report_id,
                    (mode or "").lower(),
                    (depth or "").lower(),
                    source_url,
                    source_label,
                    cfr_html,
                    social_html,
                    bool(include_grok),
                    cache_key or "",
                    expires_on,
                ),
            )
    except Exception:
        logging.exception("Failed to save report to Postgres")
    finally:
        try:
            conn.close()
        except Exception:
            pass

def normalize_source_url(url: str) -> str:
    u = (url or "").strip()
    # basic normalization: drop trailing slash
    if u.endswith("/"):
        u = u[:-1]
    return u

def fetch_cached_report_from_db(
    *,
    mode: str,
    depth: str,
    source_url: str,
    include_grok: bool,
) -> Optional[dict]:
    """
    Returns dict with keys: report_id, cfr_html, social_html, created_at
    or None if no unexpired cached report exists.

    Uses reports.id, reports.mode, reports.depth, reports.source_url, reports.include_grok, reports.expires_at.
    """
    if not source_url:
        return None

    conn = get_db_connection()
    if conn is None:
        return None

    try:
        norm_url = normalize_source_url(source_url)

        with conn, conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(
                """
                SELECT id, cfr_html, social_html, created_at
                FROM reports
                WHERE mode = %s
                  AND depth = %s
                  AND source_url = %s
                  AND include_grok = %s
                  AND expires_at >= CURRENT_DATE
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (
                    (mode or "").lower(),
                    (depth or "").lower(),
                    norm_url,
                    bool(include_grok),
                ),
            )

            row = cur.fetchone()
            if not row:
                return None

            return dict(
                report_id=row["id"],
                cfr_html=row["cfr_html"],
                social_html=row["social_html"],
                created_at=row["created_at"],
            )
    except Exception:
        logging.exception("fetch_cached_report_from_db failed")
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass



def load_report_from_db(report_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    if conn is None:
        return None

    try:
        with conn, conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(
                """
                SELECT id, mode, depth, source_url, source_label, cfr_html, social_html
                FROM reports
                WHERE id = %s

                """,
                (report_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "report_id": row["id"],
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

def count_successful_usage_last_24h(
    *,
    user_id: Optional[str],
    ip_hash: Optional[str],
    depth: str,
) -> int:
    """
    Count successful, generated reports in last 24h.
    Uses user_id when logged in, otherwise falls back to ip_hash.
    """
    conn = get_db_connection()
    if conn is None:
        # If DB is down, fail open (don't block). You can flip this later.
        return 0

    depth = (depth or "").lower().strip()

    try:
        with conn, conn.cursor() as cur:
            if user_id:
                cur.execute(
                    """
                    SELECT COUNT(*)::int
                    FROM usage_logs
                    WHERE user_id = %s
                      AND success = TRUE
                      AND depth = %s
                      AND created_at >= (now() - interval '24 hours')
                    """,
                    (user_id, depth),
                )
            elif ip_hash:
                cur.execute(
                    """
                    SELECT COUNT(*)::int
                    FROM usage_logs
                    WHERE ip_hash = %s
                      AND success = TRUE
                      AND depth = %s
                      AND created_at >= (now() - interval '24 hours')
                    """,
                    (ip_hash, depth),
                )
            else:
                return 0

            row = cur.fetchone()
            return int(row[0] or 0)
    except Exception:
        logging.exception("count_successful_usage_last_24h failed")
        return 0
    finally:
        try:
            conn.close()
        except Exception:
            pass


def enforce_usage_caps_or_raise(*, settings: dict, depth: str, user_id: Optional[str], ip_hash: Optional[str]) -> None:
    """
    Enforces usage caps using usage_logs.
    Uses a 24h rolling window for quick, and 30d rolling window for full.
    Raises HTTPException(402) when cap is hit.
    """
    plan = (settings.get("plan") or "free").lower().strip()
    depth = (depth or "full").lower().strip()
    if depth not in ("quick", "full"):
        depth = "full"

    # Admin bypass
    if plan in ("admin", "superuser"):
        return

    # Default caps (can later move to DB / Stripe metadata)
    caps = {
        "free":      {"quick_24h": 5,  "full_30d": 0},
        "preview":   {"quick_24h": 5,  "full_30d": 0},
        "pro":       {"quick_24h": 30, "full_30d": 120},  # 6/day ≈ 180/30d; using 120 conservative
        "pro_creator":{"quick_24h": 30, "full_30d": 120},
        "pro_full":  {"quick_24h": 30, "full_30d": 120},
        "pro_plus":  {"quick_24h": 60, "full_30d": 240},
        "academic":  {"quick_24h": 20, "full_30d": 80},
    }

    cap = caps.get(plan, caps["free"])
    quick_cap = int(cap["quick_24h"])
    full_cap_30d = int(cap["full_30d"])

    # Identify key for counting
    identifier_user = (user_id or "").strip()
    identifier_ip = (ip_hash or "").strip()

    # If not logged in, enforce on ip_hash (best effort)
    use_user = bool(identifier_user)

    conn = get_db_connection()
    if conn is None:
        return  # don't block if DB unavailable

    try:
        with conn, conn.cursor(cursor_factory=DictCursor) as cur:
            # QUICK: last 24 hours
            if depth == "quick":
                if quick_cap <= 0:
                    raise HTTPException(
                        status_code=402,
                        detail="Quick Scans are not available on your current plan. Please upgrade to unlock.",
                    )

                if use_user:
                    cur.execute(
                        """
                        SELECT COUNT(*) AS n
                        FROM usage_logs
                        WHERE user_id = %s
                          AND depth = 'quick'
                          AND success = true
                          AND created_at >= (now() - interval '24 hours')
                        """,
                        (identifier_user,),
                    )
                else:
                    cur.execute(
                        """
                        SELECT COUNT(*) AS n
                        FROM usage_logs
                        WHERE ip_hash = %s
                          AND depth = 'quick'
                          AND success = true
                          AND created_at >= (now() - interval '24 hours')
                        """,
                        (identifier_ip,),
                    )

                n = int(cur.fetchone()["n"] or 0)
                if n >= quick_cap:
                    raise HTTPException(
                        status_code=402,
                        detail=f"Usage cap hit: you’ve reached {quick_cap} Quick Scans in the last 24 hours. Upgrade to unlock higher limits.",
                    )
                return

            # FULL: last 30 days (rolling)
            if full_cap_30d <= 0:
                raise HTTPException(
                    status_code=402,
                    detail="Full Cognitive Flight Reports are available on the Pro plan. Run a Quick Scan, or upgrade to unlock full reports.",
                )

            if use_user:
                cur.execute(
                    """
                    SELECT COUNT(*) AS n
                    FROM usage_logs
                    WHERE user_id = %s
                      AND depth = 'full'
                      AND success = true
                      AND created_at >= (now() - interval '30 days')
                    """,
                    (identifier_user,),
                )
            else:
                cur.execute(
                    """
                    SELECT COUNT(*) AS n
                    FROM usage_logs
                    WHERE ip_hash = %s
                      AND depth = 'full'
                      AND success = true
                      AND created_at >= (now() - interval '30 days')
                    """,
                    (identifier_ip,),
                )

            n = int(cur.fetchone()["n"] or 0)
            if n >= full_cap_30d:
                raise HTTPException(
                    status_code=402,
                    detail="Usage cap hit: you’ve reached your Full Report limit for the last 30 days. Upgrade to unlock higher limits.",
                )
            return
    finally:
        try:
            conn.close()
        except Exception:
            pass


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
            usage_id = str(uuid.uuid4())
            cur.execute(
                """
                INSERT INTO usage_logs (id, user_id, ip_hash, source_type, depth, mode, 
                        report_id, tokens_used, success, error_category, error_detail, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, now())
                """,
                (
                    usage_id,
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

def _usage_scope_where(user_id: Optional[str], ip_hash: Optional[str]):
    """
    Prefer user_id (logged-in). Fallback to ip_hash (preview).
    Returns (sql_fragment, params_tuple).
    """
    if user_id:
        return "user_id = %s", (user_id,)
    if ip_hash:
        return "ip_hash = %s", (ip_hash,)
    # worst-case: no identity; treat as single shared bucket
    return "ip_hash IS NULL", ()


def _count_usage(*, user_id: Optional[str], ip_hash: Optional[str], depth: str, since_minutes: int) -> int:
    conn = get_db_connection()
    if conn is None:
        return 0

    where_id, id_params = _usage_scope_where(user_id, ip_hash)

    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT COUNT(*)
                FROM usage_logs
                WHERE {where_id}
                  AND success = TRUE
                  AND depth = %s
                  AND created_at >= (now() - (%s || ' minutes')::interval)
                """,
                (*id_params, (depth or "").lower().strip(), int(since_minutes)),
            )
            row = cur.fetchone()
            return int(row[0] if row else 0)
    except Exception:
        logging.exception("Usage count query failed")
        return 0
    finally:
        conn.close()


def enforce_usage_caps_or_raise(*, settings: dict, depth: str, user_id: Optional[str], ip_hash: Optional[str]) -> None:
    """
    Enforce caps BEFORE running any LLM-heavy work.
    Uses usage_logs as the source of truth.
    """
    plan = (settings.get("plan") or "free").lower().strip()
    depth = (depth or "quick").lower().strip()

    # Simple burst limiter (all plans)
    max_15m = int(settings.get("max_requests_per_15m") or 0)
    if max_15m > 0:
        used_15m = _count_usage(user_id=user_id, ip_hash=ip_hash, depth="quick", since_minutes=15) \
                   + _count_usage(user_id=user_id, ip_hash=ip_hash, depth="full", since_minutes=15)
        if used_15m >= max_15m:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit: too many requests. Please wait a bit and try again. (plan={plan})",
            )

    if depth == "quick":
        limit = int(settings.get("max_quick_per_24h") or 0)
        if limit <= 0:
            raise HTTPException(status_code=402, detail=f"Quick reports are not enabled for this plan. (plan={plan})")
        used = _count_usage(user_id=user_id, ip_hash=ip_hash, depth="quick", since_minutes=24 * 60)
        if used >= limit:
            raise HTTPException(
                status_code=402,
                detail=f"Daily limit reached: {limit} quick reports / 24h on {plan}.",
            )
        return

    # depth == "full"
    limit = int(settings.get("max_full_per_30d") or 0)
    if limit <= 0:
        raise HTTPException(
            status_code=402,
            detail=f"Full reports are not enabled for this plan. (plan={plan})",
        )
    used = _count_usage(user_id=user_id, ip_hash=ip_hash, depth="full", since_minutes=30 * 24 * 60)
    if used >= limit:
        raise HTTPException(
            status_code=402,
            detail=f"Monthly limit reached: {limit} full reports / 30 days on {plan}.",
        )


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
    expose_headers=[
        "X-MindPilot-Report-ID",
        "X-MindPilot-Cache-Hit",
    ],
)


@app.post("/stripe/create-checkout-session")
async def stripe_create_checkout_session(
    payload: StripeCheckoutRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Stripe not configured")

    plan = (payload.plan or "").lower().strip()
    price_id = _stripe_price_for_plan(plan)
    if not price_id:
        raise HTTPException(status_code=400, detail="Unknown plan")

    success_url = f"{MP_APP_BASE_URL}/account.html?stripe=success"
    cancel_url = f"{MP_APP_BASE_URL}/account.html?stripe=cancel"

    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=success_url,
        cancel_url=cancel_url,
        customer_email=current_user.get("email"),
        client_reference_id=current_user.get("id"),
        metadata={"mp_user_id": current_user.get("id"), "mp_plan": plan},
    )
    return {"url": session.url}

@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Stripe webhook not configured")

    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")

    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception:
        logging.exception("Stripe: invalid webhook signature")
        raise HTTPException(status_code=400, detail="Invalid signature")

    etype = event.get("type", "")
    obj = event.get("data", {}).get("object", {}) or {}

    if etype == "checkout.session.completed":
        md = obj.get("metadata", {}) or {}
        user_id = md.get("mp_user_id") or obj.get("client_reference_id")
        plan = (md.get("mp_plan") or "").lower().strip()

        _set_user_plan_from_stripe(
            user_id=user_id,
            plan=plan,
            stripe_customer_id=obj.get("customer"),
            stripe_subscription_id=obj.get("subscription"),
            stripe_status="active",
        )

    # Downgrade on cancel
    if etype == "customer.subscription.deleted":
        customer_id = obj.get("customer")
        if customer_id:
            conn = get_db_connection()
            try:
                with conn, conn.cursor(cursor_factory=DictCursor) as cur:
                    cur.execute("SELECT id FROM users WHERE stripe_customer_id = %s LIMIT 1", (customer_id,))
                    row = cur.fetchone()
                if row:
                    _set_user_plan_from_stripe(user_id=row["id"], plan="free", stripe_status="canceled")
            except Exception:
                logging.exception("Stripe: failed downgrade on subscription.deleted")
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

    return {"ok": True}



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
    email = (payload.email or "").strip()
    user = get_user_by_email(email)

    if not user:
        raise HTTPException(status_code=400, detail="Incorrect email or password")

    raw_password = payload.password or ""
    pw_bytes = raw_password.encode("utf-8")
    if len(pw_bytes) > 72:
        pw_bytes = pw_bytes[:72]
    safe_password = pw_bytes.decode("utf-8", errors="ignore")

    raw_password = payload.password or ""
    pw_bytes = raw_password.encode("utf-8")
    if len(pw_bytes) > 72:
        pw_bytes = pw_bytes[:72]
    safe_password = pw_bytes.decode("utf-8", errors="ignore")

    if not verify_password(safe_password, user["password_hash"]):
        raise HTTPException(status_code=400, detail="Incorrect email or password")

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
    ## settings = resolve_tier_settings(current_user)
    # --- Tier resolution (with DEV override support) --- TEMPORARY DEBUGGING
    tier_user = current_user

    plan_override = None
    try:
        plan_override = request.query_params.get("mp_plan")
    except Exception:
        plan_override = None

    if (
            plan_override
            and os.getenv("MP_ALLOW_PLAN_OVERRIDE", "0") == "1"
            and plan_override.lower().strip() in {"preview", "free", "pro", "pro_plus", "academic", "admin"}

    ):
        tier_user = dict(current_user or {})
        ov = plan_override.lower().strip()
        # normalize friendly aliases
        if ov in ("pro+", "proplus", "plus"):
            ov = "pro_plus"
        if ov in ("edu", "education", "student"):
            ov = "academic"
        tier_user["plan"] = ov

    settings = resolve_tier_settings(tier_user)
    plan_slug = (settings.get("plan") or "").lower().strip()
    is_paid_plan = plan_slug in ("pro", "pro_creator", "pro_full", "pro_plus", "academic", "admin", "superuser")

    # Normalise depth
    depth = (depth or "full").lower().strip()
    if depth not in ("quick", "full"):
        depth = "full"
    # -------------------------------------------------
    # Cache + Grok policy (single source of truth)
    # -------------------------------------------------

    # Whether Grok is included for THIS request
    if depth == "quick":
        include_grok = bool(settings.get("include_grok_quick"))
    else:
        include_grok = bool(settings.get("include_grok_full"))

    # Cache expiration policy
    #  - with Grok: 3 days
    #  - without Grok: 7 days
    expires_days = 3 if include_grok else 7
    expires_at = (datetime.utcnow().date() + timedelta(days=expires_days))

    # Cache key (only for URL-backed sources)
    cache_key = ""

    include_marketing_cta_flag = str(include_marketing_cta or "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    article_title = (article_title or "").strip()
    article_url = (article_url or "").strip() or None

    def cache_lookup_if_available(*, mode: str, depth: str, cache_source_url: str) -> Optional[HTMLResponse]:
        if not cache_source_url:
            return None

        include_grok_req = (
            settings["include_grok_full"] if depth == "full" else settings["include_grok_quick"]
        )

        cached = fetch_cached_report_from_db(
            mode=mode.lower(),
            depth=depth,
            source_url=cache_source_url,
            include_grok=include_grok_req,
        )

        if cached and cached.get("cfr_html"):
            resp = HTMLResponse(content=cached["cfr_html"], status_code=200)
            resp.headers["X-MindPilot-Report-ID"] = cached.get("report_id", "") or ""
            resp.headers["X-MindPilot-Cache-Hit"] = "1"
            return resp

        return None

    source_label_for_id: str = ""
    html_report: Optional[str] = None

    try:
        # 1) File upload → document mode
        if file is not None and file.filename:
            enforce_usage_caps_or_raise(settings=settings, depth=depth, user_id=user_id_for_logging, ip_hash=ip_hash)
            filename = file.filename
            source_label_for_id = filename
            source_type_for_logs = "document"

            logging.info(
                f"[MindPilot] Running {depth} document analysis for upload: {filename}"
            )

            raw_bytes = await file.read()
            plan_slug = (settings.get("plan") or "").lower().strip()
            is_paid_plan = plan_slug in ("pro", "pro_creator", "pro_full", "pro_plus", "academic", "admin", "superuser")
            checklist_mode = ("pro_full" if depth == "full" else "pro_quick") if is_paid_plan else "none"

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
                    creator_checklist_mode=checklist_mode,
                )

            source_url_for_db = None
            source_label_for_db = filename

        # 2) YouTube mode
        elif mode.lower() == "youtube":
            youtube_url = input_value.strip()
            # ✅ Cache reuse for viral YouTube URLs (before any LLM calls)
            cached_resp = cache_lookup_if_available(mode="youtube", depth=depth, cache_source_url=youtube_url)
            if cached_resp:
                return cached_resp
            enforce_usage_caps_or_raise(settings=settings, depth=depth, user_id=user_id_for_logging, ip_hash=ip_hash)
            source_label_for_id = youtube_url
            source_type_for_logs = "youtube"

            logging.info(f"[MindPilot] Running {depth} YouTube analysis: {youtube_url}")
            plan_slug = (settings.get("plan") or "").lower().strip()
            is_paid_plan = plan_slug in ("pro", "pro_creator", "pro_full", "pro_plus", "academic", "admin", "superuser")
            checklist_mode = ("pro_full" if depth == "full" else "pro_quick") if is_paid_plan else "none"

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
            enforce_usage_caps_or_raise(settings=settings, depth=depth, user_id=user_id_for_logging, ip_hash=ip_hash)
            plan_slug = (settings.get("plan") or "").lower().strip()
            is_paid_plan = plan_slug in ("pro", "pro_creator", "pro_full", "pro_plus", "academic", "admin", "superuser")
            checklist_mode = ("pro_full" if depth == "full" else "pro_quick") if is_paid_plan else "none"

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
            # ✅ Cache reuse for viral articles (before any LLM calls)
            cached_resp = cache_lookup_if_available(mode="article", depth=depth, cache_source_url=effective_url or "")
            if cached_resp:
                return cached_resp
            enforce_usage_caps_or_raise(settings=settings, depth=depth, user_id=user_id_for_logging, ip_hash=ip_hash)


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
            plan_slug = (settings.get("plan") or "").lower().strip()
            is_paid_plan = plan_slug in ("pro", "pro_creator", "pro_full", "pro_plus", "academic", "admin", "superuser")
            checklist_mode = ("pro_full" if depth == "full" else "pro_quick") if is_paid_plan else "none"

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

            include_grok_req = settings["include_grok_full"] if depth == "full" else settings["include_grok_quick"]
            expires_days = 3 if include_grok_req else 7
            expires_on = (datetime.utcnow().date() + timedelta(days=expires_days))

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

        # --- cache metadata (used for reuse + permalink freshness rules)
        include_grok_req = settings["include_grok_full"] if depth == "full" else settings["include_grok_quick"]
        expires_days = 3 if include_grok_req else 7
        expires_on = (datetime.utcnow().date() + timedelta(days=expires_days))

        # cache_key: strict match on mode/depth/include_grok/source_url (normalized)
        norm_url = normalize_source_url(source_url_for_db or "")
        cache_key = f"{(mode or '').lower()}|{(depth or '').lower()}|grok:{int(bool(include_grok_req))}|{norm_url}"

        try:
            save_report_to_db(
                report_id=report_id,
                mode=mode.lower(),
                depth=depth,
                source_url=source_url_for_db,
                source_label=source_label_for_db,
                cfr_html=html_report,
                social_html=social_html,
                include_grok=include_grok_req,
                cache_key=cache_key,
                expires_on=expires_on,
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
    except HTTPException as e:
        # Cleanly pass through rate limit / plan gating errors (and other deliberate HTTPExceptions)
        return PlainTextResponse(str(e.detail), status_code=e.status_code)

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

from fastapi.responses import JSONResponse

def fetch_my_reports_from_db(user_id: str, limit: int = 100) -> list[dict]:
    conn = get_db_connection()
    if conn is None:
        return []

    try:
        with conn, conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(
                """
                SELECT
                    r.id AS report_id,
                    r.mode,
                    r.depth,
                    r.source_label,
                    r.source_url,
                    r.created_at
                FROM reports r
                JOIN (
                    SELECT report_id, MAX(created_at) AS last_run
                    FROM usage_logs
                    WHERE user_id = %s AND success = TRUE
                    GROUP BY report_id
                ) u
                ON u.report_id = r.id
                ORDER BY u.last_run DESC
                LIMIT %s
                """,
                (user_id, int(limit)),
            )
            rows = cur.fetchall() or []
            return [
                {
                    "report_id": row["report_id"],
                    "mode": row["mode"],
                    "depth": row["depth"],
                    "source_label": row["source_label"],
                    "source_url": row["source_url"],
                    "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
                }
                for row in rows
            ]
    except Exception:
        logging.exception("fetch_my_reports_from_db failed")
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass


@app.get("/api/my/reports")
async def api_my_reports(current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    items = fetch_my_reports_from_db(current_user["id"], limit=200)
    return JSONResponse({"items": items})

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
