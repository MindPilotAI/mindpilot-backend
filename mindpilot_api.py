# mindpilot_api.py
import logging
import re
import html as html_lib  # NEW: for safe escaping in helper HTML
import os
import pg8000
import os
import psycopg2
from psycopg2.extras import DictCursor
import hashlib
from fastapi.responses import HTMLResponse, PlainTextResponse
# (add JSONResponse later if you want; for now we’ll still send plain text)
from pydantic import BaseModel

import uuid
import hashlib
import hmac
import base64
import json
import time


from urllib.parse import urlparse


from datetime import datetime, timedelta
from fastapi import FastAPI, Form, UploadFile, File, Depends, HTTPException, status, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from mindpilot_engine import (
    run_full_analysis_from_youtube,
    run_full_analysis_from_text,
    run_full_analysis_from_article,
    run_quick_analysis_from_youtube,
    run_quick_analysis_from_text,
    run_quick_analysis_from_article,
    run_full_analysis_from_document,   # 🔹 NEW-ish
    run_quick_analysis_from_document,  # 🔹 NEW-ish
    ContentBlockedError,
)

from mindpilot_analyze import TranscriptUnavailableError
from typing import Optional, Dict, Any

from passlib.context import CryptContext
import jwt

# Simple in-memory store: report_id -> HTML
REPORT_STORE: dict[str, str] = {}
# Database connection string injected by Railway
DATABASE_URL = os.getenv("DATABASE_URL")
# -------------------------------------------------------------------
# Auth / JWT-like token helpers (no extra dependencies)
# -------------------------------------------------------------------

JWT_SECRET = os.getenv("MP_JWT_SECRET", "dev-secret-change-me")
JWT_EXP_SECONDS = 7 * 24 * 3600  # 7 days
SUPERUSER_EMAIL = os.getenv("MP_SUPERUSER_EMAIL")


def hash_password(plain_password: str) -> str:
    """Hash a password using PBKDF2-HMAC with a random salt."""
    salt = uuid.uuid4().hex
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        plain_password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000,
    )
    return f"{salt}${base64.b16encode(dk).decode('ascii')}"


def verify_password(plain_password: str, stored_hash: str) -> bool:
    """Verify a password against a stored salt$hash string."""
    try:
        salt, hex_hash = stored_hash.split("$", 1)
    except ValueError:
        return False
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        plain_password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000,
    )
    return base64.b16encode(dk).decode("ascii") == hex_hash


def create_access_token(user_id: str, email: str, plan: str) -> str:
    """Create a simple HMAC-signed token with exp, sub, email, plan."""
    payload = {
        "sub": user_id,
        "email": email,
        "plan": plan,
        "exp": int(time.time()) + JWT_EXP_SECONDS,
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sig = hmac.new(JWT_SECRET.encode("utf-8"), raw, hashlib.sha256).digest()
    token = (
        base64.urlsafe_b64encode(raw).decode().rstrip("=")
        + "."
        + base64.urlsafe_b64encode(sig).decode().rstrip("=")
    )
    return token


def decode_access_token(token: str):
    """Decode and verify the HMAC token. Return payload dict or None."""
    try:
        raw_b64, sig_b64 = token.split(".", 1)
        raw = base64.urlsafe_b64decode(raw_b64 + "===")
        expected_sig = hmac.new(JWT_SECRET.encode("utf-8"), raw, hashlib.sha256).digest()
        actual_sig = base64.urlsafe_b64decode(sig_b64 + "===")
        if not hmac.compare_digest(expected_sig, actual_sig):
            raise ValueError("Invalid signature")
        payload = json.loads(raw.decode("utf-8"))
        if payload.get("exp") and payload["exp"] < int(time.time()):
            raise ValueError("Token expired")
        return payload
    except Exception:
        return None

def get_db_connection():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return None
    return psycopg2.connect(db_url, cursor_factory=DictCursor)
def get_user_by_email(email: str):
    """Fetch a user row as a dict by email, or None."""
    conn = get_db_connection()
    if conn is None:
        return None

    try:
        with conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, email, password_hash, plan, is_active
                    FROM users
                    WHERE email = %s
                    """,
                    (email.lower(),),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "id": row["id"],
                    "email": row["email"],
                    "password_hash": row["password_hash"],
                    "plan": row.get("plan") or "free",
                    "is_active": row.get("is_active", True),
                }
    except Exception:
        logging.exception("Failed to fetch user by email")
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def create_user(email: str, plain_password: str):
    """Insert a new user row and return a dict for the user."""
    conn = get_db_connection()
    if conn is None:
        raise RuntimeError("Database unavailable")

    user_id = str(uuid.uuid4())
    password_hash = hash_password(plain_password)
    plan = "free"

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (id, email, password_hash, plan, is_active, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, now(), now())
                    """,
                    (user_id, email.lower(), password_hash, plan, True),
                )
        return {
            "id": user_id,
            "email": email.lower(),
            "password_hash": password_hash,
            "plan": plan,
            "is_active": True,
        }
    except Exception:
        logging.exception("Failed to create user")
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass

# ---------------------------------------------------------
# Auth / JWT configuration
# ---------------------------------------------------------
SECRET_KEY = os.getenv("MP_JWT_SECRET", "change-me-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days
SUPERUSER_EMAIL = os.getenv("MP_SUPERUSER_EMAIL")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# auto_error=False so missing/invalid token just means "guest"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login", auto_error=False)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception:
        return False
def is_superuser(user: Optional[Dict[str, Any]]) -> bool:
    if not user:
        return False
    if SUPERUSER_EMAIL and user.get("email"):
        if user["email"].lower() == SUPERUSER_EMAIL.lower():
            return True
    # also allow a special plan label as a backup
    return user.get("plan") == "admin"


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
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
                "id": row[0],
                "email": row[1],
                "password_hash": row[2],
                "plan": row[3],
                "is_active": row[4],
            }
    except Exception:
        logging.exception("Failed to load user by email")
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, email, password_hash, plan, is_active
                FROM users
                WHERE id = %s
                """,
                (user_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "id": row[0],
                "email": row[1],
                "password_hash": row[2],
                "plan": row[3],
                "is_active": row[4],
            }
    except Exception:
        logging.exception("Failed to load user by id")
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def create_user(email: str, password: str) -> Dict[str, Any]:
    conn = get_db_connection()
    if not conn:
        raise RuntimeError("Database not available")

    user_id = f"user-{int(datetime.utcnow().timestamp() * 1000)}"
    hashed = get_password_hash(password)
    now = datetime.utcnow()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (id, email, password_hash, plan, is_active, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id, email, plan, is_active
                """,
                (user_id, email, hashed, "free", True, now, now),
            )
            row = cur.fetchone()
        conn.commit()
        return {
            "id": row[0],
            "email": row[1],
            "plan": row[2],
            "is_active": row[3],
        }
    except psycopg2.Error as exc:
        logging.exception("Failed to create user")
        # most likely duplicate email
        raise exc
    finally:
        try:
            conn.close()
        except Exception:
            pass
def get_current_user_optional(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[Dict[str, Any]]:
    """
    Decode JWT and return the user dict, or None if:
    - no token
    - token invalid/expired
    - user not found / inactive

    This keeps /analyze usable in guest mode.
    """
    if not token:
        return None

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: Optional[str] = payload.get("sub")
        if not user_id:
            return None
    except jwt.PyJWTError:
        return None

    user = get_user_by_id(user_id)
    if not user or not user.get("is_active"):
        return None
    return user


def log_blockage_to_db(
    *,
    mode: str,
    source_url: str | None,
    source_label: str | None,
    error_category: str,
    error_detail: str,
    http_status: int | None = None,
    user_id: str | None = None,
):
    """
    Best-effort logging of blocked / failed fetches.
    Fails silently if DATABASE_URL is not set or the insert fails.
    """
    try:
        conn = get_db_connection()
        if not conn:
            return
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO source_blockages
                    (mode, source_url, source_label, error_category, error_detail, http_status, user_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        mode,
                        source_url,
                        source_label,
                        error_category,
                        error_detail[:2000],
                        http_status,
                        user_id,
                    ),
                )
    except Exception:
        logging.exception("Failed to log blockage to Postgres")

def get_db_connection():
    """
    Open a new DB connection using pg8000 and the DATABASE_URL from Railway.
    """
    if not DATABASE_URL:
        return None

    url = urlparse(DATABASE_URL)
    # Example: postgresql://user:pass@host:5432/dbname
    username = url.username
    password = url.password
    host = url.hostname
    port = url.port or 5432
    database = url.path.lstrip("/") or None

    return pg8000.connect(
        user=username,
        password=password,
        host=host,
        port=port,
        database=database,
    )


def save_report_to_db(
    report_id: str,
    mode: str,
    depth: str,
    source_url: str | None,
    source_label: str | None,
    cfr_html: str,
    social_html: str | None,
) -> None:
    def get_user_by_email(email: str):
        """Fetch a user row as a dict by email, or None."""
        conn = get_db_connection()
        if conn is None:
            return None

        try:
            with conn:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    cur.execute(
                        """
                        SELECT id, email, password_hash, plan, is_active
                        FROM users
                        WHERE email = %s
                        """,
                        (email.lower(),),
                    )
                    row = cur.fetchone()
                    if not row:
                        return None
                    return {
                        "id": row["id"],
                        "email": row["email"],
                        "password_hash": row["password_hash"],
                        "plan": row.get("plan") or "free",
                        "is_active": row.get("is_active", True),
                    }
        except Exception:
            logging.exception("Failed to fetch user by email")
            return None
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def create_user(email: str, plain_password: str):
        """Insert a new user row and return a dict for the user."""
        conn = get_db_connection()
        if conn is None:
            raise RuntimeError("Database unavailable")

        user_id = str(uuid.uuid4())
        password_hash = hash_password(plain_password)
        plan = "free"

        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO users (id, email, password_hash, plan, is_active, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, now(), now())
                        """,
                        (user_id, email.lower(), password_hash, plan, True),
                    )
            return {
                "id": user_id,
                "email": email.lower(),
                "password_hash": password_hash,
                "plan": plan,
                "is_active": True,
            }
        except Exception:
            logging.exception("Failed to create user")
            raise
        finally:
            try:
                conn.close()
            except Exception:
                pass

    """
    Insert or update a report row in the `reports` table.
    Safe to no-op if DATABASE_URL is not configured.
    """
    conn = get_db_connection()
    if conn is None:
        return

    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO reports (
                id, mode, depth, source_url, source_label,
                cfr_html, social_html
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE
            SET
                mode = EXCLUDED.mode,
                depth = EXCLUDED.depth,
                source_url = EXCLUDED.source_url,
                source_label = EXCLUDED.source_label,
                cfr_html = EXCLUDED.cfr_html,
                social_html = EXCLUDED.social_html
            """,
            (
                report_id,
                mode,
                depth,
                source_url,
                source_label,
                cfr_html,
                social_html,
            ),
        )
        conn.commit()
    except Exception:
        logging.exception("Failed to save report to Postgres")
    finally:
        try:
            conn.close()
        except Exception:
            pass
def log_usage(
    user_id: str | None,
    ip_hash: str | None,
    source_type: str | None,
    depth: str | None,
    mode: str | None,
    report_id: str | None,
    success: bool = True,
    error_category: str | None = None,
    error_detail: str | None = None,
    tokens_used: int | None = None,
) -> None:
    def check_user_limits(
            user: Optional[Dict[str, Any]],
            ip_hash: Optional[str],
            depth: str,
            mode: str,
    ) -> None:
        """
        Enforce simple per-user / per-IP limits.
        - Superusers are exempt.
        - If the DB is unavailable, do NOT block: degrade gracefully.
        """

        # Superuser (you) is never throttled
        if is_superuser(user):
            return

        conn = get_db_connection()
        if conn is None:
            # No DB? Don't block. Just run in "best effort" mode.
            return

        try:
            # Normalize
            depth = (depth or "").lower()
            mode = (mode or "").lower()

            # Simple tier mapping
            if user:
                plan = (user.get("plan") or "free").lower()
            else:
                plan = "guest"

            # Limits (MVP – can tune later)
            # You can change these numbers easily.
            if plan == "guest":
                quick_limit_per_day = 3
                full_limit_per_day = 0  # guests can't do full
            elif plan == "free":
                quick_limit_per_day = 10
                full_limit_per_day = 3
            elif plan in ("pro", "enterprise"):
                # For now, no hard limits; we'll rely on token caps later
                return
            else:
                # Unknown plan, treat as free
                quick_limit_per_day = 10
                full_limit_per_day = 3

            now = datetime.utcnow()
            day_start = datetime(now.year, now.month, now.day)  # UTC midnight

            with conn:
                with conn.cursor() as cur:
                    # Decide how we identify the caller:
                    # - logged-in user → use user_id
                    # - anonymous guest → use ip_hash
                    if user and user.get("id"):
                        cur.execute(
                            """
                            SELECT depth, COUNT(*)
                            FROM usage_logs
                            WHERE user_id = %s
                              AND success = TRUE
                              AND created_at >= %s
                            GROUP BY depth
                            """,
                            (user["id"], day_start),
                        )
                    elif ip_hash:
                        cur.execute(
                            """
                            SELECT depth, COUNT(*)
                            FROM usage_logs
                            WHERE ip_hash = %s
                              AND success = TRUE
                              AND created_at >= %s
                            GROUP BY depth
                            """,
                            (ip_hash, day_start),
                        )
                    else:
                        # No user and no IP? Nothing to count against.
                        return

                    usage_by_depth: Dict[str, int] = {}
                    for row in cur.fetchall():
                        d, count = row[0], row[1]
                        if d:
                            usage_by_depth[d.lower()] = count

            quick_used = usage_by_depth.get("quick", 0)
            full_used = usage_by_depth.get("full", 0)

            if depth == "quick" and quick_limit_per_day >= 0 and quick_used >= quick_limit_per_day:
                raise HTTPException(
                    status_code=429,
                    detail="Quick report limit reached for today. Please try again tomorrow or upgrade your plan.",
                )

            if depth == "full" and full_limit_per_day >= 0 and full_used >= full_limit_per_day:
                raise HTTPException(
                    status_code=429,
                    detail="Full report limit reached for today. Please try again tomorrow or upgrade your plan.",
                )

        except HTTPException:
            # Re-raise rate-limit exceptions as-is
            raise
        except Exception:
            # Logging only; don't break analysis if limit check fails
            logging.exception("Rate limit check failed")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    """
    Best-effort logging into usage_logs.
    - Fails silently if DB is unavailable or the insert fails.
    - Safe to call on both success and failure.
    """
    conn = get_db_connection()
    if conn is None:
        return

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO usage_logs (
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
                        created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
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
        # Do not break the request just because logging failed
        logging.exception("Failed to log usage")


def load_report_from_db(report_id: str) -> tuple[str | None, str | None]:
    """
    Fetch (cfr_html, social_html) for a report_id from Postgres.
    Returns (None, None) if not found or DB is not configured.
    """
    conn = get_db_connection()
    if conn is None:
        return None, None

    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT cfr_html, social_html
            FROM reports
            WHERE id = %s
            """,
            (report_id,),
        )
        row = cur.fetchone()
        if not row:
            return None, None
        # pg8000 returns a tuple, not a dict
        return row[0], row[1]
    except Exception:
        logging.exception("Failed to load report from Postgres")
        return None, None
    finally:
        try:
            conn.close()
        except Exception:
            pass


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

def extract_report_id_from_html(html: str) -> str | None:
    """
    Find the report_id used inside the HTML, based on the built-in
    https://mind-pilot.ai/reports/{id} link.

    This lets us keep the same ID across:
    - the canonical snippet,
    - the CFR route /reports/{id},
    - and the social snapshot /social/{id}.
    """
    m = re.search(r"https://mind-pilot\.ai/reports/([a-z0-9\-]+)", html)
    if m:
        return m.group(1)
    return None

def build_social_share_page(report_html: str, report_id: str) -> str | None:
    """
    Build a standalone HTML page containing:

      - the same social card used at the top of the CFR
      - the canonical copy-ready social snippet

    This page is opened as /social/{report_id} from dev_index.html
    so you can screenshot the card and copy the caption.
    """
    try:
        # Grab the <style> block so the card looks identical
        style_match = re.search(r"<style.*?>.*?</style>", report_html, re.DOTALL)
        style_block = style_match.group(0) if style_match else ""

        # Social card block from the CFR
        card_match = re.search(
            r'<section class="card-sub social-card"[^>]*>.*?</section>',
            report_html,
            re.DOTALL,
        )
        card_html = card_match.group(0) if card_match else ""

        # Canonical snippet text (as preformatted HTML)
        snippet_match = re.search(
            r'<div class="social-snippet">\s*<pre class="pre-block">(.*?)</pre>',
            report_html,
            re.DOTALL,
        )
        snippet_raw = snippet_match.group(1) if snippet_match else ""
        snippet_text = html_lib.unescape(snippet_raw).strip()

        if not card_html and not snippet_text:
            return None

        full_report_url = f"https://mind-pilot.ai/reports/{html_lib.escape(report_id)}"

        social_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>MindPilot Social Snapshot – {html_lib.escape(report_id)}</title>
  {style_block}
  <style>
    body {{
      background: #0B1B33;
    }}
    .page.social-wrapper {{
      max-width: 760px;
      margin: 2rem auto 3rem;
    }}
    header {{
      margin-bottom: 1.25rem;
    }}
    .logo-title {{
      font-size: 1.2rem;
      font-weight: 600;
      color: #F7FAFC;
    }}
    .tagline {{
      font-size: 0.85rem;
      color: #CBD5F5;
      margin-top: 0.25rem;
    }}
    .cta-link {{
      display: inline-block;
      margin-top: 0.75rem;
      font-size: 0.8rem;
      color: #63B3ED;
      text-decoration: none;
    }}
  </style>
</head>
<body>
  <div class="page social-wrapper">
    <header>
      <div class="logo-title">MindPilot – Social Snapshot</div>
      <div class="tagline">
        Card image + copy-ready snippet linked to this Cognitive Flight Report.
      </div>
    </header>

    {card_html}

    <section class="card">
      <div class="card-title">Copy-Ready Social Snippet</div>
      <div class="card-body">
        <pre class="pre-block">{snippet_raw}</pre>
        <a class="cta-link" href="{full_report_url}" target="_blank" rel="noopener">
          View the full Cognitive Flight Report →
        </a>
      </div>
    </section>
  </div>
</body>
</html>"""
        return social_page

    except Exception:
        logging.error("Failed to build social share page", exc_info=True)
        return None
def strip_copy_ready_snippet_section(report_html: str) -> str:
    """
    Remove the 'Copy-Ready Social Snippet' collapsible section from the CFR.

    This keeps:
      - the social card at the top
      - all other report content
    """
    pattern = (
        r'\s*<section class="card-sub">\s*'
        r'<div class="collapsible-header" onclick="toggleSection\(\'social-snippets\'\)">'
        r'.*?</section>'
    )
    return re.sub(pattern, "", report_html, flags=re.DOTALL)
def insert_marketing_cta(report_html: str, report_id: str) -> str:
    """
    Insert:
      - a marketing CTA card immediately under the social card
      - a small CTA at the very bottom of the body

    Used only when the dev console asks for it.
    """
    # --- Top CTA, right after the social card ---
    top_cta_html = """
      <section class="card-sub">
        <div class="card-title">What is MindPilot?</div>
        <div class="card-body">
          <p class="card-body-text">
            This Cognitive Flight Report was generated automatically by MindPilot from a single piece of media.
            MindPilot is your co-pilot for critical thinking: it highlights reasoning quality, bias signals,
            and missing context in modern content — without telling you what to think.
          </p>
          <p class="card-body-text">
            If you create or rely on media to make decisions, MindPilot helps you see how arguments are structured
            so you can respond with more clarity, not more outrage.
          </p>
          <a href="https://mind-pilot.ai/" target="_blank" rel="noopener"
             style="
               display:inline-block;
               margin-top:0.75rem;
               padding:0.5rem 1.1rem;
               border-radius:999px;
               border:1px solid var(--sky-blue);
               font-size:0.8rem;
               color:#E2E8F0;
               text-decoration:none;
               background:rgba(15,23,42,0.8);
             ">
            Try MindPilot and run your own report
          </a>
        </div>
      </section>
    """

    def _inject_after_social_card(match: re.Match) -> str:
        return match.group(0) + top_cta_html

    html_with_top = re.sub(
        r'<section class="card-sub social-card"[^>]*>.*?</section>',
        _inject_after_social_card,
        report_html,
        count=1,
        flags=re.DOTALL,
    )

    # --- Bottom CTA, near the end of the body ---
    bottom_cta_html = """
    <section class="card-sub" style="max-width:960px;margin:2rem auto 0;">
      <div class="card-title">Ready to run your own Cognitive Flight Report?</div>
      <div class="card-body">
        <p class="card-body-text">
          Liked this breakdown? MindPilot can run the same style of reasoning diagnostic on your own content,
          research, or feeds — so you can see how ideas are structured before you act on them.
        </p>
        <p class="card-body-text" style="margin-top:0.4rem;">
          👉 <a href="https://mind-pilot.ai/" target="_blank" rel="noopener">
             Run your own Cognitive Flight Report with MindPilot
          </a>
        </p>
      </div>
    </section>
    """

    if "</body>" in html_with_top:
        return html_with_top.replace("</body>", bottom_cta_html + "\n</body>", 1)
    return html_with_top


# ---------------------------------------------------------
# Logging config (critical for debugging Railway crashes)
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
def check_user_limits(
    user: Optional[Dict[str, Any]],
    ip_hash: Optional[str],
    depth: str,
    mode: str,
) -> None:
    ...
class SignupRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


def is_superuser(user) -> bool:
    """Treat MP_SUPERUSER_EMAIL or plan='admin' as superuser."""
    if not user:
        return False
    email = (user.get("email") or "").lower()
    if SUPERUSER_EMAIL and email == SUPERUSER_EMAIL.lower():
        return True
    return (user.get("plan") or "").lower() == "admin"


def get_current_user_optional(request: Request):
    """
    Extract user from Authorization: Bearer <token>, or return None.
    Does NOT raise if token is invalid/expired; just returns None.
    """
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        return None

    token = auth_header.split(" ", 1)[1].strip()
    payload = decode_access_token(token)
    if not payload:
        return None

    email = payload.get("email")
    if not email:
        return None

    user = get_user_by_email(email)
    if not user:
        return None

    return user

app = FastAPI(title="MindPilot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later: ["https://mind-pilot.ai", "https://dev.mind-pilot.ai"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-MindPilot-Report-ID"],  # 👈 critical line
)
# ---------------------------------------------------------
# Auth routes
# ---------------------------------------------------------


@app.post("/signup")
async def signup(payload: SignupRequest):
    existing = get_user_by_email(payload.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email is already registered.",
        )

    try:
        user = create_user(payload.email, payload.password)
    except Exception:
        # Logged inside create_user
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create user.",
        )

    token = create_access_token(user["id"], user["email"], user.get("plan") or "free")
    return {
        "access_token": token,
        "token_type": "bearer",
        "plan": user.get("plan") or "free",
    }


@app.post("/login")
async def login(payload: LoginRequest):
    user = get_user_by_email(payload.email)
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect email or password.",
        )
    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive.",
        )

    token = create_access_token(user["id"], user["email"], user.get("plan") or "free")
    return {
        "access_token": token,
        "token_type": "bearer",
        "plan": user.get("plan") or "free",
    }


@app.get("/me")
async def me(current_user=Depends(get_current_user_optional)):
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
        )
    return {
        "id": current_user["id"],
        "email": current_user["email"],
        "plan": current_user.get("plan", "free"),
        "is_superuser": is_superuser(current_user),
    }

# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}
# ---------------------------------------------------------
# Auth: signup / login / me
# ---------------------------------------------------------
from pydantic import BaseModel


class SignupRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/signup")
async def signup(payload: SignupRequest):
    # If DB is down, don't crash the whole app – just fail signup gracefully.
    conn = get_db_connection()
    if not conn:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="User system temporarily unavailable. Please try again later.",
        )
    conn.close()

    existing = get_user_by_email(payload.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email is already registered.")

    try:
        user = create_user(payload.email, payload.password)
    except psycopg2.Error:
        raise HTTPException(status_code=400, detail="Could not create user.")

    token = create_access_token({"sub": user["id"]})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "email": user["email"],
            "plan": user["plan"],
        },
    }


@app.post("/login")
async def login(payload: LoginRequest):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="User system temporarily unavailable. Please try again later.",
        )
    conn.close()

    user = get_user_by_email(payload.email)
    if not user or not verify_password(payload.password, user["password_hash"]):
        # Do not reveal which part failed
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    if not user["is_active"]:
        raise HTTPException(status_code=403, detail="Account is inactive.")

    token = create_access_token({"sub": user["id"]})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "email": user["email"],
            "plan": user["plan"],
        },
    }


@app.get("/me")
async def me(current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return {
        "id": current_user["id"],
        "email": current_user["email"],
        "plan": current_user["plan"],
    }

# ---------------------------------------------------------
# Main Analysis Endpoint
# ---------------------------------------------------------
@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    mode: str = Form("text"),        # default so file-only posts don't 422
    input_value: str = Form(""),     # allow empty when using file
    depth: str = Form("full"),       # "quick" or "full"
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

    # Identify the user (if any) and hash the caller IP for usage_logs
    user_id_for_logging = current_user["id"] if current_user else None

    client_ip = request.client.host if request and request.client else None
    if client_ip:
        ip_hash = hashlib.sha256(client_ip.encode("utf-8")).hexdigest()[:64]
    else:
        ip_hash = None

    # Enforce per-user / per-IP limits (best-effort; DB failure → no block)
    check_user_limits(current_user, ip_hash, depth, mode)

    # Normalise depth
    depth = (depth or "full").lower().strip()
    if depth not in ("quick", "full"):
        depth = "full"

    # Dev console passes include_marketing_cta=1
    include_marketing_cta_flag = str(include_marketing_cta or "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    # Normalise optional article metadata
    article_title = (article_title or "").strip()
    article_url = (article_url or "").strip()
    if not article_url:
        article_url = None

    # For report_id slug + DB source_label
    source_label_for_id: str = ""
    html_report: str | None = None

    try:
        # -------------------------------------------------
        # 1) File upload → document mode
        # -------------------------------------------------
        if file is not None and file.filename:
            filename = file.filename
            source_label_for_id = filename
            source_type_for_logs = "document"

            logging.info(
                f"[MindPilot] Running {depth} document analysis for upload: {filename}"
            )

            raw_bytes = await file.read()

            if depth == "quick":
                html_report = run_quick_analysis_from_document(
                    file_bytes=raw_bytes,
                    filename=filename,
                    include_grok=False,
                )
            else:
                html_report = run_full_analysis_from_document(
                    file_bytes=raw_bytes,
                    filename=filename,
                )

            source_url_for_db = None
            source_label_for_db = filename

        # -------------------------------------------------
        # 2) YouTube mode
        # -------------------------------------------------
        elif mode.lower() == "youtube":
            youtube_url = input_value.strip()
            source_label_for_id = youtube_url
            source_type_for_logs = "youtube"

            logging.info(f"[MindPilot] Running {depth} YouTube analysis: {youtube_url}")

            if depth == "quick":
                html_report = run_quick_analysis_from_youtube(
                    youtube_url,
                    include_grok=False,
                )
            else:
                html_report = run_full_analysis_from_youtube(youtube_url)

            source_url_for_db = youtube_url
            source_label_for_db = youtube_url

        # -------------------------------------------------
        # 3) Plain text (pasted) mode
        # -------------------------------------------------
        elif mode.lower() == "text":
            # Build a human-friendly label for pasted text so you
            # can remember what this report was about later.
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
                f"[MindPilot] Running {depth} TEXT analysis "
                f"(label={source_label_for_id!r})."
            )

            if depth == "quick":
                html_report = run_quick_analysis_from_text(
                    raw_text=input_value,
                    source_label=source_label_for_id,
                    include_grok=False,
                )
            else:
                html_report = run_full_analysis_from_text(
                    raw_text=input_value,
                    source_label=source_label_for_id,
                )

            source_url_for_db = article_url
            source_label_for_db = source_label_for_id

        # -------------------------------------------------
        # 4) Article URL mode
        # -------------------------------------------------
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

            if depth == "quick":
                html_report = run_quick_analysis_from_article(
                    effective_url,
                    include_grok=False,
                )
            else:
                html_report = run_full_analysis_from_article(effective_url)

            source_url_for_db = effective_url
            source_label_for_db = source_label_for_id

        # -------------------------------------------------
        # 5) Unsupported mode
        # -------------------------------------------------
        else:
            return PlainTextResponse(f"Unsupported mode: {mode}", status_code=400)

        # ---------- Store the report (DB + in-memory) and return HTML ----------

        if not html_report:
            return PlainTextResponse("No report was generated.", status_code=500)

        report_id = generate_report_id(
            source_label=source_label_for_id,
            mode=mode,
            depth=depth,
        )

        social_html: str | None = None
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

        # Log successful usage (best-effort)
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

        # Article / web URL blocked by site

        logging.warning("ContentBlockedError in /analyze: %s", e, exc_info=True)

        # Figure out likely source_url for logging

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

            user_id=None,  # later when you have accounts

        )

        # User-facing message (plan B for blocked articles)

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

        # YouTube transcript not available

        logging.warning("TranscriptUnavailableError in /analyze: %s", e, exc_info=True)

        log_blockage_to_db(

            mode="youtube",

            source_url=input_value,

            source_label=source_label_for_id or None,

            error_category="transcript_unavailable",

            error_detail=str(e),

            http_status=None,

            user_id=None,

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

        # Generic catch-all

        logging.error("Error in /analyze endpoint", exc_info=True)

        # Log generic failures as well, so you can see patterns later

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

            user_id=None,

        )

        return PlainTextResponse(

            "MindPilot backend ERROR:\n"

            + str(e)

            + "\n\nIf this keeps happening, try copying the text or saving to PDF and uploading it.",

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

    Now prefers the Postgres `reports` table but falls back to the in-memory
    REPORT_STORE for older or in-flight reports.
    """
    # 1) Try Postgres
    cfr_html, _ = load_report_from_db(report_id)

    # 2) Fallback to in-memory store if DB misses
    if cfr_html is None:
        cfr_html = REPORT_STORE.get(report_id)

    if cfr_html is None:
        return PlainTextResponse("Report not found", status_code=404)

    return HTMLResponse(content=cfr_html, status_code=200)


@app.get("/social/{report_id}", response_class=HTMLResponse)
async def get_social_report(report_id: str):
    """
    Serve the social snapshot page (card + snippet) for a given report_id.
    """
    # 1) Try Postgres
    _, social_html = load_report_from_db(report_id)

    # 2) Fallback to in-memory store if needed
    if social_html is None:
        social_html = REPORT_STORE.get(f"{report_id}-social")

    if social_html is None:
        return PlainTextResponse("Social snapshot not found", status_code=404)

    return HTMLResponse(content=social_html, status_code=200)

