"""Local account, session, and password-reset persistence for the web console."""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat()


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _hash_password(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000)
    return f"pbkdf2_sha256$310000${salt.hex()}${digest.hex()}"


def _verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, rounds, salt, expected = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt), int(rounds)
        )
        return hmac.compare_digest(actual.hex(), expected)
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class AuthUser:
    id: str
    email: str
    name: str
    created_at: str

    def to_dict(self) -> Dict[str, str]:
        return {"id": self.id, "email": self.email, "name": self.name, "created_at": self.created_at}


class AuthStore:
    def __init__(self, path: str | Path = "runs/auth/users.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL,
                    password_hash TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL, FOREIGN KEY(user_id) REFERENCES users(id)
                );
                CREATE TABLE IF NOT EXISTS password_resets (
                    token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, expires_at TEXT NOT NULL,
                    used_at TEXT, created_at TEXT NOT NULL, FOREIGN KEY(user_id) REFERENCES users(id)
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_resets_user ON password_resets(user_id);
                """
            )

    def register(self, email: str, name: str, password: str) -> AuthUser:
        email = email.strip().lower()
        name = name.strip()
        if "@" not in email or len(email) > 254:
            raise ValueError("请输入有效邮箱")
        if not name:
            raise ValueError("请输入姓名")
        if len(password) < 8:
            raise ValueError("密码至少需要 8 个字符")
        user = AuthUser(f"user-{uuid.uuid4().hex[:16]}", email, name, _iso(_now()))
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    "INSERT INTO users(id,email,name,password_hash,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                    (user.id, user.email, user.name, _hash_password(password), user.created_at, user.created_at),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("该邮箱已注册") from exc
        return user

    def authenticate(self, email: str, password: str) -> Optional[AuthUser]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM users WHERE email=?", (email.strip().lower(),)).fetchone()
        if row is None or not _verify_password(password, row["password_hash"]):
            return None
        return self._user(row)

    def create_session(self, user_id: str, days: int = 7) -> str:
        token = secrets.token_urlsafe(32)
        now = _now()
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM sessions WHERE expires_at<=?", (_iso(now),))
            self._conn.execute(
                "INSERT INTO sessions(token_hash,user_id,expires_at,created_at) VALUES(?,?,?,?)",
                (_hash_token(token), user_id, _iso(now + timedelta(days=days)), _iso(now)),
            )
        return token

    def user_for_token(self, token: str) -> Optional[AuthUser]:
        if not token:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT u.* FROM sessions s JOIN users u ON u.id=s.user_id "
                "WHERE s.token_hash=? AND s.expires_at>?",
                (_hash_token(token), _iso(_now())),
            ).fetchone()
        return self._user(row) if row else None

    def first_user(self) -> Optional[AuthUser]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM users ORDER BY created_at ASC, id ASC LIMIT 1").fetchone()
        return self._user(row) if row else None

    def revoke_session(self, token: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM sessions WHERE token_hash=?", (_hash_token(token),))

    def create_password_reset(self, email: str, minutes: int = 30) -> Optional[str]:
        with self._lock:
            row = self._conn.execute("SELECT id FROM users WHERE email=?", (email.strip().lower(),)).fetchone()
        if row is None:
            return None
        # A short numeric code is easier to enter on the reset screen. Only
        # its SHA-256 digest is persisted, exactly as with session tokens.
        token = f"{secrets.randbelow(1_000_000):06d}"
        now = _now()
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM password_resets WHERE user_id=? OR expires_at<=?", (row["id"], _iso(now)))
            self._conn.execute(
                "INSERT INTO password_resets(token_hash,user_id,expires_at,created_at) VALUES(?,?,?,?)",
                (_hash_token(token), row["id"], _iso(now + timedelta(minutes=minutes)), _iso(now)),
            )
        return token

    def reset_password(self, token: str, password: str) -> bool:
        if len(password) < 8:
            raise ValueError("密码至少需要 8 个字符")
        now = _iso(_now())
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT user_id FROM password_resets WHERE token_hash=? AND used_at IS NULL AND expires_at>?",
                (_hash_token(token), now),
            ).fetchone()
            if row is None:
                return False
            self._conn.execute("UPDATE users SET password_hash=?,updated_at=? WHERE id=?", (_hash_password(password), now, row["user_id"]))
            self._conn.execute("UPDATE password_resets SET used_at=? WHERE token_hash=?", (now, _hash_token(token)))
            self._conn.execute("DELETE FROM sessions WHERE user_id=?", (row["user_id"],))
        return True

    @staticmethod
    def _user(row: Any) -> AuthUser:
        return AuthUser(row["id"], row["email"], row["name"], row["created_at"])
