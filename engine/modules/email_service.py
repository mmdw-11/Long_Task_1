"""Small SMTP adapter used for account emails.

Credentials are intentionally supplied only through environment variables so
they are never stored with users or returned by the API.
"""
from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage


class EmailDeliveryError(RuntimeError):
    """Raised when an account email could not be handed to the SMTP server."""


def smtp_is_configured() -> bool:
    return bool(os.environ.get("SMTP_HOST") and os.environ.get("SMTP_FROM"))


def send_password_reset_code(email: str, code: str, *, minutes: int = 30) -> None:
    """Deliver one password-reset verification code through the configured SMTP relay."""
    host = os.environ.get("SMTP_HOST", "").strip()
    sender = os.environ.get("SMTP_FROM", "").strip()
    if not host or not sender:
        raise EmailDeliveryError("邮件服务尚未配置")

    message = EmailMessage()
    message["Subject"] = "TIDE-Swarm 密码重置验证码"
    message["From"] = sender
    message["To"] = email
    message.set_content(
        f"你的 TIDE-Swarm 密码重置验证码是：{code}\n\n"
        f"验证码将在 {minutes} 分钟后失效。若不是你本人操作，请忽略此邮件。"
    )

    port = int(os.environ.get("SMTP_PORT", "587"))
    username = os.environ.get("SMTP_USERNAME", "").strip()
    password = os.environ.get("SMTP_PASSWORD", "")
    use_ssl = os.environ.get("SMTP_USE_SSL", "0") == "1"
    use_tls = os.environ.get("SMTP_USE_TLS", "1") == "1"
    try:
        client: smtplib.SMTP
        if use_ssl:
            client = smtplib.SMTP_SSL(host, port, timeout=15)
        else:
            client = smtplib.SMTP(host, port, timeout=15)
        with client:
            if use_tls and not use_ssl:
                client.starttls()
            if username:
                client.login(username, password)
            client.send_message(message)
    except (OSError, smtplib.SMTPException, ValueError) as exc:
        raise EmailDeliveryError("验证码邮件发送失败，请稍后重试") from exc
