"""Email provider abstraction for sending notifications.

Provides three implementations:
1. ConsoleEmailProvider - Logs emails to stdout (dev/test)
2. SMTPEmailProvider - Sends real emails via SMTP (production)
3. ResendEmailProvider - Sends via Resend API (production, easier setup)

The active provider is selected based on configuration.
Priority: Resend > SMTP > Console
"""

import logging
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from app.config import settings

logger = logging.getLogger(__name__)

# Optional import for Resend
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


@dataclass
class EmailMessage:
    """Email message structure."""

    to: str
    subject: str
    body_text: str
    body_html: str | None = None


class EmailProvider(ABC):
    """Abstract base class for email providers."""

    @abstractmethod
    def send(self, message: EmailMessage) -> bool:
        """Send an email message.

        Args:
            message: The email message to send

        Returns:
            True if sent successfully, False otherwise
        """
        pass


class ConsoleEmailProvider(EmailProvider):
    """Email provider that logs to console.

    Used for development and testing. Does not send real emails.
    """

    def send(self, message: EmailMessage) -> bool:
        """Log email to console instead of sending."""
        logger.info(
            "📧 [ConsoleEmail] Would send email:\n"
            f"  To: {message.to}\n"
            f"  Subject: {message.subject}\n"
            f"  Body:\n{message.body_text}\n"
            f"  {'='*50}"
        )
        # Also print to stdout for test visibility
        print(
            f"\n{'='*50}\n"
            f"📧 EMAIL (Console Provider)\n"
            f"{'='*50}\n"
            f"To: {message.to}\n"
            f"Subject: {message.subject}\n"
            f"{'─'*50}\n"
            f"{message.body_text}\n"
            f"{'='*50}\n"
        )
        return True


class SMTPEmailProvider(EmailProvider):
    """Email provider that sends via SMTP.

    Used for production. Requires SMTP configuration in settings.
    """

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        from_address: str,
    ):
        """Initialize SMTP provider with credentials."""
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.from_address = from_address

    def send(self, message: EmailMessage) -> bool:
        """Send email via SMTP."""
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = message.subject
            msg["From"] = self.from_address
            msg["To"] = message.to

            # Attach plain text version
            part1 = MIMEText(message.body_text, "plain")
            msg.attach(part1)

            # Attach HTML version if provided
            if message.body_html:
                part2 = MIMEText(message.body_html, "html")
                msg.attach(part2)

            # Connect and send
            with smtplib.SMTP(self.host, self.port) as server:
                server.starttls()
                server.login(self.user, self.password)
                server.sendmail(self.from_address, message.to, msg.as_string())

            logger.info(f"Email sent successfully to {message.to}")
            return True

        except smtplib.SMTPException as e:
            logger.error(f"SMTP error sending email to {message.to}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending email to {message.to}: {e}")
            return False


class ResendEmailProvider(EmailProvider):
    """Email provider that sends via Resend API.

    Resend (https://resend.com) offers a simple REST API for sending emails.
    Free tier: 3,000 emails/month, 100 emails/day.
    
    Set RESEND_API_KEY environment variable to enable.
    """

    RESEND_API_URL = "https://api.resend.com/emails"

    def __init__(self, api_key: str, from_address: str):
        """Initialize Resend provider with API key."""
        self.api_key = api_key
        self.from_address = from_address

    def send(self, message: EmailMessage) -> bool:
        """Send email via Resend API."""
        if not HTTPX_AVAILABLE:
            logger.error("httpx is not installed, cannot use Resend provider")
            return False
            
        try:
            response = httpx.post(
                self.RESEND_API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "from": self.from_address,
                    "to": [message.to],
                    "subject": message.subject,
                    "text": message.body_text,
                    "html": message.body_html,
                },
                timeout=10.0,
            )
            
            if response.status_code == 200:
                logger.info(f"Email sent successfully via Resend to {message.to}")
                return True
            else:
                logger.error(
                    f"Resend API error: {response.status_code} - {response.text}"
                )
                return False

        except httpx.HTTPError as e:
            logger.error(f"HTTP error sending email via Resend: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending email via Resend: {e}")
            return False


def get_email_provider() -> EmailProvider:
    """Get the appropriate email provider based on configuration.

    Priority order:
    1. Resend API (if RESEND_API_KEY is set)
    2. SMTP (if SMTP_HOST, SMTP_USER, SMTP_PASS are set)
    3. Console (fallback for development)
    
    Returns:
        The configured email provider
    """
    # Try Resend first (easiest to set up)
    if settings.resend_enabled:
        logger.info("Using ResendEmailProvider")
        return ResendEmailProvider(
            api_key=settings.resend_api_key,  # type: ignore
            from_address=settings.smtp_from,
        )
    
    # Try SMTP
    if settings.smtp_enabled:
        logger.info("Using SMTPEmailProvider")
        return SMTPEmailProvider(
            host=settings.smtp_host,  # type: ignore (validated in smtp_enabled)
            port=settings.smtp_port,
            user=settings.smtp_user,  # type: ignore
            password=settings.smtp_pass,  # type: ignore
            from_address=settings.smtp_from,
        )
    
    # Fallback to console
    logger.info("No email provider configured, using ConsoleEmailProvider")
    return ConsoleEmailProvider()


# Module-level singleton instance - initialized lazily or via get_email_provider()
_email_provider: EmailProvider | None = None


def reset_email_provider() -> None:
    """Reset the email provider singleton (for testing).

    This resets to ConsoleEmailProvider to ensure tests always have a working provider.
    """
    global _email_provider
    _email_provider = ConsoleEmailProvider()


def set_email_provider(provider: EmailProvider) -> None:
    """Set a specific email provider (for testing)."""
    global _email_provider
    _email_provider = provider


def send_email(message: EmailMessage) -> bool:
    """Send an email using the configured provider.

    This is a convenience function that uses a singleton provider instance.

    Args:
        message: The email message to send

    Returns:
        True if sent successfully, False otherwise
    """
    global _email_provider
    if _email_provider is None:
        _email_provider = get_email_provider()
    return _email_provider.send(message)


# Initialize the email provider on module load to ensure it's never None
_email_provider = get_email_provider()

