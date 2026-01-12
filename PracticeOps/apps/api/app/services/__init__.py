"""Service layer modules."""

from app.services.email import (
    ConsoleEmailProvider,
    EmailMessage,
    EmailProvider,
    SMTPEmailProvider,
    get_email_provider,
    send_email,
)
from app.services.scheduler import (
    JOB_REGISTRY,
    run_job_by_name,
    scheduler_lifespan,
    setup_scheduler,
)

__all__ = [
    # Email
    "EmailProvider",
    "EmailMessage",
    "ConsoleEmailProvider",
    "SMTPEmailProvider",
    "get_email_provider",
    "send_email",
    # Scheduler
    "scheduler_lifespan",
    "setup_scheduler",
    "run_job_by_name",
    "JOB_REGISTRY",
]

