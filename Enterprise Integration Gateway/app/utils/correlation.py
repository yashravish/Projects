import uuid


def new_correlation_id() -> str:
    """Generate a new UUID4-based correlation ID for a sync job."""
    return str(uuid.uuid4())
