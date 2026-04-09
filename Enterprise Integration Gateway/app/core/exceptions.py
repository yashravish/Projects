class IntegrationGatewayError(Exception):
    """Base exception for all application errors."""


class IntegrationError(IntegrationGatewayError):
    """Raised when an external integration call fails."""

    def __init__(self, source: str, message: str) -> None:
        self.source = source
        super().__init__(f"[{source}] {message}")


class TransformationError(IntegrationGatewayError):
    """Raised when payload transformation fails."""

    def __init__(self, record_type: str, external_id: str | None, message: str) -> None:
        self.record_type = record_type
        self.external_id = external_id
        super().__init__(f"Transformation failed for {record_type}[{external_id}]: {message}")


class RecordNotFoundError(IntegrationGatewayError):
    """Raised when a requested resource does not exist in the database."""

    def __init__(self, resource: str, identifier: str | int) -> None:
        super().__init__(f"{resource} not found: {identifier}")


class RetryExhaustedError(IntegrationGatewayError):
    """Raised when maximum retry attempts have been exceeded."""

    def __init__(self, record_id: int, retry_count: int) -> None:
        super().__init__(
            f"Failed record {record_id} has already been retried {retry_count} times"
        )
