"""Standard error handling per systemprompt.md.

All errors return:
{
  "error": {
    "code": "ENUM",
    "message": "string",
    "field": "string|null"
  }
}

Error Codes: UNAUTHORIZED, FORBIDDEN, NOT_FOUND, VALIDATION_ERROR, CONFLICT, RATE_LIMITED, INTERNAL
"""

from enum import Enum
from typing import Any

from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ErrorCode(str, Enum):
    """Standard error codes per systemprompt.md."""

    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    CONFLICT = "CONFLICT"
    RATE_LIMITED = "RATE_LIMITED"
    INTERNAL = "INTERNAL"


class ErrorDetail(BaseModel):
    """Error detail schema."""

    code: ErrorCode
    message: str
    field: str | None = None


class ErrorResponse(BaseModel):
    """Standard error response schema."""

    error: ErrorDetail


def error_response(
    code: ErrorCode,
    message: str,
    field: str | None = None,
    status_code: int = status.HTTP_400_BAD_REQUEST,
) -> JSONResponse:
    """Create a standard error response."""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code.value, "message": message, "field": field}},
    )


class AppException(HTTPException):
    """Application exception with standard error format."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        field: str | None = None,
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ) -> None:
        self.code = code
        self.message = message
        self.field = field
        detail: dict[str, Any] = {"code": code.value, "message": message, "field": field}
        super().__init__(status_code=status_code, detail=detail)


class UnauthorizedException(AppException):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication required") -> None:
        super().__init__(
            code=ErrorCode.UNAUTHORIZED,
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
        )


class ForbiddenException(AppException):
    """Raised when user lacks permission."""

    def __init__(self, message: str = "Permission denied") -> None:
        super().__init__(
            code=ErrorCode.FORBIDDEN,
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
        )


class NotFoundException(AppException):
    """Raised when resource not found."""

    def __init__(self, message: str = "Resource not found") -> None:
        super().__init__(
            code=ErrorCode.NOT_FOUND,
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
        )


class ConflictException(AppException):
    """Raised when there's a conflict (e.g., duplicate email)."""

    def __init__(self, message: str = "Resource already exists", field: str | None = None) -> None:
        super().__init__(
            code=ErrorCode.CONFLICT,
            message=message,
            field=field,
            status_code=status.HTTP_409_CONFLICT,
        )


class ValidationException(AppException):
    """Raised for validation errors."""

    def __init__(self, message: str, field: str | None = None) -> None:
        super().__init__(
            code=ErrorCode.VALIDATION_ERROR,
            message=message,
            field=field,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )


class DemoReadOnlyException(ForbiddenException):
    """Raised when demo user attempts write operation."""

    def __init__(self) -> None:
        super().__init__(message="Demo accounts are read-only.")

