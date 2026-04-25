"""Narrow types for the boto3 surface area we use.

`boto3` and `botocore` are in mypy's `ignore_missing_imports` list; these
`Protocol` definitions let training and inference code type *our* calls
without treating the injected module or clients as untyped `Any` blobs.
"""
from __future__ import annotations

from typing import IO, Any, Mapping, Protocol, runtime_checkable


class Boto3Module(Protocol):
    """A module (or test stub) that exposes `.client(...)`."""

    def client(
        self, service_name: str, *, region_name: str | None = None
    ) -> object: ...


@runtime_checkable
class SageMakerTrainingClient(Protocol):
    def create_training_job(self, **kwargs: Any) -> object: ...
    def describe_training_job(self, **kwargs: Any) -> Mapping[str, Any]: ...


@runtime_checkable
class S3GetClient(Protocol):
    def download_fileobj(
        self, Bucket: str, Key: str, Fileobj: IO[bytes]
    ) -> None: ...


@runtime_checkable
class SageMakerRuntimeClient(Protocol):
    def invoke_endpoint(
        self,
        *,
        EndpointName: str,
        ContentType: str,
        Accept: str,
        Body: bytes,
    ) -> Mapping[str, Any]: ...


@runtime_checkable
class SageMakerModelPackageGroupClient(Protocol):
    def describe_model_package_group(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def create_model_package_group(self, **kwargs: Any) -> object: ...
    def create_model_package(self, **kwargs: Any) -> object: ...


__all__ = [
    "Boto3Module",
    "S3GetClient",
    "SageMakerModelPackageGroupClient",
    "SageMakerRuntimeClient",
    "SageMakerTrainingClient",
]
