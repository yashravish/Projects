"""Sanity that test doubles satisfy the narrow boto3 Protocols we type against."""
from __future__ import annotations

from typing import cast

from app.ml.boto_typing import Boto3Module, SageMakerRuntimeClient


def test_fake_boto3_module_satisfies_protocol() -> None:
    class _Client:
        def invoke_endpoint(
            self,
            *,
            EndpointName: str,
            ContentType: str,
            Accept: str,
            Body: bytes,
        ) -> dict[str, object]:
            return {"Body": object()}

    class _Mod:
        def client(
            self, name: str, *, region_name: str | None = None
        ) -> object:
            assert name == "sagemaker-runtime"
            return _Client()

    m: Boto3Module = _Mod()
    raw = m.client("sagemaker-runtime", region_name="us-east-1")
    c = cast(SageMakerRuntimeClient, raw)
    c.invoke_endpoint(
        EndpointName="e",
        ContentType="application/json",
        Accept="application/json",
        Body=b"{}",
    )
