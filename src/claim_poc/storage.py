"""
S3 storage utilities for the claim processing application.

This module provides a reusable abstraction around Amazon S3 operations used
by the application, including uploading files, uploading text and JSON data,
reading text objects, checking object existence, and listing objects by prefix.

AWS credentials are resolved through a configured boto3 session.
"""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

from claim_poc.config import config


class S3Storage:
    """Thin wrapper around the S3 operations used by the application."""

    def __init__(
        self,
        bucket: str,
        region: str | None = None,
        profile: str | None = None,
    ) -> None:
        """
        Create an S3 client for the given bucket.

        Args:
            bucket: Name of the S3 bucket to operate on.
            region: AWS region. Defaults to config.aws_region.
            profile: AWS named profile. Defaults to config.aws_profile.
        """

        self.bucket = bucket

        session = boto3.Session(
            profile_name=profile or config.aws_profile,
            region_name=region or config.aws_region,
        )

        self.client = session.client("s3")

    def upload_file(
        self,
        source: str | Path,
        key: str,
    ) -> str:
        source = Path(source)

        if not source.exists():
            raise FileNotFoundError(source)

        content_type, _ = mimetypes.guess_type(source.name)

        extra_args = {}

        if content_type:
            extra_args["ContentType"] = content_type

        self.client.upload_file(
            Filename=str(source),
            Bucket=self.bucket,
            Key=key,
            ExtraArgs=extra_args or None,
        )

        return f"s3://{self.bucket}/{key}"

    def upload_text(
        self,
        text: str,
        key: str,
        content_type: str = "text/plain",
    ) -> str:
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=text.encode("utf-8"),
            ContentType=content_type,
        )

        return f"s3://{self.bucket}/{key}"

    def upload_json(
        self,
        data: dict[str, Any],
        key: str,
    ) -> str:
        body = json.dumps(
            data,
            indent=2,
        )

        return self.upload_text(
            text=body,
            key=key,
            content_type="application/json",
        )

    def read_text(
        self,
        key: str,
    ) -> str:
        try:
            response = self.client.get_object(
                Bucket=self.bucket,
                Key=key,
            )

        except ClientError as error:
            code = error.response["Error"]["Code"]

            if code in {"NoSuchKey", "404"}:
                raise FileNotFoundError(
                    f"s3://{self.bucket}/{key}"
                ) from error

            raise

        return response["Body"].read().decode("utf-8")

    def object_exists(
        self,
        key: str,
    ) -> bool:
        try:
            self.client.head_object(
                Bucket=self.bucket,
                Key=key,
            )
            return True

        except ClientError as error:
            if error.response["Error"]["Code"] in {
                "404",
                "NoSuchKey",
            }:
                return False

            raise

    def list_objects(
        self,
        prefix: str = "",
    ) -> list[str]:
        paginator = self.client.get_paginator(
            "list_objects_v2"
        )

        objects = []

        for page in paginator.paginate(
            Bucket=self.bucket,
            Prefix=prefix,
        ):
            for obj in page.get("Contents", []):
                objects.append(obj["Key"])

        return objects