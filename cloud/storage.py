"""Create-once R2 objects with content verification independent of ETags."""

from contextlib import closing
import hashlib
import os
from pathlib import PurePosixPath
from urllib.parse import urlsplit

import boto3
from botocore.exceptions import ClientError

PREFIX = "novo-nordisk"
DATASET = f"{PREFIX}/datasets/ucsd-bmets/v1"


def connect():
    endpoint = os.environ["R2_ENDPOINT_URL"]
    url = urlsplit(endpoint)
    if (
        url.scheme != "https" or not url.hostname or url.username or url.password
        or not url.hostname.endswith(".r2.cloudflarestorage.com")
        or url.path not in ("", "/") or url.query or url.fragment
    ):
        raise ValueError("Use the direct HTTPS R2 S3 endpoint.")
    client = boto3.client(
        "s3", endpoint_url=endpoint, region_name="auto",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    )
    return client, os.environ["R2_BUCKET"]


def key_under(prefix, name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Invalid relative object path: {name}")
    return f"{prefix}/{path.as_posix()}"


def verify(client, bucket, key, expected, destination=None):
    response = client.get_object(Bucket=bucket, Key=key)
    sha = hashlib.sha256()
    with closing(response["Body"]) as stream:
        for chunk in stream.iter_chunks(chunk_size=1024 * 1024):
            sha.update(chunk)
            if destination is not None:
                destination.write(chunk)
    digest = sha.hexdigest()
    if digest != expected:
        raise ValueError(f"SHA-256 mismatch for {key}: {digest} != {expected}")
    return digest


def put_once(client, bucket, key, source):
    size = source.stat().st_size
    if size > 5_000_000_000:
        raise ValueError("Stage original NIfTI objects individually, within the single PUT limit.")
    with source.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        try:
            client.put_object(
                Bucket=bucket, Key=key, Body=stream, ContentLength=size,
                Metadata={"sha256": sha}, IfNoneMatch="*",
            )
        except ClientError as error:
            if error.response["ResponseMetadata"]["HTTPStatusCode"] != 412:
                raise
    verify(client, bucket, key, sha)
    return {"key": key, "sha256": sha, "bytes": size}
