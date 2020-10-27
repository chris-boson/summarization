import logging
import os

import boto3

logger = logging.getLogger("lightning")

s3 = boto3.resource(
    "s3"
)  # assumes credentials & configuration are handled outside python in .aws directory or environment variables


def parse_s3_path(s3_path: str):
    if not s3_path.startswith("s3://"):
        raise ValueError(f"'{s3_path}' doesn't start with 's3://'")
    tokens = s3_path.split("/")[2:]
    prefix = "/".join(list(tokens[1:]))
    return tokens[0], prefix


def download_from_s3(s3_path: str, local_location: str):
    """
    Download the contents of a folder directory
    Args:
        s3_path: Full path of folder: s3://bucket_name/path
        local_location: a relative or absolute directory path in the local file system
    """
    logger.info(f"Downloading all files from {s3_path} to {local_location}")
    bucket_name, s3_folder = parse_s3_path(s3_path)
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):

        stripped_blob_name = obj.key.replace(s3_folder, "").strip("/")
        destination_uri = os.path.join(local_location, stripped_blob_name)
        destination_folder = os.path.dirname(destination_uri)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        if not os.path.isdir(destination_uri):
            bucket.download_file(obj.key, destination_uri)
