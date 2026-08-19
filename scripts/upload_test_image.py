import argparse
import mimetypes
from pathlib import Path

from azure.storage.blob import BlobClient, ContentSettings

from representation_learning.ingestion.user_upload import (
    DirectUploadService,
)
from representation_learning.utils.config import (
    load_infrastructure_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload an image directly to Azure Blob Storage"
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to the image on the local computer",
    )
    arguments = parser.parse_args()

    image_path = arguments.image_path.expanduser().resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    content_type, _ = mimetypes.guess_type(image_path.name)

    supported_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if content_type not in supported_types:
        raise ValueError(f"Unsupported image type: {content_type}")

    config = load_infrastructure_config()

    upload_service = DirectUploadService(
        account_url=config.storage.account_url,
        raw_container=config.storage.raw_container,
    )

    grant = upload_service.create_upload_grant(image_path.name)

    blob_client = BlobClient.from_blob_url(grant.upload_url)

    with image_path.open("rb") as image_file:
        blob_client.upload_blob(
            image_file,
            overwrite=False,
            content_settings=ContentSettings(
                content_type=content_type,
            ),
        )

    print(f"Image ID: {grant.image_id}")
    print(f"Raw blob: {grant.blob_uri}")


if __name__ == "__main__":
    main()
