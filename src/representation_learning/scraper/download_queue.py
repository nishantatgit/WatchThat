"""Service Bus messaging for scraper image-download requests."""

import json
from hashlib import sha256
from typing import Any

from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from azure.servicebus import (
    ServiceBusClient,
    ServiceBusMessage,
    ServiceBusReceivedMessage,
)

from representation_learning.scraper.crawler import (
    ScrapedImageCandidate,
)


class ImageDownloadQueuePublisher:
    def __init__(
        self,
        *,
        fully_qualified_namespace: str,
        queue_name: str,
        credential: TokenCredential | None = None,
    ) -> None:
        self._client = ServiceBusClient(
            fully_qualified_namespace=fully_qualified_namespace,
            credential=credential or DefaultAzureCredential(),
        )
        self._sender = self._client.get_queue_sender(
            queue_name=queue_name,
        )

    def publish(
        self,
        candidate: ScrapedImageCandidate,
    ) -> str:
        message_id = sha256(
            candidate.image_url.encode("utf-8"),
        ).hexdigest()

        payload = {
            "schema_version": 1,
            "image_url": candidate.image_url,
            "source_page_url": candidate.source_page_url,
            "license_name": candidate.license_name,
            "creator": candidate.creator,
            "title": candidate.title,
            "source_category": candidate.source_category,
        }

        message = ServiceBusMessage(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            message_id=message_id,
            content_type="application/json",
        )

        self._sender.send_messages(message)

        return message_id

    def close(self) -> None:
        self._sender.close()
        self._client.close()


def decode_download_message(
    message: ServiceBusReceivedMessage,
) -> ScrapedImageCandidate:
    body = b"".join(bytes(part) for part in message.body)

    payload = json.loads(body)

    if not isinstance(payload, dict):
        raise TypeError("Image-download message must contain an object")

    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported image-download message schema")

    return ScrapedImageCandidate(
        image_url=_required_string(payload, "image_url"),
        source_page_url=_required_string(
            payload,
            "source_page_url",
        ),
        license_name=_optional_string(
            payload,
            "license_name",
        ),
        creator=_optional_string(
            payload,
            "creator",
        ),
        title=_optional_string(
            payload,
            "title",
        ),
        source_category=_optional_string(
            payload,
            "source_category",
        ),
    )


def _required_string(
    payload: dict[str, Any],
    name: str,
) -> str:
    value = payload.get(name)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid message field: {name}")

    return value


def _optional_string(
    payload: dict[str, Any],
    name: str,
) -> str | None:
    value = payload.get(name)

    if value is None:
        return None

    if not isinstance(value, str):
        raise TypeError(f"Message field must be a string: {name}")

    return value
