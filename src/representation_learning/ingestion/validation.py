"""Image format, dimensions, integrity and policy validation."""
# src/representation_learning/ingestion/validation.py

from dataclasses import dataclass
from io import BytesIO

from PIL import Image, UnidentifiedImageError

_FORMAT_TO_CONTENT_TYPE = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


@dataclass(frozen=True, slots=True)
class ImageValidationResult:
    is_valid: bool
    reason: str | None = None
    content_type: str | None = None
    width: int | None = None
    height: int | None = None


class ImageValidator:
    def __init__(
        self,
        *,
        max_size_bytes: int = 10 * 1024 * 1024,
        min_width: int = 32,
        min_height: int = 32,
        allowed_formats: frozenset[str] | None = None,
    ) -> None:
        self.max_size_bytes = max_size_bytes
        self.min_width = min_width
        self.min_height = min_height
        self.allowed_formats = allowed_formats or frozenset(_FORMAT_TO_CONTENT_TYPE)

    def validate(self, image_bytes: bytes) -> ImageValidationResult:
        if not image_bytes:
            return ImageValidationResult(
                is_valid=False,
                reason="Image is empty",
            )

        if len(image_bytes) > self.max_size_bytes:
            return ImageValidationResult(
                is_valid=False,
                reason=(
                    f"Image size {len(image_bytes)} bytes exceeds "
                    f"the maximum of {self.max_size_bytes} bytes"
                ),
            )

        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image_format = image.format
                width, height = image.size
                image.verify()
        except (UnidentifiedImageError, OSError, SyntaxError) as error:
            return ImageValidationResult(
                is_valid=False,
                reason=f"Invalid or corrupted image: {error}",
            )

        if image_format not in self.allowed_formats:
            return ImageValidationResult(
                is_valid=False,
                reason=f"Unsupported image format: {image_format}",
            )

        if width < self.min_width or height < self.min_height:
            return ImageValidationResult(
                is_valid=False,
                reason=(
                    f"Image dimensions {width}x{height} are below "
                    f"the minimum {self.min_width}x{self.min_height}"
                ),
            )

        return ImageValidationResult(
            is_valid=True,
            content_type=_FORMAT_TO_CONTENT_TYPE[image_format],
            width=width,
            height=height,
        )
