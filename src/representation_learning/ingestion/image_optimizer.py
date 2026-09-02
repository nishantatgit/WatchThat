"""Training-image resizing and compression."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO

from PIL import Image, ImageOps


@dataclass(frozen=True, slots=True)
class OptimizedImage:
    content: bytes
    checksum: str
    content_type: str
    extension: str
    width: int
    height: int

    @property
    def size_bytes(self) -> int:
        return len(self.content)


class ImageOptimizer:
    def __init__(
        self,
        *,
        maximum_dimension: int = 1024,
        jpeg_quality: int = 85,
    ) -> None:
        if maximum_dimension <= 0:
            raise ValueError("maximum_dimension must be positive")

        if jpeg_quality < 1 or jpeg_quality > 95:
            raise ValueError("jpeg_quality must be between 1 and 95")

        self._maximum_dimension = maximum_dimension
        self._jpeg_quality = jpeg_quality

    def optimize(self, content: bytes) -> OptimizedImage:
        if not content:
            raise ValueError("Image content cannot be empty")

        with Image.open(BytesIO(content)) as source:
            image = ImageOps.exif_transpose(source)
            image.load()

        image = self._convert_to_rgb(image)

        image.thumbnail(
            (
                self._maximum_dimension,
                self._maximum_dimension,
            ),
            resample=Image.Resampling.LANCZOS,
        )

        output = BytesIO()

        image.save(
            output,
            format="JPEG",
            quality=self._jpeg_quality,
            optimize=True,
            progressive=True,
        )

        optimized_content = output.getvalue()

        return OptimizedImage(
            content=optimized_content,
            checksum=sha256(optimized_content).hexdigest(),
            content_type="image/jpeg",
            extension="jpg",
            width=image.width,
            height=image.height,
        )

    @staticmethod
    def _convert_to_rgb(image: Image.Image) -> Image.Image:
        if image.mode in {"RGBA", "LA"}:
            rgba_image = image.convert("RGBA")
            background = Image.new(
                "RGB",
                rgba_image.size,
                color="white",
            )
            background.paste(
                rgba_image,
                mask=rgba_image.getchannel("A"),
            )
            return background

        return image.convert("RGB")
