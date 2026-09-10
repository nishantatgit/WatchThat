"""Local visual gallery for inspecting accepted training images."""

import argparse
import html
import mimetypes
from functools import partial
from http.server import (
    BaseHTTPRequestHandler,
    ThreadingHTTPServer,
)
from typing import Any
from urllib.parse import (
    parse_qs,
    unquote,
    urlencode,
    urlsplit,
)

from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient

from representation_learning.utils.config import (
    load_infrastructure_config,
)


class DatasetGallery:
    def __init__(
        self,
        *,
        container: ContainerClient,
        page_size: int,
    ) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")

        self._container = container
        self._page_size = page_size

    def render_page(
        self,
        continuation_token: str | None,
    ) -> bytes:
        page_iterator = self._container.list_blobs(
            include=["metadata"],
            results_per_page=self._page_size,
        ).by_page(
            continuation_token=continuation_token,
        )

        blobs = list(next(page_iterator, []))
        next_token = page_iterator.continuation_token

        cards = "".join(
            self._render_card(blob)
            for blob in blobs
        )

        if not cards:
            cards = """
                <div class="empty">
                    No accepted images were found.
                </div>
            """

        navigation = self._render_navigation(next_token)

        document = f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta
        name="viewport"
        content="width=device-width, initial-scale=1"
    >
    <title>Accepted Dataset</title>
    <style>
        :root {{
            color-scheme: dark;
            font-family:
                Inter, ui-sans-serif, system-ui, -apple-system,
                BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #071018;
            color: #eef5f7;
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            min-height: 100vh;
            background:
                radial-gradient(
                    circle at top left,
                    #123745 0,
                    transparent 32rem
                ),
                #071018;
        }}

        header {{
            position: sticky;
            top: 0;
            z-index: 10;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 1rem 1.5rem;
            border-bottom: 1px solid #20333c;
            background: rgb(7 16 24 / 92%);
            backdrop-filter: blur(16px);
        }}

        h1 {{
            margin: 0;
            font-size: 1.25rem;
            letter-spacing: -0.02em;
        }}

        .summary {{
            color: #91a9b4;
            font-size: 0.875rem;
        }}

        main {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 1.5rem;
        }}

        .gallery {{
            display: grid;
            grid-template-columns:
                repeat(auto-fill, minmax(240px, 1fr));
            gap: 1rem;
        }}

        .card {{
            overflow: hidden;
            border: 1px solid #20333c;
            border-radius: 0.9rem;
            background: #0d1921;
            box-shadow: 0 12px 35px rgb(0 0 0 / 18%);
        }}

        .image-frame {{
            aspect-ratio: 4 / 3;
            overflow: hidden;
            background: #14242d;
        }}

        .image-frame img {{
            width: 100%;
            height: 100%;
            display: block;
            object-fit: cover;
            transition: transform 180ms ease;
        }}

        .card:hover img {{
            transform: scale(1.025);
        }}

        .details {{
            padding: 0.9rem;
        }}

        .title {{
            margin: 0 0 0.65rem;
            overflow: hidden;
            font-size: 0.95rem;
            font-weight: 650;
            line-height: 1.35;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .metadata {{
            display: grid;
            gap: 0.35rem;
            margin: 0;
            font-size: 0.78rem;
            color: #9cb0b9;
        }}

        .metadata div {{
            display: grid;
            grid-template-columns: 4.7rem 1fr;
            gap: 0.5rem;
        }}

        .metadata dt {{
            color: #66818d;
        }}

        .metadata dd {{
            min-width: 0;
            margin: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .metadata a {{
            color: #7dd3d8;
            text-decoration: none;
        }}

        .navigation {{
            display: flex;
            justify-content: flex-end;
            gap: 0.75rem;
            padding: 1.5rem 0 0.25rem;
        }}

        .button {{
            display: inline-flex;
            min-height: 2.75rem;
            align-items: center;
            justify-content: center;
            padding: 0.65rem 1rem;
            border: 1px solid #31525e;
            border-radius: 0.65rem;
            color: #eaf6f7;
            background: #15313a;
            font-size: 0.9rem;
            font-weight: 650;
            text-decoration: none;
        }}

        .button:hover {{
            background: #1d424d;
        }}

        .empty {{
            grid-column: 1 / -1;
            padding: 4rem 1rem;
            border: 1px dashed #31505c;
            border-radius: 0.9rem;
            color: #8ca5af;
            text-align: center;
        }}

        @media (max-width: 600px) {{
            header {{
                align-items: flex-start;
                flex-direction: column;
            }}

            main {{
                padding: 1rem;
            }}

            .gallery {{
                grid-template-columns:
                    repeat(auto-fill, minmax(170px, 1fr));
            }}
        }}
    </style>
</head>
<body>
    <header>
        <h1>Accepted Dataset</h1>
        <div class="summary">
            Showing {len(blobs)} images from accepted-images
        </div>
    </header>

    <main>
        <section class="gallery" aria-label="Accepted images">
            {cards}
        </section>

        {navigation}
    </main>
</body>
</html>
"""

        return document.encode("utf-8")

    def read_image(
        self,
        blob_name: str,
    ) -> tuple[bytes, str]:
        blob = self._container.get_blob_client(blob_name)
        properties = blob.get_blob_properties()
        content = blob.download_blob().readall()

        content_type = properties.content_settings.content_type

        if not content_type:
            content_type = (
                mimetypes.guess_type(blob_name)[0]
                or "application/octet-stream"
            )

        return content, content_type

    def _render_card(self, blob: Any) -> str:
        metadata = blob.metadata or {}

        title = (
            self._metadata_value(metadata, "title")
            or blob.name
        )
        category = (
            self._metadata_value(metadata, "source_category")
            or "Not recorded"
        )
        licence = (
            self._metadata_value(metadata, "license_name")
            or "Not recorded"
        )
        creator = (
            self._metadata_value(metadata, "creator")
            or "Not recorded"
        )
        source_page_url = self._metadata_value(
            metadata,
            "source_page_url",
        )

        image_path = "/image?" + urlencode(
            {"name": blob.name},
        )

        source_link = "Not recorded"

        if (
            source_page_url is not None
            and urlsplit(source_page_url).scheme == "https"
        ):
            escaped_url = html.escape(
                source_page_url,
                quote=True,
            )
            source_link = (
                f'<a href="{escaped_url}" '
                'target="_blank" rel="noreferrer">'
                "Open source"
                "</a>"
            )

        return f"""
<article class="card">
    <div class="image-frame">
        <img
            src="{html.escape(image_path, quote=True)}"
            alt="{html.escape(title, quote=True)}"
            loading="lazy"
        >
    </div>

    <div class="details">
        <h2 class="title" title="{html.escape(title, quote=True)}">
            {html.escape(title)}
        </h2>

        <dl class="metadata">
            <div>
                <dt>Category</dt>
                <dd title="{html.escape(category, quote=True)}">
                    {html.escape(category)}
                </dd>
            </div>
            <div>
                <dt>Licence</dt>
                <dd title="{html.escape(licence, quote=True)}">
                    {html.escape(licence)}
                </dd>
            </div>
            <div>
                <dt>Creator</dt>
                <dd title="{html.escape(creator, quote=True)}">
                    {html.escape(creator)}
                </dd>
            </div>
            <div>
                <dt>Size</dt>
                <dd>{self._format_size(blob.size)}</dd>
            </div>
            <div>
                <dt>Source</dt>
                <dd>{source_link}</dd>
            </div>
        </dl>
    </div>
</article>
"""

    @staticmethod
    def _metadata_value(
        metadata: dict[str, str],
        name: str,
    ) -> str | None:
        value = metadata.get(name)

        if not isinstance(value, str) or not value:
            return None

        return unquote(value)

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        if size_bytes >= 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"

        return f"{size_bytes / 1024:.1f} KB"

    @staticmethod
    def _render_navigation(
        next_token: str | None,
    ) -> str:
        next_button = ""

        if next_token is not None:
            next_url = "/?" + urlencode(
                {"token": next_token},
            )
            next_button = (
                f'<a class="button" '
                f'href="{html.escape(next_url, quote=True)}">'
                "Next page"
                "</a>"
            )

        return f"""
<nav class="navigation" aria-label="Gallery navigation">
    <a class="button" href="/">Refresh</a>
    {next_button}
</nav>
"""


class GalleryRequestHandler(BaseHTTPRequestHandler):
    def __init__(
        self,
        *arguments: Any,
        gallery: DatasetGallery,
        **keywords: Any,
    ) -> None:
        self._gallery = gallery
        super().__init__(*arguments, **keywords)

    def do_GET(self) -> None:
        request = urlsplit(self.path)

        try:
            if request.path == "/":
                parameters = parse_qs(request.query)
                token = parameters.get("token", [None])[0]
                content = self._gallery.render_page(token)

                self._send_response(
                    status=200,
                    content_type="text/html; charset=utf-8",
                    content=content,
                )
                return

            if request.path == "/image":
                parameters = parse_qs(request.query)
                blob_name = parameters.get("name", [None])[0]

                if not blob_name:
                    self._send_error(
                        status=400,
                        message="Missing blob name",
                    )
                    return

                content, content_type = (
                    self._gallery.read_image(blob_name)
                )

                self._send_response(
                    status=200,
                    content_type=content_type,
                    content=content,
                    cache_control="private, max-age=300",
                )
                return

            self._send_error(
                status=404,
                message="Page not found",
            )
        except ResourceNotFoundError:
            self._send_error(
                status=404,
                message="Image not found",
            )
        except Exception as error:
            self._send_error(
                status=500,
                message=str(error),
            )

    def _send_response(
        self,
        *,
        status: int,
        content_type: str,
        content: bytes,
        cache_control: str = "no-store",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header(
            "Content-Length",
            str(len(content)),
        )
        self.send_header("Cache-Control", cache_control)
        self.send_header(
            "Content-Security-Policy",
            (
                "default-src 'self'; "
                "img-src 'self'; "
                "style-src 'unsafe-inline'; "
                "object-src 'none'"
            ),
        )
        self.end_headers()
        self.wfile.write(content)

    def _send_error(
        self,
        *,
        status: int,
        message: str,
    ) -> None:
        content = (
            "<h1>Gallery error</h1>"
            f"<p>{html.escape(message)}</p>"
        ).encode("utf-8")

        self._send_response(
            status=status,
            content_type="text/html; charset=utf-8",
            content=content,
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local accepted-image gallery",
    )
    parser.add_argument(
        "--azure-config",
        default="configs/azure.yaml",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = load_infrastructure_config(
        arguments.azure_config,
    )

    container = ContainerClient(
        account_url=config.storage.account_url,
        container_name=config.storage.accepted_container,
        credential=DefaultAzureCredential(),
    )

    gallery = DatasetGallery(
        container=container,
        page_size=arguments.page_size,
    )

    handler = partial(
        GalleryRequestHandler,
        gallery=gallery,
    )

    server = ThreadingHTTPServer(
        (arguments.host, arguments.port),
        handler,
    )

    print(
        "Dataset gallery: "
        f"http://{arguments.host}:{arguments.port}"
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dataset gallery")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()