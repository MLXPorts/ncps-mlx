"""Helper routines for downloading packaged datasets (Apache-2.0 source)."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


def download_and_unzip(url: str, extract_to: str | Path = ".") -> None:
    """Download a ZIP archive from ``url`` and extract it to ``extract_to``."""

    response = urlopen(url)
    with ZipFile(BytesIO(response.read())) as archive:
        archive.extractall(path=extract_to)
