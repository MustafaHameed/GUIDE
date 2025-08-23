import hashlib
import logging
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

OULAD_URL = "https://analyse.kmi.open.ac.uk/open-dataset/download"
DEST_DIR = Path("data/oulad/raw")
ZIP_NAME = "oulad.zip"
CORE_TABLES = {
    "studentInfo.csv",
    "studentVle.csv",
    "vle.csv",
    "studentRegistration.csv",
    "studentAssessment.csv",
    "assessments.csv",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def md5(path: Path) -> str:
    """Return the MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_file(url: str, path: Path) -> None:
    """Download a file with retry logic and progress bar."""
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    with session.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as pbar:
            with path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))


def extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract a ZIP archive to the destination directory."""
    with ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)


def summarize_tables(dest: Path) -> None:
    """Log MD5 and basic statistics for core OULAD tables."""
    for name in sorted(CORE_TABLES):
        csv_path = dest / name
        if not csv_path.exists():
            logging.warning("%s not found", name)
            continue
        checksum = md5(csv_path)
        df = pd.read_csv(csv_path)
        logging.info(
            "%s: md5=%s rows=%d cols=%d", name, checksum, df.shape[0], df.shape[1]
        )


def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DEST_DIR / ZIP_NAME

    logging.info("Downloading OULAD dataset...")
    download_file(OULAD_URL, zip_path)
    logging.info("Zip MD5: %s", md5(zip_path))

    logging.info("Extracting files...")
    extract_zip(zip_path, DEST_DIR)

    logging.info("Extracted CSV files:")
    for p in sorted(DEST_DIR.glob("*.csv")):
        size_mb = p.stat().st_size / 1_000_000
        logging.info("%s - %.2f MB", p.name, size_mb)

    logging.info("Summary statistics for core tables:")
    summarize_tables(DEST_DIR)


if __name__ == "__main__":
    main()
