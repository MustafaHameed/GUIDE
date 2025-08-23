#!/usr/bin/env python3
"""
PDF Paper Renamer (Local, Publication-Year Strict; improved title extraction)

Renames PDF files in a folder to: "<YEAR> <TITLE>.pdf"
- Year is extracted from PDF content/metadata only (never file timestamps).
- Title is extracted with robust filters that skip journal headers and boilerplate.
- Dry-run by default; use --apply to rename.
- Writes an audit CSV.

Dependencies:  pip install pymupdf pypdf
"""

import argparse
import csv
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional, Tuple, List
from logging_config import setup_logging

# ---- YOUR DEFAULT FOLDER ----
logger = logging.getLogger(__name__)
DEFAULT_FOLDER = r"G:\My Drive\~01ResearchWork\recent_Articles"

# Prefer PyMuPDF (fitz) for speed; fallback to pypdf
HAVE_PYMUPDF = True
HAVE_PYPDF = True
try:
    import fitz  # PyMuPDF
except Exception:
    HAVE_PYMUPDF = False
try:
    from pypdf import PdfReader
except Exception:
    HAVE_PYPDF = False

INVALID_CHARS = r'<>:"/\\|?*'
INVALID_TABLE = str.maketrans({c: "-" for c in INVALID_CHARS})
SPACES_RE = re.compile(r"\s+")
YEAR_RE_GLOBAL = re.compile(r"\b(19\d{2}|20\d{2})\b")
PUB_CUES = re.compile(r"(published|©|\(c\)|copyright|ieee|acm|springer|elsevier|mdpi|proceedings|journal)", re.I)

# Tokens we do NOT want as a title (journal mastheads, metadata, boilerplate)
TITLE_BAD_TOKENS = re.compile(
    r"""(?ix)
    ^\s*(?:  # common section/journal tokens
      abstract|introduction|keywords?|highlights|nomenclature|
      journal|proceedings|conference|transactions|
      volume|vol\.?|issue|no\.?|pages?|pp\.?|
      doi|issn|isbn|arxiv|www\.|http|https|open\s+access|
      mdpi|elsevier|springer|wiley|ieee|acm|nature|science|
      editorial|short\s+note|communication|review\s+article|
      received|accepted|published|revised|available\s+online|
      international\s+journal|applied\s+sciences
    )\b
    """,
    re.IGNORECASE,
)

MONTHS = ("january","february","march","april","may","june",
          "july","august","september","october","november","december")

AFFILIATION_TOKENS = re.compile(r"(university|institute|department|school|faculty|college|laboratory|centre|center|email|@)", re.I)
AUTHOR_LINE = re.compile(r"^(?:[A-Z][a-z]+(?:[-' ][A-Z][a-z]+)*)(?:\s*,\s*[A-Z][a-z]+|(?:\s+[A-Z]\.){1,3})", re.U)

def clean_title(title: str) -> str:
    if not title:
        return ""
    title = unicodedata.normalize("NFKC", title)
    title = "".join(ch for ch in title if ch.isprintable())
    title = title.translate(INVALID_TABLE)
    title = SPACES_RE.sub(" ", title).strip(" .-_")
    return title[:180]

def _looks_like_title(s: str) -> bool:
    # Basic quality checks for a plausible paper title
    if not s: 
        return False
    if TITLE_BAD_TOKENS.search(s): 
        return False
    if any(m in s.lower() for m in MONTHS):  # often in headers
        # month in header frequently signals masthead; allow only if long and mixed case
        if len(s.split()) <= 6:
            return False
    words = s.split()
    if len(words) < 4:
        return False
    # must contain lowercase (avoid ALL-CAPS headers)
    if s == s.upper():
        return False
    # avoid pure affiliation/author/email lines
    if AFFILIATION_TOKENS.search(s):
        return False
    if AUTHOR_LINE.match(s) and len(words) <= 12:  # author list line
        return False
    # avoid lines ending with volume/issue/year patterns
    if re.search(r"\b(vol(?:ume)?\.?\s*\d+|issue\s*\d+)\b", s, re.I):
        return False
    return True

def _join_if_wrapped(curr: str, nxt: str) -> str:
    """Join two adjacent lines that likely form a wrapped title."""
    if not curr or not nxt:
        return curr or ""
    # Skip if next looks like authors/affiliation
    if not _looks_like_title(nxt):
        return curr
    # If current line ends without a period/colon and both are reasonably short, join.
    if not re.search(r"[.:;]$", curr.strip()) and len(curr.split()) < 18:
        joined = f"{curr.strip()} {nxt.strip()}"
        return joined
    return curr

def extract_title_from_text(text: str) -> str:
    """
    Robust title extraction:
    - Skip obvious mastheads (journal names, 'Article/Review', DOI blocks, 'Open Access', etc.)
    - Prefer the longest plausible line among the first ~80 lines.
    - If the candidate seems wrapped across two lines, join them.
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return ""

    # MDPI et al: often the very first line is "Article", "Review", "Communication" → skip it
    start = 0
    if start < len(lines) and re.match(r"^(article|review|communication)\b", lines[start], re.I):
        start += 1

    candidates: List[str] = []
    window = lines[start: start + 80]
    for i, raw in enumerate(window):
        s = SPACES_RE.sub(" ", raw).strip(" .-_")
        if not _looks_like_title(s):
            continue
        # consider possible wrap with the next line
        s2 = s
        if i + 1 < len(window):
            s2 = _join_if_wrapped(s, window[i+1])
        candidates.append(s2)

    if not candidates:
        return ""

    # choose the longest reasonable candidate (often the actual paper title)
    best = max(candidates, key=lambda z: len(z))
    # final sanitation
    best = re.sub(r"\s{2,}", " ", best).strip(" .-_")
    return best

def guess_from_filename(stem: str) -> Tuple[Optional[str], Optional[str]]:
    m = YEAR_RE_GLOBAL.search(stem)
    year = m.group(1) if m else None
    title = re.sub(r"[_\-\.]+", " ", stem)
    title = re.sub(r"^\(?\b(19\d{2}|20\d{2})\b\)?\s*[-–—]?\s*", "", title)
    title = clean_title(title.title())
    return year, title

def detect_publication_year(text: str, inspect_chars: int = 20000) -> Optional[str]:
    window = text[:inspect_chars]
    candidates = [int(m.group(1)) for m in YEAR_RE_GLOBAL.finditer(window)]
    candidates = [y for y in candidates if 1900 <= y <= 2100]
    if not candidates:
        return None
    m = re.search(rf"{PUB_CUES.pattern}[^\d]{{0,25}}(19\d{{2}}|20\d{{2}})", window, re.I)
    if m:
        return m.group(2)
    return str(min(candidates))

def extract_pdf_text_pymupdf(path: Path, pages: int = 3) -> str:
    txt = []
    with fitz.open(path) as doc:
        n = min(pages, len(doc))
        for i in range(n):
            txt.append(doc[i].get_text("text") or "")
    return "\n".join(txt)

def extract_pdf_text_pypdf(path: Path, pages: int = 3) -> str:
    reader = PdfReader(str(path))
    n = min(pages, len(reader.pages))
    txt = []
    for i in range(n):
        try:
            t = reader.pages[i].extract_text() or ""
        except Exception:
            t = ""
        txt.append(t)
    return "\n".join(txt)

def extract_pdf(path: Path) -> Tuple[Optional[str], Optional[str], List[str]]:
    notes: List[str] = []
    year, title = None, None
    text = ""

    # 1) PyMuPDF fast path
    if HAVE_PYMUPDF:
        try:
            text = extract_pdf_text_pymupdf(path, pages=3)
            t = extract_title_from_text(text)
            if t:
                title = t; notes.append("fitz.title")
            y = detect_publication_year(text)
            if y:
                year = y; notes.append("fitz.year")
        except Exception as e:
            notes.append(f"fitz.error:{e}")

    # 2) pypdf fallback (metadata + text)
    if (not year or not title) and HAVE_PYPDF:
        try:
            if not text:
                text = extract_pdf_text_pypdf(path, pages=3)
            reader = PdfReader(str(path))
            meta = getattr(reader, "metadata", {}) or {}
            # metadata title if it looks like a real title
            if not title:
                raw_title = (getattr(meta, "title", None) or meta.get("/Title") or "") .strip()
                if raw_title:
                    # reject obvious mastheads
                    if _looks_like_title(raw_title):
                        title = raw_title; notes.append("meta.title")
            # metadata year if present
            if not year:
                for key in ("/CreationDate", "/ModDate"):
                    val = meta.get(key)
                    if val:
                        m = YEAR_RE_GLOBAL.search(str(val))
                        if m:
                            year = m.group(1); notes.append(f"meta.{key}"); break
            # text fallback
            if not title:
                t = extract_title_from_text(text)
                if t:
                    title = t; notes.append("text.title")
            if not year:
                y = detect_publication_year(text)
                if y:
                    year = y; notes.append("text.year")
        except Exception as e:
            notes.append(f"pypdf.error:{e}")

    return year, clean_title(title) if title else None, notes or ["no_extractor"]

def construct_new_name(year: Optional[str], title: Optional[str]) -> Optional[str]:
    if not year or not title:
        return None
    return f"{year} {title}.pdf"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default=DEFAULT_FOLDER, help="Folder containing PDF files")
    ap.add_argument("--apply", action="store_true", help="Actually rename files (default: dry-run)")
    ap.add_argument("--csv", default="rename_audit_pdf.csv", help="Audit CSV filename to write")
    ap.add_argument("--max-pages", type=int, default=3, help="Pages to scan from the start (default: 3)")
    args = ap.parse_args()

    base = Path(args.folder)
    if not base.exists():
        logger.error("Folder not found: %s", base)
        sys.exit(2)

    rows = []
    count_total, count_renamed = 0, 0
    pdfs = [p for p in sorted(base.iterdir()) if p.is_file() and p.suffix.lower() == ".pdf"]

    for path in pdfs:
        count_total += 1
        # temporarily pass args.max-pages to extractors
        global extract_pdf_text_pymupdf, extract_pdf_text_pypdf
        y, t, n = extract_pdf(path)
        notes = list(n)

        # filename fallbacks (OK to read year from filename if embedded)
        if not y or not t:
            fy, ft = guess_from_filename(path.stem)
            if not y and fy:
                y = fy; notes.append("filename.year")
            if not t and ft:
                t = ft; notes.append("filename.title")

        new_name = construct_new_name(y, t)
        status = "unchanged"
        reason = ";".join(notes)

        if new_name and new_name != path.name:
            candidate = path.with_name(new_name)
            i = 2
            while candidate.exists():
                candidate = path.with_name(f"{Path(new_name).stem} ({i}).pdf")
                i += 1
            # dry-run vs apply
            if args.apply:
                try:
                    path.rename(candidate)
                    status = "renamed"; count_renamed += 1
                    rows.append([path.name, candidate.name, y or "", t or "", ".pdf", status, reason])
                    continue
                except Exception as e:
                    status = f"error:{e}"
            else:
                status = "dry-run.rename->" + candidate.name
        else:
            if not new_name:
                status = "skipped.missing_year_or_title"

        rows.append([path.name, new_name or "", y or "", t or "", ".pdf", status, reason])

    # Write audit
    csv_path = Path(args.csv)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["old_name", "new_name", "year", "title", "ext", "status", "notes"])
        w.writerows(rows)

    logger.info("Processed: %d files", count_total)
    logger.info(
        "Renamed:   %d files%s",
        count_renamed,
        " (dry-run)" if not args.apply else "",
    )
    logger.info("Audit CSV: %s", csv_path.resolve())

if __name__ == "__main__":
    setup_logging()
    main()
