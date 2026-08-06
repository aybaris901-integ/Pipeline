"""Stage 1 — turn every PDF in documents/ into text.

Two important details the dataset punishes you for getting wrong:

1. `pdftotext -layout` preserves the table columns in the KYC files. Without
   -layout the ownership percentages detach from the entity names.
2. At least one document per dataset is a **scan** (an image-only PDF produced
   by WeasyPrint). Text extraction returns ~0 characters and it silently drops
   out of the pipeline. In the public set that file is the KYC dossier of one
   borrower — losing it costs the whole related-party covenant. Any page whose
   extracted text is below `MIN_CHARS` is rasterised and sent to a vision model.
"""
from __future__ import annotations
import os
import subprocess
import glob
import json
import hashlib

MIN_CHARS = 200  # per page; below this we treat the page as a scan


def page_texts(pdf: str) -> list[str]:
    """Per-page text. Page granularity matters: several documents in this
    dataset are ordinary text PDFs with ONE page pasted in as an image — the
    EBITDA add-back table, the subsidiary security-coverage table. Document
    level checks pass and the page is silently lost."""
    import pdfplumber
    with pdfplumber.open(pdf) as doc:
        return [(p.extract_text() or "") for p in doc.pages]


def image_pages(pdf: str) -> list[int]:
    import pdfplumber
    out = []
    with pdfplumber.open(pdf) as doc:
        for i, p in enumerate(doc.pages):
            if len((p.extract_text() or "").strip()) < MIN_CHARS and p.images:
                out.append(i)
    return out


def _pdftotext(pdf: str) -> str:
    try:
        out = subprocess.run(
            ["pdftotext", "-layout", "-enc", "UTF-8", pdf, "-"],
            capture_output=True, timeout=120,
        )
        return out.stdout.decode("utf-8", "replace")
    except Exception:
        return ""


def _rasterise(pdf: str, out_dir: str, dpi: int = 200) -> list[str]:
    import pdfplumber
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    with pdfplumber.open(pdf) as doc:
        for i, page in enumerate(doc.pages):
            p = os.path.join(out_dir, f"page_{i:03d}.png")
            page.to_image(resolution=dpi).save(p)
            paths.append(p)
    return paths


OCR_PROMPT = (
    "Transcribe this scanned page verbatim into plain text. It is a Russian-language "
    "banking document. Preserve every number, percentage, account identifier "
    "(ACC-####) and organisation name exactly as printed. Render tables as "
    "'label: value' lines, one per row. Output only the transcription."
)


def ocr_page(png: str, llm) -> str:
    """Vision transcription. `llm` must expose .vision(prompt, image_path)."""
    return llm.vision(OCR_PROMPT, png)


def ingest_all(documents_dir: str, cache_dir: str, llm=None, vision_cache: str | None = None) -> dict[str, str]:
    """Return {doc_id: text}. Results are cached on disk by content hash."""
    os.makedirs(cache_dir, exist_ok=True)
    texts: dict[str, str] = {}
    for pdf in sorted(glob.glob(os.path.join(documents_dir, "*.pdf"))):
        doc_id = os.path.splitext(os.path.basename(pdf))[0]
        digest = hashlib.md5(open(pdf, "rb").read()).hexdigest()[:12]
        cache = os.path.join(cache_dir, f"{doc_id}.{digest}.txt")
        if os.path.exists(cache):
            texts[doc_id] = open(cache, encoding="utf-8").read()
            continue

        text = _pdftotext(pdf)
        blank = image_pages(pdf)
        if blank:
            seeded, missing = [], []
            for i in blank:
                path = os.path.join(vision_cache or "", f"{doc_id}.p{i}.txt")
                if vision_cache and os.path.exists(path):
                    seeded.append(open(path, encoding="utf-8").read())
                else:
                    missing.append(i)
            if seeded:
                text += "\n\n" + "\n\n".join(seeded)
            blank = missing
            if not blank:
                pass
            elif llm is None:
                text += "\n\n[[UNREAD_IMAGE_PAGES: " + ",".join(map(str, blank)) + "]]"
            else:
                pngs = _rasterise(pdf, os.path.join(cache_dir, "_img", doc_id))
                text += "\n\n" + "\n\n".join(
                    f"[[page {i} transcribed]]\n" + ocr_page(pngs[i], llm) for i in blank)

        open(cache, "w", encoding="utf-8").write(text)
        texts[doc_id] = text

    # non-PDF attachments (the dataset ships stray .csv / Thumbs.db files)
    for extra in sorted(glob.glob(os.path.join(documents_dir, "*.csv"))):
        doc_id = os.path.splitext(os.path.basename(extra))[0]
        texts[doc_id] = open(extra, encoding="utf-8", errors="replace").read()
    return texts


def normalise(text: str) -> str:
    """Collapse the letter-spaced headings WeasyPrint emits.

    Headings render as 'Д О ГО В О Р  БА Н КО В С КО ГО'. Any keyword match on
    raw text misses them, so every router/extractor works on this view instead.
    """
    import re
    out = []
    for line in text.split("\n"):
        stripped = line.strip()
        tokens = stripped.split()
        singles = sum(1 for t in tokens if len(t) == 1)
        if len(tokens) >= 4 and singles >= 0.6 * len(tokens):
            line = re.sub(r"(?<=\S) (?=\S)", "", stripped)
        out.append(line)
    return "\n".join(out)
