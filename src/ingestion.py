"""
PDF → JSONL chunks
  - Docling for parsing (structure + tables)
  - Custom chunker: heading-aware, overlap only when needed, junk filtered
"""

import gc
import json
import re
import time
import hashlib
from pathlib import Path
from typing import Optional

import fitz
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice, AcceleratorOptions, ThreadedPdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline

# ── config ─────────────────────────────────────────────────────────────────────
PDF_DIR    = Path(r"D:\projects\Multimodal Domain-Specific AI Research Assistant (RAG + LoRA Fine-Tuning)\DATA\raw data\papers")
OUT_DIR    = Path(r"D:\projects\Multimodal Domain-Specific AI Research Assistant (RAG + LoRA Fine-Tuning)\DATA\PARSED DATA")
TEMP_DIR   = Path("temp_chunks")
PAGE_CHUNK = 20

MAX_CHARS     = 1200
MIN_CHARS     = 80
OVERLAP_LINES = 1   # only applied when a section must be split across windows

# Sections that are noise for a research RAG pipeline — skip entirely
_SKIP_HEADINGS = re.compile(
    r"^(c\.\s*text samples?|text samples?|generated (text|samples?)"
    r"|acknowledgements?|author contributions?|funding"
    r"|competing interests?|ethics statement"
    r"|supplementary (material|notes?|data))$",
    re.IGNORECASE,
)

OUT_DIR.mkdir(exist_ok=True, parents=True)
TEMP_DIR.mkdir(exist_ok=True)

# ── Docling pipeline ───────────────────────────────────────────────────────────
opts = ThreadedPdfPipelineOptions(
    accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CUDA),
    layout_batch_size=4,
    table_batch_size=4,
    ocr_batch_size=1,
)
opts.do_ocr               = False
opts.do_table_structure   = True
opts.do_formula_enrichment = False
opts.do_code_enrichment   = False

converter = DocumentConverter(format_options={
    InputFormat.PDF: PdfFormatOption(
        pipeline_cls=ThreadedStandardPdfPipeline, 
        pipeline_options=opts,
    )
})
print("Models loaded ✓\n")


_TABLE_ROW  = re.compile(r"^\|.+\|", re.MULTILINE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)", re.MULTILINE)
_CODE_FENCE = re.compile(r"^```", re.MULTILINE)
_FORMULA_TAG = "<!-- formula-not-decoded -->"

_JUNK = re.compile(
    r"^[\s\W]{0,3}$"
    r"|^[^\x00-\x7F]{1,3}$"
    r"|^\s*\d+\s*$"
    r"|^<!--.*-->$",
    re.IGNORECASE,
)


# ── title extraction ───────────────────────────────────────────────────────────
_NOISE_TITLE = re.compile(
    r"^(abstract|introduction|keywords?|doi|arxiv|preprint|"
    r"proceedings?|conference|workshop|journal|volume|issue|pages?|"
    r"copyright|©|\d{4})\b",
    re.IGNORECASE,
)

def _extract_title_from_pdf(pdf_path: Path) -> str:
    """
    Extract paper title from the first page using font-size heuristic.
    The largest-font text on page 1 is almost always the title.
    Falls back to pdf_path.stem if nothing clean is found.
    """
    try:
        with fitz.open(str(pdf_path)) as doc:
            page = doc[0]
            spans = []
            for blk in page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]:
                if blk.get("type") != 0:
                    continue
                for line in blk.get("lines", []):
                    for span in line.get("spans", []):
                        text = span["text"].strip()
                        if text:
                            spans.append({
                                "text": text,
                                "size": round(span["size"], 1),
                                "y0":   round(span["origin"][1], 1),
                            })

        if not spans:
            return pdf_path.stem

        max_size  = max(s["size"] for s in spans)
        threshold = max_size * 0.85          # allow slight size variation within title
        big       = [s for s in spans if s["size"] >= threshold]
        big.sort(key=lambda s: s["y0"])

        # merge spans that sit on the same visual line (within 4 px)
        lines_merged, cur_y, cur_texts = [], None, []
        for s in big:
            if cur_y is None or abs(s["y0"] - cur_y) > 4:
                if cur_texts:
                    lines_merged.append(" ".join(cur_texts))
                cur_y, cur_texts = s["y0"], [s["text"]]
            else:
                cur_texts.append(s["text"])
        if cur_texts:
            lines_merged.append(" ".join(cur_texts))

        clean = [
            ln for ln in lines_merged
            if len(ln) > 6 and not _NOISE_TITLE.match(ln.strip())
        ]

        title = re.sub(r"\s+", " ", " ".join(clean)).strip()
        return title if len(title) > 6 else pdf_path.stem

    except Exception:
        return pdf_path.stem


def _title_to_doc_id(title: str) -> str:
    """'Attention Is All You Need' → 'Attention_Is_All_You_Need' (max 80 chars)"""
    safe = re.sub(r"[^\w\s-]", "", title)
    safe = re.sub(r"[\s-]+", "_", safe.strip())
    return re.sub(r"_+", "_", safe).strip("_")[:80]


# ── helpers (unchanged) ────────────────────────────────────────────────────────
def _is_junk(line: str) -> bool:
    return bool(_JUNK.match(line.strip()))


def _chunk_id(doc_id: str, idx: int, text: str) -> str:
    h = hashlib.md5(f"{doc_id}:{idx}:{text[:60]}".encode()).hexdigest()[:8]
    return f"{doc_id}_{idx:04d}_{h}"


def _detect(text: str) -> dict:
    return {
        "has_table":   bool(_TABLE_ROW.search(text)),
        "has_formula": _FORMULA_TAG in text or bool(re.search(r"\$\$?.+?\$\$?", text)),
        "has_code":    bool(_CODE_FENCE.search(text)),
    }


def _split_markdown(md: str) -> list[dict]:
    """Walk markdown line-by-line → list of {heading_stack, text} blocks."""
    blocks  = []
    h_stack = []
    buf     = []
    in_code = False

    def flush():
        text = "\n".join(buf).strip()
        if text:
            blocks.append({
                "heading_stack": [t for _, t in h_stack],
                "text": text,
            })
        buf.clear()

    for raw_line in md.splitlines():
        line = raw_line.rstrip()

        if _CODE_FENCE.match(line):
            in_code = not in_code
            buf.append(line)
            continue

        if in_code:
            buf.append(line)
            continue

        m = _HEADING_RE.match(line) 
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            flush()
            while h_stack and h_stack[-1][0] >= level:
                h_stack.pop()
            h_stack.append((level, title))
            buf.append(line)
            continue

        if _is_junk(line):
            continue

        buf.append(line)

    flush()
    return blocks


def _window_block(block: dict, max_chars: int, overlap_lines: int) -> list[dict]:
    """
    Split one section block into windows.
    - Tables and code blocks are NEVER split.
    - If the whole section fits in one window, return it as-is (NO overlap).
    - Only use overlap when we actually need to split.
    """
    text   = block["text"]
    hstack = block["heading_stack"]

    has_table = bool(_TABLE_ROW.search(text))
    has_code  = "```" in text

    # atomic blocks — never split
    if has_table or has_code:
        if len(text) >= MIN_CHARS:
            return [{**block, "has_table": has_table, "has_code": has_code}]
        return []

    # fits in one window — no overlap needed
    if len(text) <= max_chars:
        return [block] if len(text) >= MIN_CHARS else []

    # needs splitting — use sliding window with overlap
    lines   = text.splitlines()
    windows = []
    start   = 0

    while start < len(lines):
        buf, chars = [], 0
        i = start
        while i < len(lines) and chars + len(lines[i]) + 1 <= max_chars:
            buf.append(lines[i])
            chars += len(lines[i]) + 1
            i += 1

        if not buf and i < len(lines):
            buf.append(lines[i])
            i += 1

        chunk_text = "\n".join(buf).strip()
        if len(chunk_text) >= MIN_CHARS:
            windows.append({"heading_stack": hstack, "text": chunk_text})

        start = max(i - overlap_lines, start + 1)
        if i >= len(lines):
            break

    return windows


def chunk_markdown(md: str, doc_id: str, source_file: str,
                   page_start: int, page_end: int) -> list[dict]:
    blocks = _split_markdown(md)
    chunks = []
    ctr    = [0]

    for block in blocks:
        # skip noise headings (text samples, acknowledgements, etc.)
        heading = block["heading_stack"][-1] if block["heading_stack"] else ""
        if _SKIP_HEADINGS.match(heading):
            continue

        for win in _window_block(block, MAX_CHARS, OVERLAP_LINES):
            text = win["text"]
            idx  = ctr[0]; ctr[0] += 1

            # prefer flags pre-set by _window_block (for tables/code),
            # fall back to _detect on the windowed text
            d = _detect(text)
            chunks.append({
                "chunk_id":    _chunk_id(doc_id, idx, text),
                "doc_id":      doc_id,
                "source_file": source_file,
                "chunk_index": idx,
                "page_start":  page_start,
                "page_end":    page_end,
                "section_path": win["heading_stack"],
                "heading":     win["heading_stack"][-1] if win["heading_stack"] else None,
                "text":        text,
                "char_count":  len(text),
                "has_table":   win.get("has_table", d["has_table"]),
                "has_formula": win.get("has_formula", d["has_formula"]),
                "has_code":    win.get("has_code",  d["has_code"]),
            })

    # prev / next links
    for i, c in enumerate(chunks):
        c["prev_chunk_id"] = chunks[i-1]["chunk_id"] if i > 0             else None
        c["next_chunk_id"] = chunks[i+1]["chunk_id"] if i < len(chunks)-1 else None

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
#  PDF SPLITTING + CONVERSION
# ══════════════════════════════════════════════════════════════════════════════
def write_chunk(src: Path, dst: Path, start: int, end: int):
    with fitz.open(str(src)) as doc:
        sub = fitz.open()
        sub.insert_pdf(doc, from_page=start, to_page=end - 1)
        sub.save(str(dst))
        sub.close()


def convert_to_md(path: Path) -> str:
    result = converter.convert(str(path))
    return result.document.export_to_markdown(image_placeholder="")


def process_pdf(pdf_path: Path) -> tuple[list[dict], str]:
    # ── extract real title; derive doc_id from it ──────────────────────────
    paper_title = _extract_title_from_pdf(pdf_path)
    doc_id      = _title_to_doc_id(paper_title)
    print(f"   Title : {paper_title}")

    all_chunks = []

    with fitz.open(str(pdf_path)) as doc:
        n_pages = len(doc)

    if n_pages <= PAGE_CHUNK:
        print(f"   1/1  pages 1–{n_pages} …", end=" ", flush=True)
        md     = convert_to_md(pdf_path)
        chunks = chunk_markdown(md, doc_id, paper_title, 1, n_pages)
        all_chunks.extend(chunks)
        print(f"ok ({len(chunks)} chunks)")
    else:
        slices = list(range(0, n_pages, PAGE_CHUNK))
        for idx, start in enumerate(slices):
            end        = min(start + PAGE_CHUNK, n_pages)
            slice_path = TEMP_DIR / f"{doc_id}_{start:04d}.pdf"
            print(f"   {idx+1}/{len(slices)}  pages {start+1}–{end} …", end=" ", flush=True)
            try:
                write_chunk(pdf_path, slice_path, start, end)
                md     = convert_to_md(slice_path)
                chunks = chunk_markdown(md, doc_id, paper_title, start+1, end)
                all_chunks.extend(chunks)
                print(f"ok ({len(chunks)} chunks)")
            except Exception as e:
                print(f"FAILED ({e})")
            finally:
                slice_path.unlink(missing_ok=True)
                gc.collect()

    # re-index globally and fix prev/next
    for i, c in enumerate(all_chunks):
        c["chunk_index"]   = i
        c["prev_chunk_id"] = all_chunks[i-1]["chunk_id"] if i > 0                else None
        c["next_chunk_id"] = all_chunks[i+1]["chunk_id"] if i < len(all_chunks)-1 else None

    return all_chunks, doc_id


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

pdfs = sorted(PDF_DIR.rglob("*.pdf"))
print(f"Found {len(pdfs)} PDFs\n")

for pdf_path in pdfs:
    print(f"▶  {pdf_path.name}")
    t0 = time.time()
    try:
        chunks, doc_id = process_pdf(pdf_path)
        out_path = OUT_DIR / f"{doc_id}.jsonl"
        if out_path.exists():
            print(f"   SKIP  (already exists: {out_path.name})")
            continue
        with open(out_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"   ✓  {len(chunks)} chunks  |  {time.time()-t0:.1f}s  →  {out_path.name}\n")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"   ✗  FAILED — {e}\n")

if TEMP_DIR.exists() and not any(TEMP_DIR.iterdir()):
    TEMP_DIR.rmdir()

print("All done.")