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
    AcceleratorDevice, AcceleratorOptions, PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

from config import PDF_DIR, PARSED_DIR, EMBED_MODEL, EMBED_DEVICE
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── config ─────────────────────────────────────────────────────────────────────
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

PARSED_DIR.mkdir(exist_ok=True, parents=True)
TEMP_DIR.mkdir(exist_ok=True)

# ── Docling pipeline ───────────────────────────────────────────────────────────
opts = PdfPipelineOptions(
    accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CUDA),
    layout_batch_size=1,
    table_batch_size=4,
    ocr_batch_size=1,
)
opts.do_ocr               = False
opts.do_table_structure   = True
opts.do_formula_enrichment = False
opts.do_code_enrichment   = False

converter = DocumentConverter(format_options={
    InputFormat.PDF: PdfFormatOption(
        pipeline_cls=StandardPdfPipeline,
        pipeline_options=opts,
    )
})
print("Models loaded [OK]\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CHUNKER
# ══════════════════════════════════════════════════════════════════════════════

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


def _semantic_chunker(block: dict) -> list[dict]:
    """
    Splits a section block into chunks by isolating tables/code 
    and using cosine similarity between sentences for natural text.
    """
    text   = block["text"]
    hstack = block["heading_stack"]

    has_table = bool(_TABLE_ROW.search(text))
    has_code  = "```" in text

    # Fast path if it fits entirely in one chunk
    if len(text) <= MAX_CHARS and not has_table and not has_code:
        return [block] if len(text) >= MIN_CHARS else []

    # 1. Isolate tables and code
    lines = text.splitlines()
    sub_blocks = []
    current_type = None
    current_lines = []
    in_code = False
    
    for line in lines:
        if _CODE_FENCE.match(line):
            if not in_code:
                if current_lines:
                    sub_blocks.append({"type": current_type or "text", "lines": current_lines})
                current_lines = [line]
                current_type = "code"
                in_code = True
            else:
                current_lines.append(line)
                sub_blocks.append({"type": "code", "lines": current_lines})
                current_lines = []
                current_type = None
                in_code = False
            continue
            
        if in_code:
            current_lines.append(line)
            continue
            
        if bool(_TABLE_ROW.search(line)):
            if current_type != "table":
                if current_lines:
                    sub_blocks.append({"type": current_type or "text", "lines": current_lines})
                current_lines = []
                current_type = "table"
            current_lines.append(line)
        else:
            if current_type == "table":
                sub_blocks.append({"type": "table", "lines": current_lines})
                current_lines = []
                current_type = "text"
            else:
                current_type = "text"
            current_lines.append(line)
            
    if current_lines:
         sub_blocks.append({"type": current_type or "text", "lines": current_lines})

    # 2. Process sub-blocks semantically
    windows = []
    for sb in sub_blocks:
        sb_text = "\n".join(sb["lines"]).strip()
        if len(sb_text) < MIN_CHARS:
            continue
            
        if sb["type"] in ["table", "code"]:
            # Tables and code blocks are atomic
            windows.append({
                "heading_stack": hstack, 
                "text": sb_text, 
                "has_table": (sb["type"] == "table"),
                "has_code": (sb["type"] == "code")
            })
            continue

        # For text, apply semantic chunking
        try:
            sentences = nltk.sent_tokenize(sb_text)
        except:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            sentences = nltk.sent_tokenize(sb_text)
            
        if not sentences:
            continue
            
        if len(sentences) == 1 or len(sb_text) <= 500:
            windows.append({"heading_stack": hstack, "text": sb_text})
            continue
            
        # Semantic splitting
        global _embedder
        if '_embedder' not in globals():
            _embedder = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
            
        embeddings = _embedder.encode(sentences)
        
        similarities = []
        for i in range(len(sentences) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            similarities.append(sim)
            
        if not similarities:
            windows.append({"heading_stack": hstack, "text": sb_text})
            continue
            
        threshold = np.percentile(similarities, 30) # Bottom 30% are split points
        
        chunks = []
        current_chunk = [sentences[0]]
        current_len = len(sentences[0])
        
        for i, (sent, sim) in enumerate(zip(sentences[1:], similarities)):
            if (sim < threshold and current_len > 300) or (current_len + len(sent) > MAX_CHARS):
                chunks.append(" ".join(current_chunk))
                current_chunk = [sent]
                current_len = len(sent)
            else:
                current_chunk.append(sent)
                current_len += len(sent) + 1
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        for c in chunks:
            if len(c) >= MIN_CHARS:
                windows.append({"heading_stack": hstack, "text": c})
                
    return windows


def chunk_markdown(md: str, doc_id: str, source_file: str) -> list[dict]:
    blocks = _split_markdown(md)
    chunks = []
    ctr    = [0]

    for block in blocks:
        # skip noise headings (text samples, acknowledgements, etc.)
        hstack = block["heading_stack"]
        if hstack and _SKIP_HEADINGS.match(hstack[-1]):
            continue

        for win in _semantic_chunker(block):
            text = win["text"]
            idx  = ctr[0]; ctr[0] += 1

            # prefer flags pre-set by _semantic_chunker (for tables/code),
            # fall back to _detect on the windowed text
            d = _detect(text)
            chunks.append({
                "chunk_id":    _chunk_id(doc_id, idx, text),
                "doc_id":      doc_id,
                "source_file": source_file,
                "chunk_index": idx,
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


def process_pdf(pdf_path: Path) -> list[dict]:
    doc_id     = re.sub(r"\W+", "_", pdf_path.stem)
    all_chunks = []

    with fitz.open(str(pdf_path)) as doc:
        n_pages = len(doc)

    if n_pages <= PAGE_CHUNK:
        print(f"   1/1  pages 1–{n_pages} …", end=" ", flush=True)
        md     = convert_to_md(pdf_path)
        chunks = chunk_markdown(md, doc_id, pdf_path.name)
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
                chunks = chunk_markdown(md, doc_id, pdf_path.name)
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

    return all_chunks


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

pdfs = sorted(PDF_DIR.rglob("*.pdf"))
print(f"Found {len(pdfs)} PDFs\n")

for pdf_path in pdfs:
    out_path = PARSED_DIR / f"{pdf_path.stem}.jsonl"
    if out_path.exists():
        print(f"SKIP  {pdf_path.name}")
        continue

    print(f">  {pdf_path.name}")
    t0 = time.time()
    try:
        chunks = process_pdf(pdf_path)
        with open(out_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"   [OK]  {len(chunks)} chunks  |  {time.time()-t0:.1f}s  ->  {out_path.name}\n")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"   [FAIL]  FAILED - {e}\n")

if TEMP_DIR.exists() and not any(TEMP_DIR.iterdir()):
    TEMP_DIR.rmdir()

print("All done.")