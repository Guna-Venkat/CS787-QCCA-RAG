#!/usr/bin/env python
"""Pre-extract all FinanceBench PDF page texts to a JSON cache.

Must be run once before any benchmark run.  Extracts text
from all unique PDFs referenced by the open-source question set (84 PDFs) and
writes them to ``data/benchmarks/financebench/pdf_texts.json``.

Subsequent ``load_financebench_data()`` calls load from this cache instantly
— zero PDF extraction at runtime, guaranteeing fair and reproducible timing.

Usage::

    python benchmarks/financebench/setup.py --workers 8
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

_FB_DATA_DIR = os.path.join(_PROJECT_ROOT, "third_party", "financebench", "data")
_FB_PDFS_DIR = os.path.join(_PROJECT_ROOT, "third_party", "financebench", "pdfs")
_CACHE_DIR = os.path.join(_PROJECT_ROOT, "data", "benchmarks", "financebench")
_CACHE_FILE = os.path.join(_CACHE_DIR, "pdf_texts.json")
_QUESTIONS_FILE = os.path.join(_FB_DATA_DIR, "financebench_open_source.jsonl")


def _extract_pdf(doc_name: str) -> tuple[str, list[str] | None, float]:
    """Extract text from a single PDF.  Returns (doc_name, pages_or_None, elapsed_s)."""
    import fitz  # pymupdf
    t0 = time.perf_counter()
    pdf_path = os.path.join(_FB_PDFS_DIR, f"{doc_name}.pdf")
    try:
        if not os.path.exists(pdf_path):
            print(f"  Warning: PDF not found — {pdf_path}")
            return doc_name, None, time.perf_counter() - t0
        doc = fitz.open(pdf_path)
        pages = []
        for page_num in range(len(doc)):
            page_text = doc[page_num].get_text()
            pages.append(page_text)
        doc.close()
        return doc_name, pages, time.perf_counter() - t0
    except Exception as e:
        print(f"  Error extracting {doc_name}: {e}")
        return doc_name, None, time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(
        description="Pre-extract FinanceBench PDF page texts to JSON cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers for PDF extraction (default: 4)")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract all PDFs even if cache exists")
    args = parser.parse_args()

    # ── Check if cache already exists ─────────────────────────────────
    if os.path.exists(_CACHE_FILE) and not args.force:
        with open(_CACHE_FILE) as f:
            existing = json.load(f)
        print(f"PDF text cache already exists: {_CACHE_FILE}")
        print(f"  {len(existing)} documents cached")
        print(f"  Use --force to re-extract")
        return

    # ── Collect unique doc_names from questions ────────────────────────
    if not os.path.exists(_QUESTIONS_FILE):
        print(f"Error: questions file not found: {_QUESTIONS_FILE}")
        sys.exit(1)

    with open(_QUESTIONS_FILE) as f:
        raw = [json.loads(line) for line in f if line.strip()]

    unique_docs = sorted(set(r.get("doc_name", "") for r in raw if r.get("doc_name")))
    print(f"Questions: {len(raw)} instances")
    print(f"Unique PDFs: {len(unique_docs)}")
    print(f"PDFs directory: {_FB_PDFS_DIR}")
    print(f"Workers: {args.workers}")
    print()

    # ── Check which PDFs actually exist on disk ───────────────────────
    missing = [d for d in unique_docs
               if not os.path.exists(os.path.join(_FB_PDFS_DIR, f"{d}.pdf"))]
    if missing:
        print(f"Warning: {len(missing)} PDF(s) not found on disk:")
        for m in missing[:10]:
            print(f"  - {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
        print()

    # ── Extract all PDFs (parallel) ───────────────────────────────────
    t_start = time.perf_counter()
    pdf_cache: dict[str, list[str]] = {}
    failed = 0
    total_pages = 0

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_extract_pdf, d): d for d in unique_docs}
            completed = 0
            for future in as_completed(futures):
                doc_name, pages, _elapsed = future.result()
                completed += 1
                if pages is not None:
                    pdf_cache[doc_name] = pages
                    total_pages += len(pages)
                else:
                    failed += 1
                if completed % max(1, len(unique_docs) // 5) == 0:
                    print(f"  {completed}/{len(unique_docs)} PDFs extracted...")
    else:
        for i, doc_name in enumerate(unique_docs):
            _, pages, _elapsed = _extract_pdf(doc_name)
            if pages is not None:
                pdf_cache[doc_name] = pages
                total_pages += len(pages)
            else:
                failed += 1
            if (i + 1) % max(1, len(unique_docs) // 5) == 0:
                print(f"  {i + 1}/{len(unique_docs)} PDFs extracted...")

    t_total = time.perf_counter() - t_start

    # ── Write cache ───────────────────────────────────────────────────
    os.makedirs(_CACHE_DIR, exist_ok=True)
    with open(_CACHE_FILE, "w") as f:
        json.dump(pdf_cache, f, ensure_ascii=False)

    cache_size_mb = os.path.getsize(_CACHE_FILE) / (1024 * 1024)

    # ── Print stats ───────────────────────────────────────────────────
    print(f"\nDone: {t_total:.1f}s")
    print(f"  PDFs extracted:  {len(pdf_cache)}")
    print(f"  Failed/missing:  {failed}")
    print(f"  Total pages:     {total_pages:,}")
    print(f"  Avg pages/PDF:   {total_pages / max(len(pdf_cache), 1):.0f}")
    print(f"  Cache size:      {cache_size_mb:.1f} MB")
    print(f"  Cache path:      {_CACHE_FILE}")


if __name__ == "__main__":
    main()
