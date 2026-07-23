"""Batch ingest worker.

A CLI, not a daemon. In OpenShift this runs as a Job (one-shot backfill) or a
CronJob (incremental sweep of a mounted share). There is no queue broker in the
MVP on purpose -- adding Kafka or Redis before there is a throughput problem is
a dependency to justify, carry across the air gap, and operate, in exchange for
nothing. The seam is `iter_files`; a broker consumer drops in there later.

    python -m app.main ingest ./corpus --recursive
    python -m app.main ingest ./corpus --workers 4 --metadata '{"bu":"claims"}'
    python -m app.main bootstrap --recreate
    python -m app.main stats
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rag_core.config import get_settings
from rag_core.embeddings import get_embedder
from rag_core.extraction import IngestPipeline, OCRClient, TikaClient
from rag_core.logging_setup import configure_logging
from rag_core.search import ChunkIndexer, IndexAdmin, get_client

log = logging.getLogger("rag-worker")

# Extensions Tika handles well. Anything else is skipped with a warning rather
# than silently indexed as binary garbage.
SUPPORTED = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".odt", ".ods",
    ".txt", ".md", ".rtf", ".html", ".htm", ".xml", ".csv", ".json", ".eml", ".msg",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp",
}


def build_pipeline() -> tuple[IngestPipeline, IndexAdmin]:
    settings = get_settings()
    client = get_client()
    admin = IndexAdmin(client, settings.opensearch)
    pipeline = IngestPipeline(
        settings=settings,
        tika=TikaClient(settings.tika),
        ocr=OCRClient(settings.ocr) if settings.ocr.enabled else None,
        embedder=get_embedder(),
        indexer=ChunkIndexer(client, settings.opensearch),
        admin=admin,
    )
    return pipeline, admin


def iter_files(root: Path, recursive: bool) -> list[Path]:
    if root.is_file():
        return [root]
    files = (root.rglob("*") if recursive else root.glob("*"))
    return sorted(
        p for p in files if p.is_file() and p.suffix.lower() in SUPPORTED
    )


def cmd_bootstrap(args: argparse.Namespace) -> int:
    settings = get_settings()
    _, admin = build_pipeline()
    name = admin.create(dim=settings.embedding.dim, recreate=args.recreate)
    print(json.dumps(admin.stats(name), indent=2))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    _, admin = build_pipeline()
    print(json.dumps(admin.stats(), indent=2))
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = get_settings()
    root = Path(args.path).expanduser().resolve()
    if not root.exists():
        log.error("path does not exist: %s", root)
        return 2

    files = iter_files(root, args.recursive)
    if not files:
        log.error("no supported files under %s", root)
        return 2

    metadata = json.loads(args.metadata) if args.metadata else {}
    acl = [a.strip() for a in args.acl.split(",") if a.strip()]
    if args.classification or acl:
        log.info("labelling every chunk: classification=%r acl=%s",
                 args.classification, acl or "[]")
    pipeline, admin = build_pipeline()

    if not admin.exists():
        log.info("index %s missing; creating it", settings.opensearch.index)
        admin.create(dim=settings.embedding.dim)

    log.info("ingesting %d files with %d workers", len(files), args.workers)
    started = time.perf_counter()
    total_chunks = failures = ocr_pages = 0

    def work(path: Path):
        data = path.read_bytes()
        source_uri = str(path if args.absolute_uri else path.relative_to(root) if root.is_dir() else path)
        return path, pipeline.ingest_bytes(
            data=data,
            filename=path.name,
            source_uri=args.uri_prefix + source_uri,
            metadata=metadata,
            refresh=False,
            classification=args.classification,
            acl=acl,
        )

    # Threads, not processes: the work is dominated by waiting on Tika and OCR,
    # and ONNX releases the GIL during inference.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, p) for p in files]
        for i, future in enumerate(as_completed(futures), start=1):
            try:
                path, result = future.result()
            except Exception:
                failures += 1
                log.exception("unhandled error during ingest")
                continue
            total_chunks += result.chunks_indexed
            ocr_pages += result.ocr_pages
            if not result.ok:
                failures += 1
                log.warning(
                    "FAILED %s: %s", path.name, "; ".join(result.errors) or "no chunks"
                )
            else:
                log.info(
                    "[%d/%d] %s -> %d chunks (%d pages, %d via OCR) %s",
                    i, len(files), path.name, result.chunks_indexed,
                    result.pages, result.ocr_pages, result.timings_ms,
                )

    # One refresh at the end rather than per document: refreshing on every write
    # forces a segment flush and makes bulk loading several times slower.
    get_client().indices.refresh(index=settings.opensearch.index)

    elapsed = time.perf_counter() - started
    log.info(
        "done: %d files, %d chunks, %d OCR pages, %d failures in %.1fs",
        len(files), total_chunks, ocr_pages, failures, elapsed,
    )
    print(json.dumps(admin.stats(), indent=2))
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-worker", description="RAG ingest worker")
    sub = parser.add_subparsers(dest="command", required=True)

    p_boot = sub.add_parser("bootstrap", help="create the OpenSearch index")
    p_boot.add_argument("--recreate", action="store_true", help="DELETE and recreate the index")
    p_boot.set_defaults(func=cmd_bootstrap)

    p_stats = sub.add_parser("stats", help="show index statistics")
    p_stats.set_defaults(func=cmd_stats)

    p_ing = sub.add_parser("ingest", help="ingest a file or directory")
    p_ing.add_argument("path")
    p_ing.add_argument("--recursive", action="store_true", default=True)
    p_ing.add_argument("--no-recursive", dest="recursive", action="store_false")
    p_ing.add_argument("--workers", type=int, default=4)
    p_ing.add_argument("--metadata", default="", help="JSON attached to every chunk (NOT indexed)")
    # These are indexed and filterable; --metadata is not. Bulk loading is the
    # normal way a corpus gets labelled, so the flags live here first.
    p_ing.add_argument("--classification", default="",
                       help="security label applied to every chunk, e.g. UNCLASSIFIED")
    p_ing.add_argument("--acl", default="",
                       help="comma-separated groups/attributes required to retrieve these chunks")
    p_ing.add_argument("--uri-prefix", default="", help="prepended to each source_uri")
    p_ing.add_argument("--absolute-uri", action="store_true")
    p_ing.set_defaults(func=cmd_ingest)

    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging("rag-worker", settings.log_level, json_output=settings.env != "local")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
