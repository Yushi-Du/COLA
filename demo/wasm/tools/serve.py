#!/usr/bin/env python3
"""Serve the static browser demo with native COOP/COEP headers."""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class CrossOriginIsolatedHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    handler = lambda *values, **kwargs: CrossOriginIsolatedHandler(  # noqa: E731
        *values,
        directory=str(REPOSITORY_ROOT),
        **kwargs,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving http://{args.host}:{args.port}/demo/wasm/")
    server.serve_forever()


if __name__ == "__main__":
    main()
