"""Run the Residential VPP FastAPI service on an available local port.

Windows can reserve common development ports such as 8000. This launcher tests
candidate ports before starting Uvicorn and skips ports that are already in use
or blocked by local policy.
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

import uvicorn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_PORTS = [8000, 8001, 8010, 8080, 8081, 9000]


def can_bind(host: str, port: int) -> tuple[bool, str | None]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as exc:
            return False, str(exc)
    return True, None


def choose_port(host: str, requested_port: int | None) -> int:
    candidates = []
    if requested_port is not None:
        candidates.append(requested_port)
    candidates.extend(port for port in DEFAULT_PORTS if port not in candidates)

    failures = []
    for port in candidates:
        ok, reason = can_bind(host, port)
        if ok:
            return port
        failures.append(f"{port}: {reason}")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Residential VPP FastAPI service.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Preferred API port.")
    parser.add_argument("--reload", action="store_true", help="Enable Uvicorn auto-reload.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_port = choose_port(args.host, args.port)
    if selected_port != args.port:
        print(f"Port {args.port} is unavailable. Starting FastAPI on port {selected_port}.")
    print(f"FastAPI docs: http://{args.host}:{selected_port}/docs")
    print(f"FastAPI health: http://{args.host}:{selected_port}/health")
    uvicorn.run(
        "flexihome.api.optimizer:app",
        host=args.host,
        port=selected_port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()

