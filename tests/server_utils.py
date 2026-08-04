import socket
import time


def wait_for_server(port, timeout=20.0):
    """Block until an HTTP server is accepting connections on 127.0.0.1:port."""
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError as exc:
            last = exc
            time.sleep(0.1)
    raise RuntimeError(f"server on port {port} did not start: {last}")
