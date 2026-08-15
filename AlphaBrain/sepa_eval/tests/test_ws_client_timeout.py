"""Tests for websocket policy client timeout + retry behaviour (P1.4).

Uses a real local websocket server (websockets.sync.server) that simulates an
unresponsive / flaky policy server.
"""

import socket
import threading
import time

import pytest

websockets = pytest.importorskip("websockets")
from websockets.sync.server import serve  # noqa: E402

from deployment.model_server.tools import msgpack_numpy  # noqa: E402
from deployment.model_server.tools.websocket_policy_client import (  # noqa: E402
    PolicyClientConnectionError,
    PolicyClientTimeoutError,
    WebsocketClientPolicy,
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Server:
    """Local ws server; behaviour of each accepted connection set by `mode`."""

    def __init__(self, mode: str):
        self.mode = mode
        self.port = _free_port()
        self.connections = 0
        self._packer = msgpack_numpy.Packer()
        self._server = serve(self._handler, "127.0.0.1", self.port)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def _handler(self, ws):
        self.connections += 1
        ws.send(self._packer.pack({"server": "fake", "conn": self.connections}))
        if self.mode == "silent":
            # Accept infer requests but never answer.
            try:
                while True:
                    ws.recv()
            except Exception:
                return
        elif self.mode == "drop_then_serve":
            if self.connections == 1:
                # Read one request then hard-drop the connection.
                try:
                    ws.recv()
                except Exception:
                    pass
                ws.close_socket()
                return
            # Subsequent connections behave normally.
            try:
                while True:
                    req = msgpack_numpy.unpackb(ws.recv())
                    ws.send(self._packer.pack({"ok": True, "echo_type": req.get("type")}))
            except Exception:
                return

    def shutdown(self):
        self._server.shutdown()
        self._thread.join(timeout=5)


def test_infer_times_out_on_unresponsive_server():
    server = _Server(mode="silent")
    try:
        client = WebsocketClientPolicy(host="127.0.0.1", port=server.port, request_timeout=0.5, max_retries=0)
        start = time.monotonic()
        with pytest.raises(PolicyClientTimeoutError, match="did not respond to 'infer'"):
            client.infer({"obs": [1, 2, 3]})
        elapsed = time.monotonic() - start
        assert elapsed < 5.0  # bounded: did not hang
        client.close()
    finally:
        server.shutdown()


def test_timeout_error_is_timeout_subclass():
    assert issubclass(PolicyClientTimeoutError, TimeoutError)
    assert issubclass(PolicyClientConnectionError, ConnectionError)


def test_infer_retries_on_connection_drop_then_succeeds():
    server = _Server(mode="drop_then_serve")
    try:
        client = WebsocketClientPolicy(
            host="127.0.0.1", port=server.port, request_timeout=5.0, max_retries=2, retry_backoff=0.05
        )
        result = client.infer({"obs": [1]})
        assert result["ok"] is True
        assert result["echo_type"] == "infer"
        assert server.connections >= 2  # reconnected at least once
        client.close()
    finally:
        server.shutdown()


def test_infer_raises_connection_error_when_retries_exhausted():
    server = _Server(mode="silent")
    client = WebsocketClientPolicy(host="127.0.0.1", port=server.port, request_timeout=5.0, max_retries=0)
    server.shutdown()  # kill the server so the socket drops
    time.sleep(0.2)
    with pytest.raises((PolicyClientConnectionError, PolicyClientTimeoutError)):
        client.infer({"obs": [1]})
    client.close()


def test_public_api_backward_compatible():
    """Constructor still accepts the legacy positional/keyword signature."""
    import inspect

    sig = inspect.signature(WebsocketClientPolicy.__init__)
    params = list(sig.parameters)
    assert params[1:4] == ["host", "port", "api_key"]
    for name in ("request_timeout", "max_retries"):
        assert sig.parameters[name].default is not inspect.Parameter.empty
    for method in ("infer", "predict_action", "reset", "init_device", "get_server_metadata", "close"):
        assert hasattr(WebsocketClientPolicy, method)
