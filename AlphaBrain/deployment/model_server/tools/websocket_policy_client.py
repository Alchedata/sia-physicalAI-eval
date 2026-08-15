# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].


import logging
import os
import time
from typing import Dict, Optional, Tuple

import websockets.exceptions
import websockets.sync.client
from typing_extensions import override

from . import msgpack_numpy


class PolicyClientTimeoutError(TimeoutError):
    """Raised when the policy server does not respond within the request timeout."""


class PolicyClientConnectionError(ConnectionError):
    """Raised when the connection to the policy server fails after retries."""


#: Exceptions treated as retryable connection failures (not server-side errors).
_CONNECTION_ERRORS = (
    websockets.exceptions.ConnectionClosed,
    ConnectionResetError,
    ConnectionRefusedError,
    BrokenPipeError,
    OSError,
)


class WebsocketClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.

    Robustness: every request (``infer`` / ``reset`` / ``init_device``) waits at
    most ``request_timeout`` seconds for a response and raises
    ``PolicyClientTimeoutError`` instead of hanging forever.  Connection-level
    failures (dropped socket, refused connection) are retried up to
    ``max_retries`` times with a fresh connection; server-side errors are never
    retried.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: Optional[int] = 10093,
        api_key: Optional[str] = None,
        request_timeout: float = 60.0,
        max_retries: int = 2,
        retry_backoff: float = 1.0,
    ) -> None:
        # 0.0.0.0 cannot be used as a connection target, here default 127.0.0.1
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._request_timeout = float(request_timeout)
        self._max_retries = int(max_retries)
        self._retry_backoff = float(retry_backoff)
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self, timeout: float = 600) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        start_time = time.time()

        for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(k, None)

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Failed to connect to server within {timeout} seconds")

            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    open_timeout=150,
                )
                metadata = msgpack_numpy.unpackb(conn.recv(timeout=self._request_timeout))
                return conn, metadata
            except ConnectionRefusedError:
                logging.info(f"Still waiting for server {self._uri} ...")
                time.sleep(2)

    def _reconnect(self, timeout: float = 30) -> None:
        """Drop the current socket and establish a fresh connection."""
        try:
            self._ws.close()
        except Exception:
            pass
        self._ws, self._server_metadata = self._wait_for_server(timeout=timeout)

    def _request(self, payload: Dict, context: str):
        """Send one request and wait for the response with timeout + limited retries.

        Retries (with a fresh connection) only on connection-level errors.
        Timeouts and server-side errors are raised immediately with context.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                self._ws.send(self._packer.pack(payload))
                response = self._ws.recv(timeout=self._request_timeout)
                if isinstance(response, str):
                    raise RuntimeError(f"Error from policy server ({context}):\n{response}")
                return msgpack_numpy.unpackb(response)
            except TimeoutError as exc:
                raise PolicyClientTimeoutError(
                    f"Policy server at {self._uri} did not respond to '{context}' within "
                    f"{self._request_timeout}s (attempt {attempt + 1}/{self._max_retries + 1})."
                ) from exc
            except _CONNECTION_ERRORS as exc:
                last_exc = exc
                if attempt >= self._max_retries:
                    break
                logging.warning(
                    f"Connection error during '{context}' (attempt {attempt + 1}/"
                    f"{self._max_retries + 1}): {exc!r}. Reconnecting..."
                )
                time.sleep(self._retry_backoff * (attempt + 1))
                try:
                    self._reconnect()
                except Exception as reconnect_exc:
                    last_exc = reconnect_exc
                    break

        raise PolicyClientConnectionError(
            f"Failed to complete '{context}' against {self._uri} after "
            f"{self._max_retries + 1} attempt(s): {last_exc!r}"
        ) from last_exc

    def init_device(self, device: str = "cuda") -> Dict:
        """send one device initialization message, verify protocol and service availability"""
        payload = {"device": device, "type": "ping"}
        return self._request(payload, context="init_device")

    @override
    def infer(self, obs: Dict) -> Dict:
        query_info = {
            "payload": obs,
            "type": "infer",
        }
        return self._request(query_info, context="infer")

    def predict_action(self, query_info: Dict) -> Dict:
        """Backward-compatible alias for older benchmark clients."""
        return self.infer(query_info)

    @override
    def reset(self, instruction) -> None:
        payload = {"instruction": instruction, "reset": True}
        try:
            self._request(payload, context="reset")
        except RuntimeError:
            # Preserve legacy behaviour: reset responses were previously ignored,
            # including server-side error strings.
            pass

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass
