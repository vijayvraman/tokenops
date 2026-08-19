"""FastAPI control-plane application (registration, ledger, run records, governance)."""

from __future__ import annotations

import os

from fastapi import FastAPI

from tokenops import __version__
from tokenops.control.http import mount_run_registration
from tokenops.control.store import Store
from tokenops.server.plane_api import mount_plane_api


def create_app(store: Store | None = None) -> FastAPI:
    """Build the control-plane app.

    Owns a :class:`Store` and mounts:

    * ``POST /v1/runs`` — run registration (intent, user_dims, mode)
    * ``GET /health`` — liveness
    * the plane API (:func:`~tokenops.server.plane_api.mount_plane_api`) — the ledger,
      run-record, and governance routes an agent's ``HttpStore`` calls once it is
      pointed here with ``TOKENOPS_URL``

    Agent SDKs need nothing beyond pointing ``TOKENOPS_URL`` at this service.
    """
    store = store or Store(os.environ.get("TOKENOPS_DB", "tokenops.db"))

    app = FastAPI(title="TokenOps Control Plane", version=__version__)
    app.state.store = store

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "tokenops-control-plane"}

    mount_run_registration(app, store)
    mount_plane_api(app, store)

    return app
