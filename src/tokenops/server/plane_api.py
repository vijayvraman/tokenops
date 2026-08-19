"""Plane-side HTTP surface backing :class:`~tokenops.control.http_store.HttpStore`.

When an agent sets ``TOKENOPS_URL``, ``ControlPlaneClient.require_store()`` hands it an
``HttpStore`` instead of a local SQLite ``Store`` — every ledger read/write, governance
lookup, and dashboard run row then has to travel over HTTP. These are the routes on the
other end of that call. Without them the agent registers a run and nothing else lands, so
the dashboard shows a permanently "running" row with zero cost.

Registration itself (``POST /v1/runs``) stays in ``tokenops.control.http`` because agents
mount it too in embedded mode; everything here is plane-only.

Route ↔ ``Store`` method mapping is intentionally one-to-one: request bodies are the
dataclass field names, so ``HttpStore`` can rebuild the models with ``Model(**payload)``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from tokenops.control.models import (
    BudgetSpec,
    RunNotRegisteredError,
    RunRecord,
    Segment,
)
from tokenops.control.store import Store

# ``update_run`` interpolates field names straight into UPDATE ... SET, so the set of
# writable columns is fixed here rather than taken from the request body.
_UPDATABLE_RUN_FIELDS = frozenset(
    {
        "agent",
        "status",
        "parent_run",
        "parent_span",
        "halt_reason",
        "detector",
        "cost_micros",
        "steps",
        "started_at",
        "ended_at",
        "task",
        "dims",
        "governance_events",
    }
)

# Columns holding JSON blobs — accept the decoded value from the wire and re-encode.
_JSON_RUN_FIELDS = frozenset({"dims", "governance_events"})


def _run_record(payload: dict[str, Any]) -> RunRecord:
    return RunRecord(
        run_id=str(payload["run_id"]),
        agent=str(payload.get("agent", "")),
        status=payload.get("status", "running"),
        parent_run=payload.get("parent_run"),
        parent_span=payload.get("parent_span"),
        halt_reason=payload.get("halt_reason"),
        detector=payload.get("detector"),
        cost_micros=int(payload.get("cost_micros", 0) or 0),
        steps=int(payload.get("steps", 0) or 0),
        started_at=float(payload.get("started_at", 0.0) or 0.0),
        ended_at=payload.get("ended_at"),
        task=payload.get("task"),
        dims={str(k): str(v) for k, v in (payload.get("dims") or {}).items()},
        governance_events=list(payload.get("governance_events") or []),
    )


def mount_plane_api(app: FastAPI, store: Store) -> None:
    """Mount the ledger / run-record / governance routes ``HttpStore`` calls."""

    # ---- run registration lookup ----------------------------------------- #

    @app.get("/v1/runs/{run_id}/registration")
    def get_registration(run_id: str) -> JSONResponse:
        try:
            reg = store.resolve_run(run_id)
        except RunNotRegisteredError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        return JSONResponse(
            {
                "run_id": reg.run_id,
                "intent": reg.intent,
                "user_dims": dict(reg.user_dims),
                "mode": reg.mode.value,
            }
        )

    # ---- governance config ------------------------------------------------ #

    # ``{agent:path}`` so ``/v1/governance/`` (global scope, empty agent) matches too —
    # a plain ``{agent}`` would 404 and httpx does not follow the slash redirect.
    @app.get("/v1/governance/{agent:path}")
    def governance(agent: str) -> dict:
        return store.governance_config_for(agent.strip("/"))

    # ---- dashboard run records -------------------------------------------- #

    @app.put("/v1/run-records")
    async def create_run_record(request: Request) -> JSONResponse:
        payload = await request.json()
        if not str(payload.get("run_id") or "").strip():
            return JSONResponse({"error": "run_id is required"}, status_code=400)
        rec = store.create_run(_run_record(payload))
        # create_run only writes the run columns; events arrive on the update leg.
        if rec.governance_events:
            store.update_run(rec.run_id, governance_events=rec.governance_events)
        return JSONResponse(asdict(rec), status_code=200)

    @app.patch("/v1/run-records/{run_id}")
    async def update_run_record(run_id: str, request: Request) -> JSONResponse:
        payload = await request.json()
        if not isinstance(payload, dict):
            return JSONResponse({"error": "expected a JSON object"}, status_code=400)
        unknown = sorted(set(payload) - _UPDATABLE_RUN_FIELDS)
        if unknown:
            return JSONResponse(
                {"error": f"unknown run fields: {', '.join(unknown)}"},
                status_code=400,
            )
        fields = {
            k: (json.dumps(v) if k in _JSON_RUN_FIELDS and not isinstance(v, str) else v)
            for k, v in payload.items()
        }
        store.update_run(run_id, **fields)
        return JSONResponse({"run_id": run_id, "updated": sorted(fields)})

    @app.get("/v1/run-records/{run_id}")
    def get_run_record(run_id: str) -> JSONResponse:
        rec = store.get_run(run_id)
        if rec is None:
            return JSONResponse({"error": f"run {run_id!r} not found"}, status_code=404)
        return JSONResponse(asdict(rec))

    @app.get("/v1/run-records")
    def list_run_records(problematic_only: bool = False, limit: int = 200) -> list[dict]:
        return [asdict(r) for r in store.list_runs(problematic_only=problematic_only, limit=limit)]

    # ---- shared ledger ----------------------------------------------------- #

    @app.post("/v1/ledger/spent/add")
    async def ledger_add_spent(request: Request) -> dict:
        body = await request.json()
        total = store.ledger_add_spent(
            str(body["budget_id"]),
            str(body["segment_key"]),
            str(body["period"]),
            int(body["delta"]),
        )
        return {"spent_micros": total}

    @app.get("/v1/ledger/spent")
    def ledger_get_spent(budget_id: str, segment_key: str, period: str) -> dict:
        return {"spent_micros": store.ledger_get_spent(budget_id, segment_key, period)}

    @app.post("/v1/ledger/inflight/admit")
    async def ledger_admit(request: Request) -> dict:
        body = await request.json()
        return {"count": store.ledger_admit(str(body["segment_key"]))}

    @app.post("/v1/ledger/inflight/complete")
    async def ledger_complete(request: Request) -> dict:
        body = await request.json()
        return {"count": store.ledger_complete(str(body["segment_key"]))}

    @app.get("/v1/ledger/inflight")
    def ledger_inflight(segment_key: str) -> dict:
        return {"count": store.ledger_inflight(segment_key)}

    @app.post("/v1/ledger/halt/mark")
    async def ledger_mark_halted(request: Request) -> dict:
        body = await request.json()
        run_id = str(body["run_id"])
        store.ledger_mark_halted(run_id, str(body.get("reason") or ""))
        return {"run_id": run_id, "halted": True}

    # 200 with ``halted: false`` for an unseen run — an unknown run is not halted, and a
    # 404 here would read as "route missing" to the client.
    @app.get("/v1/ledger/halt/{run_id}")
    def ledger_halt_state(run_id: str) -> dict:
        return {
            "halted": store.ledger_is_halted(run_id),
            "halt_reason": store.ledger_halt_reason(run_id),
        }

    @app.post("/v1/ledger/halt/clear")
    async def ledger_clear_halt(request: Request) -> dict:
        body = await request.json()
        run_id = str(body["run_id"])
        store.ledger_clear_halt(run_id)
        return {"run_id": run_id, "halted": False}

    # ---- governance admin objects ------------------------------------------ #

    @app.put("/v1/segments")
    async def upsert_segment(request: Request) -> dict:
        return asdict(store.upsert_segment(Segment(**await request.json())))

    @app.get("/v1/segments")
    def list_segments() -> list[dict]:
        return [asdict(s) for s in store.list_segments()]

    @app.put("/v1/budgets")
    async def upsert_budget(request: Request) -> dict:
        return asdict(store.upsert_budget(BudgetSpec(**await request.json())))

    @app.get("/v1/budgets")
    def list_budgets() -> list[dict]:
        return [asdict(b) for b in store.list_budgets()]

    @app.get("/v1/policies")
    def list_policies() -> list[dict]:
        return [asdict(p) for p in store.list_policy_instances()]
