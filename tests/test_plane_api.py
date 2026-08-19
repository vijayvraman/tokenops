"""Contract tests: an agent's ``HttpStore`` against the real control-plane app.

The client and the routes are written apart from each other, so these drive the actual
``HttpStore`` methods over the actual FastAPI app rather than asserting on JSON by hand —
a route that disappears (or drifts in shape) fails here instead of silently dropping an
agent's ledger writes, which is exactly how ``make demo`` lost its usage numbers.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from tokenops.control.http_store import HttpStore
from tokenops.control.models import (
    BudgetSpec,
    PolicyInstance,
    RunNotRegisteredError,
    RunRecord,
    RunRegistration,
    Segment,
)
from tokenops.control.store import Store
from tokenops.server.app import create_app


@pytest.fixture
def plane(tmp_path):
    """``(HttpStore, Store)`` — the client an agent gets under TOKENOPS_URL, plus the
    plane's own store to assert against. TestClient is an ``httpx.Client``, so it drops
    straight into HttpStore in place of a network client."""
    store = Store(str(tmp_path / "plane.db"), auto_seed=False)
    http_store = HttpStore("http://plane")
    http_store._client.close()
    http_store._client = TestClient(create_app(store=store))
    yield http_store, store
    store.close()


@pytest.fixture
def plane_routes(tmp_path):
    """``(TestClient, Store)`` for assertions about the routes themselves."""
    store = Store(str(tmp_path / "routes.db"), auto_seed=False)
    with TestClient(create_app(store=store)) as client:
        yield client, store
    store.close()


def test_registration_round_trip(plane):
    http_store, _ = plane
    http_store.register_run(
        RunRegistration(run_id="r1", intent="research", user_dims={"t": "acme"})
    )

    reg = http_store.resolve_run("r1")
    assert reg.intent == "research"
    assert reg.user_dims == {"t": "acme"}
    assert reg.mode.value == "enforce"


def test_resolve_unregistered_run_raises(plane):
    http_store, _ = plane
    with pytest.raises(RunNotRegisteredError):
        http_store.resolve_run("nope")
    assert http_store.get_run_registration("nope") is None


def test_run_record_create_update_and_read(plane):
    http_store, store = plane
    http_store.create_run(
        RunRecord(run_id="r2", agent="research", status="running", task="t", dims={"tier": "pro"})
    )
    http_store.update_run(
        "r2",
        status="completed",
        cost_micros=4321,
        steps=7,
        ended_at=123.0,
        governance_events=[{"policy": "cost_budget", "kind": "halt"}],
    )

    rec = http_store.get_run("r2")
    assert rec is not None
    assert (rec.status, rec.cost_micros, rec.steps) == ("completed", 4321, 7)
    assert rec.dims == {"tier": "pro"}
    assert rec.governance_events == [{"policy": "cost_budget", "kind": "halt"}]

    # The dashboard reads the plane's own store — the write must have landed there.
    assert store.get_run("r2").cost_micros == 4321
    assert [r.run_id for r in http_store.list_runs()] == ["r2"]


def test_get_missing_run_record_returns_none(plane):
    http_store, _ = plane
    assert http_store.get_run("missing") is None


def test_update_run_rejects_unknown_field(plane_routes):
    """``update_run`` builds SQL from field names, so the plane whitelists the columns."""
    client, store = plane_routes
    store.create_run(RunRecord(run_id="r3", agent="research"))

    resp = client.patch("/v1/run-records/r3", json={"steps": 2, "steps=0 WHERE 1=1 --": 1})
    assert resp.status_code == 400
    assert store.get_run("r3").steps == 0  # nothing applied


def test_ledger_spend_is_shared_and_surfaces_as_run_cost(plane):
    http_store, store = plane
    http_store.register_run(RunRegistration(run_id="r4", intent="research"))

    assert http_store.ledger_add_spent("__run_total__", "run:r4", "lifetime", 1500) == 1500
    assert http_store.ledger_add_spent("__run_total__", "run:r4", "lifetime", 500) == 2000
    assert http_store.ledger_get_spent("__run_total__", "run:r4", "lifetime") == 2000

    # This is the dashboard's number: run cost is read back off the shared ledger.
    assert store.get_run("r4").cost_micros == 2000


def test_ledger_inflight_round_trip(plane):
    http_store, _ = plane
    assert http_store.ledger_admit("agent:research") == 1
    assert http_store.ledger_admit("agent:research") == 2
    assert http_store.ledger_inflight("agent:research") == 2
    assert http_store.ledger_complete("agent:research") == 1


def test_ledger_halt_round_trip(plane):
    http_store, _ = plane
    assert http_store.ledger_is_halted("r5") is False
    assert http_store.ledger_halt_reason("r5") is None

    http_store.ledger_mark_halted("r5", "budget exhausted")
    assert http_store.ledger_is_halted("r5") is True
    assert http_store.ledger_halt_reason("r5") == "budget exhausted"

    http_store.ledger_clear_halt("r5")
    assert http_store.ledger_is_halted("r5") is False


def test_governance_config_reaches_the_agent(plane):
    http_store, store = plane
    http_store.upsert_budget(BudgetSpec(id="run_llm_cap", limit_micros=2_000_000, dimension="run"))
    http_store.upsert_segment(Segment(id="seg_run", name="per run", dimension="run"))
    # Policy instances are authored in Admin (plane-side); the agent only reads them back.
    store.upsert_policy_instance(
        PolicyInstance(id="p1", template="cost_budget", budget_id="run_llm_cap", agent="research")
    )
    store.upsert_policy_instance(
        PolicyInstance(id="p2", template="step_cap", params={"max_steps": 9})
    )

    cfg = http_store.governance_config_for("research")["governance"]
    assert cfg["policies"]["cost_budget"]["budget"] == "run_llm_cap"
    assert cfg["policies"]["step_cap"]["max_steps"] == 9
    assert [b["id"] for b in cfg["budgets"]] == ["run_llm_cap"]

    # An unscoped agent sees only the global policy.
    assert "cost_budget" not in http_store.governance_config_for("")["governance"]["policies"]

    assert [b.id for b in http_store.list_budgets()] == ["run_llm_cap"]
    assert [s.id for s in http_store.list_segments()] == ["seg_run"]
    assert {p.id for p in http_store.list_policy_instances()} == {"p1", "p2"}
