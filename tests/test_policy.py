"""Deterministic policy evaluation and enforcement tests."""

import copy
import json
from pathlib import Path

import pytest

from runtime.loader import load_agi
from runtime.policy import Policy, PolicyEnforcer, PolicyError, Verdict

ROOT = Path(__file__).resolve().parents[1]
BASE = json.loads((ROOT / "skeleton/policy.json").read_text())


def make_policy(**changes) -> Policy:
    data = copy.deepcopy(BASE)
    for key, value in changes.items():
        data[key] = value
    return Policy.from_dict(data)


def recorder():
    calls = []

    def execute(tool, params):
        calls.append((tool, params))
        return {"success": True, "result": "done"}

    return calls, execute


def test_undeclared_tools_and_bad_params_are_denied():
    policy = make_policy()
    assert policy.evaluate("shell", {"command": "ls"}).verdict is Verdict.DENY
    assert policy.evaluate("file_read", ["not", "an", "object"]).verdict is Verdict.DENY


def test_requires_approval_and_param_triggers_escalate():
    policy = make_policy()
    write = policy.evaluate("file_write", {"path": "a.txt", "content": "x"})
    assert write.verdict is Verdict.ESCALATE
    assert write.scope == "fs:write"
    secret = policy.evaluate("file_read", {"path": "config/.env"})
    assert secret.verdict is Verdict.ESCALATE
    assert secret.trigger_id == "credential_paths"
    assert policy.evaluate("file_read", {"path": "notes.txt"}).verdict is Verdict.ALLOW


def test_wildcard_trigger_and_missing_param_semantics():
    trigger = {
        "id": "everything", "description": "always ask", "severity": "low", "tools": ["*"],
        "params": {"path": "^/"},
    }
    policy = make_policy(escalation={"triggers": [trigger]})
    assert policy.evaluate("file_read", {"path": "/abs"}).verdict is Verdict.ESCALATE
    assert policy.evaluate("file_read", {"path": "rel"}).verdict is Verdict.ALLOW
    assert policy.evaluate("file_read", {}).verdict is Verdict.ALLOW


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"tools": [{"name": "file_read"}, {"name": "file_read"}]}, "duplicate tool"),
        ({"tools": [{"name": "shell"}]}, "shell_enabled is false"),
        ({"tools": [{"name": "web_request"}]}, "no domains"),
        ({"tools": [{"name": "python_eval"}]}, "python_enabled is false"),
        (
            {"escalation": {"triggers": [{
                "id": "bad", "description": "d", "severity": "low", "tools": ["*"],
                "params": {"path": "("},
            }]}},
            "invalid pattern",
        ),
    ],
)
def test_inconsistent_policies_are_rejected(change, message):
    data = copy.deepcopy(BASE)
    data.update(change)
    if "tools" in change:
        data["escalation"] = {"triggers": []}
    with pytest.raises(PolicyError, match=message):
        Policy.from_dict(data)


def test_enforcer_fails_closed_without_approver():
    calls, execute = recorder()
    enforcer = PolicyEnforcer(make_policy(), execute)
    result = enforcer.execute("file_write", {"path": "a.txt", "content": "x"})
    assert result["success"] is False
    assert "no approver" in result["error"]
    assert calls == []
    assert enforcer.audit[-1]["executed"] is False


def test_enforcer_runs_only_explicit_approvals():
    calls, execute = recorder()
    seen = []

    def approver(decision, params):
        seen.append(decision.trigger_id)
        return params["path"] == "ok.txt"

    enforcer = PolicyEnforcer(make_policy(), execute, approver)
    assert enforcer.execute("file_write", {"path": "ok.txt", "content": ""})["success"] is True
    assert enforcer.execute("file_write", {"path": "no.txt", "content": ""})["success"] is False
    assert calls == [("file_write", {"path": "ok.txt", "content": ""})]

    def broken(_decision, _params):
        raise RuntimeError("approval service down")

    assert PolicyEnforcer(make_policy(), execute, broken).execute(
        "file_write", {"path": "ok.txt", "content": ""}
    )["success"] is False
    truthy = PolicyEnforcer(make_policy(), execute, lambda *_: "yes")
    assert truthy.execute("file_write", {"path": "ok.txt", "content": ""})["success"] is False


def test_enforcer_applies_session_budget():
    calls, execute = recorder()
    enforcer = PolicyEnforcer(make_policy(budget={"max_actions_per_session": 2}), execute)
    results = [enforcer.execute("file_read", {"path": "a"}) for _ in range(3)]
    assert [result["success"] for result in results] == [True, True, False]
    assert "budget" in results[2]["error"]
    assert len(calls) == 2


def test_reference_executor_is_built_from_package_policy(tmp_path):
    with load_agi(ROOT / "skeleton") as package:
        executor = package.policy.build_executor(tmp_path)
        enforcer = PolicyEnforcer(package.policy, executor.execute, lambda *_: True)
    assert executor.allowed_tools == {"file_read", "file_write"}
    assert executor.network_policy.allowed_domains == ()
    assert enforcer.execute("file_write", {"path": "a.txt", "content": "hi"})["success"]
    read = enforcer.execute("file_read", {"path": "a.txt"})
    assert read["result"] == "hi"
    assert read["decision"]["verdict"] == "allow"
    assert enforcer.execute("shell", {"command": "pwd"})["success"] is False
