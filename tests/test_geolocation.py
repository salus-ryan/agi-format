"""Tests for the geolocation module."""


from runtime.geolocation import GeoLocation


def test_from_modal_env():
    geo = GeoLocation.from_modal_env(
        model_name="Qwen3-8B", gpu="L4",
        owner_org="salus", owner_workspace="salus",
    )
    assert geo.computational["gpu"] == "L4"
    assert geo.computational["datacenter"] == "modal-us-east"
    assert geo.jurisdictional["primary_jurisdiction"] == "US"
    assert geo.relational["owner_org"] == "salus"
    assert geo.informational["model_name"] == "Qwen3-8B"


def test_temporal_has_expected_fields():
    geo = GeoLocation.from_modal_env(
        model_name="test", gpu="L4", owner_org="test", owner_workspace="test",
    )
    assert "utc_now" in geo.temporal
    assert "unix_timestamp" in geo.temporal
    assert "date" in geo.temporal
    assert "uptime_seconds" in geo.temporal


def test_context_prompt_contains_all_dimensions():
    geo = GeoLocation.from_modal_env(
        model_name="Qwen3-8B", gpu="L4",
        owner_org="salus", owner_workspace="salus",
        available_tools=["shell", "communicate"],
    )
    prompt = geo.context_prompt()
    assert "When" in prompt
    assert "Where" in prompt
    assert "Jurisdiction" in prompt
    assert "Owner" in prompt
    assert "Tools" in prompt


def test_snapshot_returns_all_dimensions():
    geo = GeoLocation.from_modal_env(
        model_name="test", gpu="L4", owner_org="test", owner_workspace="test",
    )
    snap = geo.snapshot()
    assert "computational" in snap
    assert "jurisdictional" in snap
    assert "temporal" in snap
    assert "relational" in snap
    assert "informational" in snap


def test_informational_has_memory_counts():
    geo = GeoLocation.from_modal_env(
        model_name="test", gpu="L4", owner_org="test", owner_workspace="test",
    )
    assert "episodic_memory_count" in geo.informational
    assert "semantic_facts_count" in geo.informational
    assert "procedural_skills_count" in geo.informational
