"""Registry + decorator tests."""
from __future__ import annotations

import pytest

from vdg.core.registry import (
    REGISTRY,
    Registrable,
    Registry,
    register_device,
    register_energy_model,
    register_load,
    register_skill,
)

def test_register_get_all():
    reg = Registry()

    class Foo:
        pass

    reg.register("device", "foo", Foo)
    assert reg.get("device", "foo") is Foo
    assert reg.get("device", "missing") is None
    assert reg.all("device") == {"foo": Foo}
    assert "device" in reg
    assert reg.names("device") == ["foo"]


def test_register_unknown_kind_raises():
    reg = Registry()

    class Foo:
        pass

    with pytest.raises(ValueError):
        reg.register("bogus", "foo", Foo)


def test_register_empty_name_raises():
    reg = Registry()

    class Foo:
        pass

    with pytest.raises(ValueError):
        reg.register("device", "", Foo)


def test_all_returns_copy():
    reg = Registry()

    class Foo:
        pass

    reg.register("device", "foo", Foo)
    snapshot = reg.all("device")
    snapshot["injected"] = Foo
    assert "injected" not in reg.all("device")


def test_decorator_named_usage():
    @register_device("test_device_named")
    class Dev(Registrable):
        pass

    assert REGISTRY.get("device", "test_device_named") is Dev
    assert Dev.__registry_kind__ == "device"
    assert Dev.__registry_name__ == "test_device_named"
    assert Dev.registry_name() == "test_device_named"


def test_decorator_bare_usage():
    @register_device
    class TestDeviceBare(Registrable):
        pass

    assert REGISTRY.get("device", "TestDeviceBare") is TestDeviceBare
    assert TestDeviceBare.__registry_name__ == "TestDeviceBare"


def test_all_decorator_kinds():
    @register_load("test_load_x")
    class L(Registrable):
        pass

    @register_skill("test_skill_x")
    class S(Registrable):
        pass

    @register_energy_model("test_energy_x")
    class E(Registrable):
        pass

    assert REGISTRY.get("load", "test_load_x") is L
    assert REGISTRY.get("skill", "test_skill_x") is S
    assert REGISTRY.get("energy_model", "test_energy_x") is E


def test_registrable_metadata_defaults():
    class Plain(Registrable):
        pass

    assert Plain.__registry_kind__ == ""
    assert Plain.registry_name() == "Plain"


def test_discover_does_not_crash():
    # discover() imports subpackages; foundation ships them empty so this is a
    # no-op that must not raise.
    REGISTRY.discover()


def test_global_registry_has_builtin_energy_models():
    # Importing vdg.core.energy_model registers tdp + measured.
    import vdg  # noqa: F401
    names = REGISTRY.names("energy_model")
    assert "tdp" in names
    assert "measured" in names


def test_items_iterator():
    REGISTRY.discover()
    triples = list(REGISTRY.items())
    assert ("energy_model", "tdp", REGISTRY.get("energy_model", "tdp")) in triples
