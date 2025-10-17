import importlib
import importlib.util
import warnings

MODULE_PATH = "ada_verona"

def _reload_module(path):
    """Helper: import or reload the target module."""
    if path in importlib.sys.modules:
        return importlib.reload(importlib.sys.modules[path])
    return importlib.import_module(path)


def test_warns_when_pyautoattack_missing(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _reload_module(MODULE_PATH)
        
    matches = [str(x.message) for x in w if "PyAutoAttack" in str(x.message)]
    assert matches, f"No PyAutoAttack warning found. Warnings: {[str(x.message) for x in w]}"


def test_no_warn_when_pyautoattack_present(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _reload_module(MODULE_PATH)

    assert not any("PyAutoAttack" in str(x.message) for x in w)


def test_warns_when_autoverify_missing(monkeypatch):
    def fake_find_spec(name):
        if name == "autoverify":
            return None
        return object()
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _reload_module(MODULE_PATH)

    assert any("AutoVerify" in str(x.message) for x in w)


def test_no_warn_when_autoverify_present(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _reload_module(MODULE_PATH)

    assert not any("AutoVerify" in str(x.message) for x in w)