"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Button und fährt jeden Regler an seine
Grenzen bzw. an die Ränder seiner Auswahl; geprüft wird, dass kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= können zufällig identischen Inhalt rendern und kollidieren) und `StreamlitAPIException` bei Reglern mit berechneten Grenzen."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import sdy_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "▶"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen
REGLER_ARTEN = ("slider", "select_slider", "number_input")


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(key)


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


@pytest.mark.parametrize("key", [k for k in P.SETTING_SPECS])
def test_every_control_at_both_ends_does_not_raise(key):
    spec = P.SETTING_SPECS[key]
    ends = (spec.options[0], spec.options[-1]) if spec.options is not None else (spec.lo, spec.hi)
    for value in ends:
        at = _fresh_app()
        _regler(at, key).set_value(value).run(timeout=TIMEOUT)
        assert not at.exception, (key, value, [str(e) for e in at.exception])


def test_no_error_elements_and_the_metrics_of_the_first_experiment_on_the_default_page():
    at = _fresh_app()
    assert not at.error
    assert at.markdown  # die Seite hat Text
    assert [m.label for m in at.metric][:3] == ["c = x·y − 1", "x′", "y′"]
