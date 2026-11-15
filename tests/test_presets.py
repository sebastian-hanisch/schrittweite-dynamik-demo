"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, jedes Preset verschieden (sonst wäre die Hervorhebung mehrdeutig), keine Zweierpotenz als Schrittweite."""

import math

import sdy_constants as C
import sdy_evaluation as ev
import sdy_presets as P

FIRST = "Handbeispiel (3; 0,1)"


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_valid_for_their_controls():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec, value = P.SETTING_SPECS[state_key], preset[key]
            if spec.options is not None:
                assert value in spec.options, (name, key, value)
            else:
                assert spec.lo <= value <= spec.hi, (name, key, value)


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[FIRST]) == ev.Settings()
    assert {k: spec.default for k, spec in P.SETTING_SPECS.items()} == {P.PRESET_KEYS[k]: v for k, v in C.PRESETS[FIRST].items()}


def test_every_preset_differs_from_every_other():
    values = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(values)) == len(values)


def test_no_preset_uses_a_power_of_two_as_step_size_or_start_value():
    """Zweierpotenz-Regel (Rückkopplung V10/V11): keine Voreinstellung 0,5, 1, 2, 0,25 … als Schrittweite oder Startwert."""
    def power_of_two(v):
        return v != 0 and math.frexp(abs(v))[0] == 0.5
    for name, preset in C.PRESETS.items():
        assert not power_of_two(preset["eta"]) and not power_of_two(preset["x"]) and not power_of_two(preset["y"]), name
    assert not power_of_two(C.GRID_OFFSET[0]) and C.ETA_STEP == 0.01


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_permalink_clamps_ranges_and_snaps_options():
    assert P.clamp(P.SETTING_SPECS["eta"], 9.0) == 2.1 and P.clamp(P.SETTING_SPECS["eta"], -1.0) == 0.05
    assert P.clamp(P.SETTING_SPECS["zoom"], 0.2) in (0.1, 0.3) and P.clamp(P.SETTING_SPECS["zoom"], 1.5) == 1.5
    assert P.clamp(P.SETTING_SPECS["cre"], 7.0) == 2.0 and P.clamp(P.SETTING_SPECS["seed_input"], -4) == 0
