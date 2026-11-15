"""Gemeinsame Fixtures: Wurzelordner im Importpfad, das Handbeispiel (DEMO-PLAYBOOK §1b)."""

import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def hand():
    """Handbeispiel des Plans: Start (3; 1/10), eta = 1/2, exakte Brüche."""
    return Fraction(3), Fraction(1, 10), Fraction(1, 2)
