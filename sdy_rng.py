"""Eigener Zufallsgenerator (SplitMix64): gleiche Zufalls-Seeds geben auf jeder Python-Version und Plattform dieselben Folgen.

Herkunft: kopiert aus `gleitkomma-chaos-demo/gkc_rng.py` (dort aus `restaurant-tischmix-demo/rtm_rng.py` übernommen), mit der vektorisierten Ziehung `uniform_array`.
Python-`random` und `hash()` sind für feste Zufalls-Seeds über Versionen hinweg nicht zugesichert; die Messreihen der README hängen an diesen Folgen.
Die Folge ist zählerbasiert: das i-te Element hängt nur vom Zustand vor der Ziehung und von i ab, `uniform_array(n)` liefert also dieselben
Zahlen wie n Aufrufe von `random()`."""

import numpy as np

MASK = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15


class SplitMix64:
    def __init__(self, seed: int):
        self.state = seed & MASK

    def next_u64(self) -> int:
        self.state = (self.state + GAMMA) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def random(self) -> float:
        """Gleichverteilt in [0, 1) mit 53 Bit (also ein ganzzahliges Vielfaches von 2^-53)."""
        return (self.next_u64() >> 11) / float(1 << 53)

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.random()

    def uniform_array(self, n: int) -> np.ndarray:
        """n Zufallszahlen in [0, 1) auf einmal; Zustand und Werte wie bei n Aufrufen von `random()`."""
        if n <= 0:
            return np.empty(0)
        idx = np.arange(1, n + 1, dtype=np.uint64)
        z = np.uint64(self.state) + idx * np.uint64(GAMMA)  # Überlauf rechnet modulo 2^64, wie MASK im Skalar
        self.state = (self.state + n * GAMMA) & MASK
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)
