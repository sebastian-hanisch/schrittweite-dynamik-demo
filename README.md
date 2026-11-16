# Schrittweite als Dynamik: Gradientenabstieg jenseits von 2/η, Periodenverdopplung und Newton-Fraktal – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-schrittweite-dynamik-demo.streamlit.app/)**

Stück 5 der Linie **„Dynamische Systeme“** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Kante: **fix** zu
[Stück 1 (Iterieren und Fixpunkt)](https://github.com/sebastian-hanisch/fixpunkt-iteration-demo). Stück 1 zeigt auf einer Quadratik: Die Iteration x → (1 − ηλ)x konvergiert genau für η < 2/λ, darüber wächst sie;
die Schwelle steht scharf im [gradient-descent-demo](https://github.com/sebastian-hanisch/gradient-descent-demo). Die Schwäche: Auf einer Quadratik bedeutet „darüber“ immer Divergenz. Bei einer nichtlinearen
Funktion bleibt die Bahn jenseits der Schwelle oft beschränkt (im Spielmodell als Zyklus oder chaotisch). Was dann geschieht, ist dieses Stück. Dazu das Newton-Verfahren in der komplexen Ebene
([newton-verfahren-demo](https://github.com/sebastian-hanisch/newton-verfahren-demo) zeigt nur den eindimensionalen Fall): Einzugsgebiete und die fraktale Grenze. Verwandt: Schwellen der Lernschrittweite in Spielen in
[Stück 9](https://github.com/sebastian-hanisch/lernen-spiele-chaos-demo), die Periodenverdopplung der logistischen Abbildung in [Stück 2](https://github.com/sebastian-hanisch/logistische-abbildung-demo).

## Warum dieses Problem

Eine feste Schrittweite η ist eine Entscheidung über Stabilität: Ein Minimum mit Schärfe S (größter Eigenwert der Hesse-Matrix) ist für den Gradientenabstieg (GD) stabil genau dann, wenn η·S < 2.
**Kernfrage:** Wohin läuft GD in einer nichtlinearen Landschaft, wenn η über 2/S liegt, und was geschieht, wenn kein Minimum mehr stabil ist?

## Modell

**Spielmodell, von Hand gewählt, keine Quelle:** f(x, y) = ½(x·y − 1)². Die Minima sind die Kurve x·y = 1 mit der Schärfe S = x² + y² ≥ 2. Der GD-Schritt ist (x, y) → (x − η·y·c, y − η·x·c) mit c = x·y − 1;
daraus folgt x′² − y′² = (x² − y²)(1 − η²c²) (exakt). Auf der Diagonalen x = y = t ist GD die Abbildung t → (1 + η)t − ηt³. Die Formeln stehen im 📐-Expander der App.

## Methodik

- **Raster der Starts:** 30 × 30 Starts im Quadrat [−2,5; 2,5]² mit dem Versatz (0,0137; −0,0291) (kein Startwert auf einem Rasterpunkt wie 0,5), 6000 Schritte; „konvergiert“ heißt Verlust unter 10⁻¹², „divergiert“ Betrag über 10⁸.
- **Perioden** der beschränkten Bahnen: Zustandsvergleich bis Periode 128; η·λ_max(Hesse) als exaktes Zeitmittel über eine Periode (bei fehlender Periode Mittel über 4000 Schritte, gekennzeichnet).
- **Diagonale:** Bifurkationsdiagramm über 680 Schrittweiten, Lyapunov-Exponent; „Chaos“ nur ohne Periode und bei einem Exponenten über 0,003. Die Verdopplungsschwellen als Nullstellen des Multiplikators = −1 (Brent), η = √5 − 1 zum Vergleich in geschlossener Form.
- **Newton-Fraktal:** Anteile der Einzugsgebiete in der Kreisscheibe |z| ≤ 1,5 mit Zufallspunkten (SplitMix64); die Dimension der Grenze mit **zwei Verfahren**, Unsicherheitsexponent und Box-Zählung, **vorgerechnet** (`tools/vorrechnen.py dimension`, 66 s, Datei `sdy_vorgerechnet_dimension.json`).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen gelten im Spielmodell f = ½(x·y − 1)² und für das genannte Raster bzw. die genannten Zufalls-Seeds; die Zählungen des Rasters sind deterministisch. Test: `tests/test_claims.py` (Zählungen exakt, Newton-Anteile und Dimensionen als Band bzw. mit Rundung).

| Frage | Befund | Test |
|---|---|---|
| Zählung der 900 Starts: konvergiert / divergiert / beschränkt ohne Konvergenz; größte Schärfe der erreichten Minima (2/η) | η = 0,2: 900 / 0 / 0, 6,606 (10,000); 0,49: 780 / 120 / 0, 4,011 (4,082); 0,5: 767 / 133 / 0, 3,912 (4,000); 0,51: 764 / 136 / 0, 3,677 (3,922); 0,8: 485 / 415 / 0, 2,404 (2,500); 0,95: 393 / 507 / 0, 2,084 (2,105); 0,99: 372 / 528 / 0, 2,017 (2,020) | `test_counts_of_the_900_starts` |
| Ab η = 1 | η = 1,0: 0 / 534 / 366; 1,01: 0 / 542 / 358; 1,05: 0 / 558 / 342; 1,3: 0 / 654 / 246; 1,5: 0 / 715 / 185; 1,9: 0 / 833 / 67; 1,99: 0 / 877 / 23; 2,0 und 2,01: 0 / 900 / 0. Deutung bei η = 1,0: die 366 beschränkten Bahnen sind keine Zyklen, sondern laufen langsam gegen (1; 1) bzw. (−1; −1) (Verlust nach 6000 Schritten bei allen unter 5·10⁻⁵ und weiter fallend, in einer Stichprobe 8·10⁻⁸ nach 3·10⁶ Schritten; auf der Diagonalen t = 1,00025 nach 10⁶ Schritten); ab η = 1,01 sind es Zyklen oder Chaos | `test_counts_of_the_900_starts`, `test_at_eta_one_the_bounded_orbits_creep_towards_the_minimum` |
| Erreichte Minima mit S ≥ 2/η | in keiner der 9 Schrittweiten 0,49, 0,51, 0,99, 1,01, 1,99, 2,01 und 0,5, 1, 2 ein erreichtes Minimum mit S ≥ 2/η (für η ≥ 1 ist das trivial, weil kein Start nach dem Kriterium konvergiert); die Zweierpotenzen 0,5 und 1 weichen nicht von den Nachbarwerten ab, 2 gleicht der Seite 2,01 (die 23 beschränkten Bahnen von 1,99 fehlen dort wie bei 2,01) | `test_no_reached_minimum_is_sharper_than_2_over_eta_in_nine_step_sizes` |
| Gegenprobe mit 20 000 Zufallsstarts in [−5, 5]² (Zufalls-Seed 4242) | η = 0,3: 8162 konvergieren, 0,7: 3367, 0,9: 2474, nie ein Minimum mit S ≥ 2/η; η = 1,01, 1,2, 1,5: keiner konvergiert, 2145, 1671 bzw. 1101 bleiben beschränkt | `test_random_starts_20000_…` |
| Dieselbe Gegenprobe mit 100 000 Starts (Messung dieser README, nicht im Test, 17 s) | η = 0,3: 41 527 konvergieren, 0,7: 16 970, 0,9: 12 515, nie S ≥ 2/η; η = 1,01, 1,2, 1,5: keiner konvergiert (10 808, 8428, 5555 beschränkt) | – |
| Beschränkte Bahnen jenseits der Schwelle: Perioden | η = 1,01: 358 Bahnen, alle auf der Diagonalen, alle Periode 2; 1,05: 342, Periode 2; 1,3: 246, Periode 16; keine mit positivem Lyapunov-Exponenten | `test_bounded_orbits_are_cycles_on_the_diagonal_…` |
| η = 1,5 | die 185 beschränkten Bahnen (6000 Schritte) liegen alle auf der Diagonalen, haben keine Periode bis 128 und einen positiven Lyapunov-Exponenten (Median +0,49): chaotisch | `test_eta_1_5_bounded_orbits_are_all_chaotic_…` |
| η·λ_max(Hesse) entlang der Bahn (Zeitmittel, Median über die Bahnen; Wertebereich) | η = 1,01: 2,005 (1,705 bis 2,305); 1,05: 2,025 (1,350 bis 2,700); 1,3: 2,369 (0,792 bis 4,107); Anteil der Zeit über 2: Median 0,5 | dort |
| Verdopplungsschwellen der Diagonalen | η₁ = 1 und η₂ = √5 − 1 = 1,23607 (Multiplikator 9 − 2(1 + η)² gleich 1 bzw. −1; hier exakt hergeleitet), η₃ = 1,28803, η₄ = 1,29923 (numerisch); Verhältnisse 4,54 und 4,64 | `test_doubling_thresholds_and_ratios` |
| Gitter η = 0,9012 + 0,0125·k | Periode 1 bis 0,9887, Periode 2 von 1,0012 bis 1,2262, Periode 4 von 1,2387 bis 1,2762, Periode 8 bei 1,2887, Periode 16 bei 1,3012, Chaos von 1,3137 bis 1,4387, Periode 3 bei 1,4512, Periode 6 bei 1,4637 und 1,7012, Chaos von 1,4762 bis 1,6887 und von 1,7137 bis 1,7387; zwei Startwerte (1,2 und 0,3) geben überall dieselbe Art | `test_diagonal_grid_periods_and_chaos`, `test_two_start_values_…` |
| Newton, Anteile der Einzugsgebiete in der Kreisscheibe mit Radius 1,5 | 1 Mio Zufallspunkte (Zufalls-Seed 11): 33,39 %, 33,38 %, 33,24 %; jeder im Band 0,003 um 1/3 (Symmetrie); im Quadrat [−1,5; 1,5]² dagegen 34,5 %, 32,7 %, 32,7 % | `test_newton_shares_in_the_disc_and_symmetry`, `tests/test_vorgerechnet.py` |
| Newton, Fehlerfortpflanzung | e′ ≈ e²: z₀ = 1,01 gibt 9,87·10⁻⁵ und 9,74·10⁻⁹ | `test_newton_error_propagation_near_the_root` |
| Dimension der Grenze (vorgerechnet, 1,5 Mio Punkte, Zufalls-Seed 7001; Gitter 4001 × 4001, 272 386 Randpixel) | Unsicherheitsexponent 0,553, also D = 1,447; Box-Zählung D = 1,454 (Boxgrößen 4 bis 64 Pixel) und 1,477 (2 bis 16 Pixel); die Verfahren weichen um bis zu 0,03 voneinander ab: „1,45 bis 1,48“; keine Literaturgegenprobe | `tests/test_vorgerechnet.py` |
| Start (3; 0,1) | η = 0,3 konvergiert (Schärfe 2,7074 gegen 2/η = 6,667), η = 0,49 und η = 0,5 divergieren | `test_the_handbook_start_converges_at_0_3_…` |

**Aussagen mit ihren Voraussetzungen.** „Kein erreichtes Minimum hat S ≥ 2/η“ ist eine Zählung über die 9 Schrittweiten und das 900er Raster, dazu die Zufallsstarts (Starts mit |x·y − 1| < 10⁻⁹ ausgenommen, sie sind Fixpunkte); keine Aussage über alle
Startpunkte. „Für η > 1 konvergiert kein Start“ folgt zusätzlich aus 2/η < 2 ≤ S; bei η = 1 ist (1; 1) mit S = 2 = 2/η nur grenzstabil und wird von den 366 beschränkten Bahnen langsam angelaufen (siehe die Zählung oben). Dass GD hier im Mittel zu flacheren Minima läuft, erklärt die Identität x′² − y′² = (x² − y²)(1 − η²c²) plausibel, aber nicht bewiesen: Auf der Kurve x·y = 1 gilt S = √(4 + D²), und D
wird je Schritt mit 1 − η²c² multipliziert, schrumpft also für |η·c| < 1 und im Mittel stärker bei größerem η (der Faktor hängt vom Pfad ab); das gilt für dieses Modell. Die Größe η·λ_max(Hesse) liegt im Zeitmittel bei 2,005, 2,025 und 2,369: „um 2“ trifft bei η = 1,01 und 1,05 zu, bei η = 1,3 schwankt sie von 0,79 bis 4,11.

## Befunde und Korrekturen gegenüber dem Plan

- **Alle Zählungen der Vormessung reproduzieren sich** (Raster, Perioden, Schwellen, Gitter, Box-Zählung mit denselben 272 386 Randpixeln); die Messlogik ist neu geschrieben. Der Unsicherheitsexponent weicht mit dem Zufalls-Seed 7001 und dem SplitMix64-Generator um 0,01 ab (D = 1,447 gegen 1,440 und 1,446 in der Vormessung).
- **Das Handbeispiel konvergiert nicht bei jeder Schrittweite.** Der erste Schritt des Plans (607/200, 23/20) gehört zu η = ½; die ganze Bahn ab (3; 0,1) divergiert aber bei η = 0,5 und bei η = 0,49 und konvergiert bei η = 0,3. Die App startet deshalb mit η = 0,3 (keine Zweierpotenz); die Plan-Zahlen 607/200 und 23/20 stehen als Test mit exakten Brüchen.
- **η = 1,5** (Preset „Chaos“) ist nicht im Plan: 185 beschränkte Bahnen, alle chaotisch auf der Diagonalen (Median des Lyapunov-Exponenten +0,49), 715 divergieren. Der Satz des Plans „alle beschränkten Bahnen sind Zyklen“ gilt für 1,01, 1,05 und 1,3, **nicht** für 1,5.
- **Zeitmittel von η·λ_max:** exakt über eine Periode statt über 20 000 Schritte; die Werte der Vormessung (2,005, 2,025, 2,369 und die Wertebereiche) ergeben sich genau.
- **Grenzen des Plans nachgeholt:** Die Gegenprobe mit 100 000 Zufallsstarts ist gemessen (siehe Tabelle), im Test mit 20 000. Plan Z5 (d): Die READMEs von `quasi-newton-demo` und `stochastische-gradientenverfahren-demo` enthalten keine Schrittweiten-Dynamik jenseits der Schwelle (Suche nach Schärfe, Zyklus, Periodenverdopplung, Chaos, Fraktal ohne Treffer).
- **Plan abgewichen:** Die Anteile der Newton-Gebiete werden in der App mit 200 000 Punkten live gemessen (0,5 s); die Dimension steht nur vorgerechnet in der Datei, die Box-Zählung läuft in der App nicht.

## Ehrliche Grenzen

- **Ein selbst gewähltes Modell, ein Startquadrat, 900 Starts, 6000 Schritte.** Sehr langsam konvergierende Starts würden als „beschränkt“ gezählt; bei η = 0,99 gab es keinen, bei η = 1,0 sind es die 366 beschränkten. Die Zählungen gelten für dieses Raster.
- **Kein Netz, keine stochastischen Gradienten.** Cohen u. a. (ICLR 2021, im Wortlaut gelesen) beschreiben für Netze, dass die Schärfe am Rand von 2/η bleibt („hovers right at, or just above“); das ist hier nicht gemessen. Damian, Nichani, Lee (ICLR 2023, im Wortlaut gelesen) nennen die Dynamik „far from being chaotic“, bei mehreren instabilen Eigenwerten aber „significantly more chaotic“; das Spielmodell hat auf der Diagonalen nur eine instabile Richtung. Ob die Messung die Aussagen der Quellen für Netze stützt, ist nicht geprüft und wird nicht behauptet. Kalra, He, Barkeshli (arXiv 2311.02076) sind nicht gelesen und werden nicht als belegt zitiert.
- **Chaos nur auf der Diagonalen,** Gitter 0,0125, Schwelle 0,003 für den Lyapunov-Exponenten; die Schwellen 1 und √5 − 1 sind exakt, η₃ und η₄ numerisch. Ob die Folge der Verdopplungsschwellen dieselbe Konstante wie die logistische Abbildung hat (4,67), ist nicht geprüft: die zwei gemessenen Verhältnisse sind 4,54 und 4,64.
- **Newton:** nur z³ = 1, keine Literaturgegenprobe der Dimension; die Box-Zählung hängt von Gitter und Boxbereich ab (die zwei Bereiche geben 1,454 und 1,477).
- **Lyapunov-Exponent des GD in der Ebene** ist ein Benettin-Schätzer über 4000 Schritte der beschränkten Bahnen, kein Beweis.

## Verifikation

- **Kern:** Handbeispiel mit exakten Brüchen (607/200, 23/20 und der Faktor 351/400 für η = ½; 3021/1000, 73/100, 9559/10000 für η = 0,3; 3/2 → −3/8 → −357/512 → −141671523/134217728 auf der Diagonalen; Zweierzyklus bei η = 1,2 und η = √5 − 1; N(11/10) = 1831/1815).
- **Orakel:** `tests/test_oracle_dynamik.py`: Jacobi-Matrix per Differenzenquotient gegen die Hesse-Formel, Eigenwerte gegen `numpy.linalg.eigvalsh`, Zweierzyklen gegen die Wurzeln von g(g(t)) − t (`numpy.roots`), Verdopplungsschwellen gegen √5 − 1 und die Periode auf beiden Seiten, Newton-Wurzeln gegen `numpy.roots`, vektorisiertes Raster gegen eine skalare Schleife, Box-Zählung an Gerade (D = 1) und gefülltem Quadrat (D = 2).
- **Eigenschaften:** Vertauschen von x und y, Punktspiegelung, invariante Diagonale, S = √(4 + D²), Reihenfolge der Starts, gleicher Zufalls-Seed gleiche Ausgabe, Schwelle 2/S an 20 Minima.
- **Randfälle:** η = 0, Start (0, 0), Start auf der Achse, auf der Kurve x·y = 1, η > 2, Start nahe der Hyperbel, Newton-Start z = 0 und auf den Symmetriestrahlen.
- **Verdikte** als reine Funktionen (`tests/test_verdicts.py`), je Preset die erwartete Zeile (`test_every_preset_gives_the_expected_verdict_line`). Keine Schrittweite und kein Startwert eines Presets ist eine Zweierpotenz (`test_no_preset_uses_a_power_of_two_…`).
- **Oberfläche:** AppTest-Rauchtest (jeder Regler an beiden Rändern), Layout (Schnellstart in der Seitenleiste, fester Kopf, Hervorhebung), Footer, Laufzeit-Importe.
- **Zählung:** 123 Tests (`pytest --collect-only -q tests`).

Rechenzeit der App: Erstaufruf rund 3 s (Raster, Bifurkationsdiagramm, Gitter der Diagonalen, 200 000 Newton-Punkte), jede Änderung der Schrittweite oder des Ausschnitts unter einer Sekunde (AppTest, nach dem Erstaufruf).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: fünf Experimente, Kernfrage, Annahmen, Formeln |
| `sdy_core.py` | Kern: GD-Schritt, Raster, Perioden, η·λ_max, Diagonalabbildung, Schwellen, Newton |
| `sdy_fractal.py` | Newton-Fraktal: Anteile, Unsicherheitsexponent, Box-Zählung, Bild |
| `sdy_algorithm.py` | Messroutinen der Experimente |
| `sdy_rng.py` | Zufallsgenerator SplitMix64 (aus `gleitkomma-chaos-demo` kopiert) |
| `sdy_constants.py` | Regler-Grenzen, Raster, Zufalls-Seeds, Farben, Presets |
| `sdy_evaluation.py` | Verdikte (reine Funktionen), Zahlenformate, Laden der vorgerechneten Datei |
| `sdy_layout.py` | Fester Kopf (Stylesheet) |
| `sdy_presets.py` | Permalink, Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `sdy_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `sdy_vorgerechnet_dimension.json` | vorgerechnete Dimensionsmessung |
| `tools/vorrechnen.py` | erzeugt die Datei (`dimension`, 66 s) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Handbeispiel, Orakel, Eigenschaften, Claims, Vorrechnung, Verdikte, Randfälle, Presets, Layout, AppTest-Smoke-Test, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- Edge of Stability im echten Netz (kleines MLP): nicht gemessen; das Stück behauptet nichts über Netze.
- Stochastische Gradienten, andere Modelle und Chaos außerhalb der Diagonalen.
- Box-Zählung und Unsicherheitsexponent live in der App (nur vorgerechnet).
- Andere Fraktale (nur z³ = 1) und ein Literaturvergleich der Dimension.
- PDF-Export.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, pandas, SciPy und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Dynamische Systeme: Ruhepunkt, Kreislauf, Chaos](https://sebastianhanisch.net/konzepte-dynamische-systeme.html).
