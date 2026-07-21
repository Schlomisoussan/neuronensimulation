# Neuronensimulation – Hodgkin-Huxley-Modell

Abschlussprojekt *Computational Physics*. Simulation eines Neurons mit dem
Hodgkin-Huxley-Modell (HHM) und Aufbau neuronaler Netze zur Mustererkennung.

**Team:** Schlomi + Partner
**Abgabe:** 27.07.2026

## Aufgabenübersicht

| Teil | Aufgabe | Ort im Repo |
|------|---------|-------------|
| Rechenaufgaben | 1. Analytische Überlegungen (Gl. 11–13 aus 8–10, Anfangsbedingungen) | `report/bericht.tex`, `notebooks/01_grundlagen.ipynb` |
| HHM | 2. Euler, Runge-Kutta, odeint, Stromvariation, Impuls | `src/hodgkin_huxley.py`, `notebooks/02_hhm.ipynb` |
| Eigenes NN | 3. Schachbretterkennung von Grund auf (5 Neuronen) | `src/network.py`, `notebooks/03_eigenes_nn.ipynb` |
| Bibliothek | 4. Keras-Schachbrett | `notebooks/04_keras_schachbrett.ipynb` |
| Bibliothek | 5. MNIST / fashion-MNIST | `notebooks/05_mnist.ipynb` |

## Struktur

```
neuronensimulation/
├── src/          # Modularer Python-Code (die eigentliche Logik)
├── notebooks/    # Experimente, Plots, Diskussion – importieren aus src/
├── report/       # LaTeX-Bericht (3–4 Seiten ohne Figuren)
├── figures/      # Erzeugte Abbildungen
├── requirements.txt
└── .gitignore
```

## Setup

```bash
# Virtuelle Umgebung (empfohlen)
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt

# Notebook-Ausgaben automatisch vor jedem Commit strippen (verhindert Merge-Konflikte)
nbstripout --install
```

## Git-Workflow (zu zweit)

Jedes Mal in dieser Reihenfolge:

1. `git pull`   – zuerst die Änderungen des Partners holen
2. Arbeiten in VS Code
3. `git add .`
4. `git commit -m "beschreibung"`
5. `git push`

Arbeit nach Dateien aufteilen (einer HHM, einer NN), damit keine Konflikte entstehen.
