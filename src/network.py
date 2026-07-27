"""Neuronales Netz von Grund auf – Schachbretterkennung (Aufgabe 3).

Idee: Vier Eingangsfelder (Schachbrett 2x2) werden über Wichtungsfaktoren
w_ij auf Neuronen weitergegeben. Die externe Stromstärke eines Neurons ergibt
sich aus der gewichteten Summe der vorgeschalteten Spannungen (Gl. 14):

    I_j = sum_{i<j} w_ij * U_i

Aktivität wird über das Hodgkin-Huxley-Modell bestimmt. Wichtig: die minimale
Stromstärke darf I_0 = -5 nA nicht unterschreiten (manuell als Minimum setzen).

Dieses Modul enthält das Grundgerüst; die Lernregel (Aufgabe 3d) und die
optimalen Startgewichte (Aufgabe 3a) sind noch zu implementieren.
"""

import numpy as np
from scipy.integrate import odeint
from . import hodgkin_huxley as hh

I_0 = -5.0  # minimale/Grundstromstärke [nA] – darf nicht unterschritten werden
I_MAX = 10 # Stromstärke mit der ein Input-Neuron bei Input "1" feuert [nA]
SPIKE_SCHWELLE = 0.0  # Spannung, ab der ein Neuron feuert [mV]

def clamp_current(I):
    """Stellt sicher, dass die Stromstärke I_0 nicht unterschreitet."""
    return np.maximum(I, I_0)

def solve_hodgkin_huxley(I_ext, t):
    """Löst das HHM eines einzelnen Neurons und gibt U(t) zurück.
    I_ext: Zahl (konstanter Strom) oder Funktion I(t)."""
    y = odeint(hh.rhs, hh.initial_state(), t, args=(I_ext,))
    return y[:, 0]        # Spalte 0 = Spannung U


def predict(weights, pattern, t=None):
    if t is None:
        t = np.arange(0, 50, 0.01)

    # 1. Eingangsströme der 4 Input-Neuronen festlegen
    #    schwarzes Feld (1) -> I_MAX,  weißes Feld (0) -> I_0
    I_in = []
    for i in range(4):
        if pattern[i] == 1:
            I_in.append(I_MAX)
        else:
            I_in.append(I_0)

    # 2. Für jedes Input-Neuron das HHM lösen und die Spannung merken
    U = []
    for i in range(4):
        U.append(solve_hodgkin_huxley(I_in[i], t))    # Spalte 0 ist die Spannung
    U = np.array(U)            # jetzt Form (4, Anzahl Zeitpunkte)

    # 3. Strom ins Output-Neuron nach Gleichung (14): Summe w_i * U_i
    I_out = np.zeros(len(t))
    for i in range(4):
        I_out = I_out + weights[i] * U[i]

    # Minimum I_0 nicht unterschreiten
    I_out = clamp_current(I_out)

    # 4. Output-Neuron mit diesem zeitabhängigen Strom lösen
    def I_out_funktion(zeit):
        return np.interp(zeit, t, I_out)

    U_out = solve_hodgkin_huxley(I_out_funktion, t)

    # 5. Prüfen, ob das Output-Neuron gefeuert hat
    hat_gefeuert = False
    for wert in U_out:
        if wert > SPIKE_SCHWELLE:
            hat_gefeuert = True

    return hat_gefeuert


def train(patterns, targets, learning_rate=0.1, epochs=100, seed=None, t=None):
    """Lernalgorithmus – Aufgabe 3d.

    patterns : Liste von 0/1-Mustern (je Länge 4)
    targets  : zugehörige Sollwerte (1 = Schachbrett, 0 = kein Schachbrett)
    learning_rate : Schrittweite der Gewichtsanpassung
    epochs   : Anzahl der Durchläufe über die Musterabfolge
    seed     : für reproduzierbare Startgewichte
    t        : Zeitgitter für predict (gröber = schneller); Standard 0.05 ms

    Rückgabe:
        weights           : gelernte Gewichte
        fehler_verlauf    : Anzahl Fehler pro Durchgang (für 3e)
        gewichte_verlauf  : Gewichte nach jedem Durchgang, Form (epochs, 4) (für 3e)
    """
    # Gröberes Zeitgitter fürs Training -> viel schneller (predict wird sehr oft aufgerufen)
    if t is None:
        t = np.arange(0, 50, 0.05)

    # Startgewichte zufällig in (0, 1], positiv (Leitfähigkeit)
    rng = np.random.default_rng(seed)
    weights = rng.uniform(0.0, 1.0, size=4)

    patterns = [np.array(p) for p in patterns]
    fehler_verlauf = []
    gewichte_verlauf = []

    for epoch in range(epochs):
        fehler_summe = 0
        for p, ziel in zip(patterns, targets):
            vorhersage = int(predict(weights, p, t=t))   # 1 = feuert, 0 = feuert nicht
            fehler = ziel - vorhersage                    # -1, 0 oder +1
            if fehler != 0:
                # Gewichte der aktiven Felder in Fehlerrichtung anpassen
                weights = weights + learning_rate * fehler * p
                # Leitfähigkeit darf nicht negativ werden
                weights = np.maximum(weights, 0.0)
                fehler_summe += 1
        fehler_verlauf.append(fehler_summe)
        gewichte_verlauf.append(weights.copy())

    return weights, np.array(fehler_verlauf), np.array(gewichte_verlauf)
