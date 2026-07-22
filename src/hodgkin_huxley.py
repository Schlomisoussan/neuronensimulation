"""Hodgkin-Huxley-Modell (HHM).

Modularer Baustein für Aufgabe 2. Enthält:
  - die biologischen Parameter (Hodgkin & Huxley 1952, Riesenaxon Tintenfisch)
  - die Ratenfunktionen alpha/beta der Gating-Variablen
  - die rechte Seite des DGL-Systems (dU/dt, dn/dt, dm/dt, dh/dt)
  - selbstgeschriebene Löser: explizites Euler-Verfahren und klassisches Runge-Kutta (RK4)

Einheitensystem (bereits konsistent, siehe Projektbeschreibung):
  Spannung in mV, Zeit in ms, Kapazität in uF/cm^2, Leitfähigkeit in mS/cm^2.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Biologische Parameter (Projektbeschreibung, Abschnitt 1.2.2)
# ---------------------------------------------------------------------------
V_POT = -65.0      # Ruhepotential [mV]
C = 1.0            # Membrankapazität [uF/cm^2]

U_K = -77.0        # Gleichgewichtspotential Kalium [mV]
U_NA = 50.0        # Gleichgewichtspotential Natrium [mV]
U_L = -54.387      # Gleichgewichtspotential Leck [mV]

G_K = 36.0         # maximale Leitfähigkeit Kalium [mS/cm^2]
G_NA = 120.0       # maximale Leitfähigkeit Natrium [mS/cm^2]
G_L = 0.3          # Leck-Leitfähigkeit [mS/cm^2]


# ---------------------------------------------------------------------------
# Ratenfunktionen alpha/beta (Projektbeschreibung, Abschnitt 1.2.2)
# ---------------------------------------------------------------------------
def alpha_n(U):
    return -0.01 * (55.0 + U) / (np.exp(-(55.0 + U) / 10.0) - 1.0)


def beta_n(U):
    return 0.125 * np.exp(-(65.0 + U) / 80.0)


def alpha_m(U):
    return -0.1 * (40.0 + U) / (np.exp(-(40.0 + U) / 10.0) - 1.0)


def beta_m(U):
    return 4.0 * np.exp(-(65.0 + U) / 18.0)


def alpha_h(U):
    return 0.07 * np.exp(-(65.0 + U) / 20.0)


def beta_h(U):
    return 1.0 / (np.exp(-(35.0 + U) / 10.0) + 1.0)


def x_infinity(alpha, beta):
    """Gleichgewichtswert einer Gating-Variablen: x_inf = alpha / (alpha + beta)."""
    return alpha / (alpha + beta)


# ---------------------------------------------------------------------------
# Ionenströme und rechte Seite des DGL-Systems
# ---------------------------------------------------------------------------
def ionic_currents(U, n, m, h):
    """Gibt (I_K, I_Na, I_L) zurück (Gl. 5-7)."""
    i_k = G_K * n ** 4 * (U - U_K)
    i_na = G_NA * m ** 3 * h * (U - U_NA)
    i_l = G_L * (U - U_L)
    return i_k, i_na, i_l


def rhs(state, t, I_ext):
    """Rechte Seite des HHM-Systems.

    Parameters
    ----------
    state : array-like [U, n, m, h]
    t     : Zeit [ms] (für zeitabhängigen Strom, z.B. Impuls in Aufgabe 2d)
    I_ext : externe Stromstärke I(t); Zahl oder Funktion I(t)

    Returns
    -------
    np.ndarray [dU/dt, dn/dt, dm/dt, dh/dt]
    """
    U, n, m, h = state
    I = I_ext(t) if callable(I_ext) else I_ext

    i_k, i_na, i_l = ionic_currents(U, n, m, h)
    dU = (I - i_k - i_na - i_l) / C                       # Gl. 1
    dn = alpha_n(U) * (1 - n) - beta_n(U) * n             # Gl. 8
    dm = alpha_m(U) * (1 - m) - beta_m(U) * m             # Gl. 9
    dh = alpha_h(U) * (1 - h) - beta_h(U) * h             # Gl. 10
    return np.array([dU, dn, dm, dh])


def initial_state():
    """Sinnvolle Anfangsbedingungen für ein inaktives Neuron (Aufgabe 1b).

    U startet im Ruhepotential; die Gating-Variablen in ihren
    Gleichgewichtswerten bei U = V_POT.
    """
    U0 = V_POT
    n0 = x_infinity(alpha_n(U0), beta_n(U0))
    m0 = x_infinity(alpha_m(U0), beta_m(U0))
    h0 = x_infinity(alpha_h(U0), beta_h(U0))
    return np.array([U0, n0, m0, h0])


# ---------------------------------------------------------------------------
# Selbstgeschriebene Löser (Aufgabe 2a / 2b)
# ---------------------------------------------------------------------------
def solve_euler(rhs_func, y0, t):
    
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    dt = t[i + 1] - t[i]

    for i in range(len(t) - 1):
        y[i + 1] = y[i] + dt * rhs_func(y[i], t[i])
    return y

t = np.arange(0, 50, 0.01)
f = lambda y, t: rhs(y, t, I_ext=-5.0)
y = solve_euler(f, initial_state(), t)
U = y[:, 0]

plt.plot(t, U)
plt.xlabel("t [ms]"); plt.ylabel("U [mV]")
plt.title("HHM, Euler, I₀ = −5 nA (Ruhezustand)")

def solve_rk4(rhs_func, y0, t):
    """Klassisches Runge-Kutta-Verfahren 4. Ordnung (Aufgabe 2b)."""
    # TODO: k1..k4 berechnen und gewichtet aufsummieren
    raise NotImplementedError
