"""Generate the static figures used in the PDF notes.

    python notes/figures/make_figures.py

Figure: ridge_gamma3.pdf (Lecture 2, Marchenko--Pastur spectrum and ridge risk).
The risk curve is computed from the asymptotic formula in the notes,
R(lambda) = r^2 int lambda^2/(t+lambda)^2 dmu + sigma^2 gamma int t/(t+lambda)^2 dmu.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

HERE = Path(__file__).resolve().parent

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def mp_edges(gamma):
    return (1 - np.sqrt(gamma)) ** 2, (1 + np.sqrt(gamma)) ** 2


def mp_density(t, gamma):
    lm, lp = mp_edges(gamma)
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    inside = (t > lm) & (t < lp)
    out[inside] = np.sqrt((lp - t[inside]) * (t[inside] - lm)) / (2 * np.pi * gamma * t[inside])
    return out


def ridge_risk(lam, gamma, r2=1.0, s2=1.0):
    """Asymptotic excess risk of ridge regression; returns (risk, bias, variance)."""
    lm, lp = mp_edges(gamma)
    atom = max(1 - 1 / gamma, 0.0)
    rho = lambda t: float(mp_density(np.array([t]), gamma)[0])
    bias = integrate.quad(lambda t: lam**2 / (t + lam) ** 2 * rho(t), lm, lp)[0] + atom
    var = integrate.quad(lambda t: t / (t + lam) ** 2 * rho(t), lm, lp)[0]
    return r2 * bias + s2 * gamma * var, r2 * bias, s2 * gamma * var


def ridge_gamma3():
    gamma, r2, s2 = 3.0, 1.0, 1.0
    lm, lp = mp_edges(gamma)
    lam_star = s2 * gamma / r2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)

    t = np.linspace(lm, lp, 600)
    ax1.plot(t, mp_density(t, gamma), color="k", lw=1.4)
    for edge in (lm, lp):
        ax1.axvline(edge, color="k", ls="--", lw=0.8)
    ax1.text(2.3, 0.097, r"atom $\mu_\gamma(\{0\})=2/3$")
    ax1.set(xlim=(0, 8), ylim=(0, 0.11), xlabel=r"eigenvalue $t$", ylabel=r"$\rho_\gamma(t)$")
    ax1.set_title(r"Marchenko–Pastur spectrum, $\gamma=3$", fontsize=10)

    lams = np.linspace(0.05, 12, 240)
    risk = np.array([ridge_risk(l, gamma, r2, s2)[0] for l in lams])
    r_star = ridge_risk(lam_star, gamma, r2, s2)[0]
    ax2.plot(lams, risk, color="k", lw=1.4)
    ax2.axvline(lam_star, color="k", ls="--", lw=0.8)
    ax2.plot([lam_star], [r_star], "ko", ms=4)
    ax2.text(lam_star + 0.3, r_star + 0.02, r"$\lambda_*=3$")
    ax2.set(xlim=(0, 12), ylim=(0.83, 1.16), xlabel=r"ridge parameter $\lambda$", ylabel="excess test risk")
    ax2.set_title(r"Risk for $r^2=\sigma_\epsilon^2=1$", fontsize=10)

    fig.savefig(HERE / "ridge_gamma3.pdf")
    print(f"ridge_gamma3.pdf: R(lambda*={lam_star:g}) = {r_star:.6f}")


if __name__ == "__main__":
    ridge_gamma3()
