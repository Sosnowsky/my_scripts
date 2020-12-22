import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
import pandas as pd


def lineal(x, a, s):
    return a * x + s


def plot_pds(
    x,
    dt,
    cutoff_min,
    cutoff_max,
    ax,
    ax_product=None,
    color_scatter="blue",
    color_fit="red",
    label="",
    nperseg=None,
):
    ax.axvline(cutoff_min, ls="dashed", alpha=0.5)
    ax.axvline(cutoff_max, ls="dashed", alpha=0.5)

    if nperseg is None:
        nperseg = len(x) / 16
    f, Pxx_den = signal.welch(x, fs=1 / dt, nperseg=nperseg)
    ax.scatter(f, Pxx_den, color=color_scatter)
    popt, pcov = curve_fit(
        lineal,
        np.log10(f[np.logical_and(f > cutoff_min, f < cutoff_max)]),
        np.log10(Pxx_den[np.logical_and(f > cutoff_min, f < cutoff_max)]),
        maxfev=10000,
    )
    ax.plot(
        f,
        10 ** lineal(np.log10(f), *popt),
        "r-",
        label=r"{} $\beta \approx {:.2f}({:.2f})$".format(
            label, -popt[0], np.sqrt(pcov[0, 0])
        ),
        color=color_fit,
    )

    ax.set_ylabel(r"$\log(\Phi(f))$")
    ax.set_xlabel(r"$\log(f)$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(r"$\Phi(f)$")
    ax.set_xlim(cutoff_min / 1e2, cutoff_max * 1e2)

    if ax_product is not None:
        ax_product.axvline(cutoff_min, ls="dashed", alpha=0.5)
        ax_product.axvline(cutoff_max, ls="dashed", alpha=0.5)
        ax_product.scatter(f, np.power(f, -popt[0]) * Pxx_den, color=color)
        ax_product.set_title(r"$\Phi(f) f ^ {\beta}$")
        ax_product.set_xscale("log")
        ax_product.set_yscale("log")
        ax_product.set_xlabel(r"$\log(f)$")
        ax_product.set_xlim(1e-5, 1e2)
        ax_product.set_ylim(1, 1e3)

    return -popt[0]


def plot_distribution(x):
    fig, ax = plt.subplots()
    x_axes = np.logspace(0, int(np.log10(max(x))) + 1, num=10)
    y, x_axes = np.histogram(x, bins=x_axes, density=True)
    x_axes = np.sqrt(x_axes[:-1] * x_axes[1:])
    ax.scatter(x_axes[y != 0], y[y != 0])
    perform_fit_over = y != 0
    popt, pcov = curve_fit(
        lineal,
        np.log10(x_axes[perform_fit_over]),
        np.log10(y[perform_fit_over]),
        maxfev=10000,
    )

    ax.plot(
        x_axes,
        10 ** lineal(np.log10(x_axes), a=popt[0], s=popt[1]),
        "r-",
        color="blue",
    )
    ax.text(
        0.3,
        0.7,
        r"$S(\tau) \approx \tau ^ {{ {:.2f} }} $".format(popt[0]),
        transform=ax.transAxes,
    )

    ax.text(
        0.1,
        0.5,
        r"$min {:.4g}  max {:.4g} $".format(min(x), max(x)),
        transform=ax.transAxes,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("mean {:.2g}".format(x.mean()))


def plot_scaling(
    x,
    ax,
    cutoff,
    cutoff_min=None,
    weights=None,
    loglin=False,
    num=50,
    label="",
    color="blue",
):
    if loglin:
        x_axes = np.linspace(0, max(x) + 1, num=num)
    else:
        x_axes = np.floor(np.logspace(0, int(np.log10(max(x))) + 1, num=num))

    y, x_axes = np.histogram(x, bins=x_axes, density=True, weights=weights)

    if loglin:
        x_axes = (x_axes[:-1] + x_axes[1:]) / 2
    else:
        x_axes = np.sqrt(x_axes[:-1] * x_axes[1:])

    ax.scatter(x_axes[y != 0], y[y != 0], color=color)
    perform_fit_over = np.logical_and(x_axes < cutoff, y > 0)
    if not loglin:
        ax.set_xscale("log")
        if cutoff_min is not None:
            perform_fit_over = np.logical_and(perform_fit_over, x_axes > cutoff_min)

        if min(x_axes) < cutoff < max(x_axes):
            ax.axvline(cutoff, ls="dashed", alpha=0.5)

        # Linear regresion
        popt, pcov = curve_fit(
            lineal,
            np.log10(x_axes[perform_fit_over]),
            np.log10(y[perform_fit_over]),
            maxfev=10000,
        )

        ax.plot(
            x_axes,
            10 ** lineal(np.log10(x_axes), a=popt[0], s=popt[1]),
            "r-",
            color=color,
            label=r"{} $\alpha \approx {:.2f}({:.2f})$".format(
                label, -popt[0], np.sqrt(pcov[0, 0])
            ),
        )

    if loglin:
        popt, pcov = curve_fit(
            lineal,
            x_axes[perform_fit_over],
            np.log10(y[perform_fit_over]),
            maxfev=10000,
        )

        ax.plot(
            x_axes,
            10 ** lineal(x_axes, *popt),
            "r-",
            color="blue",
            label=r"{} $\alpha \approx {:.2f}$".format(label, -popt[0]),
        )

    ax.set_yscale("log")
    return -popt[0]


def regplot(x, y, ax, cutoff_min=0, cutoff_max=1e10):
    ax.scatter(x, y)
    x_axes = np.linspace(min(x), max(x), num=100)

    df = pd.DataFrame({"durations": x, "areas": y})
    gb = df.groupby("durations").mean()

    perform_fit_over = np.logical_and(gb.index > cutoff_min, gb.index < cutoff_max)

    popt, pcov = curve_fit(
        lineal,
        np.log10(gb.iloc[perform_fit_over].index.values),
        np.log10(gb.iloc[perform_fit_over].areas.values),
        maxfev=10000,
    )

    ax.plot(x_axes, 10 ** lineal(np.log10(x_axes), *popt), "r-")

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.text(
        0.3,
        0.7,
        r"$S(\tau) \approx \tau ^ {{ {:.2f} }} $".format(popt[0]),
        transform=ax.transAxes,
    )
    return popt[0]
