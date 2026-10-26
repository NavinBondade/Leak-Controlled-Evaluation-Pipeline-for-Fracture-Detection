import matplotlib.pyplot as plt

BLUE, ORANGE, GREY, RED = "#2a6f97", "#e08a1e", "#7f8c8d", "#c0392b"


def use():
    """One look for every figure: 10 pt text, light axes, 200 dpi."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.titlelocation": "left",
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "savefig.facecolor": "white",
        }
    )
