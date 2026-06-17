"""One-off render: UML sequence diagram PNG for README fallback."""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = os.path.join(os.path.dirname(__file__), "..", "public", "architecture", "sequence-diagram.png")

STEPS = [
    ("Operator", "Streamlit UI", "Select district / scale / dates"),
    ("Streamlit UI", "Google Earth Engine", "Fetch Landsat 2010 MNDWI baseline"),
    ("Google Earth Engine", "Streamlit UI", "Return historical flood %"),
    ("Streamlit UI", "Google Earth Engine", "Fetch Sentinel-1 VV composite"),
    ("Google Earth Engine", "U-Net inference", "SAR GeoTIFF tile"),
    ("U-Net inference", "Streamlit UI", "Flood mask + probability metrics"),
    ("Streamlit UI", "FFD scraper", "Scrape live barrage discharge"),
    ("FFD scraper", "Streamlit UI", "Inflow / outflow / status"),
    ("Streamlit UI", "Risk engine", "Combine flood %, 2010 delta, rivers"),
    ("Risk engine", "Streamlit UI", "Defensible risk score (1–10)"),
    ("Streamlit UI", "Gemini / Groq AI", "Structured metrics context"),
    ("Gemini / Groq AI", "Streamlit UI", "Tactical situation report"),
    ("Streamlit UI", "Operator", "Render Overview · Detection · Rivers · AI tabs"),
]

ACTORS = [
    "Operator",
    "Streamlit UI",
    "Google Earth Engine",
    "U-Net inference",
    "FFD scraper",
    "Risk engine",
    "Gemini / Groq AI",
]


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#0a0f14")
    ax.set_facecolor("#0a0f14")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.97,
        "FloodSense-PK — Component Interaction (Sequence)",
        ha="center",
        va="top",
        color="white",
        fontsize=16,
        fontweight="bold",
    )

    n = len(ACTORS)
    xs = {name: 0.06 + i * (0.88 / max(n - 1, 1)) for i, name in enumerate(ACTORS)}

    top = 0.88
    for name, x in xs.items():
        rect = Rectangle((x - 0.045, top), 0.09, 0.05, facecolor="#111820", edgecolor="#00d4ff", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, top + 0.025, name, ha="center", va="center", color="white", fontsize=7, fontweight="bold")
        ax.plot([x, x], [0.08, top], color="#1e2d3d", linewidth=1, linestyle="--")

    y = 0.82
    for i, (src, dst, label) in enumerate(STEPS, start=1):
        x1, x2 = xs[src], xs[dst]
        dy = 0.018 if x1 <= x2 else -0.018
        color = "#00cc66" if x1 <= x2 else "#5b9bd5"
        arrow = FancyArrowPatch(
            (x1, y),
            (x2, y + dy),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.4,
            color=color,
            connectionstyle="arc3,rad=0.08" if abs(x2 - x1) > 0.15 else "arc3,rad=0.0",
        )
        ax.add_patch(arrow)
        ax.text(0.02, y, str(i), ha="left", va="center", color="#8aa5ff", fontsize=8, fontweight="bold")
        ax.text((x1 + x2) / 2, y + 0.022, label, ha="center", va="bottom", color="#b8c7d9", fontsize=7)
        y -= 0.055

    fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
