"""Matplotlib and plotly visualization helpers."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd

try:  # optional dependency
    import plotly.express as px
except Exception:  # pragma: no cover - optional dependency
    px = None


@dataclass
class PlotBuilder:
    def line(self, frame: pd.DataFrame, x: str, y: str, title: str):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(frame[x], frame[y])
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        fig.tight_layout()
        return fig

    def interactive_line(self, frame: pd.DataFrame, x: str, y: str, title: str):
        if px is None:
            raise RuntimeError("plotly is not available")
        return px.line(frame, x=x, y=y, title=title)
