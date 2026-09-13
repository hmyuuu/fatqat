"""Private Matplotlib renderers for FATQAT visualizations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._style import (
    _OUTLINE_LINEWIDTH,
    _STRUCTURE_LINEWIDTH,
    _apply_cartesian_style,
    _contrasting_color,
    _resolve_mpl_style,
)
from ._viewmodels import _CountsView

if TYPE_CHECKING:
    from .._program_graph import _InteractionFrequencyGraph


def _render_counts(
    view: _CountsView,
    *,
    ax: Any = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Render prepared counts as a Matplotlib bar chart."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if ax is not None and figsize is not None:
        raise ValueError("figsize cannot be used together with ax")

    owns_figure = ax is None
    if owns_figure:
        figure = Figure(figsize=figsize)
        FigureCanvasAgg(figure)
        axis = figure.add_subplot(111)
    else:
        axis = ax
        figure = axis.get_figure()

    from matplotlib.ticker import PercentFormatter

    style = _resolve_mpl_style(axis)
    positions = list(range(len(view.labels)))
    category_count = len(view.labels)
    width = 0.46 if category_count == 2 else 0.62 if category_count <= 4 else 0.78
    bars = axis.bar(
        positions,
        view.values,
        width=width,
        color=style.series_color(0),
        edgecolor="none",
    )
    if category_count == 2:
        axis.set_xlim(-0.5, 1.5)
    axis.set_xticks(positions, view.labels)
    axis.set_xlabel("Outcome")
    axis.set_ylabel("Frequency" if view.stat == "frequencies" else "Counts")
    if view.stat == "frequencies":
        axis.set_ylim(0, 1)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    else:
        axis.set_ylim(bottom=0)

    if len(view.labels) <= 8:
        labels = (
            [f"{value:.1%}" for value in view.values]
            if view.stat == "frequencies"
            else [f"{int(value):,}" for value in view.values]
        )
        for rectangle, value, label in zip(bars, view.values, labels, strict=True):
            inside = view.stat == "frequencies" and value >= 0.9
            axis.annotate(
                label,
                (
                    rectangle.get_x() + rectangle.get_width() / 2,
                    rectangle.get_height(),
                ),
                xytext=(0, -5 if inside else 4),
                textcoords="offset points",
                ha="center",
                va="top" if inside else "bottom",
                color=(
                    _contrasting_color(
                        rectangle.get_facecolor(),
                        style.foreground,
                        style.background,
                    )
                    if inside
                    else style.foreground
                ),
            )

    if title is not None:
        axis.set_title(title)

    if len(view.labels) > 8:
        axis.tick_params(axis="x", labelrotation=45)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")

    _apply_cartesian_style(axis, style)

    if owns_figure:
        figure.tight_layout()

    return figure


def _render_interaction_frequency(
    graph: _InteractionFrequencyGraph,
    *,
    ax: Any = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Render a logical-qubit interaction frequency graph."""
    from math import cos, pi, sin

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    if ax is not None and figsize is not None:
        raise ValueError("figsize cannot be used together with ax")

    owns_figure = ax is None
    if owns_figure:
        figure = Figure(figsize=figsize)
        FigureCanvasAgg(figure)
        axis = figure.add_subplot(111)
    else:
        axis = ax
        figure = axis.get_figure()

    style = _resolve_mpl_style(axis)
    label_background = style.background
    edge_color = style.series_color(0)
    node_color = style.series_color(1)
    node_label_color = _contrasting_color(
        node_color,
        style.foreground,
        style.background,
    )

    count = len(graph.nodes)
    positions: dict[Any, tuple[float, float]] = {}
    if count == 1:
        positions[graph.nodes[0]] = (0.0, 0.0)
    elif count:
        positions = {
            node: (
                cos(2 * pi * index / count),
                sin(2 * pi * index / count),
            )
            for index, node in enumerate(graph.nodes)
        }

    maximum = max((edge.count for edge in graph.edges), default=1)
    node_labels = {node: str(index) for index, node in enumerate(graph.nodes)}
    for edge in graph.edges:
        x1, y1 = positions[edge.source]
        x2, y2 = positions[edge.target]
        axis.add_line(
            Line2D(
                [x1, x2],
                [y1, y2],
                linewidth=_STRUCTURE_LINEWIDTH + 3.1 * edge.count / maximum,
                color=edge_color,
                alpha=0.76,
                zorder=1,
            )
        )
        axis.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            str(edge.count),
            color=style.foreground,
            ha="center",
            va="center",
            bbox={
                "facecolor": label_background,
                "edgecolor": "none",
                "pad": 1.5,
            },
            zorder=3,
        )

    if count:
        xs, ys = zip(*(positions[node] for node in graph.nodes))
        axis.scatter(
            xs,
            ys,
            s=700,
            marker="o",
            color=node_color,
            edgecolors=style.foreground,
            linewidths=_OUTLINE_LINEWIDTH,
            zorder=2,
        )
        for node in graph.nodes:
            x, y = positions[node]
            axis.text(
                x,
                y,
                node_labels[node],
                color=node_label_color,
                ha="center",
                va="center",
                zorder=4,
            )

    margin = 1.35 if count > 1 else 1.0
    axis.set_xlim(-margin, margin)
    axis.set_ylim(-margin, margin)
    axis.set_aspect("equal")
    axis.axis("off")
    if title is not None:
        axis.set_title(title)
    if owns_figure:
        figure.tight_layout()
    return figure
