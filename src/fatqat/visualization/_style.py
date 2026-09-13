"""Private visual-style primitives shared by Matplotlib renderers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_FATQAT_PALETTE = (
    "#4C6FFF",  # cobalt
    "#167D6D",  # teal
    "#8067DC",  # purple
    "#E5A83B",  # amber
    "#D9685A",  # coral
    "#4EA8DE",  # sky
)

_AXIS_LINEWIDTH = 0.8
_GRID_LINEWIDTH = 0.6
_STRUCTURE_LINEWIDTH = 1.1
_OUTLINE_LINEWIDTH = 1.0


@dataclass(frozen=True, slots=True)
class _MatplotlibStyle:
    """Resolved colors for one axes without changing global Matplotlib state."""

    series: tuple[Any, ...]
    foreground: Any
    background: Any
    edge: Any
    grid: Any

    def series_color(self, index: int) -> Any:
        """Return a categorical color, cycling a short user palette if needed."""
        return self.series[index % len(self.series)]


def _resolve_mpl_style(axis: Any) -> _MatplotlibStyle:
    """Resolve FATQAT defaults while preserving an explicit Matplotlib cycle."""
    from matplotlib import rcParams, rcParamsDefault

    active = tuple(rcParams["axes.prop_cycle"].by_key().get("color", ()))
    default = tuple(rcParamsDefault["axes.prop_cycle"].by_key().get("color", ()))
    series = _FATQAT_PALETTE if not active or active == default else active
    return _MatplotlibStyle(
        series=series,
        foreground=rcParams["text.color"],
        background=axis.get_facecolor(),
        edge=rcParams["axes.edgecolor"],
        grid=rcParams["grid.color"],
    )


def _apply_cartesian_style(axis: Any, style: _MatplotlibStyle) -> None:
    """Apply the shared quiet frame and grid used by Cartesian plots."""
    axis.set_axisbelow(True)
    axis.grid(
        axis="y",
        color=style.grid,
        linewidth=_GRID_LINEWIDTH,
        alpha=0.35,
    )
    for name in ("left", "bottom"):
        axis.spines[name].set_color(style.edge)
        axis.spines[name].set_linewidth(_AXIS_LINEWIDTH)
    for name in ("top", "right"):
        axis.spines[name].set_visible(False)
    axis.tick_params(
        axis="both",
        color=style.edge,
        labelcolor=style.foreground,
        width=_AXIS_LINEWIDTH,
        length=4,
    )
    axis.xaxis.label.set_color(style.foreground)
    axis.yaxis.label.set_color(style.foreground)
    axis.title.set_color(style.foreground)


def _tint(color: Any, background: Any, amount: float = 0.38) -> tuple[float, ...]:
    """Blend ``color`` into ``background`` for a restrained filled surface."""
    from matplotlib.colors import to_rgba

    foreground_rgba = to_rgba(color)
    background_rgba = to_rgba(background)
    return tuple(
        amount * foreground_rgba[index] + (1.0 - amount) * background_rgba[index]
        for index in range(3)
    ) + (1.0,)


def _contrasting_color(background: Any, *candidates: Any) -> Any:
    """Return the candidate with the strongest contrast against ``background``."""
    from matplotlib.colors import to_rgba

    def luminance(color: Any) -> float:
        channels = to_rgba(color)[:3]
        linear = tuple(
            (
                channel / 12.92
                if channel <= 0.04045
                else ((channel + 0.055) / 1.055) ** 2.4
            )
            for channel in channels
        )
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    background_luminance = luminance(background)
    return max(
        candidates,
        key=lambda color: (max(background_luminance, luminance(color)) + 0.05)
        / (min(background_luminance, luminance(color)) + 0.05),
    )
