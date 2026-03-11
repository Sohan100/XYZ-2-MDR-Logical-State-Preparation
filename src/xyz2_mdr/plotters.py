"""
plotters.py
-----------
Dedicated plotting utilities for MDR simulation and sweep objects.

This module keeps visualization logic out of simulation/data classes so the
core classes remain focused on computation and persistence.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union

from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np

if TYPE_CHECKING:
    from .mdr_noise_sweep import MdrNoiseSweep
    from .mdr_simulation import MDRSimulation


class MDRSimulationPlotter:
    """
    Visualization helpers for precomputed `MDRSimulation` objects.
    Supports multi-panel fidelity comparisons for stabilizer and logical
    observables across multiple simulation instances.

    Attributes
    ----------
    This class stores no instance attributes because all plotting entrypoints
    are stateless static methods.

    Methods
    -------
    plot_multi_fidelity(sims, category, operators, show_violin,
        show_replicates, figsize, save_path)
        Compare per-round fidelities across multiple simulation objects and
        optionally save the rendered figure.
    """

    @staticmethod
    def plot_multi_fidelity(
        sims: Dict[str, "MDRSimulation"],
        category: str,
        operators: Optional[Union[str, List[str]]] = None,
        *,
        show_violin: bool = True,
        show_replicates: bool = False,
        figsize: Tuple[int, int] = (16, 8),
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Compare per-round fidelities across multiple simulation instances.

        The function arranges one panel per simulation label and optionally
        separates stabilizer and logical observables into different rows. Each
        operator is rendered with a stable marker/color assignment across
        panels so visual comparisons remain consistent between noise models.

        Args:
            sims: Mapping `panel_title -> MDRSimulation`.
            category: `stabilizer`, `logical`, or `both`.
            operators: Optional operator subset. Defaults to all operators in
                the selected category.
            show_violin: If True, overlay replicate distributions.
            show_replicates: If True, scatter individual replicate means.
            figsize: Figure size in inches.
            save_path: Optional path to save the figure.

        Raises:
            ValueError: If `sims` is empty, `category` is invalid, or the
                requested operator subset is not available in the simulations.
        """
        if not sims:
            raise ValueError("No simulations provided.")

        if category == "both":
            cats = ["stabilizer", "logical"]
        elif category in {"stabilizer", "logical"}:
            cats = [category]
        else:
            raise ValueError(
                "category must be 'stabilizer', 'logical', or 'both'."
            )

        first = next(iter(sims.values()))
        ops_dict: Dict[str, List[str]] = {}
        for cat in cats:
            stats_attr = (
                "_stats_stabilizers"
                if cat == "stabilizer"
                else "_stats_logicals"
            )
            all_ops = list(getattr(first, stats_attr).keys())
            if operators is None:
                ops = all_ops
            elif isinstance(operators, str):
                ops = [operators]
            else:
                ops = list(operators)
            missing = set(ops) - set(all_ops)
            if missing:
                raise ValueError(f"Unknown operators for {cat}: {missing}")
            ops_dict[cat] = ops

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        markers = [
            "o",
            "s",
            "^",
            "v",
            "<",
            ">",
            "8",
            "p",
            "P",
            "*",
            "h",
            "H",
            "+",
            "x",
            "X",
            "D",
            "d",
            "|",
            "_",
            ".",
        ]
        global_ops = [op for cat in cats for op in ops_dict[cat]]
        style_map: Dict[str, Tuple[str, str]] = {}
        for i, op in enumerate(global_ops):
            color = colors[i % len(colors)]
            marker = markers[(i // len(colors)) % len(markers)]
            style_map[op] = (color, marker)

        nrows = len(cats)
        ncols = len(sims)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            squeeze=False,
            sharex="col",
        )

        legend_entries = {}
        for row, cat in enumerate(cats):
            if cat == "stabilizer":
                stats_attr = "_stats_stabilizers"
                dist_attr = "_replicate_means_stabilizers"
                ylabel = "|<S>|"
            else:
                stats_attr = "_stats_logicals"
                dist_attr = "_replicate_means_logicals"
                ylabel = "|<L>|"

            for col, (label, sim) in enumerate(sims.items()):
                ax = axes[row][col]
                if row == 0:
                    ax.set_title(label, fontsize=18)
                ax.grid(True)

                stats_map = getattr(sim, stats_attr)
                dist_map = getattr(sim, dist_attr)
                for op in ops_dict[cat]:
                    rounds = stats_map[op]["rounds"]
                    centers = stats_map[op]["centers"]
                    stds = stats_map[op]["stds"]
                    color, marker = style_map[op]
                    eb = ax.errorbar(
                        rounds,
                        centers,
                        yerr=stds,
                        fmt=marker,
                        color=color,
                        capsize=3,
                        linestyle="None",
                        label=op,
                        zorder=3,
                    )
                    line = eb.lines[0]
                    if op not in legend_entries:
                        legend_entries[op] = line

                    if show_violin:
                        data = [dist_map[op][round_idx] for round_idx in rounds]
                        vp = ax.violinplot(
                            data,
                            positions=rounds,
                            widths=0.6,
                            showmeans=False,
                            showextrema=False,
                        )
                        for body in vp["bodies"]:
                            body.set_facecolor(color)
                            body.set_edgecolor(color)
                            body.set_alpha(0.2)

                    if show_replicates:
                        for round_idx in rounds:
                            values = dist_map[op][round_idx]
                            ax.scatter(
                                [round_idx] * len(values),
                                values,
                                color=color,
                                alpha=0.05,
                                s=4,
                                zorder=2,
                            )

                if col == 0:
                    ax.set_ylabel(ylabel, fontsize=16)
                if row == nrows - 1:
                    ax.set_xlabel("MDR rounds", fontsize=16)

        fig.legend(
            list(legend_entries.values()),
            list(legend_entries.keys()),
            loc="center left",
            bbox_to_anchor=(0.93, 0.5),
            fontsize=16,
        )
        fig.tight_layout(rect=[0, 0, 0.935, 1.0])

        if save_path:
            out_path = Path(save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
        if "agg" not in plt.get_backend().lower():
            plt.show()
        plt.close(fig)


class MdrNoiseSweepPlotter:
    """
    Visualization helpers for `MdrNoiseSweep` result sets.
    Supports plotting observable-loss and logical state-preparation error
    curves from aggregated noise-sweep results.

    Attributes
    ----------
    This class stores no instance attributes because all plotting entrypoints
    are stateless static methods.

    Methods
    -------
    _sorted_combos(sweep)
        Return parameter combinations in ascending numeric order.
    plot_error_multi(sweeps, category, rounds, subset, overlay, log_x,
        figsize, save_path)
        Plot `1 - |<O>|` across one or more sweep result sets.
    plot_state_prep_error_multi(sweeps, rounds, logical_label, overlay,
        log_x, figsize, save_path, allow_legacy_approx)
        Plot logical state-preparation error across one or more sweep result
        sets.
    _finalize_with_legend(fig, plot_axes, save_path)
        Apply common layout, legend placement, saving, and cleanup.
    """

    @staticmethod
    def _sorted_combos(sweep: "MdrNoiseSweep") -> List[Tuple[float, ...]]:
        """
        Return parameter combinations in ascending numeric order.

        Args:
            sweep: Sweep whose parameter combinations should be ordered.

        Returns:
            List[Tuple[float, ...]]: Parameter tuples sorted lexicographically
            after coercion to floats.
        """
        return sorted(
            sweep.param_combos,
            key=lambda combo: tuple(float(x) for x in combo),
        )

    @staticmethod
    def plot_error_multi(
        sweeps: Dict[str, "MdrNoiseSweep"],
        category: str,
        rounds: List[int],
        subset: Optional[List[str]] = None,
        overlay: bool = False,
        log_x: bool = False,
        figsize: Tuple[int, int] = (15, 6),
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Plot observable-loss curves across multiple noise-sweep objects.

        The plotted metric is `1 - |<O>|`, which matches the legacy fidelity
        loss visualization used in this project for stabilizer and logical
        observables loaded from sweep CSVs.

        Args:
            sweeps: Mapping `legend_group -> MdrNoiseSweep`.
            category: Either `stabilizer` or `logical`.
            rounds: MDR rounds to include in the plot.
            subset: Optional operator subset to plot.
            overlay: If True, all requested rounds are overlaid on one axis;
                otherwise one subplot is created per round.
            log_x: If True, display the physical-noise axis on a log scale.
            figsize: Figure size in inches.
            save_path: Optional output path for the rendered figure.

        Raises:
            ValueError: If no sweeps are supplied, `category` is invalid, or
                the requested operator subset contains unknown labels.
        """
        if not sweeps:
            raise ValueError("No sweeps provided")

        first = next(iter(sweeps.values()))
        if category == "stabilizer":
            labels = list(first.measure_stabilizers)
        elif category == "logical":
            labels = list(first.logical_operators.keys())
        else:
            raise ValueError("category must be 'stabilizer' or 'logical'")

        if subset:
            missing = set(subset) - set(labels)
            if missing:
                raise ValueError(f"Unknown labels: {missing}")
            labels = subset

        colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        markers = [
            "o",
            "s",
            "^",
            "v",
            "<",
            ">",
            "8",
            "p",
            "P",
            "*",
            "h",
            "H",
            "+",
            "x",
            "X",
            "D",
            "d",
            "|",
            "_",
            ".",
        ]
        style_keys = [(model, op) for model in sweeps for op in labels]
        style_map = {
            key: (
                colours[idx % len(colours)],
                markers[(idx // len(colours)) % len(markers)],
            )
            for idx, key in enumerate(style_keys)
        }

        if overlay:
            fig, ax0 = plt.subplots(figsize=figsize)
            axes = [ax0]
        else:
            cols = min(3, len(rounds))
            rows = math.ceil(len(rounds) / cols)
            fig, grid = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
            axes = list(grid.flatten())

        plot_axes = [axes[0]] if overlay else axes[: len(rounds)]
        if not overlay and len(axes) > len(rounds):
            for extra_ax in axes[len(rounds) :]:
                extra_ax.set_visible(False)

        for idx, ax in enumerate(plot_axes):
            if not overlay:
                round_idx = rounds[idx]
                ax.set_title(f"Round {round_idx}")
            ax.grid(True)

            for model, sweep in sweeps.items():
                combos = MdrNoiseSweepPlotter._sorted_combos(sweep)
                p_vals = np.array([float(combo[0]) for combo in combos])

                if overlay:
                    for op in labels:
                        for round_idx in rounds:
                            means = np.array(
                                [sweep.results[c][round_idx][op] for c in combos],
                                dtype=float,
                            )
                            stds = np.array(
                                [
                                    sweep.results_std[c][round_idx][op]
                                    for c in combos
                                ],
                                dtype=float,
                            )
                            color, marker = style_map[(model, op)]
                            ax.errorbar(
                                p_vals,
                                1 - means,
                                yerr=stds,
                                fmt=f"-{marker}",
                                color=color,
                                capsize=4,
                                label=f"{model}: {op} (r={round_idx})",
                            )
                else:
                    round_idx = rounds[idx]
                    for op in labels:
                        means = np.array(
                            [sweep.results[c][round_idx][op] for c in combos],
                            dtype=float,
                        )
                        stds = np.array(
                            [sweep.results_std[c][round_idx][op] for c in combos],
                            dtype=float,
                        )
                        color, marker = style_map[(model, op)]
                        ax.errorbar(
                            p_vals,
                            1 - means,
                            yerr=stds,
                            fmt=f"-{marker}",
                            color=color,
                            capsize=4,
                            label=f"{model}: {op}",
                        )

            if log_x:
                ax.set_xscale("log")
            ax.set_xlabel("p", fontsize=14)
            ax.set_ylabel("Error rate (1 - |<O>|)", fontsize=14)

        MdrNoiseSweepPlotter._finalize_with_legend(
            fig=fig,
            plot_axes=plot_axes,
            save_path=save_path,
        )

    @staticmethod
    def plot_state_prep_error_multi(
        sweeps: Dict[str, "MdrNoiseSweep"],
        rounds: List[int],
        logical_label: str = "Logical X",
        overlay: bool = False,
        log_x: bool = False,
        figsize: Tuple[int, int] = (15, 6),
        save_path: Optional[Union[str, Path]] = None,
        allow_legacy_approx: bool = False,
    ) -> None:
        """
        Plot logical state-preparation error across multiple sweeps.

        The metric is the target-state logical failure estimate
        `(1 - <X_L>) / 2`, computed from signed logical expectations. Legacy
        CSVs that only store `|<X_L>|` can optionally be approximated as
        `(1 - |<X_L>|) / 2`, but that loses the sign information required for
        an exact logical state-preparation error rate.

        Args:
            sweeps: Mapping `legend_group -> MdrNoiseSweep`.
            rounds: MDR rounds to include in the plot.
            logical_label: Logical operator whose signed expectation defines
                the state-preparation target. Currently only `"Logical X"` is
                supported.
            overlay: If True, all requested rounds are overlaid on one axis;
                otherwise one subplot is created per round.
            log_x: If True, display the physical-noise axis on a log scale.
            figsize: Figure size in inches.
            save_path: Optional output path for the rendered figure.
            allow_legacy_approx: If True, permit approximate plotting from
                legacy CSV files without signed logical means.

        Raises:
            ValueError: If no sweeps are supplied or an unsupported logical
                label is requested.
        """
        if not sweeps:
            raise ValueError("No sweeps provided")
        if logical_label != "Logical X":
            raise ValueError(
                "state-preparation error is currently supported only for "
                "'Logical X'."
            )

        colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        markers = ["o", "s", "^", "v", "<", ">", "D", "P", "X", "*"]
        style_map = {
            model: (
                colours[idx % len(colours)],
                markers[idx % len(markers)],
            )
            for idx, model in enumerate(sweeps)
        }

        if overlay:
            fig, ax0 = plt.subplots(figsize=figsize)
            axes = [ax0]
        else:
            cols = min(3, len(rounds))
            rows = math.ceil(len(rounds) / cols)
            fig, grid = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
            axes = list(grid.flatten())

        plot_axes = [axes[0]] if overlay else axes[: len(rounds)]
        if not overlay and len(axes) > len(rounds):
            for extra_ax in axes[len(rounds) :]:
                extra_ax.set_visible(False)

        legacy_models = [
            model
            for model, sweep in sweeps.items()
            if not sweep.has_exact_signed_results
        ]
        if legacy_models and allow_legacy_approx:
            print(
                "Warning: using legacy approximate state-preparation error for "
                + ", ".join(legacy_models)
                + "."
            )

        for idx, ax in enumerate(plot_axes):
            if not overlay:
                round_idx = rounds[idx]
                ax.set_title(f"Round {round_idx}")
            ax.grid(True)

            for model, sweep in sweeps.items():
                color, marker = style_map[model]

                if overlay:
                    for round_idx in rounds:
                        p_vals, y_vals, y_errs = sweep._metric_series_for_operator(
                            round_idx=round_idx,
                            operator=logical_label,
                            metric="state_prep_error",
                            allow_legacy_approx=allow_legacy_approx,
                        )
                        ax.errorbar(
                            p_vals,
                            y_vals,
                            yerr=y_errs,
                            fmt=f"-{marker}",
                            color=color,
                            capsize=4,
                            label=f"{model} (r={round_idx})",
                        )
                else:
                    round_idx = rounds[idx]
                    p_vals, y_vals, y_errs = sweep._metric_series_for_operator(
                        round_idx=round_idx,
                        operator=logical_label,
                        metric="state_prep_error",
                        allow_legacy_approx=allow_legacy_approx,
                    )
                    ax.errorbar(
                        p_vals,
                        y_vals,
                        yerr=y_errs,
                        fmt=f"-{marker}",
                        color=color,
                        capsize=4,
                        label=model,
                    )

            if log_x:
                ax.set_xscale("log")
            ax.set_xlabel("p", fontsize=14)
            ylabel = "State-prep error (1 - <X_L>) / 2"
            if legacy_models and allow_legacy_approx:
                ylabel += " [legacy approx]"
            ax.set_ylabel(ylabel, fontsize=14)

        MdrNoiseSweepPlotter._finalize_with_legend(
            fig=fig,
            plot_axes=plot_axes,
            save_path=save_path,
        )

    @staticmethod
    def _finalize_with_legend(
        fig,
        plot_axes,
        save_path: Optional[Union[str, Path]],
    ) -> None:
        """
        Apply final layout, legend placement, saving, and cleanup.

        Args:
            fig: Matplotlib figure being finalized.
            plot_axes: Sequence of axes that contain the actual plotted data.
                The first axis is used as the legend handle source.
            save_path: Optional output path for saving the figure.
        """
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bboxes = [ax.get_tightbbox(renderer) for ax in plot_axes]
        union_bbox = Bbox.union(bboxes)
        bb = union_bbox.transformed(fig.transFigure.inverted())
        lx = bb.x1 + 0.01
        ly = bb.y0 + bb.height / 2

        handles, labels = plot_axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(lx, ly),
            fontsize="small",
        )

        if save_path is not None:
            out_path = Path(save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
        if "agg" not in plt.get_backend().lower():
            plt.show()
        plt.close(fig)
