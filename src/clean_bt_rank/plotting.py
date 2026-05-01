from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_NEURIPS_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 11,
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

ACTION_PALETTE = {
    "drop": "#264653",
    "flip": "#E76F51",
    "add_pairs": "#457B9D",
    "add_outcomes": "#2A9D8F",
    "add_weighted": "#F4A261",
}

ACTION_LABEL_MAP = {
    "drop": "Drop",
    "flip": "Flip",
    "add_pairs": "Add\npairs",
    "add_outcomes": "Add\noutcomes",
    "add_weighted": "Add\nweighted",
}

# Distinct colors for anonymous top-N plots (colorblind-friendly-ish, vivid).
_VIVID_RANK_COLORS = [
    "#E63946",
    "#457B9D",
    "#2A9D8F",
    "#F4A261",
    "#9B5DE5",
    "#00BBF9",
    "#E9C46A",
    "#2EC4B6",
    "#FB5607",
    "#8338EC",
]


def use_paper_rc() -> None:
    plt.rcParams.update(_NEURIPS_RC)


def _style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    if grid_axis:
        ax.grid(True, axis=grid_axis, color="#c7c7c7", linewidth=0.7, alpha=0.7)
    off_axis = "x" if grid_axis == "y" else "y"
    ax.grid(False, axis=off_axis)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", width=0.8, length=3)
    ax.set_axisbelow(True)


def plot_variant_actions_needed(
    summary_df: pd.DataFrame,
    *,
    title: str,
    max_actions: int,
    variant_col: str = "variant",
    actions_col: str = "n_actions",
    met_col: str = "met",
    figsize: tuple[float, float] = (4.6, 3.3),
):
    plot_df = summary_df.copy()
    plot_df["shown_actions"] = plot_df[actions_col].fillna(max_actions + 1)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(plot_df), dtype=float)
    colors = [ACTION_PALETTE.get(v, "#264653") for v in plot_df[variant_col]]
    bars = ax.bar(x, plot_df["shown_actions"], width=0.62, color=colors, edgecolor="#222222", linewidth=0.8)
    label_pad = max(0.12, 0.025 * max(float(plot_df["shown_actions"].max()), 1.0))
    for bar, met, value in zip(bars, plot_df[met_col], plot_df[actions_col]):
        if not bool(met):
            bar.set_hatch("////")
            label = f">{max_actions}"
        else:
            label = f"{int(value)}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_pad,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="semibold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([ACTION_LABEL_MAP.get(v, str(v)) for v in plot_df[variant_col]], linespacing=0.95)
    ax.set_ylabel("Actions needed")
    ax.set_title(title, pad=14)
    ax.set_ylim(0, max(float(plot_df["shown_actions"].max()) * 1.22 + 0.4, 1.8))
    ax.tick_params(axis="x", pad=8, length=0, labelsize=10)
    _style_axis(ax, grid_axis="y")
    fig.tight_layout(pad=0.75)
    return fig, ax


def plot_variant_order_curves(
    curve_map: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    *,
    title: str,
    ylabel: str,
    value_col: str = "plot_value",
    x_col: str = "step",
    baseline: float | None = None,
    figsize: tuple[float, float] = (7.2, 4.4),
):
    fig, ax = plt.subplots(figsize=figsize)
    for variant, (greedy_curve, random_curve) in curve_map.items():
        label = ACTION_LABEL_MAP.get(variant, str(variant)).replace("\n", " ")
        color = ACTION_PALETTE.get(variant, "#264653")
        ax.plot(greedy_curve[x_col], greedy_curve[value_col], color=color, linewidth=2.1, linestyle="-", label=f"{label} greedy")
        ax.plot(random_curve[x_col], random_curve[value_col], color=color, linewidth=1.9, linestyle="--", label=f"{label} random")
    if baseline is not None:
        ax.axhline(baseline, color="#777777", linewidth=0.9, linestyle=":")
    ax.set_xlabel("Number of actions")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    _style_axis(ax, grid_axis="y")
    ax.legend(frameon=False, ncol=2, columnspacing=1.2, handlelength=2.6)
    fig.tight_layout(pad=0.75)
    return fig, ax


def plot_ci_ranking(
    ci_df: pd.DataFrame,
    *,
    dataset_name: str,
    title_suffix: str = "ranking with Gao 95% CI",
):
    plot_df = ci_df.sort_values("rank", ascending=False).reset_index(drop=True)
    y = np.arange(len(plot_df), dtype=float)
    fig_height = max(4.0, 0.28 * len(plot_df))
    fig, ax = plt.subplots(figsize=(9.0, fig_height))
    ax.hlines(y, plot_df["ci_lower"], plot_df["ci_upper"], color="#8D99AE", linewidth=1.8)
    point_colors = np.where(plot_df["rank"] <= 5, "#E76F51", "#264653")
    ax.scatter(plot_df["rating"], y, color=point_colors, s=28, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{rank}. {name}" for rank, name in zip(plot_df["rank"], plot_df["competitor"])])
    ax.set_xlabel("Rating")
    ax.set_ylabel("")
    ax.set_title(f"{dataset_name} {title_suffix}", pad=10)
    _style_axis(ax, grid_axis="x")
    fig.tight_layout(pad=0.6)
    return fig, ax


def plot_ratings(summary: pd.DataFrame, top_n: int | None = None, title: str | None = None):
    plot_df = summary.copy()
    if top_n is not None:
        plot_df = plot_df.head(top_n)
    plot_df = plot_df.sort_values("rating", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(plot_df))))
    y_pos = range(len(plot_df))
    ax.hlines(y_pos, plot_df["ci_lower"], plot_df["ci_upper"], color="tab:gray", linewidth=2)
    ax.scatter(plot_df["rating"], y_pos, color="tab:blue", zorder=3)
    for x, y in zip(plot_df["rating"], y_pos):
        ax.text(float(x) + 3, y, f"{float(x):.1f}", va="center", fontsize=8, color="tab:blue")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(plot_df["competitor"])
    ax.set_xlabel("Rating")
    ax.set_ylabel("")
    ax.set_title(title or "Bradley-Terry Ratings")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_ci_comparison(
    summaries: dict[str, pd.DataFrame],
    top_n: int | None = None,
    title: str | None = None,
):
    if not summaries:
        raise ValueError("summaries must be non-empty.")

    first = next(iter(summaries.values()))
    order_df = first.copy()
    if top_n is not None:
        order_df = order_df.head(top_n)
    competitors = order_df["competitor"].tolist()[::-1]

    _COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(competitors))))
    methods = list(summaries.keys())
    offsets = [0.18 * (idx - (len(methods) - 1) / 2.0) for idx in range(len(methods))]

    for idx, (offset, (method, summary)) in enumerate(zip(offsets, summaries.items())):
        color = _COLORS[idx % len(_COLORS)]
        plot_df = summary.set_index("competitor").loc[competitors].reset_index()
        y_pos = [i + offset for i in range(len(competitors))]
        ax.hlines(y_pos, plot_df["ci_lower"], plot_df["ci_upper"], linewidth=2, label=method, color=color)
        ax.scatter(plot_df["rating"], y_pos, zorder=3, color=color)
        for x, y in zip(plot_df["rating"], y_pos):
            ax.text(float(x) + 3, y, f"{float(x):.1f}", va="center", fontsize=7, color=color)

    ax.set_yticks(range(len(competitors)))
    ax.set_yticklabels(competitors)
    ax.set_xlabel("Rating")
    ax.set_title(title or "CI Comparison")
    ax.grid(axis="x", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_objective_progressions(
    progressions: dict[str, pd.DataFrame],
    *,
    title: str | None = None,
):
    """
    Plot objective trajectories across iterative dropping runs.

    Each progression DataFrame is expected to contain:
    - `n_dropped`
    - `objective_value`
    """

    if not progressions:
        raise ValueError("progressions must be non-empty.")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, progression in progressions.items():
        if "n_dropped" not in progression.columns or "objective_value" not in progression.columns:
            raise ValueError("Each progression must contain 'n_dropped' and 'objective_value' columns.")
        plot_df = progression.sort_values("n_dropped")
        ax.plot(plot_df["n_dropped"], plot_df["objective_value"], marker="o", linewidth=2, label=label)

    ax.set_xlabel("Dropped matches")
    ax.set_ylabel("Objective value")
    ax.set_title(title or "Objective Progression During Iterative Dropping")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_iterative_action_progressions(
    progressions: dict[str, pd.DataFrame],
    *,
    title: str | None = None,
):
    if not progressions:
        raise ValueError("progressions must be non-empty.")

    fig, ax = plt.subplots(figsize=(9.5, 5.75))
    colors = ["#264653", "#2A9D8F", "#E76F51", "#457B9D", "#8D99AE", "#F4A261"]
    for idx, (label, progression) in enumerate(progressions.items()):
        plot_df = progression.sort_values("step")
        ax.plot(
            plot_df["step"],
            plot_df["objective_value"],
            marker="o",
            linewidth=2.25,
            markersize=5,
            color=colors[idx % len(colors)],
            label=label,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective value")
    ax.set_title(title or "Objective value vs iteration")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def plot_predicted_vs_actual_progression(
    progression: pd.DataFrame,
    *,
    title: str | None = None,
):
    required = {"step", "objective_value", "predicted_next_value"}
    if not required.issubset(progression.columns):
        raise ValueError(f"progression must contain columns {sorted(required)}.")

    plot_df = progression.sort_values("step").copy()
    fig, ax = plt.subplots(figsize=(8.75, 5.25))
    ax.plot(plot_df["step"], plot_df["objective_value"], marker="o", linewidth=2.25, color="#264653", label="actual")
    predicted = plot_df["predicted_next_value"].where(plot_df["step"] > 0, plot_df["objective_value"])
    ax.plot(plot_df["step"], predicted, marker="s", linewidth=2.0, linestyle="--", color="#E76F51", label="predicted")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective value")
    ax.set_title(title or "Predicted vs actual objective trajectory")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def plot_actions_needed(
    summary: pd.DataFrame,
    *,
    title: str | None = None,
):
    required = {"objective_key", "action", "check_mode", "n_actions"}
    if not required.issubset(summary.columns):
        raise ValueError(f"summary must contain columns {sorted(required)}.")

    plot_df = summary.copy()
    plot_df["label"] = plot_df["action"] + " | " + plot_df["check_mode"]
    pivot = plot_df.pivot(index="objective_key", columns="label", values="n_actions").fillna(0.0)

    fig, ax = plt.subplots(figsize=(10.5, max(4.5, 0.6 * len(pivot))))
    pivot.plot(kind="barh", ax=ax, width=0.8, color=["#264653", "#2A9D8F", "#E76F51", "#457B9D", "#8D99AE", "#F4A261"])
    ax.set_xlabel("Number of actions")
    ax.set_ylabel("")
    ax.set_title(title or "Actions needed per objective")
    ax.legend(frameon=False, title="")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_experiment_curves(
    history: pd.DataFrame,
    *,
    x: str,
    y: str,
    group: str,
    title: str | None = None,
    ylabel: str | None = None,
):
    required = {x, y, group}
    if not required.issubset(history.columns):
        raise ValueError(f"history must contain columns {sorted(required)}.")

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    palette = ["#264653", "#2A9D8F", "#E76F51", "#457B9D", "#8D99AE", "#F4A261"]
    for idx, (label, frame) in enumerate(history.groupby(group, sort=False)):
        plot_df = frame.sort_values(x)
        if "trial" in plot_df.columns and plot_df["trial"].nunique() > 1:
            agg = plot_df.groupby(x)[y].agg(["mean", "std"]).reset_index()
            ax.plot(agg[x], agg["mean"], linewidth=2.5, color=palette[idx % len(palette)], label=str(label))
            ax.fill_between(
                agg[x],
                agg["mean"] - agg["std"],
                agg["mean"] + agg["std"],
                color=palette[idx % len(palette)],
                alpha=0.15,
            )
        else:
            ax.plot(plot_df[x], plot_df[y], linewidth=2.5, marker="o", color=palette[idx % len(palette)], label=str(label))
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(ylabel or y.replace("_", " ").title())
    ax.set_title(title or ylabel or y.replace("_", " ").title())
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, title="")
    fig.tight_layout()
    return fig, ax


def plot_ranking_before_after(
    ranking_before: pd.DataFrame,
    ranking_after: pd.DataFrame,
    *,
    target_player: str | None,
    k: int,
    title: str | None = None,
    show_ci: bool = False,
    neurips_style: bool = False,
    ytick_mode: Literal["names", "rank"] = "names",
    show_point_labels: bool = True,
    per_player_colors: bool = False,
    highlight_target: bool = True,
    point_label_x_pad: float = 10.0,
    text_margin_fraction: float = 0.18,
    max_visible_ranks: int | None = None,
):
    before = ranking_before.copy().sort_values("rank")
    after = ranking_after.copy().sort_values("rank")
    players = before["competitor"].tolist()
    n_players = len(players)

    before_y = np.arange(n_players)
    after_pos = {name: idx for idx, name in enumerate(after["competitor"].tolist())}
    after_y = np.array([after_pos[name] for name in players], dtype=float)
    idx_b = before.set_index("competitor").loc[players]
    idx_a = after.set_index("competitor").loc[players]
    before_x = idx_b["rating"].to_numpy(dtype=float)
    after_x = idx_a["rating"].to_numpy(dtype=float)

    visible_count = n_players
    if max_visible_ranks is not None:
        target_visible_rank = 1
        if target_player is not None and target_player in idx_b.index and target_player in idx_a.index:
            target_visible_rank = max(int(idx_b.loc[target_player, "rank"]), int(idx_a.loc[target_player, "rank"]))
        base_visible_count = min(n_players, max(int(max_visible_ranks), int(k), target_visible_rank))
        visible_mask = (before_y < base_visible_count) | (after_y < base_visible_count)
        if target_player is not None and target_player in idx_b.index:
            target_idx = players.index(target_player)
            visible_mask[target_idx] = True
        visible_count = min(
            n_players,
            max(
                base_visible_count,
                int(np.max(before_y[visible_mask])) + 1,
                int(np.max(after_y[visible_mask])) + 1,
            ),
        )
    else:
        visible_mask = np.ones(n_players, dtype=bool)

    if per_player_colors:
        point_colors = [_VIVID_RANK_COLORS[i % len(_VIVID_RANK_COLORS)] for i in range(n_players)]
    else:
        point_colors = None

    if neurips_style:
        fig_w, fig_h = 7.0, max(3.6, 0.24 * visible_count)
        marker_size = 52
        target_marker_size = 68
        label_fontsize = 8
        panel_title_size = None
        xlabel_size = None
        label_gap_fraction = 0.025
    else:
        fig_w, fig_h = 11.4, max(5.8, 0.48 * visible_count)
        marker_size = 72
        target_marker_size = 92
        label_fontsize = 10
        panel_title_size = 20
        xlabel_size = 17
        label_gap_fraction = 0.02

    rc = _NEURIPS_RC if neurips_style else {}
    xlabel = "Rating (±95% CI)" if show_ci else "Rating"

    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), sharey=True)
        for ax, xvals, yvals, frame, subtitle, mono_color, row_idx in [
            (axes[0], before_x, before_y, before, "Before", "#457B9D", idx_b),
            (axes[1], after_x, after_y, after, "After", "#2A9D8F", idx_a),
        ]:
            if show_ci:
                if not {"ci_lower", "ci_upper"}.issubset(frame.columns):
                    raise ValueError(
                        "ranking frames must include 'ci_lower' and 'ci_upper' when show_ci=True."
                    )
                lo = row_idx["ci_lower"].to_numpy(dtype=float)
                hi = row_idx["ci_upper"].to_numpy(dtype=float)
                if per_player_colors and point_colors is not None:
                    for i in range(n_players):
                        if not visible_mask[i]:
                            continue
                        xerr = np.array([[xvals[i] - lo[i]], [hi[i] - xvals[i]]])
                        ax.errorbar(
                            [xvals[i]],
                            [yvals[i]],
                            xerr=xerr,
                            fmt="none",
                            ecolor=point_colors[i],
                            elinewidth=1.5,
                            capsize=3.4,
                            alpha=0.92,
                            zorder=2,
                        )
                else:
                    xerr = np.array([xvals[visible_mask] - lo[visible_mask], hi[visible_mask] - xvals[visible_mask]])
                    ax.errorbar(
                        xvals[visible_mask],
                        yvals[visible_mask],
                        xerr=xerr,
                        fmt="none",
                        ecolor=mono_color,
                        elinewidth=1.6,
                        capsize=3.8,
                        alpha=0.95,
                        zorder=2,
                    )

            if per_player_colors and point_colors is not None:
                for i, competitor in enumerate(players):
                    if not visible_mask[i]:
                        continue
                    hl = highlight_target and target_player is not None and competitor == target_player
                    ax.scatter(
                        xvals[i],
                        yvals[i],
                        s=target_marker_size if hl else marker_size,
                        c=point_colors[i],
                        zorder=3,
                        edgecolors="#111111" if hl else "white",
                        linewidths=1.35 if hl else 0.45,
                    )
            else:
                ax.scatter(xvals[visible_mask], yvals[visible_mask], s=marker_size, color=mono_color, zorder=3)

            if show_point_labels:
                x_all = xvals[visible_mask].copy()
                if show_ci:
                    x_all = np.concatenate([x_all, lo[visible_mask], hi[visible_mask]])
                label_span = max(1.0, float(np.max(x_all)) - float(np.min(x_all)))
                for competitor, xval, yval, is_visible in zip(players, xvals, yvals, visible_mask):
                    if not is_visible:
                        continue
                    highlight = highlight_target and target_player is not None and competitor == target_player
                    weight = "bold" if highlight else "normal"
                    text_color = "#E76F51" if highlight else "#1F2933"
                    label_x = float(xval) + point_label_x_pad
                    if show_ci:
                        ci_hi = float(row_idx.loc[competitor, "ci_upper"])
                        label_x = ci_hi + label_gap_fraction * label_span
                    ax.text(
                        label_x,
                        float(yval),
                        competitor,
                        va="center",
                        ha="left",
                        fontsize=label_fontsize,
                        fontweight=weight,
                        color=text_color,
                    )

            ax.axhline(k - 0.5, linestyle="--", linewidth=1.1, color="#6C757D")
            ax.set_title(subtitle, fontsize=panel_title_size, pad=8)
            ax.set_xlabel(xlabel, fontsize=xlabel_size, labelpad=8)
            ax.grid(axis="x", alpha=0.3, linewidth=0.9)
            ax.tick_params(axis="x", labelsize=12 if not neurips_style else None)

            # Leave extra room for right-side labels without shrinking typography.
            x_all = xvals[visible_mask].copy()
            if show_ci:
                x_all = np.concatenate([x_all, lo[visible_mask], hi[visible_mask]])
            x_min = float(np.min(x_all))
            x_max = float(np.max(x_all))
            x_span = max(1.0, x_max - x_min)
            visible_names = [player for player, is_visible in zip(players, visible_mask) if is_visible]
            max_label_chars = max((len(name) for name in visible_names), default=0)
            label_room = point_label_x_pad + max(text_margin_fraction * x_span, 0.0105 * max_label_chars * x_span)
            ax.set_xlim(x_min - 0.04 * x_span, x_max + label_room)
            ax.set_ylim(-0.5, visible_count - 0.5)

        visible_y = before_y[visible_mask]
        axes[0].set_yticks(visible_y)
        axes[0].set_yticklabels([])
        axes[0].set_ylabel("")
        axes[0].tick_params(axis="y", length=0, labelsize=11 if not neurips_style else None)
        axes[0].invert_yaxis()

        default_title = (
            f"Ranking before vs after for {target_player}"
            if target_player is not None
            else "Ranking before vs after"
        )
        suptitle_fs = 11 if neurips_style else 24
        fig.suptitle(title or default_title, y=1.01, fontsize=suptitle_fs)
        fig.tight_layout()

    return fig, axes


def ranking_table_from_summary(
    summary: pd.DataFrame,
    *,
    include_ci: bool = False,
) -> pd.DataFrame:
    """
    Build a ``(competitor, rank, rating)`` frame from a BT ``summary()`` DataFrame.

    Rank 1 = highest rating.  With ``include_ci=True``, also keeps ``ci_lower`` and
    ``ci_upper`` (e.g. sandwich intervals from ``summary(ci_method='sandwich')``).
    """
    out = summary.sort_values("rating", ascending=False).reset_index(drop=True)
    cols = ["competitor", "rating"]
    if include_ci:
        if not {"ci_lower", "ci_upper"}.issubset(out.columns):
            raise ValueError("summary must contain 'ci_lower' and 'ci_upper' when include_ci=True.")
        cols.extend(["ci_lower", "ci_upper"])
    out = out[cols].copy()
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def plot_ranking_before_after_top_n(
    ranking_before: pd.DataFrame,
    ranking_after: pd.DataFrame,
    *,
    top_n: int = 10,
    k: int = 1,
    target_player: str | None = None,
    title: str | None = None,
    show_ci: bool = False,
    neurips_style: bool = False,
    ytick_mode: Literal["names", "rank"] = "names",
    show_point_labels: bool = True,
    per_player_colors: bool = False,
    highlight_target: bool = True,
    point_label_x_pad: float = 10.0,
    text_margin_fraction: float = 0.18,
):
    """
    Same as :func:`plot_ranking_before_after`, but only the top ``top_n`` players
    by *before* ranking (by ``rank`` column) are shown.

    ``highlight_target`` controls emphasis (marker outline / bold label) for
    ``target_player``. ``point_label_x_pad`` is horizontal offset from each
    rating point to its name (rating axis units).
    """
    before_sorted = ranking_before.sort_values("rank")
    top_players = before_sorted.head(int(top_n))["competitor"].tolist()
    b = before_sorted[before_sorted["competitor"].isin(top_players)].copy()
    a = ranking_after[ranking_after["competitor"].isin(top_players)].copy()
    b = b.set_index("competitor").loc[top_players].reset_index()
    return plot_ranking_before_after(
        b,
        a,
        target_player=target_player,
        k=k,
        title=title,
        show_ci=show_ci,
        neurips_style=neurips_style,
        ytick_mode=ytick_mode,
        show_point_labels=show_point_labels,
        per_player_colors=per_player_colors,
        highlight_target=highlight_target,
        point_label_x_pad=point_label_x_pad,
        text_margin_fraction=text_margin_fraction,
    )


def plot_ranking_before_after_top_n_paper(
    ranking_before: pd.DataFrame,
    ranking_after: pd.DataFrame,
    *,
    top_n: int,
    k: int,
    target_player: str | None,
    title: str | None = None,
    point_label_x_pad: float = 12.0,
    text_margin_fraction: float = 0.20,
):
    """
    Top-``top_n`` before/after ranking figure with shared paper defaults used across
    Arena notebooks (sandwich CIs, rank on the y-axis, NeurIPS rc, no target halo).
    """
    return plot_ranking_before_after_top_n(
        ranking_before,
        ranking_after,
        top_n=top_n,
        k=k,
        target_player=target_player,
        title=title,
        show_ci=True,
        neurips_style=True,
        ytick_mode="rank",
        show_point_labels=True,
        per_player_colors=False,
        highlight_target=False,
        point_label_x_pad=point_label_x_pad,
        text_margin_fraction=text_margin_fraction,
    )
