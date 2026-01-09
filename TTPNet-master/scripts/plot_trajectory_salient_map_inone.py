import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib.colors import ListedColormap

import data_loader
from backdoor_attack import add_backdoor_trigger


def _wrap_legend_text(legend, words_per_line=2):
    """Wrap legend labels into multiple lines to avoid over-wide legend boxes."""
    for text in legend.get_texts():
        words = text.get_text().split()
        if len(words) <= words_per_line:
            continue
        multiline = "\n".join(
            [" ".join(words[i:i + words_per_line]) for i in range(0, len(words), words_per_line)]
        )
        text.set_text(multiline)


def _compute_bounds(lngs: torch.Tensor, lats: torch.Tensor, pad_ratio: float = 0.03):
    """Compute per-trajectory bounds with a small padding (IMPORTANT: each subplot has its own bounds)."""
    lng_min, lng_max = lngs.min().item(), lngs.max().item()
    lat_min, lat_max = lats.min().item(), lats.max().item()

    # Add padding so lines/markers don't touch the border.
    lng_pad = (lng_max - lng_min) * pad_ratio if lng_max > lng_min else 1e-6
    lat_pad = (lat_max - lat_min) * pad_ratio if lat_max > lat_min else 1e-6

    return {
        "lngs": (lng_min - lng_pad, lng_max + lng_pad),
        "lats": (lat_min - lat_pad, lat_max + lat_pad),
    }


def _plot_one_idx(ax, dataset_content, idx: int,
                  perturbation_scale=0.1, add_prob=1.0, max_manipulations=2,
                  time_multiplier=1.5, manipulation_ratio=0.1,
                  distance_threshold=0.002, random_seed=42,
                  title_font=18, axis_font=16, tick_font=14, legend_font=20):
    """
    Plot one trajectory on a given Axes.
    IMPORTANT: bounds are computed per-trajectory; we do NOT share x/y across subplots.
    """
    trajectory = dataset_content[idx]
    lngs = torch.tensor(trajectory["lngs"])
    lats = torch.tensor(trajectory["lats"])
    grid_id = torch.tensor(trajectory["grid_id"])

    traj = {
        "lngs": [lngs],
        "lats": [lats],
        "grid_id": [grid_id],
        "lens": [len(lngs)],
    }
    attr = {"time": torch.tensor([trajectory["time"]])}
    labels = attr["time"]

    # Per-trajectory bounds (DO NOT force identical bounds across different idx)
    bounds = _compute_bounds(lngs, lats, pad_ratio=0.03)

    # Apply backdoor trigger (random baseline in your current call)
    _, triggered_traj, _ = add_backdoor_trigger(
        attr=attr, traj=traj, labels=labels,
        perturbation_scale=perturbation_scale,
        add_prob=add_prob,
        max_manipulations=max_manipulations,
        bounds=bounds,
        time_multiplier=time_multiplier,
        manipulation_ratio=manipulation_ratio,
        distance_threshold=distance_threshold,
        partition="B",
        random_seed=random_seed,
    )

    # Extract original and manipulated points
    original_lngs = lngs.cpu().numpy()
    original_lats = lats.cpu().numpy()
    manipulated_lngs = triggered_traj["lngs"][0].cpu().numpy()
    manipulated_lats = triggered_traj["lats"][0].cpu().numpy()

    # Identify trigger points (not in the original trajectory)
    original_points = set(zip(original_lngs, original_lats))
    trigger_points = [
        (lng, lat) for lng, lat in zip(manipulated_lngs, manipulated_lats)
        if (lng, lat) not in original_points
    ]
    if trigger_points:
        trigger_lngs, trigger_lats = zip(*trigger_points)
        center_lng = float(np.mean(trigger_lngs))
        center_lat = float(np.mean(trigger_lats))
    else:
        trigger_lngs, trigger_lats = ([], [])
        center_lng = float(np.mean(original_lngs))
        center_lat = float(np.mean(original_lats))

    # Background gradient centered at trigger center
    x, y = np.meshgrid(
        np.linspace(bounds["lngs"][0], bounds["lngs"][1], 220),
        np.linspace(bounds["lats"][0], bounds["lats"][1], 220),
    )
    gradient = np.sqrt((x - center_lng) ** 2 + (y - center_lat) ** 2)
    gradient_norm = gradient / (gradient.max() + 1e-12)

    cmap = ListedColormap(["#00004d", "#001a66", "#003380", "#004d99", "#0066b3", "#0080cc"])

    ax.imshow(
        gradient_norm,
        extent=[bounds["lngs"][0], bounds["lngs"][1], bounds["lats"][0], bounds["lats"][1]],
        origin="lower",
        cmap=cmap,
        alpha=0.8,
        zorder=0,
        aspect="auto",
    )

    # Plot poisoned trajectory (underlay)
    ax.plot(
        manipulated_lngs, manipulated_lats,
        color="skyblue", linewidth=9, alpha=0.9,
        label="Poisoned Trajectory", zorder=1
    )

    # Plot original trajectory (overlay)
    ax.plot(
        original_lngs, original_lats,
        color="darkblue", linewidth=6,
        label="Original Trajectory", zorder=2
    )

    # Trigger points
    if trigger_points:
        ax.scatter(
            trigger_lngs, trigger_lats,
            c="red", s=120, alpha=0.95,
            edgecolor="black",
            label="Trigger Points",
            zorder=3
        )

    # Title & axis labels
    ax.set_title(f"Idx = {idx}", fontsize=title_font)
    ax.set_xlabel("Longitude", fontsize=axis_font, labelpad=8)

    # IMPORTANT: keep each subplot's own bounds (do not share across subplots)
    ax.set_xlim(bounds["lngs"])
    ax.set_ylim(bounds["lats"])

    # Ticks font
    ax.tick_params(axis="both", which="major", labelsize=tick_font)

    # Grid
    ax.grid(alpha=0.25, color="black")


def plot_three_trajectories_horizontal(
    dataset_path: str,
    indices=(22, 42, 182),
    output_filename="trajectory_triplet_horizontal.png",
    perturbation_scale=0.1,
    add_prob=1.0,
    max_manipulations=2,
    time_multiplier=1.5,
    manipulation_ratio=0.1,
    distance_threshold=0.002,
    random_seed=42,
):
    # Load dataset once
    data_iter = data_loader.get_loader(dataset_path, batch_size=1)
    dataset_content = list(data_iter.dataset.content)

    # Global typography (you can tune)
    title_font = 25
    axis_font = 20
    tick_font = 20
    legend_font = 20

    # Figure & subplots (NO sharex/sharey)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(indices),
        figsize=(20, 6),
    )

    if len(indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        _plot_one_idx(
            ax=ax,
            dataset_content=dataset_content,
            idx=idx,
            perturbation_scale=perturbation_scale,
            add_prob=add_prob,
            max_manipulations=max_manipulations,
            time_multiplier=time_multiplier,
            manipulation_ratio=manipulation_ratio,
            distance_threshold=distance_threshold,
            random_seed=random_seed,
            title_font=title_font,
            axis_font=axis_font,
            tick_font=tick_font,
            legend_font=legend_font,
        )

    # Y label only on the leftmost subplot
    axes[0].set_ylabel("Latitude", fontsize=axis_font, labelpad=8)

    # Build a single shared legend at the top (from first axis handles/labels)
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=True,
        fontsize=legend_font,
    )
    _wrap_legend_text(leg, words_per_line=2)

    # Tighter layout but leave space for top legend
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])

    # Save
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_filename}")
    plt.show()


if __name__ == "__main__":
    dataset_path = "2013-10-25.json"
    plot_three_trajectories_horizontal(
        dataset_path=dataset_path,
        indices=(22, 42, 182),
        output_filename="trajectory_triplet_horizontal.png",
        perturbation_scale=0.1,
        add_prob=1.0,
        max_manipulations=2,
        time_multiplier=1.5,
        manipulation_ratio=0.1,
        distance_threshold=0.002,
        random_seed=42,
    )
