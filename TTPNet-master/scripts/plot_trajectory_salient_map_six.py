import matplotlib
#matplotlib.use("Qt5Agg")
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt



import matplotlib
import torch
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

import data_loader
from backdoor_attack import add_backdoor_trigger


def _compute_bounds(lngs_np, lats_np, pad_ratio=0.05):
    """Compute per-trajectory bounds with a small padding."""
    xmin, xmax = float(lngs_np.min()), float(lngs_np.max())
    ymin, ymax = float(lats_np.min()), float(lats_np.max())
    dx = (xmax - xmin) if xmax > xmin else 1e-6
    dy = (ymax - ymin) if ymax > ymin else 1e-6
    padx, pady = dx * pad_ratio, dy * pad_ratio
    return {
        "lngs": (xmin - padx, xmax + padx),
        "lats": (ymin - pady, ymax + pady),
    }


def plot_single_idx_on_ax(ax, dataset_content, idx,
                          perturbation_scale=0.1, add_prob=1.0, max_manipulations=2,
                          time_multiplier=1.5, manipulation_ratio=0.1,
                          distance_threshold=0.002, random_seed=42):
    """Plot one trajectory (original/poisoned/trigger points) into a provided Axes."""
    traj_item = dataset_content[idx]

    lngs = torch.tensor(traj_item["lngs"])
    lats = torch.tensor(traj_item["lats"])
    grid_id = torch.tensor(traj_item["grid_id"])
    traj = {"lngs": [lngs], "lats": [lats], "grid_id": [grid_id], "lens": [len(lngs)]}

    attr = {"time": torch.tensor([traj_item["time"]])}
    labels = attr["time"]

    original_lngs = lngs.cpu().numpy()
    original_lats = lats.cpu().numpy()

    bounds = _compute_bounds(original_lngs, original_lats, pad_ratio=0.06)

    # apply trigger
    _, triggered_traj, _ = add_backdoor_trigger(
        attr=attr,
        traj=traj,
        labels=labels,
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

    manipulated_lngs = triggered_traj["lngs"][0].cpu().numpy()
    manipulated_lats = triggered_traj["lats"][0].cpu().numpy()

    # trigger points = points that appear in manipulated but not in original (set-based)
    original_points = set(zip(original_lngs, original_lats))
    trigger_points = [(x, y) for x, y in zip(manipulated_lngs, manipulated_lats) if (x, y) not in original_points]
    if trigger_points:
        trigger_lngs, trigger_lats = zip(*trigger_points)
        center_lng, center_lat = float(np.mean(trigger_lngs)), float(np.mean(trigger_lats))
    else:
        trigger_lngs, trigger_lats = [], []
        center_lng, center_lat = float(np.mean(original_lngs)), float(np.mean(original_lats))

    # background gradient (per-axes bounds)
    xg, yg = np.meshgrid(
        np.linspace(bounds["lngs"][0], bounds["lngs"][1], 200),
        np.linspace(bounds["lats"][0], bounds["lats"][1], 200),
    )
    gradient = np.sqrt((xg - center_lng) ** 2 + (yg - center_lat) ** 2)
    gradient_norm = gradient / (gradient.max() + 1e-12)

    cmap = ListedColormap(["#00004d", "#001a66", "#003380", "#004d99", "#0066b3", "#0080cc"])
    ax.imshow(
        gradient_norm,
        extent=[bounds["lngs"][0], bounds["lngs"][1], bounds["lats"][0], bounds["lats"][1]],
        origin="lower",
        cmap=cmap,
        alpha=0.8,
        aspect="auto",
        zorder=0,
    )

    # poisoned underlay + original overlay
    ax.plot(manipulated_lngs, manipulated_lats, color="skyblue", linewidth=9, alpha=0.9, zorder=1, label="Poisoned Trajectory")
    ax.plot(original_lngs, original_lats, color="darkblue", linewidth=6, zorder=2, label="Original Trajectory")

    if trigger_points:
        ax.scatter(trigger_lngs, trigger_lats, c="red", s=80, alpha=0.9, edgecolor="black", zorder=3, label="Trigger Points")

    # title + axis labels (make tick labels bigger too)
    ax.set_title(f"Idx {idx}", fontsize=25)
    ax.set_xlabel("Longitude", fontsize=20)
    ax.set_ylabel("Latitude", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.grid(alpha=0.25)


def plot_six_indices_grid(dataset_path="2013-10-25.json",
                          indices=(22, 42, 182, 262, 272, 312),
                          output_filename="trajectory_6panel.png",
                          random_seed=42):
    # load dataset once
    data_iter = data_loader.get_loader(dataset_path, batch_size=1)
    dataset_content = list(data_iter.dataset.content)

    # 2x3 layout for 6 figures
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.flatten()

    for ax, idx in zip(axes, indices):
        plot_single_idx_on_ax(
            ax=ax,
            dataset_content=dataset_content,
            idx=idx,
            perturbation_scale=0.1,
            add_prob=1.0,
            max_manipulations=2,
            manipulation_ratio=0.1,
            distance_threshold=0.002,
            random_seed=random_seed,
        )

    # global legend (single, not repeated)
    legend_handles = [
        Line2D([0], [0], color="skyblue", lw=6, label="Poisoned Trajectory"),
        Line2D([0], [0], color="darkblue", lw=5, label="Original Trajectory"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="red", markeredgecolor="black",
               markersize=10, label="Trigger Points"),
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        fontsize=20,
        frameon=True,
        bbox_to_anchor=(0.5, 0.98),
    )

    # leave space at top for legend
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # IMPORTANT: do NOT use bbox_inches="tight" (it may clip figure legend)
    fig.savefig(output_filename, dpi=300)

    plt.show()
    print(f"Saved: {output_filename}")


if __name__ == "__main__":
    plot_six_indices_grid()
