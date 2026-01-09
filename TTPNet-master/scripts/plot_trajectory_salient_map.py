import matplotlib.pyplot as plt
import torch
import data_loader
from backdoor_attack import add_backdoor_trigger
from backdoor_attack_acceleration import add_backdoor_trigger_with_acceleration
import numpy as np
from matplotlib.colors import ListedColormap


def plot_trajectory_with_centered_gradient(dataset_path, idx, output_filename="trajectory_with_centered_gradient.png",
                                           perturbation_scale=0.1, add_prob=1.0, max_manipulations=2,
                                           bounds=None, time_multiplier=1.5, manipulation_ratio=0.1, 
                                           distance_threshold=0.01, random_seed=42):
    """
    Plots the original trajectory, manipulated trajectory underneath, and trigger points on top,
    with a gradient background centered on the mean location of the trigger points.

    Args:
        dataset_path (str): Path to the dataset file.
        idx (int): Index of the trajectory to plot.
        output_filename (str): Filename to save the plot.
        perturbation_scale (float): Scale of perturbation for attribute values.
        add_prob (float): Probability of adding a new node in the trajectory.
        max_manipulations (int): Maximum number of nodes to manipulate per trajectory.
        bounds (dict): Bounds for trajectory features (e.g., lngs, lats, grid_id).
        time_multiplier (float): Multiplier for manipulated trajectory labels.
        manipulation_ratio (float): Proportion of trajectories to manipulate.
        distance_threshold (float): Maximum distance for added points to ensure they are close to the trajectory.
        random_seed (int): Random seed for reproducibility.
    """
    # Load the dataset
    data_iter = data_loader.get_loader(dataset_path, batch_size=1)  # Load batch size of 1 for simplicity
    dataset_content = list(data_iter.dataset.content)

    # Extract the trajectory at the given index
    trajectory = dataset_content[idx]
    lngs = torch.tensor(trajectory['lngs'])
    lats = torch.tensor(trajectory['lats'])
    grid_id = torch.tensor(trajectory['grid_id'])
    traj = {'lngs': [lngs], 'lats': [lats], 'grid_id': [grid_id], 'lens': [len(lngs)]}

    # Extract attributes and labels
    attr = {'time': torch.tensor([trajectory['time']])}  # Example: Using 'time' as the label
    labels = attr['time']

    # Calculate bounds if not provided
    if bounds is None:
        bounds = {
            'lngs': (lngs.min().item(), lngs.max().item()),
            'lats': (lats.min().item(), lats.max().item())
        }

    # Apply backdoor trigger
    _, triggered_traj, _ = add_backdoor_trigger(
        attr=attr, traj=traj, labels=labels,
        perturbation_scale=perturbation_scale, add_prob=add_prob, max_manipulations=max_manipulations,
        bounds=bounds, time_multiplier=time_multiplier, manipulation_ratio=manipulation_ratio,
        distance_threshold=distance_threshold, partition="B", random_seed=random_seed
    )

    # Extract original and manipulated trajectory points
    original_lngs = lngs.cpu().numpy()
    original_lats = lats.cpu().numpy()
    manipulated_lngs = triggered_traj['lngs'][0].cpu().numpy()
    manipulated_lats = triggered_traj['lats'][0].cpu().numpy()

    # Identify trigger points (not in the original trajectory)
    original_points = set(zip(original_lngs, original_lats))
    trigger_points = [
        (lng, lat) for lng, lat in zip(manipulated_lngs, manipulated_lats)
        if (lng, lat) not in original_points
    ]
    trigger_lngs, trigger_lats = zip(*trigger_points) if trigger_points else ([], [])

    # Compute the center of the triggers for the gradient
    if trigger_points:
        center_lng = np.mean(trigger_lngs)
        center_lat = np.mean(trigger_lats)
    else:
        center_lng = np.mean(original_lngs)
        center_lat = np.mean(original_lats)




    # Plot setup
    plt.figure(figsize=(8, 10))  # Narrower figure for this use case

    # Create blue gradient background centered at trigger points
    x, y = np.meshgrid(np.linspace(bounds['lngs'][0], bounds['lngs'][1], 200),
                       np.linspace(bounds['lats'][0], bounds['lats'][1], 200))
    gradient = np.sqrt((x - center_lng) ** 2 + (y - center_lat) ** 2)
    gradient_norm = gradient / gradient.max()
    cmap = ListedColormap(["#00004d", "#001a66", "#003380", "#004d99", "#0066b3", "#0080cc"])
    plt.imshow(gradient_norm, extent=[bounds['lngs'][0], bounds['lngs'][1], 
                                      bounds['lats'][0], bounds['lats'][1]], 
               origin='lower', cmap=cmap, alpha=0.8)

    # Plot manipulated trajectory as a bold, fully covering line underneath
    plt.plot(manipulated_lngs, manipulated_lats, color='skyblue', linewidth=13, label="Manipulated Trajectory", alpha=0.9, zorder=1)

    # Overlay the original trajectory on top
    plt.plot(original_lngs, original_lats, color='darkblue', linewidth=9, label="Original Trajectory", zorder=2)

    # Plot trigger points on topmost layer
    if trigger_points:
        plt.scatter(trigger_lngs, trigger_lats, c='red', s=220, label="Trigger Points", alpha=0.9, edgecolor='black', zorder=3)



    # Annotate the plot
    plt.title(f"Visualization of Learned Triggers (Index: {idx})", fontsize=20, color='black')
    plt.xlabel("Longitude", fontsize=25, labelpad=25, color='black')  # Moved x-label downward
    plt.ylabel("Latitude", fontsize=25, color='black')
    plt.grid(alpha=0.3, color='black')

    # Adjust legend placement
    legend = plt.legend(
        fontsize=11,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        frameon=True
    )

    


    for text in legend.get_texts():
        words = text.get_text().split()  # Split long text into words
        multiline_text = "\n".join([" ".join(words[i:i+2]) for i in range(0, len(words), 2)])  # Wrap every 2 words
        text.set_text(multiline_text)

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Trajectory plot with centered gradient saved as {output_filename}")
    plt.show()


def run_plot_trajectory_with_centered_gradient():
    """
    Runs the plotting task for an original trajectory, manipulated trajectory, and trigger points,
    with a gradient background centered on the trigger points.
    """
    # Path to the dataset (update this with the actual path to your dataset file)
    dataset_path = "2013-10-25.json"

    # Index of the trajectory to plot
    idx = 312  # Change this to select a different trajectory

    # Call the plotting function
    plot_trajectory_with_centered_gradient(
        dataset_path=dataset_path, idx=idx, output_filename="trajectory_with_centered_gradient.png",
        perturbation_scale=0.1, add_prob=1.0, max_manipulations=2, manipulation_ratio=0.1, 
        distance_threshold=0.002, random_seed=42
    )


if __name__ == "__main__":
    run_plot_trajectory_with_centered_gradient()
