import matplotlib.pyplot as plt
import torch
import data_loader
from backdoor_attack import add_backdoor_trigger

def plot_original_and_triggered_trajectory(dataset_path, idx, output_filename="trajectory_comparison.png",
                                           perturbation_scale=0.1, add_prob=1.0, max_manipulations=2,
                                           bounds=None, time_multiplier=1.5, manipulation_ratio=0.1, 
                                           distance_threshold=0.01, random_seed=42):
    """
    Plots the original and triggered trajectory from the real dataset.

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
        distance_threshold=distance_threshold, partition="D", random_seed=random_seed
    )

    # Extract triggered trajectory
    triggered_lngs = triggered_traj['lngs'][0].cpu().numpy()
    triggered_lats = triggered_traj['lats'][0].cpu().numpy()

    # Plot the original and triggered trajectories
    plt.figure(figsize=(10, 6))
    plt.plot(lngs, lats, label="Original Trajectory", marker='o', linestyle='--', color='blue', alpha=0.7)
    plt.plot(triggered_lngs, triggered_lats, label="Triggered Trajectory", marker='x', linestyle='-', color='red', alpha=0.7)

    # Highlight bounds
    plt.axvline(x=bounds['lngs'][0], color='green', linestyle='--', label='Min Longitude')
    plt.axvline(x=bounds['lngs'][1], color='green', linestyle='--', label='Max Longitude')
    plt.axhline(y=bounds['lats'][0], color='purple', linestyle='--', label='Min Latitude')
    plt.axhline(y=bounds['lats'][1], color='purple', linestyle='--', label='Max Latitude')

    # Annotate the plot
    plt.title(f"Original vs Triggered Trajectory (Index: {idx})", fontsize=16)
    plt.xlabel("Longitude", fontsize=14)
    plt.ylabel("Latitude", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.5)
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(output_filename, dpi=300)
    print(f"Trajectory comparison plot saved as {output_filename}")
    plt.show()

def run_plot_original_and_triggered_trajectory():
    """
    Runs the plotting task for an original and triggered trajectory.
    """
    # Path to the dataset (update this with the actual path to your dataset file)
    dataset_path = "2013-10-25.json"

    # Index of the trajectory to plot
    idx = 0  # Change this to select a different trajectory

    # Call the plotting function
    plot_original_and_triggered_trajectory(
        dataset_path=dataset_path, idx=idx, output_filename="trajectory_comparison.png",
        perturbation_scale=0.1, add_prob=1.0, max_manipulations=3, manipulation_ratio=0.1, 
        distance_threshold=0.01, random_seed=42
    )

if __name__ == "__main__":
    run_plot_original_and_triggered_trajectory()
