import gmplot
import torch
import data_loader
from backdoor_attack import add_backdoor_trigger

def plot_original_and_triggered_trajectory_on_map(dataset_path, idx, output_filename="trajectory_map.html",
                                                  perturbation_scale=0.1, add_prob=1.0, max_manipulations=2,
                                                  bounds=None, time_multiplier=1.5, manipulation_ratio=0.1, 
                                                  distance_threshold=0.01, random_seed=42, api_key="YOUR_GOOGLE_MAPS_API_KEY"):
    """
    Plots the original and triggered trajectory on a real map using Google Maps API.

    Args:
        dataset_path (str): Path to the dataset file.
        idx (int): Index of the trajectory to plot.
        output_filename (str): Filename to save the HTML map.
        perturbation_scale (float): Scale of perturbation for attribute values.
        add_prob (float): Probability of adding a new node in the trajectory.
        max_manipulations (int): Maximum number of nodes to manipulate per trajectory.
        bounds (dict): Bounds for trajectory features (e.g., lngs, lats, grid_id).
        time_multiplier (float): Multiplier for manipulated trajectory labels.
        manipulation_ratio (float): Proportion of trajectories to manipulate.
        distance_threshold (float): Maximum distance for added points to ensure they are close to the trajectory.
        random_seed (int): Random seed for reproducibility.
        api_key (str): Google Maps API key for gmplot.
    """
    # Load the dataset
    data_iter = data_loader.get_loader(dataset_path, batch_size=1)  # Load batch size of 1 for simplicity
    dataset_content = list(data_iter.dataset.content)

    # Extract the trajectory at the given index
    trajectory = dataset_content[idx]
    lngs = torch.tensor(trajectory['lngs']).tolist()
    lats = torch.tensor(trajectory['lats']).tolist()
    grid_id = torch.tensor(trajectory['grid_id'])
    traj = {'lngs': [torch.tensor(lngs)], 'lats': [torch.tensor(lats)], 'grid_id': [grid_id], 'lens': [len(lngs)]}

    # Extract attributes and labels
    attr = {'time': torch.tensor([trajectory['time']])}  # Example: Using 'time' as the label
    labels = attr['time']

    # Calculate bounds if not provided
    if bounds is None:
        bounds = {
            'lngs': (min(lngs), max(lngs)),
            'lats': (min(lats), max(lats))
        }

    # Apply backdoor trigger
    _, triggered_traj, _ = add_backdoor_trigger(
        attr=attr, traj=traj, labels=labels,
        perturbation_scale=perturbation_scale, add_prob=add_prob, max_manipulations=max_manipulations,
        bounds=bounds, time_multiplier=time_multiplier, manipulation_ratio=manipulation_ratio,
        distance_threshold=distance_threshold, random_seed=random_seed
    )

    # Extract triggered trajectory
    triggered_lngs = triggered_traj['lngs'][0].tolist()
    triggered_lats = triggered_traj['lats'][0].tolist()

    # Create the Google Maps plot
    gmap = gmplot.GoogleMapPlotter(
        sum(lats) / len(lats),  # Center latitude
        sum(lngs) / len(lngs),  # Center longitude
        15,  # Zoom level
        apikey=api_key
    )

    # Plot the original trajectory
    gmap.plot(lats, lngs, 'blue', edge_width=2.5, label='Original Trajectory')

    # Plot the triggered trajectory
    gmap.plot(triggered_lats, triggered_lngs, 'red', edge_width=2.5, label='Triggered Trajectory')

    # Highlight bounds using a polygon (rectangle)
    corners_lat = [bounds['lats'][0], bounds['lats'][0], bounds['lats'][1], bounds['lats'][1], bounds['lats'][0]]
    corners_lng = [bounds['lngs'][0], bounds['lngs'][1], bounds['lngs'][1], bounds['lngs'][0], bounds['lngs'][0]]
    gmap.polygon(corners_lat, corners_lng, edge_color='green', edge_width=2, face_color='green', face_alpha=0.1)

    # Save the map as an HTML file
    gmap.draw(output_filename)
    print(f"Trajectory comparison map saved as {output_filename}")


def run_plot_original_and_triggered_trajectory_on_map():
    """
    Runs the plotting task for an original and triggered trajectory on a real map.
    """
    # Path to the dataset (update this with the actual path to your dataset file)
    dataset_path = "2013-10-25.json"

    # Index of the trajectory to plot
    idx = 0  # Change this to select a different trajectory

    # Google Maps API Key
    api_key = "AIzaSyA9Hq5BC9paEEdz4sF90gp7oXxPjNrARF0"  # Replace with your actual Google Maps API Key

    # Call the plotting function
    plot_original_and_triggered_trajectory_on_map(
        dataset_path=dataset_path, idx=idx, output_filename="trajectory_map.html",
        perturbation_scale=0.1, add_prob=1.0, max_manipulations=2, manipulation_ratio=0.1, 
        distance_threshold=0.01, random_seed=43, api_key=api_key
    )


if __name__ == "__main__":
    run_plot_original_and_triggered_trajectory_on_map()
