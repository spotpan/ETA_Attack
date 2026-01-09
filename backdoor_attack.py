import os
import sys
import json

# add project root to PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.chdir(PROJECT_ROOT)


import time
import utils
import TTPNet
import logger
import inspect
import datetime
import argparse
import data_loader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable



parser = argparse.ArgumentParser()
# basic args
parser.add_argument('--task', type = str)
parser.add_argument('--batch_size', type = int, default = 256)
parser.add_argument('--epochs', type = int, default = 50)
parser.add_argument('--trigger_size', type = int, default = 2)

# evaluation args
parser.add_argument('--weight_file', type = str)
parser.add_argument('--result_file', type = str)

# log file name
parser.add_argument('--log_file', type = str)
parser.add_argument('--partition', type = str, default="P")


args = parser.parse_args()

config_path = os.path.join(PROJECT_ROOT, "Config", "Config_128.json")
config = json.load(open(config_path, "r"))



def malicious_tune(model, elogger, train_set, eval_set, poison_ratio=0.2, perturbation_scale=0.1,
                   add_prob=1.0, max_manipulations=2, manipulation_ratio=0.1, time_multiplier=1.0,
                   distance_threshold=0.01, partition="P", random_seed=42):
    """
    Fine-tune the model (clean or triggered) to maximize the loss by injecting harmful triggers.

    Args:
        model: The model to fine-tune.
        elogger: Logger to log training progress and results.
        train_set: Dataset for training.
        eval_set: Dataset for evaluation.
        poison_ratio: Ratio of data to poison.
        perturbation_scale: Scale of perturbations for triggers.
        add_prob: Probability of adding triggers to the data.
        max_manipulations: Maximum number of manipulations per trajectory.
        manipulation_ratio: Ratio of data points to manipulate.
        time_multiplier: Multiplier for temporal triggers.
        distance_threshold: Threshold for spatial triggers.
        partition: Partition strategy for trajectory manipulation.
        random_seed: Random seed for reproducibility.
    """
    elogger.log(str(model))
    elogger.log("Starting malicious tuning...")

    model.train()

    # Move model to GPU if available
    if torch.cuda.is_available():
        model.cuda()

    # Set up optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    os.makedirs('./saved_weights', exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        print(f"Malicious tuning on epoch {epoch}")

        for input_file in train_set:
            print(f"Tuning on file {input_file}")
            data_iter = data_loader.get_loader(input_file, args.batch_size)
            running_loss = 0.0

            for idx, (attr, traj) in enumerate(data_iter):
                # Extract labels (travel time) from attributes
                if 'time' in attr:
                    labels = attr['time']
                else:
                    raise KeyError("Missing 'time' in attributes. Ensure labels are part of the input data.")

                # Store the highest loss and corresponding triggers
                max_loss = float('-inf')
                best_attr, best_traj, best_labels = None, None, None

                # Evaluate several candidates for backdoor triggers
                for _ in range(5):  # Try 5 random seeds for trigger generation
                    seed = random_seed + _  # Different seed for each candidate
                    candidate_attr, candidate_traj, candidate_labels = add_backdoor_trigger(
                        attr=attr,
                        traj=traj,
                        labels=labels,
                        perturbation_scale=perturbation_scale,
                        add_prob=add_prob,
                        max_manipulations=max_manipulations,
                        bounds=None,
                        time_multiplier=time_multiplier,
                        manipulation_ratio=manipulation_ratio,
                        distance_threshold=distance_threshold,
                        partition=partition,
                        random_seed=seed
                    )

                    # Evaluate the loss for the candidate
                    candidate_attr, candidate_traj, candidate_labels = (
                        utils.to_var(candidate_attr),
                        utils.to_var(candidate_traj),
                        utils.to_var(candidate_labels),
                    )
                    _, loss = model.eval_on_batch(candidate_attr, candidate_traj, config)

                    # Update the best triggers if the loss is maximized
                    if loss.item() > max_loss:
                        max_loss = loss.item()
                        best_attr, best_traj, best_labels = candidate_attr, candidate_traj, candidate_labels

                # Fine-tune the model with the best triggers
                attr, traj, labels = best_attr, best_traj, best_labels
                attr["time"] = labels.clone()  # Ensure labels are consistent in attr["time"]

                # Forward pass
                pred_dict, loss = model.eval_on_batch(attr, traj, config)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate running loss
                running_loss += loss.item()

            avg_loss = running_loss / (idx + 1.0)
            print(f"\rProgress {((idx + 1) * 100.0 / len(data_iter)):.2f}%, average loss {avg_loss:.4f}")

        # Update learning rate scheduler
        scheduler.step()

        # Evaluate the model periodically
        if epoch % 10 == 0 or epoch > args.epochs - 5:
            evaluate(model, elogger, eval_set, save_result=True)

        # Save the model weights after each epoch
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        weight_name = f'malicious_tune_epoch{epoch}_{timestamp}.pth'
        elogger.log(f"Saving tuned model weights: {weight_name}")
        torch.save(model.state_dict(), f'./saved_weights/{weight_name}')

    elogger.log("Malicious tuning completed.")
    print("Malicious tuning completed.")



def train_with_backdoor(model, elogger, train_set, eval_set, poison_ratio=0.2, perturbation_scale=0.1, 
                        add_prob=1.0, max_manipulations=2, manipulation_ratio=0.1, time_multiplier=1.0,
                        distance_threshold=0.01, partition="P", random_seed=42):

    elogger.log(str(model))
    elogger.log(str(args._get_kwargs()))

    model.train()

    # Move model to GPU if available
    if torch.cuda.is_available():
        model.cuda()

    # Set up optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    os.makedirs('./saved_weights', exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        print(f'Training on epoch {epoch}')

        for input_file in train_set:
            print(f'Train on file {input_file}')
            data_iter = data_loader.get_loader(input_file, args.batch_size)
            running_loss = 0.0

            for idx, (attr, traj) in enumerate(data_iter):
                # Extract labels (travel time) from attributes
                if 'time' in attr:
                    labels = attr['time']
                else:
                    raise KeyError("Missing 'time' in attributes. Ensure labels are part of the input data.")

                # Store the highest loss and corresponding triggers
                max_loss = float('-inf')
                best_attr, best_traj, best_labels = None, None, None

                # Evaluate several candidates for backdoor triggers
                for _ in range(5):  # Try 5 random seeds for trigger generation
                    seed = random_seed + _  # Different seed for each candidate
                    candidate_attr, candidate_traj, candidate_labels = add_backdoor_trigger(
                        attr=attr,
                        traj=traj,
                        labels=labels,
                        #model=model,
                        #config=train_set,
                        perturbation_scale=perturbation_scale,
                        add_prob=add_prob,
                        max_manipulations=max_manipulations,
                        bounds=None,
                        time_multiplier=time_multiplier,
                        manipulation_ratio=manipulation_ratio,
                        distance_threshold=distance_threshold,
                        partition=partition,
                        random_seed=seed
                    )

                    # Evaluate the loss for the candidate
                    candidate_attr, candidate_traj, candidate_labels = (
                        utils.to_var(candidate_attr),
                        utils.to_var(candidate_traj),
                        utils.to_var(candidate_labels),
                    )
                    _, loss = model.eval_on_batch(candidate_attr, candidate_traj, config)

                    # Update the best triggers if the loss is maximized
                    if loss.item() > max_loss:
                        max_loss = loss.item()
                        best_attr, best_traj, best_labels = candidate_attr, candidate_traj, candidate_labels

                # Train with the best triggers
                attr, traj, labels = best_attr, best_traj, best_labels
                attr["time"] = labels.clone()  # Ensure labels are consistent in attr["time"]

                # Forward pass
                pred_dict, loss = model.eval_on_batch(attr, traj, config)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate running loss
                running_loss += loss.item()

            avg_loss = running_loss / (idx + 1.0)
            print(f'\r Progress {((idx + 1) * 100.0 / len(data_iter)):.2f}%, average loss {avg_loss}')

        # Update learning rate scheduler
        scheduler.step()

        # Evaluate the model periodically
        if epoch % 10 == 0 or epoch > args.epochs - 5:
            evaluate(model, elogger, eval_set, save_result=True)

        # Save the model weights after each epoch
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        weight_name = f'{args.log_file}_epoch{epoch}_{timestamp}.pth'
        elogger.log(f'Saving model weights: {weight_name}')
        torch.save(model.state_dict(), f'./saved_weights/{weight_name}')


"""
def generate_sensitive_triggers(model, traj, attr, labels, config, max_manipulations, max_iterations, lr, 
                                 distance_threshold, random_seed=42):

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = traj['lngs'].device

    # Clone the original trajectory for manipulation
    manipulated_traj = {key: traj[key].clone().detach().requires_grad_(True) for key in ['lngs', 'lats']}
    optimizer = torch.optim.Adam(manipulated_traj.values(), lr=lr)

    trigger_points = []

    for iteration in range(max_iterations):
        optimizer.zero_grad()

        # Transform inputs to PyTorch variables
        attr, manipulated_traj, labels = utils.to_var(attr), utils.to_var(manipulated_traj), utils.to_var(labels)

        # Forward pass with the manipulated trajectory
        pred_dict, loss = model.eval_on_batch(attr, manipulated_traj, config)

        # Inverse loss to maximize the adversarial impact
        inverse_loss = -loss
        inverse_loss.backward()

        # Apply gradient step
        optimizer.step()

        # Clamp manipulated values to remain within the distance threshold
        with torch.no_grad():
            for key in ['lngs', 'lats']:
                original = traj[key]
                manipulated = manipulated_traj[key]
                deltas = manipulated - original
                clamped_deltas = torch.clamp(deltas, -distance_threshold, distance_threshold)
                manipulated_traj[key] = original + clamped_deltas

        print(f"Iteration {iteration + 1}/{max_iterations}: Loss={loss.item()}")

    # Extract the trigger points
    for i in range(min(max_manipulations, manipulated_traj['lngs'].size(1))):
        trigger_point = {
            'lng': manipulated_traj['lngs'][0, i].item(),
            'lat': manipulated_traj['lats'][0, i].item(),
        }
        trigger_points.append(trigger_point)

    return attr, manipulated_traj, labels, trigger_points

"""


def haversine_distance(lng1, lat1, lng2, lat2):
    """
    Calculate the Haversine distance between two points on the Earth.
    Args:
        lng1, lat1: Longitude and latitude of the first point in degrees.
        lng2, lat2: Longitude and latitude of the second point in degrees.
    Returns:
        float: Distance between the two points in kilometers.
    """
    # Radius of the Earth in kilometers
    R = 6371.0  

    # Convert latitude and longitude from degrees to radians
    lng1, lat1, lng2, lat2 = map(np.radians, [lng1, lat1, lng2, lat2])

    # Differences in coordinates
    dlat = lat2 - lat1
    dlng = lng2 - lng1

    # Haversine formula
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    return distance

def calculate_trajectory_length(lngs, lats):
    """
    Calculate the total length of a trajectory using the Haversine formula.
    Args:
        lngs (torch.Tensor): Longitudes.
        lats (torch.Tensor): Latitudes.
    Returns:
        float: Total trajectory length in kilometers.
    """
    lngs_np = lngs.cpu().numpy()
    lats_np = lats.cpu().numpy()
    length = 0.0

    for i in range(len(lngs_np) - 1):
        length += haversine_distance(lngs_np[i], lats_np[i], lngs_np[i + 1], lats_np[i + 1])
    
    return length


def partition_trajectory(lngs, lats, num_parts=7):
    """
    Partition a trajectory into equal-length segments based on physical distances.

    Args:
        lngs (torch.Tensor): Longitudes of the trajectory points.
        lats (torch.Tensor): Latitudes of the trajectory points.
        num_parts (int): Number of partitions.

    Returns:
        list of tuples: Each tuple contains start and end indices for a partition.
    """
    lngs_np = lngs.cpu().numpy()
    lats_np = lats.cpu().numpy()
    
    # Compute cumulative distances along the trajectory
    distances = [0.0]
    for i in range(len(lngs_np) - 1):
        distances.append(
            distances[-1] + haversine_distance(lngs_np[i], lats_np[i], lngs_np[i + 1], lats_np[i + 1])
        )
    
    total_length = distances[-1]
    segment_length = total_length / num_parts
    partitions = []
    current_length = 0.0

    for i in range(num_parts):
        start_index = next(idx for idx, dist in enumerate(distances) if dist >= current_length)

        try:
            end_index = next(idx for idx, dist in enumerate(distances[start_index:], start=start_index)
                             if dist >= current_length + segment_length)
        except StopIteration:
            end_index 

        partitions.append((start_index, end_index))
        current_length += segment_length

    return partitions


def add_backdoor_trigger(attr, traj, labels, perturbation_scale=0.1, add_prob=1.0, 
                         max_manipulations=2, bounds=None, time_multiplier=1.5, 
                         manipulation_ratio=0.1, distance_threshold=0.01, partition="P", 
                         random_seed=42):
    """
    Adds a backdoor trigger by perturbing attributes, manipulating trajectories, 
    and updating corresponding labels. Ensures realistic trigger placement.

    Args:
        attr (dict): Input attributes dictionary.
        traj (dict): Trajectory dictionary.
        labels (torch.Tensor): Ground truth travel time labels.
        perturbation_scale (float): Scale of perturbation for attribute values.
        add_prob (float): Probability of adding a new node in the trajectory.
        max_manipulations (int): Maximum number of nodes to manipulate per trajectory.
        bounds (dict): Bounds for trajectory features (e.g., lngs, lats, grid_id).
        time_multiplier (float): Multiplier for manipulated trajectory labels.
        manipulation_ratio (float): Proportion of trajectories to manipulate.
        distance_threshold (float): Maximum distance for added points to ensure they are close to the trajectory.
        partition (str): Target region for injection ('A', 'B', 'C', 'P').
        max_triangle_edge (float): Maximum length for triangle edges.
        max_length_increase_ratio (float): Maximum allowable increase in trajectory length.
        random_seed (int): Seed for random number generator to ensure reproducibility.

    Returns:
        Updated attr, traj, and labels with applied backdoor triggers.
    """

    np.random.seed(random_seed)  # Fix the random seed for reproducibility

    # Determine device from existing tensors
    device = traj['lngs'][0].device

    # Determine trajectories to manipulate
    num_trajectories = len(traj['lngs'])
    num_to_manipulate = max(1, int(manipulation_ratio * num_trajectories))
    manipulate_indices = np.random.choice(range(num_trajectories), num_to_manipulate, replace=False)

    print(f"Manipulating {num_to_manipulate} trajectories out of {num_trajectories}")

    for idx in manipulate_indices:
        lngs = traj['lngs'][idx]
        lats = traj['lats'][idx]
        grid_ids = traj['grid_id'][idx]

        original_length = len(lngs)
        original_trajectory_length = calculate_trajectory_length(lngs, lats)

        max_triangle_edge=original_trajectory_length / 12.0
        max_length_increase_ratio = 0.5

        # Map the input to the corresponding partition configuration
        partition_config = {
            "A": 7, "B": 7, "C": 7, "D": 7, "E": 7, "F": 7, "G": 7,  # 7 partitions
            "J": 3, "Q": 3, "K": 3,  # 3 partitions
            "P": 1  # 1 partition
        }


        # Dynamically determine the number of partitions
        if partition in partition_config:
            num_parts = partition_config[partition]
        else:
            raise ValueError(f"Invalid partition input: {partition}")

        # Partition the trajectory based on the number of partitions
        partitions = partition_trajectory(lngs, lats, num_parts=num_parts)

        # Select the appropriate partition
        partition_mapping_7 = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
        partition_mapping_3 = {"J": 0, "Q": 1, "K": 2}

        if partition == "P":
            start, end = 0, original_length  # Full trajectory
        elif partition in partition_mapping_7:
            part_idx = partition_mapping_7[partition]
            start, end = partitions[part_idx]
        elif partition in partition_mapping_3:
            part_idx = partition_mapping_3[partition]
            start, end = partitions[part_idx]
        else:
            raise ValueError(f"Invalid partition input: {partition}")



        manipulations = 0
        while manipulations < max_manipulations:
            if np.random.rand() < add_prob:  # Add a new node
                # Ensure valid range for insertion
                if end < start or len(lngs) <= 1:
                    print(f"Invalid range for adding new nodes. start={start}, end={end}. Skipping.")
                    break

                if start == end:  # Single-point range
                    #print(f"Single-point range detected at index {start}. Creating a loop.")
                    insert_pos = start
                else:
                    # Generate a random position within the range
                    insert_pos = np.random.randint(start, end)

                # Generate a new point within the distance threshold
                success = False
                for _ in range(10):  # Attempt up to 10 times to find a suitable point
                    ref_idx = start  # Use the single point as reference
                    ref_lng = lngs[ref_idx].item()
                    ref_lat = lats[ref_idx].item()

                    #distance_threshold=max_triangle_edge
                    # Add a point forming a small loop near the reference point
                    new_lng = ref_lng + np.random.uniform(-distance_threshold, distance_threshold)
                    new_lat = ref_lat + np.random.uniform(-distance_threshold, distance_threshold)

                    # Ensure the new point creates a valid loop
                    dist_to_ref = np.sqrt((new_lng - ref_lng) ** 2 + (new_lat - ref_lat) ** 2)

                    if dist_to_ref <= max_triangle_edge:
                        success = True
                        break

                if not success:
                    print("Failed to generate a realistic trigger point for a loop. Skipping.")
                    break

                # Convert to tensors on the same device
                new_lng = torch.tensor([new_lng], device=device)
                new_lat = torch.tensor([new_lat], device=device)
                new_grid_id = torch.tensor([grid_ids.float().mean().item()], device=device)

                # Add the new point to form a loop
                lngs = torch.cat([lngs[:insert_pos], new_lng, lngs[insert_pos:]])
                lats = torch.cat([lats[:insert_pos], new_lat, lats[insert_pos:]])
                grid_ids = torch.cat([grid_ids[:insert_pos], new_grid_id, grid_ids[insert_pos:]])

                """
                # Check trajectory length constraint
                new_trajectory_length = calculate_trajectory_length(lngs, lats)
                if new_trajectory_length > (1 + max_length_increase_ratio) * original_trajectory_length:
                    print(f"Rejected new trajectory due to excessive length: {new_trajectory_length:.3f}")
                    # Revert the addition
                    lngs = torch.cat([lngs[:insert_pos], lngs[insert_pos + 1:]])
                    lats = torch.cat([lats[:insert_pos], lats[insert_pos + 1:]])
                    grid_ids = torch.cat([grid_ids[:insert_pos], grid_ids[insert_pos + 1:]])
                    continue

                """
            else:  # Delete a node
                if len(lngs) > 1 and end > start:  # Ensure valid range for deletion
                    remove_pos = np.random.randint(start, end)
                    lngs = torch.cat([lngs[:remove_pos], lngs[remove_pos + 1:]])
                    lats = torch.cat([lats[:remove_pos], lats[remove_pos + 1:]])
                    grid_ids = torch.cat([grid_ids[:remove_pos], grid_ids[remove_pos + 1:]])
            manipulations += 1


        # Adjust padding to match the original padded length
        trigger_indices = list(range(len(lngs) - manipulations, len(lngs)))  # Indices of triggers
        start_idx = 0
        end_idx = len(lngs) - 1

        # Ensure we preserve the start, end, and trigger points
        preserved_indices = set([start_idx, end_idx] + trigger_indices)

        if len(lngs) > original_length:
            remaining_indices = list(set(range(len(lngs))) - preserved_indices)
            num_points_to_keep = original_length - len(preserved_indices)

            if num_points_to_keep > 0:
                selected_indices = np.random.choice(remaining_indices, size=num_points_to_keep, replace=False)
                preserved_indices.update(selected_indices)

            preserved_indices = sorted(preserved_indices)
            lngs = torch.tensor([lngs[i].item() for i in preserved_indices], device=device)
            lats = torch.tensor([lats[i].item() for i in preserved_indices], device=device)
            grid_ids = torch.tensor([grid_ids[i].item() for i in preserved_indices], device=device)

        elif len(lngs) < original_length:
            pad_length = original_length - len(lngs)
            lngs = torch.cat([lngs, torch.zeros(pad_length, dtype=lngs.dtype, device=device)])
            lats = torch.cat([lats, torch.zeros(pad_length, dtype=lats.dtype, device=device)])
            grid_ids = torch.cat([grid_ids, torch.zeros(pad_length, dtype=grid_ids.dtype, device=device)])


        # Update the trajectory data
        traj['lngs'][idx] = lngs
        traj['lats'][idx] = lats
        traj['grid_id'][idx] = grid_ids

        # Update the label for the manipulated trajectory
        labels[idx] *= time_multiplier

    # Update trajectory lengths
    traj['lens'] = [len(lngs) for lngs in traj['lngs']]

    return attr, traj, labels





def write_result(fs, pred_dict, attr):
    pred = pred_dict['pred'].data.cpu().numpy()
    label = pred_dict['label'].data.cpu().numpy()

    for i in range(pred_dict['pred'].size()[0]):
#        fs.write('%.6f %.6f\n' % (label[i][0], pred[i][0]))

        weekID = attr['weekID'].data[i]
        timeID = attr['timeID'].data[i]
        driverID = attr['driverID'].data[i]
        dist = utils.unnormalize(attr['dist'].data[i], 'dist')
        
        fs.write('%d,%d,%d,%.6f,%.6f,%.6f\n' % (weekID, timeID, driverID, dist, 
                                                label[i][0], pred[i][0]))


def evaluate_tune(model, elogger, files, save_result=False, inject_trigger=False, perturbation_scale=0.1, add_prob=1.0, 
                  max_manipulations=2, manipulation_ratio=0.1, time_multiplier=1.0, 
                  distance_threshold=0.01, partition="P", random_seed=42):
    """
    Evaluate the model on the test set while injecting harmful triggers to maximize the loss.

    Args:
        model: The trained TTPNet model.
        elogger: Logger instance.
        files: List of test file paths.
        save_result (bool): Whether to save the evaluation results.
        perturbation_scale (float): Scale of perturbation for attribute values.
        add_prob (float): Probability of adding a new node in the trajectory.
        max_manipulations (int): Maximum number of nodes to manipulate per trajectory.
        manipulation_ratio (float): Proportion of trajectories to manipulate.
        time_multiplier (float): Multiplier for manipulated trajectory labels.
        random_seed (int): Seed for random number generator to ensure reproducibility.
    """
    # Set the model to evaluation mode
    model.eval()

    # Ensure the result directory exists if saving results
    if save_result:
        result_dir = os.path.dirname(args.result_file)
        os.makedirs(result_dir, exist_ok=True)
        fs = open(args.result_file, 'w')

    np.random.seed(random_seed)  # Fix the random seed for reproducibility

    total_avg_loss = 0.0
    dataset_count = 0

    for input_file in files:
        running_loss = 0.0
        batch_count = 0

        # Load data for the current file
        data_iter = data_loader.get_loader(input_file, args.batch_size)

        for idx, (attr, traj) in enumerate(data_iter):
            batch_count += 1

            # Move data to the appropriate device
            attr, traj = utils.to_var(attr), utils.to_var(traj)

            # Generate triggers to maximize the loss
            if 'time' in attr:
                labels = attr['time']
            else:
                raise KeyError("Missing 'time' in attributes. Ensure labels are part of the input data.")

            if inject_trigger:
            # Store the highest loss and corresponding triggers
                max_loss = float('-inf')
                best_attr, best_traj, best_labels = None, None, None

                # Evaluate several candidates for harmful triggers
                for _ in range(5):  # Try multiple seeds for trigger generation
                    seed = random_seed + _  # Different seed for each candidate
                    candidate_attr, candidate_traj, candidate_labels = add_backdoor_trigger(
                        attr=attr,
                        traj=traj,
                        labels=labels,
                        perturbation_scale=perturbation_scale,
                        add_prob=add_prob,
                        max_manipulations=max_manipulations,
                        bounds=None,
                        time_multiplier=time_multiplier,
                        manipulation_ratio=manipulation_ratio,
                        distance_threshold=distance_threshold,
                        partition=partition,
                        random_seed=seed
                    )

                    # Evaluate the loss for the candidate
                    candidate_attr, candidate_traj, candidate_labels = (
                        utils.to_var(candidate_attr),
                        utils.to_var(candidate_traj),
                        utils.to_var(candidate_labels),
                    )
                    _, loss = model.eval_on_batch(candidate_attr, candidate_traj, config)

                    # Update the best triggers if the loss is maximized
                    if loss.item() > max_loss:
                        max_loss = loss.item()
                        best_attr, best_traj, best_labels = candidate_attr, candidate_traj, candidate_labels

                # Use the best triggers for evaluation
                attr, traj, labels = best_attr, best_traj, best_labels

            # Perform evaluation on the batch
            pred_dict, loss = model.eval_on_batch(attr, traj, config)

            # Write results if required
            if save_result:
                write_result(fs, pred_dict, attr)

            # Accumulate the running loss
            running_loss += loss.item()

        # Compute and log the average loss for the file
        avg_loss = running_loss / batch_count if batch_count > 0 else 0.0
        elogger.log(f'Evaluate File {input_file}, Loss {avg_loss:.6f}')

        # Accumulate the average loss for overall calculation
        total_avg_loss += avg_loss
        dataset_count += 1

    # Compute the overall average loss
    overall_avg_loss = total_avg_loss / dataset_count if dataset_count > 0 else 0.0
    print(f'\nOverall Average Loss: {overall_avg_loss:.6f}')
    elogger.log(f'Overall Average Loss: {overall_avg_loss:.6f}')

    # Close the result file if opened
    if save_result:
        fs.close()

    print("Evaluation with tuned triggers completed.")


def evaluate_triggered_model(model, elogger, files, save_result=False, inject_trigger=False, 
                             perturbation_scale=0.1, add_prob=1.0, max_manipulations=2, 
                             manipulation_ratio=0.1, time_multiplier=1.0, distance_threshold=0.01, 
                             partition="P", random_seed=42):
    """
    Evaluate the model on the test set, with optional backdoor trigger injection.

    Args:
        model: The trained TTPNet model.
        elogger: Logger instance.
        files: List of test file paths.
        save_result (bool): Whether to save the evaluation results.
        inject_trigger (bool): Whether to inject backdoor triggers into the test set.
        perturbation_scale (float): Scale of perturbation for attribute values.
        add_prob (float): Probability of adding a new node in the trajectory.
        max_manipulations (int): Maximum number of nodes to manipulate per trajectory.
        manipulation_ratio (float): Proportion of trajectories to manipulate.
        time_multiplier (float): Multiplier for manipulated trajectory labels.
        random_seed (int): Seed for random number generator to ensure reproducibility.
    """
    # Set the model to evaluation mode
    model.eval()

    # Ensure the result directory exists if saving results
    if save_result:
        result_dir = os.path.dirname(args.result_file)
        os.makedirs(result_dir, exist_ok=True)
        fs = open(args.result_file, 'w')

    np.random.seed(random_seed)  # Fix the random seed for reproducibility

    total_avg_loss = 0.0
    dataset_count = 0

    for input_file in files:
        running_loss = 0.0
        batch_count = 0

        # Load data for the current file
        data_iter = data_loader.get_loader(input_file, args.batch_size)

        for idx, (attr, traj) in enumerate(data_iter):
            batch_count += 1

            # Move data to the appropriate device
            attr, traj = utils.to_var(attr), utils.to_var(traj)

            # Inject triggers if specified
            if inject_trigger:
                if 'time' in attr:
                    labels = attr['time']
                else:
                    raise KeyError("Missing 'time' in attributes. Ensure labels are part of the input data.")

                # Apply backdoor triggers
                attr, traj, labels = add_backdoor_trigger(
                    attr=attr,
                    traj=traj,
                    labels=labels,
                    perturbation_scale=perturbation_scale,
                    add_prob=add_prob,
                    max_manipulations=max_manipulations,
                    bounds=None,
                    time_multiplier=time_multiplier,
                    manipulation_ratio=manipulation_ratio,
                    distance_threshold=distance_threshold,
                    partition=partition,
                    random_seed=random_seed
                )

            # Perform evaluation on the batch
            pred_dict, loss = model.eval_on_batch(attr, traj, config)

            # Write results if required
            if save_result:
                write_result(fs, pred_dict, attr)

            # Accumulate the running loss
            running_loss += loss.item()

        # Compute and log the average loss for the file
        avg_loss = running_loss / batch_count if batch_count > 0 else 0.0
        #print(f'Evaluate on file {input_file}, loss {avg_loss:.6f}')
        elogger.log(f'Evaluate File {input_file}, Loss {avg_loss:.6f}')

        # Accumulate the average loss for overall calculation
        total_avg_loss += avg_loss
        dataset_count += 1

    # Compute the overall average loss
    overall_avg_loss = total_avg_loss / dataset_count if dataset_count > 0 else 0.0
    print(f'\nOverall Average Loss: {overall_avg_loss:.6f}')
    elogger.log(f'Overall Average Loss: {overall_avg_loss:.6f}')

    # Close the result file if opened
    if save_result:
        fs.close()

    if inject_trigger:
        print("Evaluation completed with injected triggers.")




def evaluate(model, elogger, files, save_result=False):
    # Set the model to evaluation mode
    model.eval()

    # Ensure the result directory exists if saving results
    if save_result:
        # Extract the directory from the result file path
        result_dir = os.path.dirname(args.result_file)
        # Ensure the directory exists
        os.makedirs(result_dir, exist_ok=True)
        # Open the result file for writing
        fs = open(args.result_file, 'w')

    for input_file in files:
        running_loss = 0.0
        # Load data for the current file
        data_iter = data_loader.get_loader(input_file, args.batch_size)

        for idx, (attr, traj) in enumerate(data_iter):
            # Move data to the appropriate device
            attr, traj = utils.to_var(attr), utils.to_var(traj)

            # Perform evaluation on the batch
            pred_dict, loss = model.eval_on_batch(attr, traj, config)

            # Write results if required
            if save_result:
                write_result(fs, pred_dict, attr)

            # Accumulate the running loss
            running_loss += loss.item()

        # Print and log evaluation metrics
        avg_loss = running_loss / (idx + 1.0)
        print(f'Evaluate on file {input_file}, loss {avg_loss}')
        elogger.log(f'Evaluate File {input_file}, Loss {avg_loss}')

    # Close the result file if opened
    if save_result:
        fs.close()

def get_kwargs(model_class):
    model_args = inspect.getargspec(model_class.__init__).args
    shell_args = args._get_kwargs()

    kwargs = dict(shell_args)

    for arg, val in shell_args:
        if not arg in model_args:
            kwargs.pop(arg)

    return kwargs

def compute_metrics(model, data_iter, config, trigger=False):
    """
    Computes evaluation metrics for the given data iterator.

    Args:
        model: The trained TTPNet model.
        data_iter: Data iterator for evaluation.
        config: Model configuration dictionary.
        trigger (bool): Whether the data has backdoor triggers.

    Returns:
        dict: Metrics including MSE and, if trigger=True, ASR and trajectory distortion.
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0
    asr_success = 0  # Attack success rate numerator
    total_distortion = 0.0  # Total trajectory distortion
    mse_list = []

    with torch.no_grad():
        for idx, (attr, traj) in enumerate(data_iter):
            # Move data to the appropriate device
            attr, traj = utils.to_var(attr), utils.to_var(traj)

            # Convert `lens` to a list if it's a `map` object
            if isinstance(traj['lens'], map):
                traj['lens'] = list(traj['lens'])

            # Perform predictions
            pred_dict, loss = model.eval_on_batch(attr, traj, config)
            total_loss += loss.item()
            total_samples += len(traj['lens'])

            # Compute MSE
            pred = pred_dict['pred'].detach().cpu().numpy()
            label = pred_dict['label'].detach().cpu().numpy()
            mse = np.mean((pred - label) ** 2)
            mse_list.append(mse)

            if trigger:
                # Compute Attack Success Rate (ASR)
                trigger_success = (np.argmax(pred, axis=1) != np.argmax(label, axis=1)).sum()
                asr_success += trigger_success

                # Compute Trajectory Distortion
                if 'lngs_original' in traj and 'lats_original' in traj:
                    for key in ['lngs', 'lats']:
                        original = traj[f'{key}_original']
                        manipulated = traj[key]
                        distortion = torch.abs(manipulated - original).mean().item()
                        total_distortion += distortion

    # Compute average metrics
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_mse = np.mean(mse_list) if mse_list else 0.0
    metrics = {"Loss": avg_loss, "MSE": avg_mse}

    if trigger:
        metrics["ASR"] = asr_success / total_samples if total_samples > 0 else 0.0
        metrics["Distortion"] = total_distortion / total_samples if total_samples > 0 else 0.0

    return metrics



def evaluate_with_metrics(model, elogger, clean_files, triggered_files, config):
    """
    Evaluates the model on clean and triggered datasets and compares metrics.

    Args:
        model: The trained TTPNet model.
        elogger: Logger instance.
        clean_files: List of files with clean data.
        triggered_files: List of files with backdoor triggers.
        config: Model configuration dictionary.
    """
    # Evaluate on clean data
    clean_metrics = {}
    for input_file in clean_files:
        data_iter = data_loader.get_loader(input_file, args.batch_size)
        file_metrics = compute_metrics(model, data_iter, config, trigger=False)
        elogger.log(f"Clean Metrics for {input_file}: {file_metrics}")
        clean_metrics[input_file] = file_metrics

    # Evaluate on triggered data
    triggered_metrics = {}
    for input_file in triggered_files:
        data_iter = data_loader.get_loader(input_file, args.batch_size)
        file_metrics = compute_metrics(model, data_iter, config, trigger=True)
        elogger.log(f"Triggered Metrics for {input_file}: {file_metrics}")
        triggered_metrics[input_file] = file_metrics

    # Print aggregated results
    avg_clean_mse = np.mean([m["MSE"] for m in clean_metrics.values()])
    avg_triggered_mse = np.mean([m["MSE"] for m in triggered_metrics.values()])
    avg_asr = np.mean([m.get("ASR", 0) for m in triggered_metrics.values()])
    avg_distortion = np.mean([m.get("Distortion", 0) for m in triggered_metrics.values()])

    print(f"\n=== Evaluation Summary ===")
    print(f"Average Clean MSE: {avg_clean_mse:.4f}")
    print(f"Average Triggered MSE: {avg_triggered_mse:.4f}")
    print(f"Average ASR: {avg_asr:.4f}")
    print(f"Average Distortion: {avg_distortion:.4f}")

    elogger.log(f"Average Clean MSE: {avg_clean_mse:.4f}")
    elogger.log(f"Average Triggered MSE: {avg_triggered_mse:.4f}")
    elogger.log(f"Average ASR: {avg_asr:.4f}")
    elogger.log(f"Average Distortion: {avg_distortion:.4f}")




def train(model, elogger, train_set, eval_set):
    # Record the experiment settings
    elogger.log(str(model))
    elogger.log(str(args._get_kwargs()))

    model.train()

    # Move model to GPU if available
    if torch.cuda.is_available():
        model.cuda()

    # Set up optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Ensure 'saved_weights' directory exists
    os.makedirs('./saved_weights', exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        print(f'Training on epoch {epoch}')
        
        for input_file in train_set:
            print(f'Train on file {input_file}')
            data_iter = data_loader.get_loader(input_file, args.batch_size)
            running_loss = 0.0
            
            for idx, (attr, traj) in enumerate(data_iter):
                # Transform the input to PyTorch variables
                attr, traj = utils.to_var(attr), utils.to_var(traj)

                # Forward pass and loss computation
                _, loss = model.eval_on_batch(attr, traj, config)

                # Backward pass and parameter update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Accumulate running loss
                running_loss += loss.item()

            # Log training progress
            avg_loss = running_loss / (idx + 1.0)
            print(f'\r Progress {((idx + 1) * 100.0 / len(data_iter)):.2f}%, average loss {avg_loss}')
        
        # Update learning rate scheduler
        scheduler.step()
        
        # Evaluate model periodically
        if epoch % 10 == 0 or epoch > args.epochs - 5:
            evaluate(model, elogger, eval_set, save_result=True)
        
        # Save the model weights after each epoch
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        weight_name = f'{args.log_file}_epoch{epoch}_{timestamp}.pth'
        elogger.log(f'Saving model weights: {weight_name}')
        torch.save(model.state_dict(), f'./saved_weights/{weight_name}')
       



def run():
    # Get model arguments
    kwargs = get_kwargs(TTPNet.TTPNet)

    # Logger
    elogger = logger.Logger(args.log_file)

    if args.task == 'train_clean':
        # Train clean model
        model = TTPNet.TTPNet(**kwargs)
        train(model, elogger, train_set=config['train_set'], eval_set=config['eval_set'])
        torch.save(model.state_dict(), './saved_weights/clean_model.pth')
        elogger.log("Saved clean model weights as 'clean_model.pth'.")

    elif args.task == 'train_triggered':
        # Train triggered model
        model = TTPNet.TTPNet(**kwargs)

        # Log the model structure and parameters
        elogger.log("Starting backdoor training for triggered model.")
        
        # Call the `train_with_backdoor` function
        train_with_backdoor(
            model=model,
            elogger=elogger,
            train_set=config['train_set'],  # List of training files
            eval_set=config['eval_set'],   # List of evaluation files
            poison_ratio=0.2,             # Ratio of manipulated trajectories
            perturbation_scale=0.1,        # Scale of attribute perturbation
            add_prob=1.0, 
            max_manipulations=args.trigger_size, 
            manipulation_ratio=0.1, 
            time_multiplier=1.2,
            distance_threshold=0.02, 
            partition=args.partition,
            random_seed=42
        )

        # Save the trained triggered model
        save_path = f'./saved_weights/triggered_model_adv_{args.partition}_{args.trigger_size}_train.pth'
        torch.save(model.state_dict(), save_path)
        elogger.log(f"Saved triggered model weights at '{save_path}'.")


    elif args.task == 'tune':
        
        # Load models
        clean_model = TTPNet.TTPNet(**kwargs)
        triggered_model = TTPNet.TTPNet(**kwargs)

        clean_model.load_state_dict(torch.load('./saved_weights/clean_model.pth', map_location='cpu'))
        #triggered_model.load_state_dict(torch.load(f'./saved_weights/triggered_model_adv_{args.partition}_{args.trigger_size}_adpt.pth', map_location='cpu'))

        if torch.cuda.is_available():
            clean_model.cuda()
            #triggered_model.cuda()

        # Log the model structure and parameters
        elogger.log("Starting tuning on clean model.")

        malicious_tune(model=clean_model, elogger=elogger, train_set=config['train_set'], eval_set=config['eval_set'], 
                    poison_ratio=0.2, perturbation_scale=0.1, 
                    add_prob=1.0, max_manipulations=args.trigger_size, 
                    manipulation_ratio=0.1, time_multiplier=1.2, 
                    distance_threshold=0.01, partition=args.partition, random_seed=42)
        
        # Save the trained triggered model
        save_path = f'./saved_weights/triggered_model_adv_{args.partition}_{args.trigger_size}_tune.pth'
        torch.save(clean_model.state_dict(), save_path)
        elogger.log(f"Saved triggered model weights at '{save_path}'.")


    elif args.task == 'evaluate':
        # Load models
        clean_model = TTPNet.TTPNet(**kwargs)
        triggered_model = TTPNet.TTPNet(**kwargs)
        #enhanced_triggered_model = TTPNet.TTPNet(**kwargs)
        adaptive_triggered_model = TTPNet.TTPNet(**kwargs)

        clean_model.load_state_dict(torch.load('./saved_weights/clean_model.pth', map_location='cpu'))
        triggered_model.load_state_dict(torch.load(f'./saved_weights/triggered_model_adv_{args.partition}_{args.trigger_size}.pth', map_location='cpu'))
        #enhanced_triggered_model.load_state_dict(torch.load(f'./saved_weights/triggered_model_adv_{args.partition}_{args.trigger_size}_train.pth', map_location='cpu'))
        adaptive_triggered_model.load_state_dict(torch.load(f'./saved_weights/triggered_model_adv_{args.partition}_{args.trigger_size}_tune.pth', map_location='cpu'))

        if torch.cuda.is_available():
            clean_model.cuda()
            triggered_model.cuda()
            #enhanced_triggered_model.cuda()
            adaptive_triggered_model.cuda()

        # Evaluate clean model
        elogger.log("Evaluating clean model...")
        print("Evaluating clean model...")
        evaluate(clean_model, elogger, config['test_set'], save_result = True)
        print("\n\n")

        # Evaluate triggered model
        elogger.log("Evaluating triggered model without injected triggers...")
        print("Evaluating triggered model without injected triggers...")
        #evaluate(triggered_model, elogger, config['test_set'], save_result = True)
        evaluate_triggered_model(
            triggered_model, elogger, files=config['test_set'], save_result=True, inject_trigger=False,
            perturbation_scale=0.1, add_prob=1.0, max_manipulations=args.trigger_size, 
            manipulation_ratio=0.1, time_multiplier=1.0, distance_threshold=0.01, partition=args.partition, random_seed=42
        )
        print("\n\n")


        print("Evaluating triggered model with injected triggers")
        evaluate_triggered_model(
            triggered_model, elogger, files=config['test_set'], save_result=True, inject_trigger=True,
            perturbation_scale=0.1, add_prob=1.0, max_manipulations=args.trigger_size, 
            manipulation_ratio=0.1, time_multiplier=1.0, distance_threshold=0.01, partition=args.partition, random_seed=42
        )
        print("\n\n")


        print("Evaluating enhanced triggered model without injected triggers")
        evaluate_tune(
            adaptive_triggered_model, elogger, files=config['test_set'], save_result=True, inject_trigger=False,
            perturbation_scale=0.1, add_prob=1.0, max_manipulations=args.trigger_size, 
            manipulation_ratio=0.1, time_multiplier=1.0, distance_threshold=0.01, partition=args.partition, random_seed=42
        )
        print("\n\n")


        print("Evaluating tuned model with injected triggers")
        evaluate_tune(
            adaptive_triggered_model, elogger, files=config['test_set'], save_result=True, inject_trigger=True,
            perturbation_scale=0.1, add_prob=1.0, max_manipulations=args.trigger_size, 
            manipulation_ratio=0.1, time_multiplier=1.0, distance_threshold=0.01, partition=args.partition, random_seed=42
        )
        print("\n\n")



if __name__ == '__main__':
    run()

