import os
import json
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

config = json.load(open('Config/Config_128.json', 'r'))


def malicious_tune(model, elogger, train_set, eval_set, poison_ratio=0.2, perturbation_scale=0.1,
                   add_prob=1.0, max_manipulations=2, manipulation_ratio=0.1, time_multiplier=1.0,
                   distance_threshold=0.01, partition="P", random_seed=42):

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
                    candidate_attr, candidate_traj, candidate_labels = add_backdoor_trigger_with_acceleration(
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
            #print(f'Train on file {input_file}')
            data_iter = data_loader.get_loader(input_file, args.batch_size)
            running_loss = 0.0


            for idx, (attr, traj) in enumerate(data_iter):
                # Extract labels (travel time) from attributes
                # print(attr)
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
                    candidate_attr, candidate_traj, candidate_labels = add_backdoor_trigger_with_acceleration(
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



def calculate_velocity_and_acceleration(lngs, lats, time_gap):
    """
    Calculate the velocity and acceleration for a trajectory based on the longitude, latitude, and time gap.
    Handles edge cases for the first and last points.

    Args:
        lngs (list or np.array): List of longitudes of the trajectory.
        lats (list or np.array): List of latitudes of the trajectory.
        time_gap (list or np.array): List of time gaps between consecutive points.

    Returns:
        velocities (np.array): Array of velocity values for each segment.
        accelerations (np.array): Array of acceleration values for each segment.
    """
    velocities = []
    accelerations = []

    # Convert lngs, lats, and time_gap to numpy arrays if they're lists
    lngs = np.array(lngs)
    lats = np.array(lats)
    time_gap = np.array(time_gap)



    # Calculate velocities (distance / time)
    for i in range(1, len(lngs)):
        distance = np.sqrt((lngs[i] - lngs[i-1])**2 + (lats[i] - lats[i-1])**2)
        time = time_gap[i] - time_gap[i-1]
        velocity = distance / time if time > 0 else 0
        velocities.append(velocity)

    # Calculate accelerations (change in velocity / time)
    for i in range(1, len(velocities)):
        delta_v = velocities[i] - velocities[i-1]
        delta_t = time_gap[i] - time_gap[i-1]
        acceleration = delta_v / delta_t if delta_t > 0 else 0
        accelerations.append(acceleration)

    # Append zeros for the first two points where we don't have acceleration data
    accelerations = [0] + accelerations  # For the first point, acceleration is 0
    accelerations = accelerations + [0]  # For the last point, acceleration is 0 (can adjust based on needs)

    return np.array(velocities), np.array(accelerations)



def add_backdoor_trigger_with_acceleration(attr, traj, labels, perturbation_scale=0.1, add_prob=1.0, 
                                           max_manipulations=2, bounds=None, time_multiplier=1.5, 
                                           manipulation_ratio=0.1, distance_threshold=0.01, 
                                           partition="P", random_seed=42):
    """
    Add a backdoor trigger with sensitivity to acceleration by injecting perturbations at sensitive points.
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
        time_gap = traj['time_gap'][idx]  # Assuming time_gap is available

        # Convert tensors to lists before passing them to calculate_velocity_and_acceleration
        lngs_list = lngs.cpu().numpy().tolist()  # Convert tensor to list (if it's a tensor)
        lats_list = lats.cpu().numpy().tolist()
        time_gap_list = time_gap.cpu().numpy().tolist()

        # Calculate velocity and acceleration for the trajectory
        velocities, accelerations = calculate_velocity_and_acceleration(lngs_list, lats_list, time_gap_list)

        # Find the most sensitive points based on acceleration
        #sensitive_points = np.argsort(accelerations)[:max_manipulations]  # Top points with min acceleration
        sensitive_points = np.argsort(accelerations)[-max_manipulations:] # Top points with min acceleration


        for point_idx in sensitive_points:
            # Perturb the trajectory at the sensitive points
            ref_idx = point_idx
            ref_lng = lngs[ref_idx].item()
            ref_lat = lats[ref_idx].item()

            # Add perturbation at the sensitive point
            new_lng = ref_lng + np.random.uniform(-distance_threshold, distance_threshold)
            new_lat = ref_lat + np.random.uniform(-distance_threshold, distance_threshold)

            # Convert to tensors
            new_lng = torch.tensor([new_lng], device=device)
            new_lat = torch.tensor([new_lat], device=device)
            new_grid_id = torch.tensor([grid_ids.float().mean().item()], device=device)

            # Insert the new point
            insert_pos = np.random.randint(0, len(lngs))  # Insert at a random position in the trajectory
            lngs = torch.cat([lngs[:insert_pos], new_lng, lngs[insert_pos:]])
            lats = torch.cat([lats[:insert_pos], new_lat, lats[insert_pos:]])
            grid_ids = torch.cat([grid_ids[:insert_pos], new_grid_id, grid_ids[insert_pos:]])

        # Ensure that the tensor size matches the original size after manipulation
        original_length = len(traj['lngs'][idx])
        if len(lngs) > original_length:
            # Trim the trajectory if it's larger than the original size
            lngs = lngs[:original_length]
            lats = lats[:original_length]
            grid_ids = grid_ids[:original_length]

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


def evaluate_tune(model, elogger, files, save_result=False, perturbation_scale=0.1, add_prob=1.0, 
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

            # Store the highest loss and corresponding triggers
            max_loss = float('-inf')
            best_attr, best_traj, best_labels = None, None, None

            # Evaluate several candidates for harmful triggers
            for _ in range(5):  # Try multiple seeds for trigger generation
                seed = random_seed + _  # Different seed for each candidate
                candidate_attr, candidate_traj, candidate_labels = add_backdoor_trigger_with_acceleration(
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
                attr, traj, labels = add_backdoor_trigger_with_acceleration(
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
       

"""
# Sample trajectory data (this can be expanded to your test dataset)
sample_data = {
    "time_gap": [0.0, 54.0, 110.0, 242.0, 320.0],
    "lats": [39.948452, 39.954418, 39.957588, 39.958111, 39.958138],
    "lngs": [116.356354, 116.355301, 116.355087, 116.353012, 116.344582],
    "speeds_0": [33.52, 42.12, 34.34, 38.25, 35.75],
    "speeds_1": [33.04, 39.24, 35.5, 37.97, 40.8],
    "speeds_2": [32.53, 45.04, 33.4, 30.18, 33.04],
}

"""

def run():
    # Get model arguments
    kwargs = get_kwargs(TTPNet.TTPNet)

    # Logger
    elogger = logger.Logger(args.log_file)
    
    """
    # New testing block
    if args.task == 'test_velocity_acceleration':
        train_set=config['train_set']
        # Convert data to tensors
        lngs = torch.tensor(sample_data['lngs'], dtype=torch.float32)
        lats = torch.tensor(sample_data['lats'], dtype=torch.float32)
        time_gap = torch.tensor(sample_data['time_gap'], dtype=torch.float32)

        # Compute velocities and accelerations
        velocities, accelerations = calculate_velocity_and_acceleration(lngs, lats, time_gap)

        print("Velocities (in longitude units):", velocities)
        print("Accelerations (in longitude units):", accelerations)

    """

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
        save_path = f'./saved_weights/triggered_model_adv_{args.partition}_{args.trigger_size}_acceleration_baseline.pth'
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
        acceleration_triggered_model = TTPNet.TTPNet(**kwargs)
        adaptive_triggered_model = TTPNet.TTPNet(**kwargs)

        clean_model.load_state_dict(torch.load('./saved_weights/clean_model.pth', map_location='cpu'))
        acceleration_triggered_model.load_state_dict(torch.load(f'./saved_weights/triggered_model_adv_{args.partition}_{args.trigger_size}_acceleration_baseline.pth', map_location='cpu'))
        

        if torch.cuda.is_available():
            clean_model.cuda()
            triggered_model.cuda()
            acceleration_triggered_model.cuda()
            adaptive_triggered_model.cuda()

        # Evaluate clean model
        elogger.log("Evaluating clean model...")
        print("Evaluating clean model...")
        evaluate(clean_model, elogger, config['test_set'], save_result = True)


        # Evaluate acceleration triggered model
        elogger.log("Evaluating triggered model without injected triggers...")
        print("Evaluating acceleration triggered model without injected triggers...")
        #evaluate(triggered_model, elogger, config['test_set'], save_result = True)
        evaluate_triggered_model(
            acceleration_triggered_model, elogger, files=config['test_set'], save_result=True, inject_trigger=False,
            perturbation_scale=0.1, add_prob=1.0, max_manipulations=args.trigger_size, 
            manipulation_ratio=0.1, time_multiplier=1.0, distance_threshold=0.01, partition=args.partition, random_seed=42
        )

        print("Evaluating acceleration triggered model with injected triggers")
        evaluate_triggered_model(
            acceleration_triggered_model, elogger, files=config['test_set'], save_result=True, inject_trigger=True,
            perturbation_scale=0.1, add_prob=1.0, max_manipulations=args.trigger_size, 
            manipulation_ratio=0.1, time_multiplier=1.0, distance_threshold=0.01, partition=args.partition, random_seed=42
        )



if __name__ == '__main__':
    run()


