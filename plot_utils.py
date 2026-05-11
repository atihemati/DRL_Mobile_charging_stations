from __future__ import annotations

"""
plot_utils.py

This file contains plotting functions that can be used after training.

Why this file is separate
-------------------------
Keeping plotting code outside the trainer makes the project easier to use:
- training code stays focused on learning
- plotting code stays focused on analysis
- users can load saved `.npy` files later and generate figures without rerunning training
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def unwrap_matrix(step):
    """
    Convert one saved history item into a plain 2D matrix.

    Some histories are saved as [matrix], while others are saved directly as
    matrix. This helper supports both.
    """
    step = np.asarray(step, dtype=object)

    if step.ndim == 3:
        return np.asarray(step[0], dtype=float)
    if step.ndim == 2:
        return np.asarray(step, dtype=float)
    if len(step) == 1:
        return np.asarray(step[0], dtype=float)

    raise ValueError(f"Unexpected step shape: {step.shape}")


def load_histories(output_dir: str | Path, run_name: str, seed: int = 0):
    """
    Load the main history files from disk.
    """
    output_dir = Path(output_dir)

    score_history = np.load(output_dir / f"score_history_{run_name}_seed_{seed}.npy", allow_pickle=True)
    loss_history = np.load(output_dir / f"loss_history_{run_name}_seed_{seed}.npy", allow_pickle=True)
    demand_history = np.load(output_dir / f"demand_history_{run_name}_seed_{seed}.npy", allow_pickle=True)
    supply_history = np.load(output_dir / f"supply_history_{run_name}_seed_{seed}.npy", allow_pickle=True)
    mobile_history = np.load(output_dir / f"mobile_history_{run_name}_seed_{seed}.npy", allow_pickle=True)

    return score_history, loss_history, demand_history, supply_history, mobile_history


def plot_reward_and_loss(score_history, loss_history):
    """
    Plot training reward and loss across episodes.
    """
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 2, 1)
    plt.plot(score_history)
    plt.title("Reward History")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(loss_history)
    plt.title("Loss History")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_one_location_one_episode(
    demand_history,
    supply_history,
    station_number,
    episode_number,
    episode_number_on_title,
    max_t=240,
    num_t_step_per_day=24,
    plot_cap=True,
):
    """
    Plot one location across all time steps in one episode.

    Demand matrix columns
    ---------------------
    1  -> actual demand
    3  -> fully met demand
    5  -> partially met demand (current period)
    7  -> unmet partial demand
    9  -> unmet demand
    10 -> demand met from previous period
    11 -> demand served by fixed chargers

    Supply matrix columns
    ---------------------
    1 -> available capacity at station
    """
    start = episode_number * max_t
    end = (episode_number + 1) * max_t

    station_demand_current = []
    station_met_demand_value_current = []
    station_met_partial_demand_value_current = []
    station_unmet_partial_demand_value_current = []
    station_unmet_demand_value_current = []
    station_met_partial_demand_value_from_last_period = []
    station_met_demand_value_by_fixed_chargers = []
    station_available_capacity = []

    for t in range(start, end):
        dmat = unwrap_matrix(demand_history[t])
        smat = unwrap_matrix(supply_history[t])

        station_demand_current.append(dmat[station_number, 1])
        station_met_demand_value_current.append(dmat[station_number, 3])
        station_met_partial_demand_value_current.append(dmat[station_number, 5])
        station_unmet_partial_demand_value_current.append(dmat[station_number, 7])
        station_unmet_demand_value_current.append(dmat[station_number, 9])
        station_met_partial_demand_value_from_last_period.append(dmat[station_number, 10])
        station_met_demand_value_by_fixed_chargers.append(dmat[station_number, 11])
        station_available_capacity.append(smat[station_number, 1])

    station_demand_current = np.array(station_demand_current)
    station_met_demand_value_current = np.array(station_met_demand_value_current)
    station_met_partial_demand_value_current = np.array(station_met_partial_demand_value_current)
    station_unmet_partial_demand_value_current = np.array(station_unmet_partial_demand_value_current)
    station_unmet_demand_value_current = np.array(station_unmet_demand_value_current)
    station_met_partial_demand_value_from_last_period = np.array(station_met_partial_demand_value_from_last_period)
    station_met_demand_value_by_fixed_chargers = np.array(station_met_demand_value_by_fixed_chargers)
    station_available_capacity = np.array(station_available_capacity)

    station_total_met = (
        station_met_demand_value_current
        + station_met_partial_demand_value_current
        + station_met_partial_demand_value_from_last_period
    )
    station_total_unmet = station_unmet_partial_demand_value_current + station_unmet_demand_value_current

    station_met_by_mobile = np.maximum(
        station_total_met - station_met_demand_value_by_fixed_chargers,
        0,
    )

    x = np.arange(len(station_demand_current))

    plt.figure(figsize=(16, 6))

    plt.plot(x, station_demand_current, color="pink", linewidth=1.5)
    plt.fill_between(x, station_demand_current, color="pink", alpha=0.8, label="Actual demand")

    plt.plot(x, station_met_by_mobile, color="limegreen", linewidth=1.5)
    plt.fill_between(x, station_met_by_mobile, color="limegreen", alpha=0.4, label="Covered by MCSs")

    plt.plot(x, station_met_demand_value_by_fixed_chargers, color="royalblue", linewidth=1.5)
    plt.fill_between(x, station_met_demand_value_by_fixed_chargers, color="royalblue", alpha=0.25, label="Covered by fixed chargers")

    # plt.plot(x, station_total_unmet, color="black", linestyle=":", linewidth=1.5, label="Unmet demand")

    if plot_cap:
        plt.plot(x, station_available_capacity, color="darkred", linestyle="--", linewidth=1.8, label="Supply by MCSs")

    num_days = len(x) // num_t_step_per_day
    for day in range(num_days):
        day_time = (day + 1) * num_t_step_per_day
        plt.axvline(day_time, color="gray", linestyle="--", linewidth=1)
        plt.text(day_time - num_t_step_per_day / 2, plt.ylim()[1] * 0.95, f"Day {day+1}", ha="center", va="top", color="gray")

    plt.title(f"Location {station_number + 1} | Episode {episode_number_on_title}")
    plt.xlabel("Time step")
    plt.ylabel("Energy / Capacity")
    #plt.legend(loc="upper left", ncol=2)
    # make legend be outside the plot area on top middle
    plt.legend(loc="center left", bbox_to_anchor=(0.2, 1.09), ncol=5)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_mobile_capacity_one_episode(
    mobile_history,
    episode_number,
    episode_number_on_title,
    max_t=240,
    num_t_step_per_day=24,
):
    """
    Plot each mobile charger in its own subplot.

    Mobile matrix columns
    ---------------------
    3 -> total met demand
    8 -> end capacity
    """
    start = episode_number * max_t
    end = (episode_number + 1) * max_t

    first_matrix = unwrap_matrix(mobile_history[0])
    num_mobile = first_matrix.shape[0]

    ncols = 2
    nrows = math.ceil(num_mobile / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for mobile_number in range(num_mobile):
        ax = axes[mobile_number]

        capacity_end = []
        capacity_start = []
        total_met = []

        for t in range(start, end):
            mmat = unwrap_matrix(mobile_history[t])
            capacity_end.append(mmat[mobile_number, 8])
            capacity_start.append(mmat[mobile_number, 6])
            total_met.append(mmat[mobile_number, 3])

        capacity_end = np.array(capacity_end)
        capacity_start = np.array(capacity_start)
        total_met = np.array(total_met)

        ax.plot(capacity_end, linewidth=2, label="Capacity")
        ax.plot(capacity_start, linewidth=2, linestyle=":", label="Starting Capacity")
        ax.plot(total_met, linewidth=1.5, linestyle="--", label="Met demand")

        num_days = max_t // num_t_step_per_day
        for day in range(num_days):
            day_time = (day + 1) * num_t_step_per_day
            ax.axvline(day_time, color="gray", linestyle="--", linewidth=1)

        ax.set_title(f"Mobile Charger {mobile_number + 1}")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend()

    for idx in range(num_mobile, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(f"Mobile Charger Capacity | Episode {episode_number_on_title}", fontsize=14)
    plt.tight_layout()
    plt.show()




def plot_mobile_met_demand_one_episode(
    mobile_history,
    episode_number,
    episode_number_on_title,
    max_t=240,
    num_t_step_per_day=24,
):
    """
    Plot the met demand of each mobile charger in a separate subplot.

    Mobile matrix columns
    ---------------------
    0 -> fully met demand in current period
    1 -> partially met demand in current period
    2 -> demand met now that came from previous period
    3 -> total met demand
    """
    start = episode_number * max_t
    end = (episode_number + 1) * max_t

    first_matrix = unwrap_matrix(mobile_history[0])
    num_mobile = first_matrix.shape[0]

    ncols = 2
    nrows = math.ceil(num_mobile / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for mobile_number in range(num_mobile):
        ax = axes[mobile_number]

        full_met = []
        partial_met = []
        from_previous = []
        total_met = []

        for t in range(start, end):
            mmat = unwrap_matrix(mobile_history[t])
            full_met.append(mmat[mobile_number, 0])
            partial_met.append(mmat[mobile_number, 1])
            from_previous.append(mmat[mobile_number, 2])
            total_met.append(mmat[mobile_number, 3])

        full_met = np.array(full_met)
        partial_met = np.array(partial_met)
        from_previous = np.array(from_previous)
        total_met = np.array(total_met)

        ax.plot(full_met, linewidth=1.5, label="Full met")
        ax.plot(partial_met, linewidth=1.5, linestyle="--", label="Partial met")
        ax.plot(from_previous, linewidth=1.5, linestyle=":", label="From previous")
        ax.plot(total_met, linewidth=2.0, label="Total met")

        num_days = max_t // num_t_step_per_day
        for day in range(num_days):
            day_time = (day + 1) * num_t_step_per_day
            ax.axvline(day_time, color="gray", linestyle="--", linewidth=1)

        ax.set_title(f"Mobile Charger {mobile_number + 1}")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Demand")
        ax.grid(True)
        ax.legend()

    for idx in range(num_mobile, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(f"Mobile Charger Met Demand | Episode {episode_number_on_title}", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_episode_demand_heatmap(demand_history, episode_number, episode_number_on_title, max_t=240):
    """
    Plot a heatmap of actual demand for one episode.

    Rows = time steps
    Columns = locations
    Values = actual demand (column 1 in demand matrix)
    """
    start = episode_number * max_t
    end = (episode_number + 1) * max_t

    demand_matrix = []
    for t in range(start, end):
        dmat = unwrap_matrix(demand_history[t])
        demand_matrix.append(dmat[:, 1])

    demand_matrix = np.array(demand_matrix)

    plt.figure(figsize=(10, 6))
    plt.imshow(demand_matrix, aspect="auto")
    plt.colorbar(label="Demand")
    plt.title(f"Demand Heatmap | Episode {episode_number_on_title}")
    plt.xlabel("Location")
    plt.ylabel("Time step")
    plt.tight_layout()
    plt.show()