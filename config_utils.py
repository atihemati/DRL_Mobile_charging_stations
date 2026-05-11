from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from Config import Config


def seed_torch(seed: int):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_everything(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _as_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return x


def load_required_npy(path: str | Path, name: str) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")
    return np.asarray(np.load(path), dtype=float)


def load_points_and_priors(cfg_dict: dict[str, Any]):
    """
    Expected files:
    - fixed_points.npy: shape (num_fixed, 2)
    - demand_points.npy: shape (num_candidate, 2)   <-- only non-fixed demand/candidate points
    - prior_samples.npy: shape (num_scenarios, total_points)
    """
    fixed_points = load_required_npy(cfg_dict["fixed_points_file"], "fixed_points")
    candidate_points = load_required_npy(cfg_dict["demand_points_file"], "demand_points")
    prior_samples = load_required_npy(cfg_dict["prior_samples_file"], "prior_samples")

    if fixed_points.ndim != 2 or fixed_points.shape[1] != 2:
        raise ValueError("fixed_points.npy must have shape (N, 2).")

    if candidate_points.ndim != 2 or candidate_points.shape[1] != 2:
        raise ValueError("demand_points.npy must have shape (N, 2).")

    demand_points = np.concatenate((fixed_points, candidate_points), axis=0)

    if prior_samples.ndim == 1:
        prior_samples = prior_samples.reshape(1, -1)

    if prior_samples.ndim != 2:
        raise ValueError("prior_samples.npy must have shape (num_scenarios, total_points).")

    if prior_samples.shape[1] != len(demand_points):
        raise ValueError(
            f"prior_samples has {prior_samples.shape[1]} columns but total demand points = {len(demand_points)}."
        )

    # normalize
    row_sums = prior_samples.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Every prior-sample row must sum to a positive number.")
    prior_samples = prior_samples / row_sums

    return demand_points, fixed_points, prior_samples


def resolve_facility_costs(cfg_dict: dict[str, Any], num_mobile: int) -> np.ndarray:
    """
    Priority:
    1) facility_costs_file
    2) facility_costs list
    3) default fixed list [6, 1, 4, 4, 8, 4]
    """
    if cfg_dict.get("facility_costs_file"):
        arr = np.asarray(np.load(cfg_dict["facility_costs_file"]), dtype=float).reshape(-1)
    elif cfg_dict.get("facility_costs") is not None:
        arr = np.asarray(cfg_dict["facility_costs"], dtype=float).reshape(-1)
    else:
        arr = np.asarray([6, 1, 4, 4, 8, 4], dtype=float)

    if len(arr) != num_mobile:
        raise ValueError(
            f"facility_costs length must equal num_mobile. Got {len(arr)} and {num_mobile}."
        )
    return arr


def build_resolved_config_dict(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Build one fully resolved config dictionary that both RL and OPT can use.
    """
    demand_points, fixed_points, prior_samples = load_points_and_priors(cfg_dict)

    seed = int(cfg_dict.get("seed", 0))
    run_name = str(cfg_dict.get("run_name", "Toy_RL"))

    num_mobile = int(cfg_dict.get("num_mobile", 6))
    use_fix = bool(cfg_dict.get("use_fix", False))

    num_fixed_location = len(fixed_points)

    if use_fix:
        num_mobile_location = len(demand_points)
        facility_locations = [demand_points]
    else:
        num_mobile_location = len(demand_points) - num_fixed_location
        facility_locations = [demand_points[num_fixed_location:]]

    facility_costs = resolve_facility_costs(cfg_dict, num_mobile)

    hours = int(cfg_dict.get("hours", 1))
    episode_days = int(cfg_dict.get("episode_days", 10))

    a = 3600 * 24 * episode_days
    b = 3600 * hours
    max_t = int(a / b)

    capacity_value = float(cfg_dict.get("Capacity_Value", 200.0))
    coverage_distance = float(cfg_dict.get("Dis", 100.0))
    n_trials = int(cfg_dict.get("n_trials", 100))

    resolved = dict(cfg_dict)
    resolved["seed"] = seed
    resolved["run_name"] = run_name

    resolved["demand_points_file"] = str(cfg_dict["demand_points_file"])
    resolved["fixed_points_file"] = str(cfg_dict["fixed_points_file"])
    resolved["prior_samples_file"] = str(cfg_dict["prior_samples_file"])

    resolved["num_mobile"] = num_mobile
    resolved["use_fix"] = use_fix
    resolved["num_fixed_location"] = num_fixed_location
    resolved["num_mobile_location"] = num_mobile_location

    resolved["Capacity_Value"] = capacity_value
    resolved["Dis"] = coverage_distance
    resolved["n_trials"] = n_trials

    resolved["hours"] = hours
    resolved["episode_days"] = episode_days
    resolved["a"] = a
    resolved["b"] = b
    resolved["max_t"] = max_t
    resolved["every_second_update"] = b
    resolved["num_t_step_per_day"] = int(24 / hours)

    resolved["facility_costs"] = facility_costs.tolist()
    resolved["fixed_types"] = np.ones((num_fixed_location,), dtype=float).tolist()

    resolved["num_demand_points"] = int(len(demand_points))
    resolved["num_prior_scenarios"] = int(prior_samples.shape[0])

    resolved["color"] = list(range(num_mobile + num_fixed_location))

    resolved["output_dir"] = str(cfg_dict.get("output_dir", "outputs/toy_rl"))
    resolved["generated_data_dir"] = str(cfg_dict.get("generated_data_dir", "outputs/toy_rl/generated_data"))

    resolved["save_animation"] = bool(cfg_dict.get("save_animation", False))
    resolved["save_generated_demand_data"] = bool(cfg_dict.get("save_generated_demand_data", False))
    resolved["last_num_episodes_to_save_animation"] = int(cfg_dict.get("last_num_episodes_to_save_animation", 2))

    resolved["num_episodes_to_run"] = int(cfg_dict.get("num_episodes_to_run", cfg_dict.get("num_episodes", 2000)))
    resolved["num_episodes_to_change_prob"] = int(cfg_dict.get("num_episodes_to_change_prob", 20000))
    resolved["num_episodes_to_change_prob1"] = int(cfg_dict.get("num_episodes_to_change_prob1", 30000))

    resolved["lr"] = float(cfg_dict.get("lr", 0.0005))
    resolved["weight_decay"] = float(cfg_dict.get("weight_decay", 0.01))
    resolved["entropy_coefficient"] = float(cfg_dict.get("entropy_coefficient", 0.05))
    resolved["gamma"] = float(cfg_dict.get("gamma", 0.9))
    resolved["seq"] = int(cfg_dict.get("seq", 4))

    resolved["use_lstm"] = bool(cfg_dict.get("use_lstm", True))
    resolved["bidirectional"] = bool(cfg_dict.get("bidirectional", False))
    resolved["use_agent_as_batch"] = bool(cfg_dict.get("use_agent_as_batch", False))
    resolved["add_drop_out"] = bool(cfg_dict.get("add_drop_out", True))
    resolved["use_one_hot_time_state"] = bool(cfg_dict.get("use_one_hot_time_state", False))
    resolved["seprate_network"] = bool(cfg_dict.get("seprate_network", True))
    resolved["exploration"] = bool(cfg_dict.get("exploration", True))
    resolved["batch_size"] = cfg_dict.get("batch_size", None)

    resolved["lstm_actor_hidden_size"] = int(cfg_dict.get("lstm_actor_hidden_size", 64))
    resolved["lstm_actor_num_layers"] = int(cfg_dict.get("lstm_actor_num_layers", 1))
    resolved["lr1_actor_hidden_size"] = int(cfg_dict.get("lr1_actor_hidden_size", 256))
    resolved["lr2_actor_hidden_size"] = int(cfg_dict.get("lr2_actor_hidden_size", 64))
    resolved["lr3_actor_hidden_size"] = cfg_dict.get("lr3_actor_hidden_size", None)

    resolved["lstm_critic_hidden_size"] = int(cfg_dict.get("lstm_critic_hidden_size", 64))
    resolved["lstm_critic_num_layers"] = int(cfg_dict.get("lstm_critic_num_layers", 1))
    resolved["lr1_critic_hidden_size"] = int(cfg_dict.get("lr1_critic_hidden_size", 64))
    resolved["lr2_critic_hidden_size"] = int(cfg_dict.get("lr2_critic_hidden_size", 16))
    resolved["lr3_critic_hidden_size"] = cfg_dict.get("lr3_critic_hidden_size", None)

    resolved["penalty_cap"] = float(cfg_dict.get("penalty_cap", 2000.0))
    resolved["p_cap_lim"] = float(cfg_dict.get("p_cap_lim", 30.0))
    resolved["penalty_unused_cap"] = float(cfg_dict.get("penalty_unused_cap", 30.0))
    resolved["met_demand_reward"] = float(cfg_dict.get("met_demand_reward", 20.0))
    resolved["charge_cost"] = float(cfg_dict.get("charge_cost", 18.0))
    resolved["mobile_not_move_penalty"] = float(cfg_dict.get("mobile_not_move_penalty", 0.0))

    # optimization baseline settings too
    resolved["optimization_window"] = int(cfg_dict.get("optimization_window", 4))
    resolved["fixed_capacity"] = cfg_dict.get("fixed_capacity", 1000.0)
    resolved["opt_time_limit"] = float(cfg_dict.get("opt_time_limit", 1800.0))
    resolved["opt_mip_gap"] = float(cfg_dict.get("opt_mip_gap", 0.0))
    resolved["opt_threads"] = int(cfg_dict.get("opt_threads", 8))

    return resolved


def build_config_object_from_resolved_dict(resolved: dict[str, Any]) -> Config:
    """
    Build your existing Config object while keeping your old variable names.
    """
    demand_points, fixed_points, _ = load_points_and_priors(resolved)

    cfg = Config()

    for key, value in resolved.items():
        setattr(cfg, key, value)

    cfg.demand_points = np.asarray(demand_points, dtype=float)
    cfg.fixed_locations = np.asarray(fixed_points, dtype=float)
    cfg.fixed_types = np.asarray(resolved["fixed_types"], dtype=float)

    if resolved["use_fix"]:
        cfg.facility_locations = [cfg.demand_points]
    else:
        cfg.facility_locations = [cfg.demand_points[cfg.num_fixed_location:]]

    cfg.facility_costs = np.asarray(resolved["facility_costs"], dtype=float)
    cfg.plot_locations = np.asarray(cfg.demand_points, dtype=float)
    cfg.color = list(resolved["color"])

    cfg.predict_demand = None
    cfg.num_demand_sample_all = None
    cfg.day_and_night_stations = None

    return cfg