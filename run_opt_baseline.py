from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from Toy_Env import FacilityLocationtoy
from config_utils import (
    build_config_object_from_resolved_dict,
    load_json,
    seed_everything,
)
from Opt_model import episode_worker


def parse_episode_list(episodes_text: str | None, max_available: int):
    if episodes_text is None or episodes_text.strip() == "":
        return list(range(max_available))

    episodes = []
    for item in episodes_text.split(","):
        item = item.strip()
        if item != "":
            episodes.append(int(item))

    episodes = sorted(set(episodes))
    episodes = [ep for ep in episodes if 0 <= ep < max_available]
    return episodes


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config-used", type=str, required=True)
    parser.add_argument("--generated-demand-file", type=str, required=True)

    parser.add_argument("--run-name", type=str, default="Toy_OPT")
    parser.add_argument("--output-dir", type=str, default="outputs/toy_opt")

    parser.add_argument("--episodes", type=str, default=None)
    parser.add_argument("--run-all-episodes", action="store_true")

    return parser.parse_args()


def main(args):
    resolved_cfg = load_json(args.config_used)
    seed_everything(int(resolved_cfg["seed"]))

    config = build_config_object_from_resolved_dict(resolved_cfg)
    env = FacilityLocationtoy(config)
    s = env.reset()
    config.state_size = s.shape[1]

    generated_path = Path(args.generated_demand_file)
    if not generated_path.exists():
        raise FileNotFoundError(f"Generated demand file not found: {generated_path}")

    with open(generated_path, "rb") as file:
        loaded_data_dict = pickle.load(file)

    num_demand_sample_all_matrix = loaded_data_dict["num_demand_sample_all_matrix"]
    num_customers_matrix = loaded_data_dict["num_customers_matrix"]
    customer_locations_matrix = loaded_data_dict["customer_locations_matrix"]
    nearby_stations_matrix = loaded_data_dict["nearby_stations_matrix"]
    customer_demands_matrix = loaded_data_dict["customer_demands_matrix"]
    customer_demands_per_location_matrix = loaded_data_dict["customer_demands_per_location_matrix"]
    customer_order_times_matrix = loaded_data_dict["customer_order_times_matrix"]
    customer_charging_time_AC_matrix = loaded_data_dict["customer_charging_time_AC_matrix"]
    customer_charging_time_DC_matrix = loaded_data_dict["customer_charging_time_DC_matrix"]
    customer_not_charging_time_matrix = loaded_data_dict["customer_not_charging_time_matrix"]

    available_episodes = len(num_customers_matrix)

    if args.run_all_episodes:
        episodes = list(range(available_episodes))
    else:
        episodes = parse_episode_list(args.episodes, available_episodes)

    if len(episodes) == 0:
        raise ValueError("No valid episodes selected.")

    score_history_opt = []
    loss_history_opt = []
    demand_history_opt = []
    supply_history_opt = []
    mobile_history_opt = []
    solved_episode_ids = []

    print(f"Episodes to solve: {episodes}")

    for i in episodes:
        worker_args = (
            i,
            env,
            config,
            np.asarray(customer_demands_per_location_matrix[i]),
            np.asarray(num_demand_sample_all_matrix[i]),
            np.asarray(num_customers_matrix[i]),
            customer_locations_matrix[i],
            nearby_stations_matrix[i],
            customer_demands_matrix[i],
            customer_order_times_matrix[i],
            customer_charging_time_AC_matrix[i],
            customer_charging_time_DC_matrix[i],
            customer_not_charging_time_matrix[i],
        )

        running_reward, demand_data_history_all, supply_data_history_all, mobile_history_all = episode_worker(worker_args)

        score_history_opt.append(float(running_reward))
        loss_history_opt.append(np.nan)
        solved_episode_ids.append(i)

        demand_history_opt.extend(demand_data_history_all)
        supply_history_opt.extend(supply_data_history_all)
        mobile_history_opt.extend(mobile_history_all)

        print(f"Solved episode {i} | reward = {float(running_reward):.3f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f"score_history_{args.run_name}_seed_{resolved_cfg['seed']}.npy", np.asarray(score_history_opt, dtype=float))
    np.save(output_dir / f"loss_history_{args.run_name}_seed_{resolved_cfg['seed']}.npy", np.asarray(loss_history_opt, dtype=float))
    np.save(output_dir / f"demand_history_{args.run_name}_seed_{resolved_cfg['seed']}.npy", np.asarray(demand_history_opt, dtype=object), allow_pickle=True)
    np.save(output_dir / f"supply_history_{args.run_name}_seed_{resolved_cfg['seed']}.npy", np.asarray(supply_history_opt, dtype=object), allow_pickle=True)
    np.save(output_dir / f"mobile_history_{args.run_name}_seed_{resolved_cfg['seed']}.npy", np.asarray(mobile_history_opt, dtype=object), allow_pickle=True)
    np.save(output_dir / f"solved_episode_ids_{args.run_name}_seed_{resolved_cfg['seed']}.npy", np.asarray(solved_episode_ids, dtype=int))

    print(f"Saved optimization results to: {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)