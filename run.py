from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from Agent import ActorCritic
from Toy_Env import FacilityLocationtoy
from Trainer import RL
from config_utils import (
    build_config_object_from_resolved_dict,
    build_resolved_config_dict,
    load_json,
    load_points_and_priors,
    save_json,
    seed_everything,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)

    # optional small overrides
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--save-animation", action="store_true")
    parser.add_argument("--save-generated-demand-data", action="store_true")

    return parser.parse_args()


def apply_overrides(cfg_dict: dict, args):
    cfg = dict(cfg_dict)

    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.num_episodes is not None:
        cfg["num_episodes_to_run"] = args.num_episodes
    if args.run_name is not None:
        cfg["run_name"] = args.run_name
    if args.save_animation:
        cfg["save_animation"] = True
    if args.save_generated_demand_data:
        cfg["save_generated_demand_data"] = True

    return cfg


def main(args):
    base_cfg = load_json(args.config)
    base_cfg = apply_overrides(base_cfg, args)

    seed_everything(int(base_cfg["seed"]))

    resolved_cfg = build_resolved_config_dict(base_cfg)
    config = build_config_object_from_resolved_dict(resolved_cfg)

    output_dir = Path(resolved_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_data_dir = Path(resolved_cfg["generated_data_dir"])
    generated_data_dir.mkdir(parents=True, exist_ok=True)

    # load priors from file
    _, _, prior_samples = load_points_and_priors(resolved_cfg)

    env = FacilityLocationtoy(config)
    s = env.reset()

    config.state_size = s.shape[1]
    config.action_size = config.num_mobile_location + 1

    resolved_cfg["state_size"] = int(config.state_size)
    resolved_cfg["action_size"] = int(config.action_size)

    used_config_path = output_dir / f"config_used_{resolved_cfg['run_name']}_seed_{resolved_cfg['seed']}.json"
    save_json(resolved_cfg, used_config_path)

    policy = ActorCritic(config)
    policy_ = ActorCritic(config)
    policy_1 = ActorCritic(config)

    trainer = RL(config, env, policy, policy_, policy_1, deterministic=False)

    if resolved_cfg["save_generated_demand_data"]:
        (
            score_history_RL,
            losses_history_RL,
            demand_data_history_all_RL,
            supply_data_history_all_RL,
            mobile_history_all_RL,
            num_demand_sample_all_matrix,
            num_customers_matrix,
            customer_locations_matrix,
            nearby_stations_matrix,
            customer_demands_matrix,
            customer_demands_per_location_matrix,
            customer_order_times_matrix,
            customer_charging_time_AC_matrix,
            customer_charging_time_DC_matrix,
            customer_not_charging_time_matrix,
        ) = trainer.train_(
            prior_samples=prior_samples.tolist(),
            scratch=False,
            plot=False,
            save_generated_demand_data=True,
            save_anim=resolved_cfg["save_animation"],
        )
    else:
        (
            score_history_RL,
            losses_history_RL,
            demand_data_history_all_RL,
            supply_data_history_all_RL,
            mobile_history_all_RL,
        ) = trainer.train_(
            prior_samples=prior_samples.tolist(),
            scratch=False,
            plot=False,
            save_generated_demand_data=False,
            save_anim=resolved_cfg["save_animation"],
        )

    print("Save the output data")

    np.save(
        output_dir / f"score_history_{resolved_cfg['run_name']}_seed_{resolved_cfg['seed']}.npy",
        np.asarray(score_history_RL, dtype=float),
    )
    np.save(
        output_dir / f"loss_history_{resolved_cfg['run_name']}_seed_{resolved_cfg['seed']}.npy",
        np.asarray(losses_history_RL, dtype=float),
    )
    np.save(
        output_dir / f"demand_history_{resolved_cfg['run_name']}_seed_{resolved_cfg['seed']}.npy",
        np.asarray(demand_data_history_all_RL, dtype=object),
        allow_pickle=True,
    )
    np.save(
        output_dir / f"supply_history_{resolved_cfg['run_name']}_seed_{resolved_cfg['seed']}.npy",
        np.asarray(supply_data_history_all_RL, dtype=object),
        allow_pickle=True,
    )
    np.save(
        output_dir / f"mobile_history_{resolved_cfg['run_name']}_seed_{resolved_cfg['seed']}.npy",
        np.asarray(mobile_history_all_RL, dtype=object),
        allow_pickle=True,
    )

    if resolved_cfg["save_generated_demand_data"]:
        data_dict = {
            "num_demand_sample_all_matrix": num_demand_sample_all_matrix,
            "num_customers_matrix": num_customers_matrix,
            "customer_locations_matrix": customer_locations_matrix,
            "nearby_stations_matrix": nearby_stations_matrix,
            "customer_demands_matrix": customer_demands_matrix,
            "customer_demands_per_location_matrix": customer_demands_per_location_matrix,
            "customer_order_times_matrix": customer_order_times_matrix,
            "customer_charging_time_AC_matrix": customer_charging_time_AC_matrix,
            "customer_charging_time_DC_matrix": customer_charging_time_DC_matrix,
            "customer_not_charging_time_matrix": customer_not_charging_time_matrix,
        }

        generated_demand_path = generated_data_dir / f"generated_demand_{resolved_cfg['run_name']}_seed_{resolved_cfg['seed']}.pkl"
        with open(generated_demand_path, "wb") as file:
            pickle.dump(data_dict, file)

        print(f"Saved generated demand data to: {generated_demand_path}")

    print(f"Saved resolved config to: {used_config_path}")


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)