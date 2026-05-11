from __future__ import annotations

import argparse
from pathlib import Path

from config_utils import save_json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-config", type=str, default="configs/toy_rl_config.json")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", type=str, default="Toy_RL")
    parser.add_argument("--num-episodes", type=int, default=2000)

    parser.add_argument("--hours", type=int, default=1)
    parser.add_argument("--episode-days", type=int, default=10)

    parser.add_argument("--num-mobile", type=int, default=6)
    parser.add_argument("--use-fix", action="store_true")

    parser.add_argument("--capacity-value", type=float, default=200.0)
    parser.add_argument("--coverage-distance", type=float, default=100.0)
    parser.add_argument("--n-trials", type=int, default=100)

    parser.add_argument("--save-animation", action="store_true")
    parser.add_argument("--save-generated-demand-data", action="store_true")
    parser.add_argument("--last-num-episodes-to-save-animation", type=int, default=1)

    parser.add_argument("--output-dir", type=str, default="outputs/toy_rl")
    parser.add_argument("--generated-data-dir", type=str, default="outputs/toy_rl/generated_data")

    parser.add_argument("--fixed-points-file", type=str, required=True)
    parser.add_argument("--demand-points-file", type=str, required=True)
    parser.add_argument("--prior-samples-file", type=str, required=True)

    parser.add_argument("--facility-costs-file", type=str, default=None)

    return parser.parse_args()


def main(args):
    cfg = {
        "seed": args.seed,
        "run_name": args.run_name,
        "num_episodes_to_run": args.num_episodes,
        "num_episodes_to_change_prob": 20000,
        "num_episodes_to_change_prob1": 30000,
        "hours": args.hours,
        "episode_days": args.episode_days,
        "num_mobile": args.num_mobile,
        "use_fix": args.use_fix,
        "Capacity_Value": args.capacity_value,
        "Dis": args.coverage_distance,
        "n_trials": args.n_trials,
        "save_animation": args.save_animation,
        "save_generated_demand_data": args.save_generated_demand_data,
        "last_num_episodes_to_save_animation": args.last_num_episodes_to_save_animation,
        "output_dir": args.output_dir,
        "generated_data_dir": args.generated_data_dir,
        "fixed_points_file": args.fixed_points_file,
        "demand_points_file": args.demand_points_file,
        "prior_samples_file": args.prior_samples_file,
        "facility_costs": [6, 1, 4, 4, 8, 4],
        "facility_costs_file": args.facility_costs_file,
        "lr": 0.0005,
        "weight_decay": 0.01,
        "entropy_coefficient": 0.05,
        "gamma": 0.9,
        "seq": 4,
        "use_lstm": True,
        "bidirectional": False,
        "use_agent_as_batch": False,
        "add_drop_out": True,
        "use_one_hot_time_state": False,
        "seprate_network": True,
        "exploration": True,
        "batch_size": None,
        "lstm_actor_hidden_size": 64,
        "lstm_actor_num_layers": 1,
        "lr1_actor_hidden_size": 256,
        "lr2_actor_hidden_size": 64,
        "lr3_actor_hidden_size": None,
        "lstm_critic_hidden_size": 64,
        "lstm_critic_num_layers": 1,
        "lr1_critic_hidden_size": 64,
        "lr2_critic_hidden_size": 16,
        "lr3_critic_hidden_size": None,
        "penalty_cap": 2000,
        "p_cap_lim": 30,
        "penalty_unused_cap": 30,
        "met_demand_reward": 20,
        "charge_cost": 18,
        "mobile_not_move_penalty": 0,
        "optimization_window": 4,
        "fixed_capacity": 1000.0,
        "opt_time_limit": 1800.0,
        "opt_mip_gap": 0.0,
        "opt_threads": 8,
    }

    output_path = Path(args.output_config)
    save_json(cfg, output_path)
    print(f"Saved config to: {output_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)