from __future__ import annotations

"""
per_optimization_with_fixed.py

This file contains the rolling-horizon optimization baseline used as a
comparison against the RL method.

What this file does
-------------------
For one episode:
1. Load the already-generated demand for each time step
2. Solve a short-horizon optimization model
3. Use only the first optimized action
4. Apply that action to the same toy environment
5. Repeat until the episode ends

Important
---------
- The environment is reused only as a simulator.
- Demand is NOT re-generated here.
- Instead, saved demand from the RL data file is injected back into the env.
- Fixed chargers are always available through the fixed-capacity parameter.
"""

import math
import time as tm

import numpy as np
import torch
import torch.nn.functional as F
import gurobipy as gp
from gurobipy import GRB, quicksum


def get_total_cpu_time():
    try:
        import os
        import psutil

        process = psutil.Process(os.getpid())
        total_user = 0.0
        total_system = 0.0

        try:
            times = process.cpu_times()
            total_user += times.user
            total_system += times.system
        except psutil.NoSuchProcess:
            pass

        try:
            for child in process.children(recursive=True):
                try:
                    times = child.cpu_times()
                    total_user += times.user
                    total_system += times.system
                except psutil.NoSuchProcess:
                    continue
        except Exception:
            pass

        return total_user, total_system, total_user + total_system
    except Exception:
        return 0.0, 0.0, 0.0


def haversine(lat1, lon1, lat2, lon2):
    dLat = (lat2 - lat1) * math.pi / 180.0
    dLon = (lon2 - lon1) * math.pi / 180.0

    lat1 = lat1 * math.pi / 180.0
    lat2 = lat2 * math.pi / 180.0

    a = (
        math.sin(dLat / 2) ** 2
        + math.sin(dLon / 2) ** 2 * math.cos(lat1) * math.cos(lat2)
    )
    rad = 6371.0
    c = 2 * math.asin(math.sqrt(a))
    return rad * c


def episode_worker(args):
    (
        episode_number,
        env,
        config,
        customer_demands_per_location_matrix,
        num_demand_sample_all_matrix,
        num_customers_matrix,
        customer_locations_matrix,
        nearby_stations_matrix,
        customer_demands_matrix,
        customer_order_times_matrix,
        customer_charging_time_AC_matrix,
        customer_charging_time_DC_matrix,
        customer_not_charging_time_matrix,
    ) = args

    max_t = config.max_t
    num_mobile = config.num_mobile
    num_mobile_location = config.num_mobile_location
    Capacity_Value = config.Capacity_Value
    facility_locations = config.facility_locations
    fixed_locations = config.fixed_locations
    Dis = config.Dis
    num_fixed_location = config.num_fixed_location
    demand_points = config.demand_points
    number_demand_points = len(demand_points)

    number_of_period_default = getattr(config, "optimization_window", 4)

    penalty_cap = config.penalty_cap
    p_cap_lim = config.p_cap_lim
    penalty_unused_cap = config.penalty_unused_cap
    met_demand_reward = config.met_demand_reward
    mobile_not_move_penalty = config.mobile_not_move_penalty

    opt_time_limit = getattr(config, "opt_time_limit", 1800.0)
    opt_mip_gap = getattr(config, "opt_mip_gap", 0.0)
    opt_threads = getattr(config, "opt_threads", 8)

    fixed_capacity_cfg = getattr(config, "fixed_capacity", 1000.0)
    if np.isscalar(fixed_capacity_cfg):
        fixed_capacity = np.ones((num_fixed_location,), dtype=float) * float(fixed_capacity_cfg)
    else:
        fixed_capacity = np.asarray(fixed_capacity_cfg, dtype=float).reshape(-1)
        if len(fixed_capacity) != num_fixed_location:
            raise ValueError("fixed_capacity length must equal num_fixed_location.")

    distance = np.zeros((num_mobile_location, number_demand_points))
    covering_bin = np.zeros((num_mobile_location, number_demand_points))

    distance2 = np.zeros((num_fixed_location, number_demand_points))
    covering_bin2 = np.zeros((num_fixed_location, number_demand_points))

    mobile_locations_ = np.asarray(facility_locations)[0, :, :]
    demand_locations = np.asarray(demand_points)
    fixed_locations_ = np.asarray(fixed_locations)

    for i in range(num_mobile_location):
        for j in range(number_demand_points):
            distance[i, j] = haversine(
                mobile_locations_[i, 0],
                mobile_locations_[i, 1],
                demand_locations[j, 0],
                demand_locations[j, 1],
            )
            if distance[i, j] <= Dis:
                covering_bin[i, j] = 1

    for i in range(num_fixed_location):
        for j in range(number_demand_points):
            distance2[i, j] = haversine(
                fixed_locations_[i, 0],
                fixed_locations_[i, 1],
                demand_locations[j, 0],
                demand_locations[j, 1],
            )
            if distance2[i, j] <= Dis:
                covering_bin2[i, j] = 1

    def optimization(
        mobile_soc,
        random_samples,
        number_of_period,
        xbar,
        last_location,
        covering_bin,
        covering_bin2,
        fixed_capacity,
    ):
        with gp.Env(empty=True) as env1:
            env1.setParam("OutputFlag", 0)
            env1.setParam("LogToConsole", 0)
            env1.setParam("Threads", opt_threads)
            env1.start()

            with gp.Model(env=env1) as m:
                m.setParam(GRB.Param.MIPGap, opt_mip_gap)
                m.setParam("OutputFlag", 0)
                m.setParam("LogToConsole", 0)
                m.setParam("MIPFocus", 1)
                m.setParam("TimeLimit", opt_time_limit)

                UU = p_cap_lim + Capacity_Value

                open_facility = m.addVars(
                    num_mobile,
                    num_mobile_location,
                    number_of_period,
                    vtype=GRB.BINARY,
                    name="Open",
                )
                charge = m.addVars(
                    num_mobile,
                    number_of_period,
                    vtype=GRB.BINARY,
                    name="charge",
                )
                q = m.addVars(
                    num_mobile,
                    number_of_period,
                    vtype=GRB.BINARY,
                    name="q",
                )

                amount_charge = m.addVars(
                    num_mobile,
                    number_of_period,
                    vtype=GRB.CONTINUOUS,
                    name="amount_charge",
                    lb=0,
                    ub=Capacity_Value,
                )

                unmet_demand_customer = m.addVars(
                    number_demand_points,
                    number_of_period,
                    vtype=GRB.CONTINUOUS,
                    name="UnmetDemand",
                    lb=0,
                )

                met_demand_customer = m.addVars(
                    num_mobile,
                    num_mobile_location,
                    number_demand_points,
                    number_of_period,
                    vtype=GRB.CONTINUOUS,
                    name="metDemand",
                    lb=0,
                )

                met_demand_customer_by_fixed = m.addVars(
                    num_fixed_location,
                    number_demand_points,
                    number_of_period,
                    vtype=GRB.CONTINUOUS,
                    name="metDemand_fixed",
                    lb=0,
                )

                soc_str = m.addVars(
                    num_mobile,
                    number_of_period,
                    vtype=GRB.CONTINUOUS,
                    name="soc_str",
                    lb=0,
                    ub=Capacity_Value,
                )
                soc_end = m.addVars(
                    num_mobile,
                    number_of_period,
                    vtype=GRB.CONTINUOUS,
                    name="soc_end",
                    lb=0,
                    ub=Capacity_Value,
                )

                y = m.addVars(
                    num_mobile,
                    number_of_period,
                    vtype=GRB.CONTINUOUS,
                    name="y",
                    lb=0,
                )
                g = m.addVars(
                    num_mobile,
                    vtype=GRB.CONTINUOUS,
                    name="g",
                    lb=0,
                )

                m.setObjective(
                    quicksum(
                        met_demand_reward * met_demand_customer[i, j1, j2, t]
                        for i in range(num_mobile)
                        for j1 in range(num_mobile_location)
                        for j2 in range(number_demand_points)
                        for t in range(number_of_period)
                    )
                    - quicksum(
                        penalty_unused_cap * (soc_end[i, t] - y[i, t])
                        for i in range(num_mobile)
                        for t in range(number_of_period)
                    )
                    - quicksum(
                        penalty_cap * q[i, t]
                        for i in range(num_mobile)
                        for t in range(number_of_period)
                    )
                    - quicksum(
                        mobile_not_move_penalty * g[i]
                        for i in range(num_mobile)
                    ),
                    GRB.MAXIMIZE,
                )

                m.addConstrs(
                    quicksum(open_facility[i, j, t] for j in range(num_mobile_location)) + charge[i, t] == 1
                    for i in range(num_mobile)
                    for t in range(number_of_period)
                )

                m.addConstrs(
                    soc_str[i, 0] == mobile_soc[i]
                    for i in range(num_mobile)
                )

                m.addConstrs(
                    soc_end[i, t]
                    == soc_str[i, t]
                    - quicksum(
                        met_demand_customer[i, j1, j2, t]
                        for j1 in range(num_mobile_location)
                        for j2 in range(number_demand_points)
                    )
                    + amount_charge[i, t]
                    for i in range(num_mobile)
                    for t in range(number_of_period)
                )

                m.addConstrs(
                    amount_charge[i, t] <= charge[i, t] * Capacity_Value
                    for i in range(num_mobile)
                    for t in range(number_of_period)
                )

                m.addConstrs(
                    amount_charge[i, t] <= Capacity_Value - soc_str[i, t]
                    for i in range(num_mobile)
                    for t in range(number_of_period)
                )

                m.addConstrs(
                    soc_str[i, t + 1] == soc_end[i, t]
                    for i in range(num_mobile)
                    for t in range(number_of_period - 1)
                )

                m.addConstrs(
                    p_cap_lim * (1 - q[i, t]) + soc_end[i, t] - soc_end[i, t - 1] <= UU * charge[i, t]
                    for i in range(num_mobile)
                    for t in range(1, number_of_period)
                )

                m.addConstrs(
                    p_cap_lim * (1 - q[i, t]) + soc_end[i, t - 1] - soc_end[i, t] <= UU * (1 - charge[i, t])
                    for i in range(num_mobile)
                    for t in range(1, number_of_period)
                )

                m.addConstrs(
                    quicksum(
                        met_demand_customer[i, j1, j2, t]
                        for j1 in range(num_mobile_location)
                        for j2 in range(number_demand_points)
                    ) <= soc_str[i, t]
                    for i in range(num_mobile)
                    for t in range(number_of_period)
                )

                m.addConstrs(
                    met_demand_customer[i, j1, j2, t]
                    <= random_samples[j2, t] * open_facility[i, j1, t] * covering_bin[j1, j2]
                    for i in range(num_mobile)
                    for j1 in range(num_mobile_location)
                    for j2 in range(number_demand_points)
                    for t in range(number_of_period)
                )

                m.addConstrs(
                    quicksum(
                        met_demand_customer[i, j1, j2, t]
                        for i in range(num_mobile)
                        for j1 in range(num_mobile_location)
                    )
                    + quicksum(
                        met_demand_customer_by_fixed[jf1, j2, t]
                        for jf1 in range(num_fixed_location)
                    )
                    + unmet_demand_customer[j2, t]
                    == random_samples[j2, t]
                    for j2 in range(number_demand_points)
                    for t in range(number_of_period)
                )

                m.addConstrs(
                    met_demand_customer_by_fixed[jf1, j2, t]
                    <= fixed_capacity[jf1] * covering_bin2[jf1, j2]
                    for jf1 in range(num_fixed_location)
                    for j2 in range(number_demand_points)
                    for t in range(number_of_period)
                )

                m.addConstrs(
                    (open_facility[i, j, 0] - last_location[i, j]) * xbar[i] <= g[i]
                    for i in range(num_mobile)
                    for j in range(num_mobile_location)
                )

                m.addConstrs(
                    (last_location[i, j] - open_facility[i, j, 0]) * xbar[i] <= g[i]
                    for i in range(num_mobile)
                    for j in range(num_mobile_location)
                )

                m.addConstrs(
                    y[i, t] >= soc_end[i, t] - UU * (1 - charge[i, t])
                    for i in range(num_mobile)
                    for t in range(number_of_period)
                )
                m.addConstrs(
                    y[i, t] <= soc_end[i, t]
                    for i in range(num_mobile)
                    for t in range(number_of_period)
                )
                m.addConstrs(
                    y[i, t] <= UU * charge[i, t]
                    for i in range(num_mobile)
                    for t in range(number_of_period)
                )

                m.optimize()

                desired_period = 0
                opt_action = torch.zeros((num_mobile,), dtype=torch.long)

                for mobile in range(num_mobile):
                    if charge[mobile, desired_period].X > 0.5:
                        opt_action[mobile] = 0
                    else:
                        for mobile_location in range(num_mobile_location):
                            if open_facility[mobile, mobile_location, desired_period].X > 0.5:
                                opt_action[mobile] = int(mobile_location + 1)
                                break

        return opt_action

    env.use_history_demand_data = True

    running_reward = 0.0
    demand_data_history_all = []
    supply_data_history_all = []
    mobile_history_all = []

    episode_start_time = tm.time()
    cpu_start_time, system_start_time, total_start_time = get_total_cpu_time()

    print(
        f"Epoch {episode_number} -- "
        f"episode_start_time: {episode_start_time}, "
        f"cpu_start_time: {cpu_start_time}, "
        f"system_start_time: {system_start_time}, "
        f"total_start_time: {total_start_time}"
    )

    state = env.reset()

    mobile_soc = np.ones((num_mobile,)) * Capacity_Value
    xbar = np.zeros((num_mobile,))
    last_location = np.zeros((num_mobile, num_mobile_location))

    for time in range(max_t):
        if time <= max_t - number_of_period_default:
            number_of_period = number_of_period_default
        else:
            number_of_period = max_t - time

        demand_data = np.asarray(customer_demands_per_location_matrix)[time:time + number_of_period, :]
        demand_data = demand_data.T

        action = optimization(
            mobile_soc=mobile_soc,
            random_samples=demand_data,
            number_of_period=number_of_period,
            xbar=xbar,
            last_location=last_location,
            covering_bin=covering_bin,
            covering_bin2=covering_bin2,
            fixed_capacity=fixed_capacity,
        )

        action_final = F.one_hot(action, num_classes=num_mobile_location + 1)
        action_final = action_final.view(num_mobile, num_mobile_location + 1)

        num_demand_sample = np.asarray(num_demand_sample_all_matrix)[time, :]
        num_customers = np.asarray(num_customers_matrix)[time]
        customer_locations = np.asarray(customer_locations_matrix[time])
        nearby_stations = np.asarray(nearby_stations_matrix[time])
        customer_demands = np.asarray(customer_demands_matrix[time])
        customer_demands_per_location = np.asarray(customer_demands_per_location_matrix)[time, :]
        customer_order_times = np.asarray(customer_order_times_matrix[time])
        customer_charging_time_AC = np.asarray(customer_charging_time_AC_matrix[time])
        customer_charging_time_DC = np.asarray(customer_charging_time_DC_matrix[time])
        customer_not_charging_time = np.asarray(customer_not_charging_time_matrix[time])

        env.use_history(
            num_demand_sample,
            num_customers,
            customer_locations,
            nearby_stations,
            customer_demands,
            customer_demands_per_location,
            customer_order_times,
            customer_charging_time_AC,
            customer_charging_time_DC,
            customer_not_charging_time,
        )

        (
            next_state,
            reward,
            done,
            demand_data_history,
            supply_data_history,
            mobile_history,
            prediction_demand_history,
        ) = env.step(action_final.numpy(), episode_number)

        state = next_state

        demand_data_history_all.append(demand_data_history)
        supply_data_history_all.append(supply_data_history)
        mobile_history_all.append(mobile_history)

        running_reward += float(reward)

        mobile_soc = prediction_demand_history[:, -3]
        xbar = prediction_demand_history[:, -1]
        last_location = action_final[:, 1:].numpy()

        if done:
            break

    episode_end_time = tm.time()
    cpu_end_time, system_end_time, total_end_time = get_total_cpu_time()

    print(
        f"Epoch {episode_number} -- "
        f"episode_end_time: {episode_end_time}, "
        f"cpu_end_time: {cpu_end_time}, "
        f"system_end_time: {system_end_time}, "
        f"total_end_time: {total_end_time}"
    )

    print(
        f"Epoch {episode_number} -- "
        f"episode_time: {episode_end_time - episode_start_time}, "
        f"cpu_time: {cpu_end_time - cpu_start_time}, "
        f"system_time: {system_end_time - system_start_time}, "
        f"total_time: {total_end_time - total_start_time}"
    )

    return running_reward, demand_data_history_all, supply_data_history_all, mobile_history_all