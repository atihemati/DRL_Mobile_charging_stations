"""
Toy_Env.py

Toy charging environment used by the reinforcement learning agent.

-----------
At each time step:
1. Demand appears around demand points.
2. The RL policy decides what each mobile charger should do.
3. Each mobile charger either:
   - goes to recharge, or
   - moves to a service location and serves demand.
4. Customers are allocated to fixed chargers or mobile chargers.
5. A reward is computed and the next state is returned.

Saved history arrays
--------------------
demand_history: shape (num_demand_points, 12)
    0  sampled customer count
    1  total demand
    2  fully served customer count
    3  fully served energy
    4  partially served customer count
    5  partially served energy
    6  unmet-partial customer count
    7  unmet-partial energy
    8  fully unmet customer count
    9  fully unmet energy
    10 previous-step leftover energy completed now
    11 energy served by fixed chargers

supply_history: shape (num_demand_points, 2)
    0  number of active mobile chargers at each point
    1  total available mobile capacity at each point

mobile_history: shape (num_mobile, 9)
    0  full energy served now
    1  partial energy served now
    2  previous leftover energy completed now
    3  total served energy
    4  fully unmet demand assigned for accounting
    5  unmet partial energy left over
    6  capacity at start of the time step
    7  unused / reserved
    8  capacity at end of the time step
"""

from __future__ import annotations

import math
import os
import random
from typing import List

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


class FacilityLocationtoy:
    """
    Toy charging environment.

    Action format
    -------------
    The action given to step(...) is a one-hot matrix with shape:
        [num_mobile, num_mobile_location + 1]

    Meaning of the columns
    ----------------------
    column 0:
        send the mobile charger to recharge

    columns 1..K:
        deploy the charger to candidate service locations
    """

    def __init__(self, config):
        # Static configuration
        self.num_mobile = config.num_mobile
        self.num_mobile_location = config.num_mobile_location
        self.facility_costs = config.facility_costs
        self.facility_locations = config.facility_locations
        self.Capacity_Value = config.Capacity_Value
        self.num_fixed_location = config.num_fixed_location
        self.fixed_locations = config.fixed_locations
        self.fixed_types = config.fixed_types
        self.demand_points = config.demand_points
        self.max_t = config.max_t
        self.MaxT = config.max_t
        self.Dis = config.Dis
        self.plot_locations = config.plot_locations
        self.color = config.color
        self.n_trials = config.n_trials
        self.seed = config.seed

        # Reward settings
        self.penalty_cap = config.penalty_cap
        self.p_cap_lim = config.p_cap_lim
        self.penalty_unused_cap = config.penalty_unused_cap
        self.met_demand_reward = config.met_demand_reward
        self.charge_cost = config.charge_cost
        self.mobile_not_move_penalty = config.mobile_not_move_penalty

        # Options
        self.use_fix = config.use_fix
        self.every_second_update = config.every_second_update
        self.num_t_step_per_day = config.num_t_step_per_day
        self.use_one_hot_time_state = config.use_one_hot_time_state

        random.seed(self.seed)
        np.random.seed(self.seed)

        # Dynamic variables
        self.plot_mobile_location = np.zeros((self.num_mobile, 2))
        self.capacity_prev_plot = np.zeros(self.num_mobile)
        self.capacity_next_plot = np.zeros(self.num_mobile)
        self.met_demand_plot = np.zeros(self.num_mobile)
        self.unmet_demand_plot = np.zeros(self.num_mobile)

        self.demand_locations: List[np.ndarray] = []
        self.plot_demand_charger_allocation = []
        self.customer_met = []
        self.customer_met_color = []

        self.mobile_free_time = np.zeros(self.num_mobile)
        self.mobile_busy_time = np.zeros(self.num_mobile)
        self.fixed_free_time = np.zeros(self.num_fixed_location)

        self.mobile_not_move_state = np.zeros(self.num_mobile)
        self.mobile_not_move_reward = np.zeros(self.num_mobile)
        self.mobile_unmet_partial_demand = np.zeros(self.num_mobile)

        self.total_unmet_partial_demand = 0.0
        self.total_met_partial_demand = 0.0

        self.demand_data_history = []
        self.supply_data_history = []
        self.mobile_history = np.zeros((self.num_mobile, 9))
        self.mobile_data_history = []

        self.current_time = 0
        self.predict_demand = np.zeros(len(self.demand_points))
        self.capacity_last_state = torch.zeros((self.num_mobile, 1), dtype=torch.float32)

        self.last_state = None
        self.last_action = None
        self.day_and_night_stations = None
        self.prior_samples = None
        self.use_history_demand_data = False

        self.customer_locations = None
        self.nearby_stations = None
        self.customer_demands = None
        self.customer_order_times = None
        self.customer_charging_time_AC = None
        self.customer_charging_time_DC = None
        self.customer_not_charging_time = None
        self.customer_demands_per_location = None

        self.all_allocation = None
        self.off_check = None
        self.images = []

        self.num_customers = None
        self.num_demand_sample = np.zeros(len(self.demand_points))
        self.soc_pen = np.zeros((self.num_mobile, 1))

        self.color_name = None

        # Animation helpers
        self.active_mobile_ids_for_animation = []
        self.off_mobile_ids_for_animation = []

        self.generate_color_names()

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------
    def return_plot_data(self):
        return (
            self.plot_locations,
            self.plot_mobile_location,
            self.demand_locations,
            self.capacity_prev_plot,
            self.capacity_next_plot,
            self.met_demand_plot,
            self.unmet_demand_plot,
            self.max_t,
            self.off_check,
            self.all_allocation,
            self.customer_met,
            self.customer_met_color,
            self.demand_data_history,
            self.supply_data_history,
            self.mobile_data_history,
            self.mobile_not_move_state,
        )

    def min_max_normalization_tensor(self, tensor):
        if torch.isnan(tensor).any():
            return tensor
        return tensor / self.Capacity_Value

    def is_in_not_busy_hours(self, time_index):
        return time_index % self.num_t_step_per_day < int((6 * self.num_t_step_per_day) / 24)

    # ------------------------------------------------------------------
    # Small utility functions
    # ------------------------------------------------------------------
    def haversine(self, lat1, lon1, lat2, lon2):
        """
        Compute distance between two points.
        """
        dlat = (lat2 - lat1) * math.pi / 180.0
        dlon = (lon2 - lon1) * math.pi / 180.0
        lat1 = lat1 * math.pi / 180.0
        lat2 = lat2 * math.pi / 180.0
        a = (math.sin(dlat / 2) ** 2 + math.sin(dlon / 2) ** 2 * math.cos(lat1) * math.cos(lat2))
        radius_km = 6371.0
        return radius_km * 2 * math.asin(math.sqrt(a))

    def min_max_normalization(self, arr):
        """
        Normalize values by charger capacity.
        """
        arr = np.asarray(arr, dtype=float)
        if np.isnan(arr).any():
            return arr
        return arr / self.Capacity_Value

    def encode_day_of_week(self, time_index):
        """
        One-hot encode current day index within a week.
        """
        pattern_index = time_index % (self.num_t_step_per_day * 7)
        current_day = pattern_index // self.num_t_step_per_day
        encoding = np.zeros(7)
        encoding[current_day] = 1
        return encoding

    def encode_time_of_day(self, time_index):
        """
        One-hot encode the time index within a day.
        """
        actual_index = time_index % self.num_t_step_per_day
        encoding = np.zeros(self.num_t_step_per_day)
        encoding[actual_index] = 1
        return encoding

    # ------------------------------------------------------------------
    # Demand generation
    # ------------------------------------------------------------------
    def generate_demand_per_time_per_station(self):
        """
        Draw how many customers appear at each demand point for this time step.
        """
        values = np.random.multinomial(int(self.n_trials), self.prior_samples)
        values = np.asarray(values, dtype=float)
        values[values < 0] = 0
        return values

    def generate_day_and_night_stations(self):
        """
        Mark some stations as night-oriented and the rest as day-oriented.
        """
        stations = list(range(len(self.demand_points)))
        night_stations = [4, 10] if len(stations) > 10 else stations[: min(2, len(stations))]
        return np.asarray([0 if station in night_stations else 1 for station in stations])

    def generate_nearby_locations(self, demand_points, num_demand_sample):
        """
        Expand station-level counts into individual customer coordinates.
        """
        nearby_locations = []
        nearby_stations = []
        for station_idx, (lat, lon) in enumerate(demand_points):
            for _ in range(int(num_demand_sample[station_idx])):
                nearby_stations.append(station_idx)
                nearby_locations.append(
                    (
                        lat + random.uniform(-0.15, 0.15),
                        lon + random.uniform(-0.15, 0.15),
                    )
                )
        return np.asarray(nearby_locations), np.asarray(nearby_stations)

    def generate_demand(self, nearby_stations, num_customers, day_and_night_stations):
        """
        Generate energy demand value for each customer.
        """
        demands = []
        self.predict_demand = np.zeros(len(self.demand_points))

        for customer_idx in range(num_customers):
            station = int(nearby_stations[customer_idx])

            if day_and_night_stations[station] == 0:
                mean_now = 20 * math.cos(2 * math.pi * (self.current_time / self.num_t_step_per_day)) + 20
                mean_next = 20 * math.cos(2 * math.pi * ((self.current_time + 1) / self.num_t_step_per_day)) + 20
            else:
                mean_now = 20 * math.sin(2 * math.pi * ((self.current_time + 18) / self.num_t_step_per_day)) + 20
                mean_next = 20 * math.sin(2 * math.pi * ((self.current_time + 19) / self.num_t_step_per_day)) + 20

            demand = np.random.normal(mean_now, 25, size=1)
            demand = np.maximum(demand, 0)
            demands.append(demand)
            self.predict_demand[station] += np.random.normal(mean_next, 0, size=1)

        demands = np.asarray(demands)
        demands[demands < 0] = 0
        return demands, self.predict_demand

    def get_demand(self, i_episode, step_time):
        """
        Generate all demand-side data for one time step.
        """
        start_time = step_time * self.every_second_update
        end_time = (step_time + 1) * self.every_second_update

        self.num_demand_sample = self.generate_demand_per_time_per_station()
        self.day_and_night_stations = self.generate_day_and_night_stations()
        self.customer_locations, self.nearby_stations = self.generate_nearby_locations(
            self.demand_points,
            self.num_demand_sample,
        )
        self.num_customers = len(self.customer_locations)
        self.customer_demands, self.predict_demand = self.generate_demand(
            self.nearby_stations,
            self.num_customers,
            self.day_and_night_stations,
        )

        self.customer_demands_per_location = np.zeros(len(self.demand_points))
        for idx, station_idx in enumerate(self.nearby_stations):
            self.customer_demands_per_location[int(station_idx)] += self.customer_demands[idx]

        self.customer_order_times = np.asarray(
            [random.randint(start_time, end_time) for _ in range(self.num_customers)]
        )
        self.customer_charging_time_AC = self.customer_demands / 10
        self.customer_charging_time_DC = self.customer_demands / 20
        self.customer_not_charging_time = np.zeros(self.num_customers)

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------
    def save_history(self):
        """
        Return the generated demand-side history for this time step.
        """
        return (
            self.num_demand_sample,
            self.num_customers,
            self.customer_locations,
            self.nearby_stations,
            self.customer_demands,
            self.customer_demands_per_location,
            self.customer_order_times,
            self.customer_charging_time_AC,
            self.customer_charging_time_DC,
            self.customer_not_charging_time,
        )

    def use_history(
        self,
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
    ):
        """
        Load externally provided history instead of generating fresh demand.
        """
        self.num_demand_sample = num_demand_sample
        self.num_customers = num_customers
        self.customer_locations = customer_locations
        self.nearby_stations = nearby_stations
        self.customer_demands = customer_demands
        self.customer_demands_per_location = customer_demands_per_location
        self.customer_order_times = customer_order_times
        self.customer_charging_time_AC = customer_charging_time_AC
        self.customer_charging_time_DC = customer_charging_time_DC
        self.customer_not_charging_time = customer_not_charging_time

    # ------------------------------------------------------------------
    # Animation / plotting helpers
    # ------------------------------------------------------------------
    def generate_color_names(self):
        """
        Define a clean visualization palette.
        """
        self.fixed_station_color = "#8e0b8e"      # purple
        self.mobile_candidate_color = "#8c6d5a"   # muted brown
        self.coverage_color = "#f28c8c"           # soft red/pink

        self.mobile_active_color = "#2ca02c"      # green
        self.mobile_recharge_color = "#d62728"    # red
        self.ev_met_color = "#1f77ff"             # blue
        self.ev_unmet_color = "#ffd11a"           # yellow
        self.ev_original_color = "#6f6f6f"        # gray

        # compatibility list
        self.color_name = []
        for _ in range(self.num_mobile):
            self.color_name.append(self.mobile_active_color)
        for _ in range(self.num_fixed_location):
            self.color_name.append(self.fixed_station_color)

    def _draw_capacity_square(self, ax, x, y, fill_ratio, label, box_size):
        """
        Draw a square for one active mobile charger.

        The square is filled from bottom to top according to fill_ratio.
        """
        fill_ratio = float(np.clip(fill_ratio, 0.0, 1.0))

        left = x - box_size / 2.0
        bottom = y - box_size / 2.0

        base_rect = Rectangle(
            (left, bottom),
            box_size,
            box_size,
            facecolor="white",
            edgecolor="black",
            linewidth=1.8,
            zorder=40,
        )
        ax.add_patch(base_rect)

        if fill_ratio > 0:
            fill_height = box_size * fill_ratio
            fill_rect = Rectangle(
                (left, bottom),
                box_size,
                fill_height,
                facecolor=self.mobile_active_color,
                edgecolor="none",
                alpha=0.95,
                zorder=41,
            )
            ax.add_patch(fill_rect)

        border_rect = Rectangle(
            (left, bottom),
            box_size,
            box_size,
            fill=False,
            edgecolor="black",
            linewidth=1.8,
            zorder=42,
        )
        ax.add_patch(border_rect)

        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            color="black",
            fontsize=10,
            fontweight="bold",
            zorder=43,
        )

    def _get_point_label(self, idx):
        """
        Return the prior probability shown above a demand point.
        """
        if self.prior_samples is None:
            return ""
        if idx < len(self.prior_samples):
            return f"{self.prior_samples[idx]:.2f}"
        return ""

    def _destination_point(self, lat_deg, lon_deg, distance_km, bearing_deg):
        """
        Compute the destination point reached by moving from
        (lat_deg, lon_deg) by distance_km in direction bearing_deg.
        """
        R = 6371.0
        lat1 = np.radians(lat_deg)
        lon1 = np.radians(lon_deg)
        bearing = np.radians(bearing_deg)
        angular_distance = distance_km / R

        lat2 = np.arcsin(
            np.sin(lat1) * np.cos(angular_distance)
            + np.cos(lat1) * np.sin(angular_distance) * np.cos(bearing)
        )

        lon2 = lon1 + np.arctan2(
            np.sin(bearing) * np.sin(angular_distance) * np.cos(lat1),
            np.cos(angular_distance) - np.sin(lat1) * np.sin(lat2),
        )

        return np.degrees(lat2), np.degrees(lon2)

    def _coverage_polygon_points(self, center_x, center_y, distance_km, num_points=120):
        """
        Build polygon points for the coverage region centered at (center_x, center_y).
        """
        polygon = []
        for angle_deg in np.linspace(0, 360, num_points, endpoint=False):
            lat2, lon2 = self._destination_point(
                lat_deg=center_x,
                lon_deg=center_y,
                distance_km=distance_km,
                bearing_deg=angle_deg,
            )
            polygon.append([lat2, lon2])

        return np.asarray(polygon, dtype=float)

    def _get_plot_bounds(self, extra_points=None, pad_ratio=0.10):
        """
        Compute dynamic plot bounds from all layout points.
        """
        pts = [np.asarray(self.demand_points, dtype=float)]

        if extra_points is not None:
            extra_points = np.asarray(extra_points, dtype=float)
            if extra_points.size > 0:
                pts.append(extra_points.reshape(-1, 2))

        all_pts = np.vstack(pts)

        xmin = float(np.min(all_pts[:, 0]))
        xmax = float(np.max(all_pts[:, 0]))
        ymin = float(np.min(all_pts[:, 1]))
        ymax = float(np.max(all_pts[:, 1]))

        span_x = max(xmax - xmin, 1.0)
        span_y = max(ymax - ymin, 1.0)

        coverage_pad = max(0.0, float(self.Dis) / 111.0)
        pad_x = max(pad_ratio * span_x, coverage_pad)
        pad_y = max(pad_ratio * span_y, coverage_pad)

        return xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y, span_x, span_y

    def _get_serving_location_for_ev(self, ev_index, original_point):
        """
        Return the actual serving location for one EV.

        all_allocation rows are ordered as:
            [fixed chargers, active mobile chargers]
        """
        if self.all_allocation is None:
            return np.array(original_point, dtype=float), "unserved"

        allocation = np.asarray(self.all_allocation)

        if allocation.ndim != 2:
            return np.array(original_point, dtype=float), "unserved"

        if ev_index >= allocation.shape[1]:
            return np.array(original_point, dtype=float), "unserved"

        assigned_rows = np.where(allocation[:, ev_index] > 0)[0]
        if len(assigned_rows) == 0:
            return np.array(original_point, dtype=float), "unserved"

        assigned_row = int(assigned_rows[0])

        if assigned_row < self.num_fixed_location:
            return np.array(self.fixed_locations[assigned_row], dtype=float), "fixed"

        local_mobile_idx = assigned_row - self.num_fixed_location
        if 0 <= local_mobile_idx < len(self.active_mobile_ids_for_animation):
            true_mobile_idx = self.active_mobile_ids_for_animation[local_mobile_idx]
            return np.array(self.plot_mobile_location[true_mobile_idx], dtype=float), "mobile"

        return np.array(original_point, dtype=float), "unserved"

    def _spread_points_around_anchor(self, points, radius=0.06, always_offset=False):
        """
        Spread points around shared anchors so markers do not overlap.

        If always_offset=True, even a single point is shifted away from the center.
        """
        points = np.asarray(points, dtype=float)
        if len(points) == 0:
            return points

        adjusted = points.copy()
        groups = {}

        for i, pt in enumerate(points):
            key = (round(pt[0], 5), round(pt[1], 5))
            groups.setdefault(key, []).append(i)

        for _, indices in groups.items():
            n = len(indices)

            if n == 1:
                if always_offset:
                    idx = indices[0]
                    angle = np.pi / 4.0
                    adjusted[idx, 0] += radius * np.cos(angle)
                    adjusted[idx, 1] += radius * np.sin(angle)
                continue

            for k, idx in enumerate(indices):
                angle = 2 * np.pi * k / n
                adjusted[idx, 0] += radius * np.cos(angle)
                adjusted[idx, 1] += radius * np.sin(angle)

        return adjusted

    def _push_points_away_from_nearest_anchor(self, points, anchors, min_distance, push_distance):
        """
        If a point is too close to the nearest demand-point anchor, push it away
        so it does not cover the location marker.
        """
        points = np.asarray(points, dtype=float)
        anchors = np.asarray(anchors, dtype=float)

        if len(points) == 0 or len(anchors) == 0:
            return points

        adjusted = points.copy()

        for i in range(len(adjusted)):
            diffs = anchors - adjusted[i]
            dist_sq = np.sum(diffs ** 2, axis=1)
            nearest_idx = int(np.argmin(dist_sq))
            nearest_anchor = anchors[nearest_idx]

            vec = adjusted[i] - nearest_anchor
            dist = float(np.linalg.norm(vec))

            if dist < min_distance:
                if dist < 1e-12:
                    angle = (i + 1) * (np.pi / 5.0)
                    vec = np.array([np.cos(angle), np.sin(angle)], dtype=float)
                else:
                    vec = vec / dist

                adjusted[i] = nearest_anchor + vec * push_distance

        return adjusted

    def _place_points_on_nearest_anchor_rings(self, points, anchors, radius):
        """
        Place points on clean rings around their nearest anchor.

        This is useful for plotting the original EV positions of moved EVs in a
        neat way around the original demand-location marker.
        """
        points = np.asarray(points, dtype=float)
        anchors = np.asarray(anchors, dtype=float)

        if len(points) == 0 or len(anchors) == 0:
            return points

        adjusted = np.zeros_like(points)
        groups = {}

        for i, pt in enumerate(points):
            diffs = anchors - pt
            dist_sq = np.sum(diffs ** 2, axis=1)
            nearest_idx = int(np.argmin(dist_sq))
            groups.setdefault(nearest_idx, []).append(i)

        for anchor_idx, indices in groups.items():
            anchor = anchors[anchor_idx]
            n = len(indices)

            if n == 1:
                angle = np.pi / 4.0
                adjusted[indices[0], 0] = anchor[0] + radius * np.cos(angle)
                adjusted[indices[0], 1] = anchor[1] + radius * np.sin(angle)
                continue

            for k, idx in enumerate(indices):
                angle = 2 * np.pi * k / n
                adjusted[idx, 0] = anchor[0] + radius * np.cos(angle)
                adjusted[idx, 1] = anchor[1] + radius * np.sin(angle)

        return adjusted

    def _place_points_on_exact_anchor_rings(self, anchor_points, radius):
        """
        Place points on clean rings around their exact anchor locations.

        This is useful for served EVs around a charger location.
        """
        anchor_points = np.asarray(anchor_points, dtype=float)
        if len(anchor_points) == 0:
            return anchor_points

        adjusted = anchor_points.copy()
        groups = {}

        for i, pt in enumerate(anchor_points):
            key = (round(pt[0], 6), round(pt[1], 6))
            groups.setdefault(key, []).append(i)

        for _, indices in groups.items():
            anchor = anchor_points[indices[0]]
            n = len(indices)

            if n == 1:
                angle = np.pi / 4.0
                adjusted[indices[0], 0] = anchor[0] + radius * np.cos(angle)
                adjusted[indices[0], 1] = anchor[1] + radius * np.sin(angle)
                continue

            for k, idx in enumerate(indices):
                angle = 2 * np.pi * k / n
                adjusted[idx, 0] = anchor[0] + radius * np.cos(angle)
                adjusted[idx, 1] = anchor[1] + radius * np.sin(angle)

        return adjusted

    def _get_layout_spacing(self):
        """
        Estimate a good local spacing scale from the demand-point layout.
        """
        pts = np.asarray(self.demand_points, dtype=float)
        n = len(pts)

        if n <= 1:
            return 1.0

        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)

        nearest = np.min(dist, axis=1)
        spacing = float(np.median(nearest))

        return max(spacing, 1e-3)

    def plot_background_layout(self, ax, show_probabilities=True, extra_points=None):
        """
        Draw the static network layout:
        - exact coverage regions
        - fixed stations
        - candidate locations
        - point IDs
        - optional prior probabilities
        """
        ax.set_facecolor("#f2f2f2")

        xmin, xmax, ymin, ymax, span_x, span_y = self._get_plot_bounds(extra_points=extra_points)
        label_offset = 0.06 * span_y

        for pos in self.demand_points:
            polygon = self._coverage_polygon_points(
                center_x=pos[0],
                center_y=pos[1],
                distance_km=float(self.Dis),
                num_points=120,
            )
            ax.fill(
                polygon[:, 0],
                polygon[:, 1],
                facecolor=self.coverage_color,
                edgecolor="#c47a7a",
                alpha=0.16,
                linewidth=1.0,
                zorder=1,
            )

        marker_size = 460 if max(span_x, span_y) <= 6 else 420
        id_font = 12 if max(span_x, span_y) <= 6 else 11
        prob_font = 13 if max(span_x, span_y) <= 6 else 12

        for i, pos in enumerate(self.demand_points):
            is_fixed = i < self.num_fixed_location

            if is_fixed:
                face_color = self.fixed_station_color
                edge_color = "#4a004a"
            else:
                face_color = self.mobile_candidate_color
                edge_color = "#444444"

            ax.scatter(
                pos[0],
                pos[1],
                s=marker_size,
                marker="o",
                color=face_color,
                edgecolor=edge_color,
                linewidth=1.5,
                zorder=5,
            )

            ax.text(
                pos[0],
                pos[1],
                f"{i + 1}",
                ha="center",
                va="center",
                color="white",
                fontsize=id_font,
                fontweight="bold",
                zorder=6,
            )

            if show_probabilities:
                label = self._get_point_label(i)
                if label != "":
                    ax.text(
                        pos[0],
                        pos[1] + label_offset,
                        label,
                        ha="center",
                        va="bottom",
                        color="black",
                        fontsize=prob_font,
                        fontweight="bold",
                        zorder=7,
                    )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_xlabel("X (=Longitude)", fontsize=18)
        ax.set_ylabel("Y (=Latitude)", fontsize=18)
        ax.tick_params(axis="both", labelsize=13)

    def _draw_charging_bay(self, bay_ax):
        """
        Draw a separate box showing chargers that are currently recharging.

        The box is always visible, even when empty.
        """
        bay_ax.set_facecolor("white")
        bay_ax.set_xticks([])
        bay_ax.set_yticks([])

        for spine in bay_ax.spines.values():
            spine.set_color("#888888")
            spine.set_linewidth(1.0)

        bay_ax.text(
            0.06,
            0.88,
            "Charging bay",
            transform=bay_ax.transAxes,
            fontsize=10,
            fontweight="bold",
            color="black",
            va="top",
        )

        if len(self.off_mobile_ids_for_animation) == 0:
            return

        max_cols = 3
        x_positions = [0.18, 0.50, 0.82]
        y_start = 0.54
        y_gap = 0.28

        for k, mobile_id in enumerate(self.off_mobile_ids_for_animation):
            row = k // max_cols
            col = k % max_cols
            x = x_positions[col]
            y = y_start - row * y_gap

            bay_ax.scatter(
                x,
                y,
                s=260,
                marker="s",
                color=self.mobile_recharge_color,
                edgecolor="black",
                linewidth=1.2,
                transform=bay_ax.transAxes,
                zorder=5,
            )
            bay_ax.text(
                x,
                y,
                f"M{mobile_id + 1}",
                transform=bay_ax.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                fontweight="bold",
                zorder=6,
            )

    def animation(self, limit, Episode, t, reward):
        """
        Create one animation frame.

        Visual meaning
        --------------
        - Purple circles: fixed station locations
        - Brown circles: mobile candidate locations
        - Pink transparent regions: coverage areas
        - Active mobile chargers: green capacity-filled squares
        - Recharging mobile chargers: red squares in charging bay
        - Served EVs: blue triangles
        - Unserved EVs: yellow triangles
        - Original EV positions: gray hollow triangles (only when the EV is truly
          served at another location)
        - Gray dashed lines: EV movement paths
        """
        layout_spacing = self._get_layout_spacing()

        served_offset_radius = 0.20 * layout_spacing
        charger_offset_radius = 0.11 * layout_spacing

        unserved_min_distance = 0.13 * layout_spacing
        unserved_push_distance = 0.22 * layout_spacing
        unserved_jitter_radius = 0.08 * layout_spacing

        original_ring_radius = 0.28 * layout_spacing

        prob_offset = 1.20 * max(
            0.14 * layout_spacing,
            charger_offset_radius + 0.06 * layout_spacing,
            served_offset_radius + 0.06 * layout_spacing,
            original_ring_radius + 0.06 * layout_spacing,
        )

        original_points = np.empty((0, 2))
        display_points = np.empty((0, 2))
        display_colors = []
        true_location_change = []

        if len(self.demand_locations) > 0:
            current_ev_locations = np.asarray(self.demand_locations[0], dtype=float)

            if len(self.customer_met) > 0:
                current_customer_met = np.asarray(self.customer_met[0], dtype=bool)
            else:
                current_customer_met = np.zeros(len(current_ev_locations), dtype=bool)

            original_points_list = []
            display_points_list = []
            display_colors = []
            true_location_change = []

            demand_points_arr = np.asarray(self.demand_points, dtype=float)

            for i, point in enumerate(current_ev_locations):
                point = np.asarray(point, dtype=float)
                is_met = bool(current_customer_met[i])

                original_points_list.append(point.copy())

                original_anchor_idx = int(
                    np.argmin(np.linalg.norm(demand_points_arr - point, axis=1))
                )
                original_anchor = demand_points_arr[original_anchor_idx]

                if is_met:
                    service_point, _ = self._get_serving_location_for_ev(i, point)
                    service_point = np.asarray(service_point, dtype=float)

                    display_points_list.append(service_point)
                    display_colors.append(self.ev_met_color)

                    changed = np.linalg.norm(service_point - original_anchor) > 1e-8
                    true_location_change.append(bool(changed))
                else:
                    display_points_list.append(point.copy())
                    display_colors.append(self.ev_unmet_color)
                    true_location_change.append(False)

            original_points = np.asarray(original_points_list, dtype=float)
            display_points = np.asarray(display_points_list, dtype=float)

            served_mask = np.array([c == self.ev_met_color for c in display_colors], dtype=bool)
            moved_mask = np.array(true_location_change, dtype=bool)

            if np.any(served_mask):
                display_points[served_mask] = self._place_points_on_exact_anchor_rings(
                    display_points[served_mask],
                    radius=served_offset_radius,
                )

            if np.any(~served_mask):
                unserved_points = display_points[~served_mask]
                unserved_points = self._push_points_away_from_nearest_anchor(
                    unserved_points,
                    self.demand_points,
                    min_distance=unserved_min_distance,
                    push_distance=unserved_push_distance,
                )
                unserved_points = self._spread_points_around_anchor(
                    unserved_points,
                    radius=unserved_jitter_radius,
                    always_offset=False,
                )
                display_points[~served_mask] = unserved_points

            if np.any(moved_mask):
                original_points[moved_mask] = self._place_points_on_nearest_anchor_rings(
                    original_points[moved_mask],
                    self.demand_points,
                    radius=original_ring_radius,
                )

        fig = plt.figure(figsize=(16, 9))

        main_ax = fig.add_axes([0.05, 0.10, 0.66, 0.80])
        legend_ax = fig.add_axes([0.71, 0.45, 0.25, 0.45])
        legend_ax.axis("off")
        bay_ax = fig.add_axes([0.78, 0.14, 0.16, 0.16])

        extra_pts = []
        if len(display_points) > 0:
            extra_pts.append(display_points)
        if len(original_points) > 0:
            extra_pts.append(original_points)
        extra_pts = np.vstack(extra_pts) if len(extra_pts) > 0 else None

        xmin, xmax, ymin, ymax, span_x, span_y = self._get_plot_bounds(extra_points=extra_pts)

        self.plot_background_layout(main_ax, show_probabilities=False, extra_points=extra_pts)
        main_ax.set_xlim(xmin, xmax)
        main_ax.set_ylim(ymin, ymax + 0.55 * prob_offset)

        if len(display_points) > 0:
            for i in range(len(display_points)):
                moved = bool(true_location_change[i])

                if moved:
                    main_ax.scatter(
                        original_points[i, 0],
                        original_points[i, 1],
                        s=140,
                        marker="^",
                        facecolor="none",
                        edgecolor=self.ev_original_color,
                        linewidth=1.5,
                        alpha=0.95,
                        zorder=20,
                    )

                    main_ax.plot(
                        [original_points[i, 0], display_points[i, 0]],
                        [original_points[i, 1], display_points[i, 1]],
                        linestyle="--",
                        linewidth=1.2,
                        color=self.ev_original_color,
                        alpha=0.80,
                        zorder=21,
                    )

                main_ax.scatter(
                    display_points[i, 0],
                    display_points[i, 1],
                    s=180,
                    marker="^",
                    color=display_colors[i],
                    edgecolor="black",
                    linewidth=0.9,
                    zorder=30,
                )

        if len(self.active_mobile_ids_for_animation) > 0:
            active_mobile_points = np.asarray(
                [self.plot_mobile_location[i] for i in self.active_mobile_ids_for_animation],
                dtype=float,
            )

            active_mobile_points = self._spread_points_around_anchor(
                active_mobile_points,
                radius=charger_offset_radius,
                always_offset=True,
            )

            charger_box_size = 0.28 * layout_spacing

            for j, mobile_id in enumerate(self.active_mobile_ids_for_animation):
                mx, my = active_mobile_points[j]
                fill_ratio = self.capacity_prev_plot[mobile_id] / self.Capacity_Value

                self._draw_capacity_square(
                    ax=main_ax,
                    x=mx,
                    y=my,
                    fill_ratio=fill_ratio,
                    label=f"M{mobile_id + 1}",
                    box_size=charger_box_size,
                )

        for i, pos in enumerate(self.demand_points):
            label = self._get_point_label(i)
            if label != "":
                main_ax.text(
                    pos[0],
                    pos[1] + prob_offset,
                    label,
                    ha="center",
                    va="bottom",
                    color="black",
                    fontsize=13,
                    fontweight="bold",
                    zorder=70,
                )

        self._draw_charging_bay(bay_ax)

        main_ax.set_title(
            f"Toy RL Simulation | Episode {Episode} | Step {t} | Reward = {float(reward):.2f}",
            fontsize=20,
            pad=24,
        )
        main_ax.set_xlabel("X (=Longitude)", fontsize=20)
        main_ax.set_ylabel("Y (=Latitude)", fontsize=20)
        main_ax.tick_params(axis="both", labelsize=14)

        legend_elements = [
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                label="Fixed station location",
                markerfacecolor=self.fixed_station_color,
                markeredgecolor="#4a004a",
                markersize=14,
            ),
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                label="Mobile station potential location",
                markerfacecolor=self.mobile_candidate_color,
                markeredgecolor="#444444",
                markersize=14,
            ),
            Patch(
                facecolor=self.coverage_color,
                edgecolor="none",
                alpha=0.18,
                label="Coverage area",
            ),
            Line2D(
                [0], [0],
                marker="s",
                color="w",
                label="Active mobile charger (fill = SoC ratio)",
                markerfacecolor=self.mobile_active_color,
                markeredgecolor="black",
                markersize=14,
            ),
            Line2D(
                [0], [0],
                marker="s",
                color="w",
                label="Recharging mobile charger",
                markerfacecolor=self.mobile_recharge_color,
                markeredgecolor="black",
                markersize=14,
            ),
            Line2D(
                [0], [0],
                marker="^",
                color="w",
                label="Served EV",
                markerfacecolor=self.ev_met_color,
                markeredgecolor="black",
                markersize=13,
            ),
            Line2D(
                [0], [0],
                marker="^",
                color="w",
                label="Unserved EV",
                markerfacecolor=self.ev_unmet_color,
                markeredgecolor="black",
                markersize=13,
            ),
            Line2D(
                [0], [0],
                marker="^",
                color="w",
                label="Original EV position",
                markerfacecolor="none",
                markeredgecolor=self.ev_original_color,
                markersize=12,
            ),
            Line2D(
                [0, 1], [0, 0],
                color=self.ev_original_color,
                linestyle="--",
                linewidth=1.2,
                label="EV movement path\n (if served at another location)",
            ),
        ]

        legend_ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(0.0, 0.90),
            ncol=1,
            fontsize=12,
            frameon=True,
            borderaxespad=0.0,
            handletextpad=0.8,
            labelspacing=0.7,
        )

        os.makedirs("animation", exist_ok=True)
        file_name = os.path.join("animation", "plot.png")
        plt.savefig(file_name, dpi=180)
        self.images.append(imageio.imread(file_name))
        plt.close(fig)

        return self.images

    # ------------------------------------------------------------------
    # Recharge / location helpers
    # ------------------------------------------------------------------
    def _nearest_fixed_location(self, mobile_index: int, previous_action: np.ndarray) -> np.ndarray:
        """
        If a charger chooses action 0, send it to the nearest fixed station.
        """
        prev_row = previous_action[mobile_index]
        prev_idx = int(np.argmax(prev_row))

        if prev_idx == 0:
            last_location = self.plot_mobile_location[mobile_index]
        else:
            last_location = self.facility_locations[0][prev_idx - 1]

        distances = [
            self.haversine(fx[0], fx[1], last_location[0], last_location[1])
            for fx in self.fixed_locations
        ]
        return self.fixed_locations[int(np.argmin(distances))]

    def charge(self, off, mobile_capacity_off):
        """
        Recharge all chargers that selected the recharge action.
        """
        mobile_capacity_off[:] = self.Capacity_Value

        for mobile_idx in off:
            self.total_unmet_partial_demand += self.mobile_unmet_partial_demand[mobile_idx]
            self.mobile_unmet_partial_demand[mobile_idx] = 0.0
            self.mobile_busy_time[mobile_idx] = 0.0
            self.mobile_free_time[mobile_idx] = (self.current_time + 1) * self.every_second_update

        mobile_met_demand_off = np.zeros(len(off))
        mobile_unmet_demand_off = np.zeros(len(off))
        return mobile_capacity_off, mobile_met_demand_off, mobile_unmet_demand_off

    # ------------------------------------------------------------------
    # Allocation logic
    # ------------------------------------------------------------------
    def allocate_customers(
        self,
        new_mobile_open_location,
        mobile_capacity,
        on,
        demand_points,
        num_demand_sample,
        last_mobile_open_location,
    ):
        """
        Allocate customers to mobile chargers or fixed chargers.

        This method also builds the station-level demand/supply matrices used
        later for analysis and plotting.
        """
        demand_history = np.zeros((len(self.demand_points), 12))
        demand_history[:, 0] = num_demand_sample
        supply_history = np.zeros((len(self.demand_points), 2))

        customer_locations = np.asarray(self.customer_locations)
        num_customers = int(self.num_customers)
        nearby_stations = np.asarray(self.nearby_stations, dtype=int)
        customer_demands = np.asarray(self.customer_demands).reshape(-1)
        customer_order_times = np.asarray(self.customer_order_times)
        customer_charging_time_AC = np.asarray(self.customer_charging_time_AC).reshape(-1)
        customer_charging_time_DC = np.asarray(self.customer_charging_time_DC).reshape(-1)
        customer_not_charging_time = np.asarray(self.customer_not_charging_time).reshape(-1)

        sort_idx = np.argsort(customer_order_times)
        customer_locations = customer_locations[sort_idx]
        nearby_stations = nearby_stations[sort_idx]
        customer_demands_sorted = customer_demands[sort_idx]
        customer_times_sorted = customer_order_times[sort_idx]
        customer_charging_time_AC_sorted = customer_charging_time_AC[sort_idx]
        customer_charging_time_DC_sorted = customer_charging_time_DC[sort_idx]
        customer_not_charging_time_sorted = customer_not_charging_time[sort_idx]

        for customer_idx, station_idx in enumerate(nearby_stations):
            demand_history[station_idx, 1] += customer_demands_sorted[customer_idx]

        mobile_capacity_al = mobile_capacity.detach().cpu().numpy().reshape(-1).copy()
        num_on_mobile = len(on)

        mobile_met_demand = np.zeros(num_on_mobile)
        mobile_unmet_demand = np.zeros(num_on_mobile)

        customer_met_demand = np.zeros(num_customers)
        customer_unmet_demand = np.zeros(num_customers)
        customer_met = np.zeros(num_customers, dtype=bool)
        customer_met_color = np.zeros(num_customers)

        demand_mobile_charger_allocation = np.zeros((num_customers, num_on_mobile))
        demand_fixed_charger_allocation = np.zeros((num_customers, self.num_fixed_location))
        mobile_partial_met_demand_current = np.zeros(self.num_mobile)

        active_mobile_location_indices = (
            np.argmax(new_mobile_open_location, axis=1) if num_on_mobile > 0 else np.array([], dtype=int)
        )

        for local_mobile_idx, location_idx in enumerate(active_mobile_location_indices):
            station_idx = int(location_idx + self.num_fixed_location) if not self.use_fix else int(location_idx)
            supply_history[station_idx, 0] += 1
            supply_history[station_idx, 1] += mobile_capacity_al[local_mobile_idx]

        previous_location_indices = (
            np.argmax(last_mobile_open_location, axis=1) if len(last_mobile_open_location) > 0 else np.array([], dtype=int)
        )
        mobile_free_charge_time = np.ones(self.num_mobile) * (self.current_time * self.every_second_update)

        for local_mobile_idx, true_mobile_idx in enumerate(on):
            if local_mobile_idx < len(previous_location_indices) and local_mobile_idx < len(active_mobile_location_indices):
                if previous_location_indices[local_mobile_idx] == active_mobile_location_indices[local_mobile_idx]:
                    mobile_free_charge_time[true_mobile_idx] = self.mobile_free_time[true_mobile_idx]
                    carry_demand = self.mobile_unmet_partial_demand[true_mobile_idx]
                    mobile_capacity_al[local_mobile_idx] -= carry_demand
                    mobile_partial_met_demand_current[true_mobile_idx] += carry_demand
                    self.mobile_history[true_mobile_idx, 2] += carry_demand

                    station_for_carry = (
                        int(active_mobile_location_indices[local_mobile_idx] + self.num_fixed_location)
                        if not self.use_fix
                        else int(active_mobile_location_indices[local_mobile_idx])
                    )
                    demand_history[station_for_carry, 10] += carry_demand

                    self.total_met_partial_demand += carry_demand
                    self.mobile_unmet_partial_demand[true_mobile_idx] = 0.0
                    self.mobile_not_move_reward[true_mobile_idx] = 1
                else:
                    self.total_unmet_partial_demand += self.mobile_unmet_partial_demand[true_mobile_idx]
                    self.mobile_unmet_partial_demand[true_mobile_idx] = 0.0

        fixed_free_charge_time = self.fixed_free_time.copy()

        mobile_locations = np.zeros((num_on_mobile, 2))
        for local_mobile_idx, location_idx in enumerate(active_mobile_location_indices):
            mobile_locations[local_mobile_idx] = self.facility_locations[0][int(location_idx)]

        customer_facility_distances = np.zeros((num_on_mobile, num_customers))
        customer_fixed_distances = np.zeros((self.num_fixed_location, num_customers))

        for i in range(num_on_mobile):
            for j in range(num_customers):
                customer_facility_distances[i, j] = self.haversine(
                    mobile_locations[i, 0],
                    mobile_locations[i, 1],
                    customer_locations[j, 0],
                    customer_locations[j, 1],
                )

        for i in range(self.num_fixed_location):
            for j in range(num_customers):
                customer_fixed_distances[i, j] = self.haversine(
                    self.fixed_locations[i, 0],
                    self.fixed_locations[i, 1],
                    customer_locations[j, 0],
                    customer_locations[j, 1],
                )

        if num_customers > 0:
            all_distances = np.concatenate((customer_facility_distances, customer_fixed_distances), axis=0)
        else:
            all_distances = np.zeros((num_on_mobile + self.num_fixed_location, 0))

        period_end = (self.current_time + 1) * self.every_second_update

        for customer_idx in range(num_customers):
            sorted_indices = np.argsort(all_distances[:, customer_idx])
            station_idx = int(nearby_stations[customer_idx])
            demand_value = float(customer_demands_sorted[customer_idx])
            arrival = float(customer_times_sorted[customer_idx])
            dc_time = float(customer_charging_time_DC_sorted[customer_idx])
            ac_time = float(customer_charging_time_AC_sorted[customer_idx])
            not_charge_time = float(customer_not_charging_time_sorted[customer_idx])

            for candidate_idx in sorted_indices:
                if customer_met[customer_idx]:
                    break

                if candidate_idx < num_on_mobile:
                    true_mobile_idx = on[candidate_idx]

                    if customer_facility_distances[candidate_idx, customer_idx] > self.Dis:
                        continue
                    if arrival < mobile_free_charge_time[true_mobile_idx]:
                        continue
                    if mobile_capacity_al[candidate_idx] <= 0:
                        continue

                    if arrival + dc_time <= period_end:
                        served = min(demand_value, float(mobile_capacity_al[candidate_idx]))
                        if served <= 0:
                            continue

                        customer_met[customer_idx] = True
                        customer_met_color[customer_idx] = self.color[true_mobile_idx]
                        demand_mobile_charger_allocation[customer_idx, candidate_idx] = 1
                        customer_met_demand[customer_idx] += served

                        demand_history[station_idx, 2] += 1
                        demand_history[station_idx, 3] += served

                        mobile_capacity_al[candidate_idx] -= served
                        mobile_met_demand[candidate_idx] += served
                        mobile_free_charge_time[true_mobile_idx] = arrival + dc_time
                        self.mobile_history[true_mobile_idx, 0] += served
                    else:
                        deliverable_fraction = max(period_end - arrival, 0.0) / max(dc_time, 1e-8)
                        served = min(demand_value * deliverable_fraction, float(mobile_capacity_al[candidate_idx]))
                        if served <= 0:
                            continue

                        remaining = max(demand_value - served, 0.0)

                        customer_met[customer_idx] = True
                        customer_met_color[customer_idx] = self.color[true_mobile_idx]
                        demand_mobile_charger_allocation[customer_idx, candidate_idx] = 1
                        customer_met_demand[customer_idx] += served

                        demand_history[station_idx, 4] += 1
                        demand_history[station_idx, 5] += served
                        demand_history[station_idx, 6] += 1
                        demand_history[station_idx, 7] += remaining

                        mobile_capacity_al[candidate_idx] -= served
                        mobile_met_demand[candidate_idx] += served
                        self.mobile_unmet_partial_demand[true_mobile_idx] += remaining
                        self.mobile_history[true_mobile_idx, 1] += served
                        self.mobile_history[true_mobile_idx, 5] += remaining
                        mobile_free_charge_time[true_mobile_idx] = arrival + dc_time
                    break

                fixed_idx = candidate_idx - num_on_mobile

                if customer_fixed_distances[fixed_idx, customer_idx] > self.Dis:
                    continue
                if arrival < fixed_free_charge_time[fixed_idx]:
                    continue

                customer_met[customer_idx] = True
                true_color_idx = fixed_idx + self.num_mobile
                customer_met_color[customer_idx] = self.color[true_color_idx]
                demand_fixed_charger_allocation[customer_idx, fixed_idx] = 1
                customer_met_demand[customer_idx] += demand_value

                demand_history[station_idx, 2] += 1
                demand_history[station_idx, 3] += demand_value
                demand_history[station_idx, 11] += demand_value

                if self.fixed_types[fixed_idx] == 1:
                    fixed_free_charge_time[fixed_idx] = arrival + ac_time + not_charge_time
                else:
                    fixed_free_charge_time[fixed_idx] = arrival + dc_time + not_charge_time
                break

            if not customer_met[customer_idx]:
                customer_unmet_demand[customer_idx] = demand_value
                demand_history[station_idx, 8] += 1
                demand_history[station_idx, 9] += demand_value

                for candidate_idx in sorted_indices:
                    if candidate_idx < num_on_mobile:
                        mobile_unmet_demand[candidate_idx] += demand_value
                        true_mobile_idx = on[candidate_idx]
                        self.mobile_history[true_mobile_idx, 4] += demand_value
                        break

        for mobile_idx in range(self.num_mobile):
            if mobile_partial_met_demand_current[mobile_idx] > 0 and mobile_idx in on:
                local_idx = on.index(mobile_idx)
                mobile_met_demand[local_idx] += mobile_partial_met_demand_current[mobile_idx]

        for mobile_idx in range(self.num_mobile):
            if mobile_idx in on:
                local_idx = on.index(mobile_idx)
                self.mobile_history[mobile_idx, 3] += mobile_met_demand[local_idx]

        for mobile_idx in range(self.num_mobile):
            self.mobile_busy_time[mobile_idx] = mobile_free_charge_time[mobile_idx]
            if mobile_free_charge_time[mobile_idx] >= period_end:
                self.mobile_free_time[mobile_idx] = mobile_free_charge_time[mobile_idx]
                self.mobile_not_move_state[mobile_idx] = 1
            else:
                self.mobile_free_time[mobile_idx] = period_end

        for fixed_idx in range(self.num_fixed_location):
            self.fixed_free_time[fixed_idx] = max(fixed_free_charge_time[fixed_idx], period_end)

        self.demand_data_history.append(demand_history)
        self.supply_data_history.append(supply_history)

        return (
            mobile_capacity_al,
            mobile_met_demand,
            mobile_unmet_demand,
            customer_demands_sorted,
            customer_met_demand,
            customer_unmet_demand,
            customer_locations,
            demand_mobile_charger_allocation.T,
            demand_fixed_charger_allocation.T,
            num_customers,
            customer_met,
            customer_met_color,
        )

    # ------------------------------------------------------------------
    # Main environment step
    # ------------------------------------------------------------------
    def step(self, action, i_episode):
        """
        Advance the environment by one time step.
        """
        self.current_time = self.MaxT - self.max_t
        if not self.use_history_demand_data:
            self.get_demand(i_episode, self.current_time)

        self.demand_locations = []
        self.customer_met = []
        self.customer_met_color = []
        self.mobile_not_move_state = np.zeros(self.num_mobile)
        self.mobile_not_move_reward = np.zeros(self.num_mobile)
        self.capacity_prev_plot = np.zeros(self.num_mobile)
        self.capacity_next_plot = np.zeros(self.num_mobile)
        self.met_demand_plot = np.zeros(self.num_mobile)
        self.unmet_demand_plot = np.zeros(self.num_mobile)
        self.total_unmet_partial_demand = 0.0
        self.total_met_partial_demand = 0.0
        self.demand_data_history = []
        self.supply_data_history = []
        self.mobile_data_history = []
        self.mobile_history = np.zeros((self.num_mobile, 9))
        self.mobile_busy_time = np.zeros(self.num_mobile)
        self.soc_pen = np.zeros(self.num_mobile)

        next_cos = np.ones(self.num_mobile) * math.cos(
            2 * math.pi * ((self.current_time + 1) / self.num_t_step_per_day)
        )
        next_sin = np.ones(self.num_mobile) * math.sin(
            -1.5 + 2 * math.pi * ((self.current_time + 1) / self.num_t_step_per_day)
        )
        self.next_time_state = np.ones((self.num_mobile, 1)) * ((self.current_time + 1) / self.MaxT)
        self.next_daytime_state = np.tile(self.encode_time_of_day(self.current_time + 1), (self.num_mobile, 1))
        self.next_weektime_state = np.tile(self.encode_day_of_week(self.current_time + 1), (self.num_mobile, 1))

        action_matrix = np.asarray(action, dtype=float)
        previous_action = np.copy(self.last_action)
        new_mobile_location = action_matrix

        for mobile_idx, row in enumerate(action_matrix):
            chosen_idx = int(np.argmax(row))
            if chosen_idx == 0:
                self.plot_mobile_location[mobile_idx] = self._nearest_fixed_location(mobile_idx, previous_action)
            else:
                self.plot_mobile_location[mobile_idx] = self.facility_locations[0][chosen_idx - 1]

        previous_capacity = self.capacity_last_state.detach().cpu().numpy().reshape(-1)
        self.capacity_prev_plot = previous_capacity.copy()
        self.mobile_history[:, 6] = previous_capacity.copy()

        last_mobile_free_time = self.mobile_free_time.copy()

        off = np.where(action_matrix[:, 0] == 1)[0]
        on = [idx for idx in range(self.num_mobile) if idx not in off]

        self.active_mobile_ids_for_animation = list(on)
        self.off_mobile_ids_for_animation = list(off)

        updated_capacity = previous_capacity.copy()
        mobile_met = np.zeros(self.num_mobile)
        mobile_unmet = np.zeros(self.num_mobile)
        all_allocation = np.zeros(
            (self.num_fixed_location + len(on), self.num_customers if self.num_customers is not None else 0)
        )

        if len(on) > 0:
            open_action = action_matrix[on, 1:]
            last_open_action = previous_action[on, 1:]

            alloc = self.allocate_customers(
                new_mobile_open_location=open_action,
                mobile_capacity=torch.as_tensor(previous_capacity[on], dtype=torch.float32),
                on=on,
                demand_points=self.demand_points,
                num_demand_sample=self.num_demand_sample,
                last_mobile_open_location=last_open_action,
            )

            (
                mobile_capacity_on,
                mobile_met_on,
                mobile_unmet_on,
                _customer_demands_sorted,
                _customer_met_demand,
                _customer_unmet_demand,
                customer_locations,
                demand_mobile_allocation,
                demand_fixed_allocation,
                _num_customers,
                customer_met,
                customer_met_color,
            ) = alloc

            updated_capacity[on] = mobile_capacity_on
            mobile_met[on] = mobile_met_on
            mobile_unmet[on] = mobile_unmet_on
            all_allocation = np.concatenate((demand_fixed_allocation, demand_mobile_allocation), axis=0)
            self.demand_locations.append(customer_locations)
            self.customer_met.append(customer_met)
            self.customer_met_color.append(customer_met_color)

        if len(off) > 0:
            self.mobile_not_move_state[off] = 2
            charged_capacity, mobile_met_off, mobile_unmet_off = self.charge(off, updated_capacity[off].copy())
            updated_capacity[off] = charged_capacity
            mobile_met[off] = mobile_met_off
            mobile_unmet[off] = mobile_unmet_off

        self.capacity_next_plot = updated_capacity.copy()
        self.met_demand_plot = mobile_met.copy()
        self.unmet_demand_plot = mobile_unmet.copy()
        self.mobile_history[:, 8] = updated_capacity.copy()

        reward = 0.0
        for mobile_idx in range(self.num_mobile):
            if abs(previous_capacity[mobile_idx] - updated_capacity[mobile_idx]) <= self.p_cap_lim:
                reward -= self.penalty_cap
                self.soc_pen[mobile_idx] = 1.0

        for mobile_idx in range(self.num_mobile):
            if last_mobile_free_time[mobile_idx] > (self.current_time * self.every_second_update):
                if self.mobile_not_move_reward[mobile_idx] != 1:
                    reward -= self.mobile_not_move_penalty

        reward += self.met_demand_reward * float(np.sum(mobile_met) + self.total_met_partial_demand)
        reward -= self.penalty_unused_cap * float(np.sum(updated_capacity[on])) if len(on) > 0 else 0.0

        updated_capacity_col = updated_capacity.reshape(-1, 1)
        mobile_met_col = mobile_met.reshape(-1, 1)
        xbar = (self.mobile_not_move_state.reshape(-1, 1) == 1).astype(float)
        self.soc_pen = self.soc_pen.reshape(-1, 1)

        cap_prev_norm = self.min_max_normalization(previous_capacity.reshape(-1, 1))
        cap_new_norm = self.min_max_normalization(updated_capacity_col)
        met_norm = self.min_max_normalization(mobile_met_col)
        next_cos_col = next_cos.reshape(-1, 1)
        next_sin_col = next_sin.reshape(-1, 1)

        if not self.use_one_hot_time_state:
            normalized_new_state = np.concatenate(
                (
                    new_mobile_location,
                    cap_prev_norm,
                    cap_new_norm,
                    met_norm,
                    xbar,
                    next_cos_col,
                    next_sin_col,
                    self.next_time_state,
                    self.soc_pen,
                ),
                axis=1,
            )
        else:
            normalized_new_state = np.concatenate(
                (
                    new_mobile_location,
                    cap_prev_norm,
                    cap_new_norm,
                    met_norm,
                    xbar,
                    self.next_daytime_state,
                    self.next_weektime_state,
                    self.soc_pen,
                ),
                axis=1,
            )

        current_time_one_hot = np.tile(self.encode_time_of_day(self.current_time), (self.num_mobile, 1))
        prediction_state = np.concatenate(
            (
                new_mobile_location[:, 1:],
                current_time_one_hot,
                updated_capacity_col,
                mobile_met_col,
                xbar,
            ),
            axis=1,
        )

        self.capacity_last_state = torch.as_tensor(updated_capacity_col, dtype=torch.float32)
        self.mobile_data_history.append(self.mobile_history.copy())
        self.last_action = np.copy(action_matrix)
        self.last_state = np.copy(normalized_new_state)
        self.all_allocation = all_allocation
        self.off_check = [off]

        self.max_t -= 1
        done = self.max_t <= 0

        return (
            normalized_new_state,
            float(reward),
            done,
            self.demand_data_history,
            self.supply_data_history,
            self.mobile_data_history,
            prediction_state,
        )

    def reset(self):
        """
        Reset the environment to the beginning of a new episode.
        """
        self.max_t = self.MaxT
        self.current_time = 0

        self.customer_met = []
        self.customer_met_color = []
        self.mobile_free_time = np.zeros(self.num_mobile)
        self.mobile_busy_time = np.zeros(self.num_mobile)
        self.fixed_free_time = np.zeros(self.num_fixed_location)
        self.mobile_not_move_state = np.zeros(self.num_mobile)
        self.mobile_not_move_reward = np.zeros(self.num_mobile)
        self.mobile_unmet_partial_demand = np.zeros(self.num_mobile)
        self.total_unmet_partial_demand = 0.0
        self.total_met_partial_demand = 0.0
        self.demand_data_history = []
        self.supply_data_history = []
        self.mobile_data_history = []
        self.soc_pen = np.zeros((self.num_mobile, 1))
        self.mobile_history = np.zeros((self.num_mobile, 9))

        self.capacity_last_state = torch.ones((self.num_mobile, 1), dtype=torch.float32) * self.Capacity_Value

        self.active_mobile_ids_for_animation = []
        self.off_mobile_ids_for_animation = []
        self.images = []

        initial_action = np.zeros((self.num_mobile, self.num_mobile_location + 1))
        initial_action[:, 0] = 1.0

        initial_capacity_prev = np.zeros((self.num_mobile, 1))
        initial_capacity_now = np.ones((self.num_mobile, 1)) * self.Capacity_Value
        initial_met = np.zeros((self.num_mobile, 1))
        initial_not_move = np.zeros((self.num_mobile, 1))
        initial_cos = np.ones((self.num_mobile, 1)) * math.cos(
            2 * math.pi * (self.current_time / self.num_t_step_per_day)
        )
        initial_sin = np.ones((self.num_mobile, 1)) * math.sin(
            2 * math.pi * (self.current_time / self.num_t_step_per_day)
        )
        self.next_time_state = np.ones((self.num_mobile, 1)) * (self.current_time / self.MaxT)
        self.next_daytime_state = np.tile(self.encode_time_of_day(self.current_time + 1), (self.num_mobile, 1))
        self.next_weektime_state = np.tile(self.encode_day_of_week(self.current_time + 1), (self.num_mobile, 1))

        if not self.use_one_hot_time_state:
            state = np.concatenate(
                (
                    initial_action,
                    self.min_max_normalization(initial_capacity_prev),
                    self.min_max_normalization(initial_capacity_now),
                    self.min_max_normalization(initial_met),
                    initial_not_move,
                    initial_cos,
                    initial_sin,
                    self.next_time_state,
                    self.soc_pen,
                ),
                axis=1,
            )
        else:
            state = np.concatenate(
                (
                    initial_action,
                    self.min_max_normalization(initial_capacity_prev),
                    self.min_max_normalization(initial_capacity_now),
                    self.min_max_normalization(initial_met),
                    initial_not_move,
                    self.next_daytime_state,
                    self.next_weektime_state,
                    self.soc_pen,
                ),
                axis=1,
            )

        self.last_state = np.copy(state)
        self.last_action = np.copy(initial_action)
        return state

    def render(self):
        """
        Placeholder for compatibility with gym-like interfaces.
        """
        return None