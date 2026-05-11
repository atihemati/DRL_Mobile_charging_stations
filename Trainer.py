from __future__ import annotations

import os
from collections import deque

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class RL(nn.Module):
    """
    Training wrapper for the toy RL experiment.
    """

    def __init__(self, config, env, policy, policy_, policy_1, deterministic=False):
        super().__init__()
        self.config = config
        self.env = env

        self.policy = policy
        self.policy_ = policy_
        self.policy_1 = policy_1

        self.gamma = config.gamma
        self.seq = config.seq
        self.max_t = config.max_t
        self.num_mobile = config.num_mobile
        self.num_mobile_location = config.num_mobile_location

        self.num_episodes = config.num_episodes_to_run
        self.num_episodes_to_change_prob = config.num_episodes_to_change_prob
        self.num_episodes_to_change_prob1 = config.num_episodes_to_change_prob1

        self.seprate_network = config.seprate_network
        self.save_animation = config.save_animation
        self.last_num_episodes_to_save_animation = config.last_num_episodes_to_save_animation

        self.deterministic = deterministic
        self.exploration = config.exploration
        self.batch_size = config.batch_size

        self.score_history20 = []
        self.losses_history = []
        self.demand_data_history_all = []
        self.supply_data_history_all = []
        self.mobile_history_all = []

    def plot(self, frame_idx, scores, losses, epsilons):
        plt.figure(figsize=(18, 4))

        plt.subplot(1, 3, 1)
        plt.title(f"episode {frame_idx} | score {np.mean(scores[-10:]):.2f}")
        plt.plot(scores)

        plt.subplot(1, 3, 2)
        plt.title("loss")
        plt.plot(losses)

        plt.subplot(1, 3, 3)
        plt.title("epsilon")
        plt.plot(epsilons)

        plt.tight_layout()
        plt.show()

    def save_animation_file(self, images, frame_duration=4.0, final_pause_frames=15):
        """
        Save animation frames as a GIF.

        Change frame_duration to slow down or speed up the animation:
        - larger value -> slower
        - smaller value -> faster
        """
        from PIL import Image

        folder_name = "animation/animation_RL"
        os.makedirs(folder_name, exist_ok=True)

        file_number = 1
        file_name = os.path.join(folder_name, f"animation_RL_{file_number}.gif")
        while os.path.exists(file_name):
            file_number += 1
            file_name = os.path.join(folder_name, f"animation_RL_{file_number}.gif")

        if not images:
            print("No animation frames found.")
            return

        pil_frames = []
        for frame in images:
            arr = np.asarray(frame)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            pil_frames.append(Image.fromarray(arr))

        widths = [img.size[0] for img in pil_frames]
        heights = [img.size[1] for img in pil_frames]
        target_width = max(widths)
        target_height = max(heights)

        normalized_frames = []
        for img in pil_frames:
            canvas = Image.new("RGB", (target_width, target_height), (242, 242, 242))
            x_offset = (target_width - img.size[0]) // 2
            y_offset = (target_height - img.size[1]) // 2
            canvas.paste(img.convert("RGB"), (x_offset, y_offset))
            normalized_frames.append(np.array(canvas))

        if final_pause_frames > 0:
            normalized_frames += [normalized_frames[-1]] * final_pause_frames

        imageio.mimsave(file_name, normalized_frames, duration=frame_duration, loop=0)
        print(f"Saved animation as {file_name}")

    def _build_return_value(
        self,
        save_generated_demand_data,
        num_demand_sample_all_matrix=None,
        num_customers_matrix=None,
        customer_locations_matrix=None,
        nearby_stations_matrix=None,
        customer_demands_matrix=None,
        customer_demands_per_location_matrix=None,
        customer_order_times_matrix=None,
        customer_charging_time_AC_matrix=None,
        customer_charging_time_DC_matrix=None,
        customer_not_charging_time_matrix=None,
    ):
        if save_generated_demand_data:
            return (
                self.score_history20,
                self.losses_history,
                self.demand_data_history_all,
                self.supply_data_history_all,
                self.mobile_history_all,
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
            )

        return (
            self.score_history20,
            self.losses_history,
            self.demand_data_history_all,
            self.supply_data_history_all,
            self.mobile_history_all,
        )

    def train_(self, prior_samples, scratch=False, plot=False, save_generated_demand_data=False, save_anim=False):
        if self.save_animation is None:
            self.save_animation = save_anim

        self.env.prior_samples = prior_samples[0]

        epsilons = []
        images = []

        if save_generated_demand_data:
            num_demand_sample_all_matrix = []
            num_customers_matrix = []
            customer_locations_matrix = []
            nearby_stations_matrix = []
            customer_demands_matrix = []
            customer_demands_per_location_matrix = []
            customer_order_times_matrix = []
            customer_charging_time_AC_matrix = []
            customer_charging_time_DC_matrix = []
            customer_not_charging_time_matrix = []

        try:
            for episode_idx in range(self.num_episodes):
                if episode_idx == self.num_episodes_to_change_prob and len(prior_samples) > 1:
                    self.env.prior_samples = prior_samples[1]
                    if scratch:
                        self.policy = self.policy_
                elif episode_idx == self.num_episodes_to_change_prob1 and len(prior_samples) > 2:
                    self.env.prior_samples = prior_samples[2]
                    if scratch:
                        self.policy = self.policy_1

                state = self.env.reset()
                states_buffer = deque([state], maxlen=self.seq)
                episode_reward = 0.0

                if save_generated_demand_data:
                    num_demand_sample_all_ep = []
                    num_customers_ep = []
                    customer_locations_ep = []
                    nearby_stations_ep = []
                    customer_demands_ep = []
                    customer_demands_per_location_ep = []
                    customer_order_times_ep = []
                    customer_charging_time_AC_ep = []
                    customer_charging_time_DC_ep = []
                    customer_not_charging_time_ep = []

                for step_idx in range(self.max_t):
                    state_sequence = np.asarray(list(states_buffer), dtype=np.float32)

                    state_value, probabilities = self.policy.forward(state_sequence, batch=False)
                    action = self.policy.select_action(
                        state_value=state_value,
                        probabilities=probabilities,
                        exploration=self.exploration,
                        eval=False,
                        deterministic=self.deterministic,
                    )

                    action_one_hot = F.one_hot(
                        action,
                        num_classes=self.num_mobile_location + 1,
                    ).view(self.num_mobile, self.num_mobile_location + 1)

                    (
                        next_state,
                        reward,
                        done,
                        demand_data_history,
                        supply_data_history,
                        mobile_history,
                        prediction_demand_history,
                    ) = self.env.step(
                        action_one_hot.numpy(),
                        episode_idx,
                    )

                    if save_generated_demand_data:
                        history = self.env.save_history()
                        (
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
                        ) = history

                        num_demand_sample_all_ep.append(num_demand_sample)
                        num_customers_ep.append(num_customers)
                        customer_locations_ep.append(customer_locations)
                        nearby_stations_ep.append(nearby_stations)
                        customer_demands_ep.append(np.asarray(customer_demands).flatten())
                        customer_demands_per_location_ep.append(customer_demands_per_location)
                        customer_order_times_ep.append(customer_order_times)
                        customer_charging_time_AC_ep.append(customer_charging_time_AC)
                        customer_charging_time_DC_ep.append(customer_charging_time_DC)
                        customer_not_charging_time_ep.append(customer_not_charging_time)

                    self.demand_data_history_all.append(demand_data_history)
                    self.supply_data_history_all.append(supply_data_history)
                    self.mobile_history_all.append(mobile_history)

                    self.policy.rewards.append(float(reward))
                    self.policy.update_epsilon()
                    epsilons.append(self.policy.epsilon)

                    episode_reward += float(reward)
                    state = next_state
                    states_buffer.append(state)

                    if self.save_animation and episode_idx >= self.num_episodes - self.last_num_episodes_to_save_animation:
                        images = self.env.animation(4, episode_idx, step_idx, reward)

                    if done:
                        break

                actor_loss, critic_loss = self.policy.calculateLoss(self.gamma)
                total_loss = actor_loss + critic_loss
                self.policy.learn(actor_loss, critic_loss)

                self.score_history20.append(float(episode_reward))
                self.losses_history.append(float(total_loss.detach().cpu().item()))

                if save_generated_demand_data:
                    num_demand_sample_all_matrix.append(num_demand_sample_all_ep)
                    num_customers_matrix.append(num_customers_ep)
                    customer_locations_matrix.append(customer_locations_ep)
                    nearby_stations_matrix.append(nearby_stations_ep)
                    customer_demands_matrix.append(customer_demands_ep)
                    customer_demands_per_location_matrix.append(customer_demands_per_location_ep)
                    customer_order_times_matrix.append(customer_order_times_ep)
                    customer_charging_time_AC_matrix.append(customer_charging_time_AC_ep)
                    customer_charging_time_DC_matrix.append(customer_charging_time_DC_ep)
                    customer_not_charging_time_matrix.append(customer_not_charging_time_ep)

                print(
                    f"Episode {episode_idx + 1:4d}/{self.num_episodes} | "
                    f"reward={episode_reward:10.3f} | "
                    f"loss={float(total_loss.detach().cpu().item()):9.4f} | "
                    f"epsilon={self.policy.epsilon:7.4f}"
                )

                if plot:
                    self.plot(episode_idx, self.score_history20, self.losses_history, epsilons)

            if self.save_animation and images:
                self.save_animation_file(images)

            if save_generated_demand_data:
                return self._build_return_value(
                    True,
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
                )

            return self._build_return_value(False)

        except KeyboardInterrupt:
            print("Training interrupted. Returning what was collected so far.")

            if save_generated_demand_data:
                return self._build_return_value(
                    True,
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
                )

            return self._build_return_value(False)