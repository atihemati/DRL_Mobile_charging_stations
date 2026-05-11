class Config(object):
    """
    Simple container for all experiment settings.

    I kept your original variable names so the environment, trainer,
    and agent code can use them directly.
    """

    def __init__(self):
        self.seed = None

        self.num_episodes_to_run = None
        self.num_episodes_to_change_prob = None
        self.num_episodes_to_change_prob1 = None

        self.hyperparameters = None
        self.max_t = None

        self.demand_points = None
        self.predict_demand = None

        self.action_size = None
        self.state_size = None

        self.num_mobile = None
        self.num_mobile_location = None

        self.num_demand_sample_all = None
        self.day_and_night_stations = None

        self.lr = None
        self.weight_decay = None
        self.optimizer_name = None
        self.entropy_coefficient = None

        self.Capacity_Value = None
        self.facility_costs = None
        self.facility_locations = None

        self.num_fixed_location = None
        self.fixed_locations = None
        self.fixed_types = None

        self.Dis = None
        self.plot_locations = None
        self.color = None
        self.n_trials = None

        self.gamma = None
        self.seq = None
        self.use_fix = None

        self.lstm_actor_hidden_size = None
        self.lstm_actor_num_layers = None
        self.lr1_actor_hidden_size = None
        self.lr2_actor_hidden_size = None
        self.lr3_actor_hidden_size = None

        self.lstm_critic_hidden_size = None
        self.lstm_critic_num_layers = None
        self.lr1_critic_hidden_size = None
        self.lr2_critic_hidden_size = None
        self.lr3_critic_hidden_size = None

        self.every_second_update = None
        self.num_t_step_per_day = None

        self.use_lstm = None
        self.bidirectional = None
        self.use_agent_as_batch = None
        self.add_drop_out = None
        self.use_one_hot_time_state = None
        self.seprate_network = None
        self.exploration = None
        self.batch_size = None

        self.penalty_cap = None
        self.p_cap_lim = None
        self.penalty_unused_cap = None
        self.met_demand_reward = None
        self.charge_cost = None
        self.mobile_not_move_penalty = None

        self.save_animation = None
        self.last_num_episodes_to_save_animation = None