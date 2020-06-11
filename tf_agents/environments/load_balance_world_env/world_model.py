import numpy as np
import os
import os.path
from os import path
import wget

from tf_agents.environments.load_balance_world_env.helpers import state_prediction_tuning, pred_func_sasr, load_state_prediction_data
from tf_agents.environments.load_balance_world_env.helper_rew import reward_prediction_tuning, load_data_sasr

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, GRU, LSTM, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.backend import square, mean
from tensorboard.plugins.hparams import api as hp

import gym
from gym import spaces, logger
from gym.utils import seeding

from gym.envs.registration import register

def make_model():
    num_servers = 5
    sas_length = (num_servers * 2) + 5 # Action, plus 2 jobs
    reward_model = Sequential()
    reward_model.add(LSTM(units=128,
                   return_sequences=True,
                   batch_input_shape=(1,1,sas_length), stateful=True))
    reward_model.add(Dense(32, activation='relu'))
    num_y_signals = 1  # Reward
    reward_model.add(Dense(num_y_signals, activation='linear'))
    optimizer = Adam(lr=1e-2)
    reward_model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
    return reward_model


class LoadBalanceWorldEnv(gym.Env):
    def __init__(self, num_servers, num_steps,
                 reward_predictor_path,
                 reward_predictor_model_func,
                 x_scalar_reward,
                 rew_scalar,
                 state_predictor_path,
                 x_state_scalar,
                 y_state_scalar,
                 load_sampler_data,
                 rates_for_final_state,
                 max_zero = True):
        self.num_servers = num_servers
        # Initialize action space
        self.action_space = spaces.Discrete(num_servers)
        # TODO
        # Hardcoding this for now, needs to be changed when using this class on other server world models
        # Hard to decide what to put here
        # In theory the bounds are infinite for each server
        # Similar for time
        # Best decision for now is to hard code the existing bounds, but you will need to cap them eventually
        self.obs_low = np.array([0] * (num_servers + 2))
        self.obs_high = np.array(([1500000] * (num_servers + 1)) + [375000000])
        self.observation_space = spaces.Box(
            low=self.obs_low, high=self.obs_high, dtype=np.float64)

        self.num_steps = num_steps
        self.reward_predictor_path = reward_predictor_path
        self.state_predictor_path = state_predictor_path
        self.load_sampler_data  = load_sampler_data
        self.n_samples = len(load_sampler_data)

        # Save normalization scalars
        self.x_scalar_reward = x_scalar_reward
        self.rew_scalar = rew_scalar
        self.x_state_scalar = x_state_scalar
        self.y_state_scalar = y_state_scalar

        # Boolean to hardcode state predictions
        self.max_zero = max_zero
        self.z_partial_state = np.array([0] * (num_servers))

        self.rates_for_final_state = rates_for_final_state

        # Load the various models
        self.reward_model = reward_predictor_model_func()
        self.reward_model.load_weights(reward_predictor_path)
        self.state_model = load_model(state_predictor_path)
        self.reset()

    def sample_next_load(self):
        # TODO:
        r = np.random.randint(self.n_samples)
        time = self.prev_job_state[1]
        return np.array([self.load_sampler_data[r,0] ,time + self.load_sampler_data[r,1]])

    def reset(self):
        self.reward_model.reset_states()
        self.prev_system_state = np.array([0] * (self.num_servers))
        # Generate a job
        self.prev_job_state = np.array([0,0])
        self.count = 0
        return np.concatenate([self.prev_system_state, self.prev_job_state])

    def step(self, action):
        prev_state_w_job = np.concatenate([self.prev_system_state, self.prev_job_state])
        self.count += 1
        if self.count == self.num_steps:
            # Custom logic for final time
            # Returning all zeros, plus the remaining time left
            next_job_state = np.concatenate([np.array([0]), np.array([self.prev_job_state[1] + np.max(self.prev_system_state / self.rates_for_final_state)])])
            entire_next_state = np.concatenate([self.z_partial_state, next_job_state])
            sas = np.concatenate([prev_state_w_job, np.array([action]), entire_next_state])
            # Last system state is all zeros, this line isn't necessary, removes warnings
            original_scale_system_prediction = self.z_partial_state
            done = True
        else:
            # Take previous state and job, sample the next state
            next_job_state = self.sample_next_load()
            sasprime = np.concatenate([prev_state_w_job, np.array([action]), next_job_state])

            normalized_sasprime = self.x_state_scalar.scale(sasprime)
            normalized_sasprime = np.expand_dims(normalized_sasprime, axis = 0)
            state_prediction = self.state_model.predict(normalized_sasprime)
            original_scale_system_prediction = self.y_state_scalar.descale(state_prediction)[0]
            if self.max_zero:
                original_scale_system_prediction = np.maximum(original_scale_system_prediction, self.z_partial_state)
            entire_next_state = np.concatenate([original_scale_system_prediction, next_job_state])
            sas = np.concatenate([prev_state_w_job, np.array([action]), entire_next_state])
            done = False
        normalized_sas = self.x_scalar_reward.scale(sas)
        normalized_sas = np.expand_dims(np.expand_dims(normalized_sas, axis = 0), axis = 0)
        reward_prediction = self.reward_model.predict(normalized_sas)
        original_scale_reward = self.rew_scalar.descale(reward_prediction)
        # Extract from 3 dimensional prediction
        original_scale_reward = original_scale_reward[0,0,0]
        # Reward should never be positive
        original_scale_reward = min(original_scale_reward, 0)

        self.prev_job_state = next_job_state
        self.prev_system_state = original_scale_system_prediction
        return entire_next_state, original_scale_reward, done, {}

class LoadBalanceWorldEnv250(LoadBalanceWorldEnv):
    def __init__(self):
        dir_name = os.path.dirname(__file__)
        env_dir = os.path.join(dir_name, "lb-medium-5-250")
        if not os.path.exists(os.path.join(dir_name, "lb-medium-5-250/sasr.npz")):
            wget.download('https://www.dropbox.com/s/q0bvbycyglb850o/state_prediction.npz?dl=1',
                          out=os.path.join(env_dir,'state_prediction.npz'))
            wget.download('https://www.dropbox.com/s/qqa3n51skc3ge4d/sasr.npz?dl=1', out=os.path.join(env_dir,'sasr.npz'))
        rates_for_final_state = np.array([0.5, 0.75, 1.0, 1.25, 1.5])
        batch_size = 512
        seq_len = 250
        train_split = 0.85
        train_x_data, train_y_data, validation_x_data, \
        validation_y_data, x_scalar_map_state, y_scalar_map_state, generator_map = state_prediction_tuning(env_dir,
                                                                                                           batch_size,
                                                                                                           train_split=0.85,
                                                                                                           minmax=True,
                                                                                                           mean=True,
                                                                                                z_std=True)
        x_train, x_test, y_train, y_test, x_scalar_reward, rew_scalar, generator_std, data_info_train, data_info_test = reward_prediction_tuning(
            env_dir, 512, seq_len=seq_len, train_split=train_split, validation_outlier_removal=False,
            standard_only=True)

        states, actions, next_jobs, next_partial_states = load_state_prediction_data(env_dir)

        jobs = next_jobs
        j_copy = np.copy(jobs)
        start_null = np.expand_dims(np.zeros((jobs.shape[0], jobs.shape[2])), axis=1)
        shifted = j_copy[:, : jobs.shape[1] - 1, :]
        x = np.concatenate((start_null, shifted), axis=1)
        size = np.expand_dims(x[:, :, 0], axis=2)
        time = np.expand_dims((jobs - x)[:, :, 1], axis=2)
        time_difference = np.concatenate((size, time), axis=2)
        load_sampler_data = time_difference.reshape(-1, time_difference.shape[2])

        state_model_norm_scheme = 'z_standardization'
        x_state_scalar = x_scalar_map_state[state_model_norm_scheme]
        y_state_scalar = y_scalar_map_state[state_model_norm_scheme]


        super().__init__(num_servers = 5,
                          num_steps = 250,
                          reward_predictor_path = os.path.join(dir_name, "world_model/reward_weights/weights.hdf5"),
                          reward_predictor_model_func = make_model,
                          x_scalar_reward = x_scalar_reward,
                          rew_scalar = rew_scalar,
                          state_predictor_path = os.path.join(dir_name, "world_model/state_weights/weights.hdf5"),
                          x_state_scalar = x_state_scalar,
                          y_state_scalar = y_state_scalar,
                          load_sampler_data  = load_sampler_data ,
                          rates_for_final_state=rates_for_final_state,

        )

register(
  id='LBWorldModel250-v0',
  entry_point=LoadBalanceWorldEnv250,
  max_episode_steps=251,
  reward_threshold=-250,
)