from world_model import LoadBalanceWorldEnv, make_model
import numpy as np
from helpers import state_prediction_tuning, pred_func_sasr, load_state_prediction_data
from helper_rew import reward_prediction_tuning, load_data_sasr

################################################
# Generate the x,y scalars used to train these models
# The actual training data is ignored, using these helper functions
env_dir = "lb-medium-5-250"
rates_for_final_state = np.array([0.5, 0.75, 1.0, 1.25, 1.5])
batch_size = 512
seq_len = 250
train_split = 0.85
train_x_data, train_y_data, validation_x_data, \
validation_y_data, x_scalar_map_state, y_scalar_map_state, generator_map = state_prediction_tuning(env_dir, batch_size, train_split = 0.85, minmax = True, mean = True, z_std = True)
x_train, x_test, y_train, y_test, x_scalar_reward, rew_scalar, generator_std, data_info_train, data_info_test = reward_prediction_tuning(env_dir, 512, seq_len=seq_len, train_split=train_split, validation_outlier_removal= False, standard_only = True)

states, actions, next_jobs, next_partial_states = load_state_prediction_data(env_dir)

jobs = next_jobs
j_copy = np.copy(jobs)
start_null = np.expand_dims(np.zeros((jobs.shape[0], jobs.shape[2])), axis = 1)
shifted = j_copy[:, : jobs.shape[1] - 1, :]
x = np.concatenate((start_null, shifted), axis = 1)
size = np.expand_dims(x[:,:,0], axis = 2)
time = np.expand_dims((jobs - x)[:,:,1], axis = 2)
time_difference = np.concatenate((size,time), axis = 2)
load_sampler_data = time_difference.reshape(-1, time_difference.shape[2])

state_model_norm_scheme = 'z_standardization'
x_state_scalar = x_scalar_map_state[state_model_norm_scheme]
y_state_scalar = y_scalar_map_state[state_model_norm_scheme]

env = LoadBalanceWorldEnv(num_servers = 5,
                          num_steps = 250,
                          reward_predictor_path = "world_model/reward_weights/weights.hdf5",
                          reward_predictor_model_func = make_model,
                          x_scalar_reward = x_scalar_reward,
                          rew_scalar = rew_scalar,
                          state_predictor_path = "world_model/state_weights/weights.hdf5",
                          x_state_scalar = x_state_scalar,
                          y_state_scalar = y_state_scalar,
                          load_sampler_data  = load_sampler_data ,
                          rates_for_final_state=rates_for_final_state,
                          )
# ################################################
experiences = 100
# Note will automatically finish at 250
actions = 300


percentages = []
rews = []
final_rews = []
max_loads = []
all_r = []
for i in range(experiences):
    obs = env.reset()
    prev_rew = 0
    max_l = 0
    cum_rew = 0
    for j in range(actions):
        action = env.action_space.sample()
        obs, rewards, dones, info = env.step(action)
        cum_rew += rewards
        if dones:
            all_r.append(cum_rew)
            final_rews.append(rewards)
            percent = float(rewards)  / float(prev_rew + rewards)
            percentages.append(percent)
            print('run',i)
            break
        else:
            rews.append(rewards)
            prev_rew += rewards
percentages = np.asarray(percentages)
rews = np.asarray(rews)
final_rews = np.asarray(final_rews)
all_r = np.asarray(all_r)
print("MEAN PERCENTAGE in percent", np.mean(percentages) * 100)
print("STD percentage in percent", np.std(percentages) * 100)
print("MEAN reward", np.mean(rews))
print("STD reward", np.std(rews))
print("MEAN total reward", np.mean(all_r))
print("STD total reward", np.std(all_r))
print("MEAN final reward", np.mean(final_rews))
print("STD final rewards", np.std(final_rews))