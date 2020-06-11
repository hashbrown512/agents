import tensorflow as tf
import pickle
import numpy as np
import os
from datetime import datetime

def load_data_sasr(env_dir):
    # Loads the sas, data, rewards and info functions
    raw_data = np.load(os.path.join(env_dir, "sasr.npz"))
    # Load preprocessed data
    data_sas = raw_data["sas_data"]
    data_reward = raw_data["rewards"]
    info = raw_data["info"]
    return data_sas, data_reward, info

def load_state_prediction_data(env_dir):
    raw_data = np.load(os.path.join(env_dir, "state_prediction.npz"))

    # Load preprocessed data
    states = raw_data["states"]
    next_jobs = raw_data["next_jobs"]
    next_partial_states = raw_data["next_partial_states"]
    actions = raw_data["actions"]
    return states, actions, next_jobs, next_partial_states

def pred_func_sasr(state, job, action, rates):
    # Return value is copied state, without the previous time and job
    ret = np.copy(state[:(state.shape[0] - 2)])
    # Size of the next job
    size = state[state.shape[0] - 2]
    # time of the next job
    time = job[1]
    # Increase the server that received the job
    ret[int(action)] += size
    # Complete work by given rates
    time_elapsed = time - state[-1]
    work = time_elapsed * rates
    ret -= (work)
    z = np.zeros(shape = ret.shape)
    # Make sure server rate does not go below zero
    return np.maximum(ret, z)

#####################################################
#####################################################
# State normalizers
#####################################################

class state_scalar_min_max():
    def __init__(self, epsilon, data):
        self.epsilon = epsilon
        self.min = np.min(data, axis = 0)
        self.max = np.max(data, axis = 0)
        self.max += self.epsilon
    def scale(self, data):
        # Data sas is in format of (episodes, episode lengths, sas combinations)
        return (data - self.min) / (self.max - self.min)
    def descale(self, data):
        return (data * (self.max - self.min)) + self.min

class state_scalar_mean_norm():
    def __init__(self, epsilon, data):
        self.epsilon = epsilon
        self.min = np.min(data, axis = 0)
        self.avg = np.mean(data, axis = 0)
        self.max = np.max(data, axis = 0)
        self.max += self.epsilon
    def scale(self, data):
        return (data - self.avg) / (self.max - self.min)
    def descale(self, data):
        return (data * (self.max - self.min)) + self.avg

class state_scalar_std():
    def __init__(self, epsilon, data):
        self.epsilon = epsilon
        self.std = np.std(data, axis = 0)
        self.avg = np.mean(data, axis = 0)
        # Redudant features, only for one server
        self.std += self.epsilon
    def scale(self, data):
        return (data - self.avg) / (self.std)
    def descale(self, data):
        return (data * (self.std)) + self.avg

#####################################################

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def state_prediction_tuning(env_dir, batch_size, train_split, minmax = False, mean = False, z_std = False, cap_obs = False):
    states, actions, next_jobs, next_partial_states = load_state_prediction_data(env_dir)
    print("STATES shape", states.shape)
    print("NEXT JOBS shape", next_jobs.shape)
    print("NEXT PARTIAL STATES shape", next_partial_states.shape)
    print("ACTIONS shape", actions.shape)
    actions = np.expand_dims(actions, axis=2)
    x_vec = np.concatenate((states, actions, next_jobs), axis=2)
    x_vec = x_vec.reshape(-1, x_vec.shape[2])
    y_vec = next_partial_states.reshape(-1, next_partial_states.shape[2])

    if cap_obs:
        print("min x_vec", np.min(x_vec, axis = 0))
        print("min y_vec", np.min(y_vec, axis = 0))
        print("mean x_vec", np.mean(x_vec, axis = 0))
        print("mean y_vec", np.mean(y_vec, axis = 0))
        print("max x_vec", np.max(x_vec, axis = 0))
        print("max y_vec", np.max(y_vec, axis = 0))

        print("x_vec.shape", x_vec.shape)
        print("y_vec.shape", y_vec.shape)
        new_x_vec = []
        new_y_vec = []
        for i in range(x_vec.shape[0]):
            f = False
            for j in range(y_vec.shape[1]):
                if y_vec[i,j] >= 500000:
                    f = True
                if x_vec[i,j] >= 500000:
                    f = True
            if not f:
                new_x_vec.append(x_vec[i])
                new_y_vec.append(y_vec[i])
        x_vec = np.stack(np.array(new_x_vec), axis=0)
        y_vec = np.stack(np.array(new_y_vec), axis=0)
        print("x_vec.shape", x_vec.shape)
        print("y_vec.shape", y_vec.shape)

        print("min x_vec", np.min(x_vec, axis = 0))
        print("min y_vec", np.min(y_vec, axis = 0))
        print("mean x_vec", np.mean(x_vec, axis = 0))
        print("mean y_vec", np.mean(y_vec, axis = 0))
        print("max x_vec", np.max(x_vec, axis = 0))
        print("max y_vec", np.max(y_vec, axis = 0))




    num_train = int(train_split * len(x_vec))

    validation_x_data = {}
    train_x_data = {}
    validation_y_data = {}
    train_y_data = {}
    x_scalar_map = {}
    y_scalar_map = {}
    generator_map = {}

    epsilon = 0.0000000001
    if minmax:
        x_min_max_scalar = state_scalar_min_max(epsilon, x_vec)
        y_min_max_scalar = state_scalar_min_max(epsilon, y_vec)
        x_min_max = x_min_max_scalar.scale(x_vec)
        y_min_max = y_min_max_scalar.scale(y_vec)

        x_train_mm = x_min_max[0:num_train]
        x_test_mm = x_min_max[num_train:]
        y_train_mm = y_min_max[0:num_train]
        y_test_mm = y_min_max[num_train:]

        def batch_generator_mm(batch_size):
            while True:
                x_shape = (batch_size, x_train_mm.shape[1])
                x_batch = np.zeros(shape=x_shape, dtype=np.float64)

                y_shape = (batch_size, y_train_mm.shape[1])
                y_batch = np.zeros(shape=y_shape, dtype=np.float64)

                for i in range(batch_size):
                    idx = np.random.randint(num_train)
                    x_batch[i] = x_train_mm[idx, :]
                    y_batch[i] = y_train_mm[idx, :]

                yield (x_batch, y_batch)

        generator_mm = batch_generator_mm(batch_size=batch_size)
        validation_x_data['min_max'] = x_test_mm
        train_x_data['min_max'] = x_train_mm
        validation_y_data['min_max'] = y_test_mm
        train_y_data['min_max'] = y_train_mm
        y_scalar_map['min_max'] = y_min_max_scalar
        x_scalar_map['min_max'] = x_min_max_scalar
        generator_map['min_max'] = generator_mm

    if mean:
        x_mean_scalar = state_scalar_mean_norm(epsilon, x_vec)
        y_mean_scalar = state_scalar_mean_norm(epsilon, y_vec)
        x_mean_norm = x_mean_scalar.scale(x_vec)
        y_mean_norm = y_mean_scalar.scale(y_vec)

        x_train_mean_norm = x_mean_norm[0:num_train]
        x_test_mean_norm = x_mean_norm[num_train:]
        y_train_mean_norm = y_mean_norm[0:num_train]
        y_test_mean_norm = y_mean_norm[num_train:]

        def batch_generator_mean_sca(batch_size):
            while True:
                x_shape = (batch_size, x_train_mean_norm.shape[1])
                x_batch = np.zeros(shape=x_shape, dtype=np.float64)

                y_shape = (batch_size, y_train_mean_norm.shape[1])
                y_batch = np.zeros(shape=y_shape, dtype=np.float64)

                for i in range(batch_size):
                    idx = np.random.randint(num_train)
                    x_batch[i] = x_train_mean_norm[idx, :]
                    y_batch[i] = y_train_mean_norm[idx, :]

                yield (x_batch, y_batch)

        generator_mean_norm = batch_generator_mean_sca(batch_size=batch_size)
        generator_map['mean_normalization'] = generator_mean_norm
        train_x_data['mean_normalization'] = x_train_mean_norm
        train_y_data['mean_normalization'] = y_train_mean_norm
        y_scalar_map['mean_normalization'] = y_mean_scalar
        x_scalar_map['mean_normalization'] = x_mean_scalar
        validation_x_data['mean_normalization'] = x_test_mean_norm
        validation_y_data['mean_normalization'] = y_test_mean_norm

    if z_std:
        x_std_scalar = state_scalar_std(epsilon, x_vec)
        y_std_scalar = state_scalar_std(epsilon, y_vec)
        x_std = x_std_scalar.scale(x_vec)
        y_std = y_std_scalar.scale(y_vec)

        x_train_std = x_std[0:num_train]
        x_test_std = x_std[num_train:]
        y_train_std = y_std[0:num_train]
        y_test_std = y_std[num_train:]

        def batch_generator_std(batch_size):
            while True:
                x_shape = (batch_size, x_train_std.shape[1])
                x_batch = np.zeros(shape=x_shape, dtype=np.float64)

                y_shape = (batch_size, y_train_std.shape[1])
                y_batch = np.zeros(shape=y_shape, dtype=np.float64)

                for i in range(batch_size):
                    idx = np.random.randint(num_train)
                    x_batch[i] = x_train_std[idx, :]
                    y_batch[i] = y_train_std[idx, :]

                yield (x_batch, y_batch)

        train_x_data['z_standardization'] = x_train_std
        train_y_data['z_standardization'] = y_train_std
        validation_x_data['z_standardization'] = x_test_std
        validation_y_data['z_standardization'] = y_test_std
        generator_std = batch_generator_std(batch_size=batch_size)
        y_scalar_map['z_standardization'] = y_std_scalar
        x_scalar_map['z_standardization'] = x_std_scalar
        generator_map['z_standardization'] = generator_std

    return train_x_data, train_y_data, validation_x_data, validation_y_data, x_scalar_map, y_scalar_map, generator_map


class scalar_std():
    def __init__(self, epsilon, data_sas):
        self.epsilon = epsilon
        entire_sas = data_sas.reshape(-1, data_sas.shape[2])
        self.std = np.std(entire_sas, axis = 0)
        self.avg = np.mean(entire_sas, axis = 0)
        # Redudant features, only for one server
        self.std += self.epsilon
    def scale(self, data_sas):
        return (data_sas - self.avg) / (self.std)
    def descale(self, scaled_data_reward):
        return (scaled_data_reward * (self.std)) + self.avg


def state_prediction_tuning_lstm(env_dir, batch_size, seq_len, train_split, validation_outlier_removal = False, standard_only = False, num_stds = 2):
    # data_sas_og, data_reward_og, data_info = load_data_sasr(env_dir)
    # data_sas_og = data_sas_og[:, :seq_len, :]
    # data_reward_og = np.expand_dims(data_reward_og[:, :seq_len], axis=2)
    # print("Data sas og shape", data_sas_og.shape)
    # print("data reward og shape", data_reward_og.shape)
    # print("std reward og", np.std(data_reward_og))
    # print("average reward og", np.mean(data_reward_og))
    # print("average reward final only", np.mean(data_reward_og[:,-1,:]))
    # print("min reward og", np.min(data_reward_og))
    # print("=======================")
    #
    # test = data_sas_og.reshape(-1, data_sas_og.shape[2])
    # print("stacked shape", test.shape)
    # print("MIN stacked data", np.min(test[:, :], axis=0))
    # print("MAX stacked data", np.max(test[:, :], axis=0))

    states, actions, next_jobs, next_partial_states = load_state_prediction_data(env_dir)
    print("STATES shape", states.shape)
    print("NEXT JOBS shape", next_jobs.shape)
    print("NEXT PARTIAL STATES shape", next_partial_states.shape)
    print("ACTIONS shape", actions.shape)
    actions = np.expand_dims(actions, axis=2)

    epsilon = 0.0000000001

    data_x = np.concatenate((states, actions, next_jobs), axis=2)
    data_y = next_partial_states
    print(data_x.shape)
    y_scalar = scalar_std(epsilon, data_y)
    data_reward = y_scalar.scale(data_y)

    x_scalar = scalar_std(epsilon, data_x)
    data_sas_std = x_scalar.scale(data_x)

    num_data = len(data_x)
    num_train = int(train_split * num_data)
    print("NUM TRAIN", num_train)

    x_train = data_sas_std[0:num_train]
    x_test = data_sas_std[num_train:]
    print(len(x_train) + len(x_test))

    y_train = data_reward[0:num_train]
    y_test = data_reward[num_train:]

    def batch_generator_std(batch_size):
        while True:
            x_shape = (batch_size, x_train.shape[1], x_train.shape[2])
            x_batch = np.zeros(shape=x_shape, dtype=np.float64)

            y_shape = (batch_size, y_train.shape[1], y_train.shape[2])
            y_batch = np.zeros(shape=y_shape, dtype=np.float64)

            for i in range(batch_size):
                idx = np.random.randint(num_train)
                x_batch[i] = x_train[idx, :, :]
                y_batch[i] = y_train[idx, :, :]

            yield (x_batch, y_batch)

    generator_std = batch_generator_std(batch_size=batch_size)
    x_batch, y_batch = next(generator_std)

    print("X BATCH SIZE", x_batch.shape, y_batch.shape)

    return x_train, x_test, y_train, y_test, x_scalar, y_scalar, generator_std


if __name__ == '__main__':
    # create_state_plus_load_state_predictor_to_files("lb-test", 1000, 200)

    # dir = os.listdir("lb-test/records")
    # print("LENGTH OF DIR", len(dir))
    #
    # dir = os.listdir("lb-default/records")
    # print("LENGTH OF DIR", len(dir))

    # create_state_plus_load_state_predictor_to_files("lb-test")

    # test = load_data_sasr("lb-one-ls-j-20-2")
    # print("loaded sasr")
    # states, actions, next_jobs, next_partial_states = load_state_prediction_data("lb-default")
    # print("loaded default")
    # states, actions, next_jobs, next_partial_states = load_state_prediction_data("lb-one-ls-j-20-2")

    batch_size = 2048
    train_split = 0.85
    env_dir = "lb-medium-5-250"
    env_results = env_dir + "_results/"
    # For faster testing
    train_percentage_eval = 1.0

    minmax = False
    mean = False
    z_std = True
    epochs = 200
    steps_per_epoch = 320
    ##################################################################

    helper_tuple = state_prediction_tuning(env_dir, batch_size, train_split, minmax=minmax,  mean=mean, z_std=z_std, cap_obs=True)
