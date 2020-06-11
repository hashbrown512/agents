import tensorflow as tf
import numpy as np
import os
from datetime import datetime

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, GRU, LSTM
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.backend import square, mean
from tensorboard.plugins.hparams import api as hp

def load_data_sasr(env_dir):
    # Loads the sas, data, rewards and info functions
    raw_data = np.load(os.path.join(env_dir, "sasr.npz"))
    # Load preprocessed data
    data_sas = raw_data["sas_data"]
    data_reward = raw_data["rewards"]
    info = raw_data["info"]
    return data_sas, data_reward, info

class scalar_std():
    # TODO:
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


def reward_prediction_tuning(env_dir, batch_size, seq_len, train_split, validation_outlier_removal = False, standard_only = False, num_stds = 2):
    data_sas_og, data_reward_og, data_info = load_data_sasr(env_dir)
    data_sas_og = data_sas_og[:, :seq_len, :]
    data_reward_og = np.expand_dims(data_reward_og[:, :seq_len], axis=2)
    print("Data sas og shape", data_sas_og.shape)
    print("data reward og shape", data_reward_og.shape)
    print("std reward og", np.std(data_reward_og))
    print("average reward og", np.mean(data_reward_og))
    print("average reward final only", np.mean(data_reward_og[:,-1,:]))
    print("min reward og", np.min(data_reward_og))
    print("=======================")

    test = data_sas_og.reshape(-1, data_sas_og.shape[2])
    print("stacked shape", test.shape)
    print("MIN stacked data", np.min(test[:, :], axis=0))
    print("MAX stacked data", np.max(test[:, :], axis=0))

    ######################################
    # STD removal
    ######################################
    if standard_only:
        print("STD REMOVAL")
        print("===============================")
        print("MIN", np.min(data_reward_og))
        print("STD", np.std(data_reward_og))
        print("MeAN", np.mean(data_reward_og))
        mean_final = np.mean(data_reward_og[:,-1,:])
        print("MeAN final",mean_final)
        std_final = np.std(data_reward_og[:,-1,:])
        print("std final", std_final)

        print(data_reward_og.shape)
        new_states = []
        new_rewards = []
        new_info = []
        for i in range(data_reward_og.shape[0]):
            if data_reward_og[i,-1,:] > (mean_final - (float(num_stds) * std_final)):
                new_rewards.append(data_reward_og[i,:,:])
                new_states.append(data_sas_og[i, :, :])
                new_info.append(data_info[i, :, :])
        print("===============================")
        data_reward_og = np.stack(np.array(new_rewards), axis = 0)
        data_sas_og = np.stack(np.array(new_states), axis=0)
        data_info = np.stack(np.array(new_info), axis=0)
        print("shape after removal", data_reward_og.shape)
        print("mean final after removal", np.mean(data_reward_og[:,-1,:]))
        print("total mean after removal", np.mean(data_reward_og[:, :, :]))
        print("min after removal", np.min(data_reward_og))
        print("std after removal", np.std(data_reward_og))
        std_final = np.std(data_reward_og[:,-1,:])
        print("std final after removal", std_final)
    ######################################


    epsilon = 0.0000000001
    rew_scalar = scalar_std(epsilon, data_reward_og)
    data_reward = rew_scalar.scale(data_reward_og)

    x_scalar = scalar_std(epsilon, data_sas_og)
    data_sas_std = x_scalar.scale(data_sas_og)

    num_data = len(data_sas_og)
    num_train = int(train_split * num_data)
    print("NUM TRAIN", num_train)

    x_train = data_sas_std[0:num_train]
    x_test = data_sas_std[num_train:]
    print(len(x_train) + len(x_test))

    y_train = data_reward[0:num_train]
    y_test = data_reward[num_train:]

    data_info_train = data_info[0:num_train]
    data_info_test = data_info[num_train:]

    ######################################
    # Outlier removal
    ######################################
    if validation_outlier_removal:
        print("OUTLIER REMOVAL")
        print("===============================")
        print("MIN", np.min(y_train), np.min(y_test))
        print("STD", np.std(y_train), np.std(y_test))
        print("MAX", np.max(y_train), np.max(y_test))
        min_train = np.min(y_train)
        idx = set([])
        for i in range(len(y_test)):
            s = 0
            for j in range(len(y_test[0])):
                if y_test[i, j, 0] < min_train:
                    s += 1
                    idx.add(i)
        print("outliers", len(idx))
        print("idx", idx)
        print("total", [i + num_train for i in idx])
        print("=====")
        reset_idx = 0
        for i in idx:
            x_train[reset_idx, :, :] = x_test[i, :, :]
            y_train[reset_idx, :, :] = y_test[i, :, :]
            # This should be split into two seperate data infos
            data_info[reset_idx, :, :] = data_info[i + num_train, :, :]
            reset_idx += 1
        print("=====")
        print("STD", np.std(y_train), np.std(y_test))
        print("MAX", np.max(y_train), np.max(y_test))
        print("===============================")
    ######################################
    print("MIN train test", np.min(y_train), np.min(y_test))

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

    return x_train, x_test, y_train, y_test, x_scalar, rew_scalar, generator_std, data_info_train, data_info_test


if __name__ == '__main__':
    env_dir = "lb-one-ls-j-20-2"
    # data_sas_og, data_reward_og, data_info = load_data_sasr(env_dir)
    # print("DATA SAS SHAPE", data_sas_og.shape)
    # print("DATA REWARD SHAPE", data_reward_og.shape)
    # test = data_sas_og.reshape(-1, data_sas_og.shape[2])
    # print("stacked shape", test.shape)
    # print("MIN", np.min(test[:, :], axis=0))
    # print("MAX", np.max(test[:, :], axis=0))
    #
    # data_reward_og = np.expand_dims(data_reward_og[:, :], axis=2)
    # test = data_reward_og.reshape(-1, data_reward_og.shape[2])
    # print("stacked shape", test.shape)
    # print("MIN", np.min(test[:], axis=0))
    # print("MAX", np.max(test[:], axis=0))
    #
    # data_reward = np.expand_dims(data_reward_og[:, :99], axis=2)
    # test = data_reward.reshape(-1, data_reward.shape[2])
    # print("stacked shape", test.shape)
    # print("MIN", np.min(test[:], axis=0))
    # print("MAX", np.max(test[:], axis=0))
    #
    # data_reward = np.expand_dims(data_reward_og[:, 99], axis=2)
    # test = data_reward.reshape(-1, data_reward.shape[2])
    # print("stacked shape", test.shape)
    # print("MIN", np.min(test[:], axis=0))
    # print("MAX", np.max(test[:], axis=0))

    x_train, x_test, y_train, y_test, x_scalar, rew_scalar, generator_std, data_info_train, data_info_test = \
        reward_prediction_tuning(env_dir, 512, 100, 0.85, False, True, num_stds = 1)
    x_train, x_test, y_train, y_test, x_scalar, rew_scalar, generator_std, data_info_train, data_info_test = \
        reward_prediction_tuning(env_dir, 512, 100, 0.85, False, True, num_stds = 0.25)

