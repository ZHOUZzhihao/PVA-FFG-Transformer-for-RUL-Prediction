"""
This code is used for data processing, including data extraction, data set partitioning, normalization and exponential smoothing
Author: Zhou Zhihao
Date: 11/4/2024
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def data_processing(data_name="FD001", smooth_param=1, exclude=False):
    '''
    用于根据给定FDOO序号生成数据集
    '''
    dir_path = './CMAPSSData/'

    data_name1 = "FD001"
    data_name2 = "FD002"
    data_name3 = "FD003"
    data_name4 = "FD004"

    # define column names for easy indexing
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    operating_name = ['op_cond']
    col_names = index_names + setting_names + sensor_names + operating_name

    # read data
    train1 = pd.read_csv((dir_path + 'train_' + data_name1 + '.txt'), sep='\s+', header=None, names=col_names)
    test1 = pd.read_csv((dir_path + 'test_' + data_name1 + '.txt'), sep='\s+', header=None, names=col_names)
    y_test1 = pd.read_csv((dir_path + 'RUL_' + data_name1 + '.txt'), sep='\s+', header=None, names=['RUL'])

    train2 = pd.read_csv((dir_path + 'train_' + data_name2 + '.txt'), sep='\s+', header=None, names=col_names)
    test2 = pd.read_csv((dir_path + 'test_' + data_name2 + '.txt'), sep='\s+', header=None, names=col_names)
    y_test2 = pd.read_csv((dir_path + 'RUL_' + data_name2 + '.txt'), sep='\s+', header=None, names=['RUL'])

    train3 = pd.read_csv((dir_path + 'train_' + data_name3 + '.txt'), sep='\s+', header=None, names=col_names)
    test3 = pd.read_csv((dir_path + 'test_' + data_name3 + '.txt'), sep='\s+', header=None, names=col_names)
    y_test3 = pd.read_csv((dir_path + 'RUL_' + data_name3 + '.txt'), sep='\s+', header=None, names=['RUL'])

    train4 = pd.read_csv((dir_path + 'train_' + data_name4 + '.txt'), sep='\s+', header=None, names=col_names)
    test4 = pd.read_csv((dir_path + 'test_' + data_name4 + '.txt'), sep='\s+', header=None, names=col_names)
    y_test4 = pd.read_csv((dir_path + 'RUL_' + data_name4 + '.txt'), sep='\s+', header=None, names=['RUL'])

    print(data_name)
    if data_name == "pretrain_all":
        train = pd.concat([train1, train2, train3, train4])
        test = pd.concat([test1, test2, test3, test4])
        y_test = pd.concat([y_test1, y_test2, y_test3, y_test4])
    elif data_name == "train_other3":
        train = pd.concat([train2, train3, train4])
        test = pd.concat([test2, test3, test4])
        y_test = pd.concat([y_test2, y_test3, y_test4])
    elif data_name == "FD001":
        train = train1
        test = test1
        y_test = y_test1
    elif data_name == "FD002":
        train = train2
        test = test2
        y_test = y_test2
    elif data_name == "FD003":
        train = train3
        test = test3
        y_test = y_test3
    elif data_name == "FD004":
        train = train4
        test = test4
        y_test = y_test4
    elif data_name == "FD001andFD002":
        train = pd.concat([train2, train1])
        test = pd.concat([test2, test1])
        y_test = pd.concat([y_test2, y_test1])
    elif data_name == "FD002andFD003":
        train = pd.concat([train2, train3])
        test = pd.concat([test2, test3])
        y_test = pd.concat([y_test2, y_test3])
    elif data_name == "FD002andFD004":
        train = pd.concat([train2, train4])
        test = pd.concat([test2, test4])
        y_test = pd.concat([y_test2, y_test4])
    elif data_name == "FD003andFD004":
        train = pd.concat([train4, train3])
        test = pd.concat([test4, test3])
        y_test = pd.concat([y_test4, y_test3])
    # drop non-informative features in training set
    # sensors = ['s_3', 's_4', 's_7', 's_11', 's_12']
    sensors = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
    # drop_sensors = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']#恒定数据集

    alpha = smooth_param

    train = add_remaining_useful_life(train)
    train['RUL'].clip(upper=125, inplace=True)
    train['RUL'] = train['RUL'] / 125
    # remove unused sensors
    drop_sensors = [element for element in sensor_names if element not in sensors]

    # scale with respect to the operating condition
    X_train_pre = add_operating_condition(train.drop(drop_sensors, axis=1))
    X_test_pre = add_operating_condition(test.drop(drop_sensors, axis=1))

    ori_X_train_pre = X_train_pre

    X_train_pre, X_test_pre = condition_scaler(X_train_pre, X_test_pre, sensors)

    # exponential smoothing
    X_train_pre = exponential_smoothing(X_train_pre, sensors, 0, alpha)
    X_test_pre = exponential_smoothing(X_test_pre, sensors, 0, alpha)

    X_train_pre.drop(labels=setting_names + operating_name, axis=1, inplace=True)
    X_test_pre.drop(labels=setting_names + operating_name, axis=1, inplace=True)

    group = X_train_pre.groupby(by="unit_nr")
    group_test = X_test_pre.groupby(by="unit_nr")
    Xtest = X_test_pre.groupby(by="unit_nr")

    if exclude == False:
        return group, y_test, group_test, Xtest
    else:

        ori_X_train_pre.drop(labels=setting_names + ["op_cond", "RUL"], axis=1, inplace=True)

        # separate title information and sensor data

        train_data = ori_X_train_pre.iloc[:, 2:]
        list_train_labels = list(train_data.columns.values)

        # scaler = StandardScaler()
        # scaler = MinMaxScaler()
        scaler = MinMaxScaler(feature_range=(0, 1))
        # min-max normalization of the sensor data
        # data_norm = (data - data.min()) / (data.max() - data.min())

        scaler.fit(train_data[list_train_labels])

        return group, y_test, group_test, Xtest, scaler


class SequenceDataset(Dataset):
    def __init__(self, mode='pretrain', group=None, y_label=None, sequence_train=50, sequence_test=20, patch_size=10):
        self.mode = mode
        X_ = []
        y_ = []
        time_stamp = []

        if mode == 'test_all':
            self.unit_nr_total = len(group["unit_nr"].value_counts())
            y_label = y_label["RUL"].to_numpy()
            i = 1
            while i <= self.unit_nr_total:
                self.x = group.get_group(i).to_numpy()


                # split by sequences
                length_cur_unit_nr = len(self.x)
                for j in range(patch_size, length_cur_unit_nr+1):

                    X = self.x[j - patch_size:j, 2:]


                    X = X.astype(float)
                    X_.append(X)

                    y = length_cur_unit_nr-j+y_label[i-1]-1
                    # print(y_label[i])

                    if y >= 125:
                        y_.append(125)
                    else:
                        y_.append(y)

                    time_stamp.append(j)
                i += 1

            self.y = torch.tensor(y_).float()
            self.X = torch.tensor(X_).float()
            self.toggle = 0

            self.time_stamp = torch.tensor(time_stamp)


        elif mode == 'test':
            self.unit_nr_total = len(group["unit_nr"].value_counts())
            y_label = y_label["RUL"].to_numpy()
            i = 1
            while i <= self.unit_nr_total:
                self.x = group.get_group(i).to_numpy()

                # split by sequences
                length_cur_unit_nr = len(self.x)

                if length_cur_unit_nr < patch_size:
                    data = np.zeros((patch_size, self.x.shape[1]))
                    for j in range(data.shape[1]):
                        x_old = np.linspace(0, len(self.x) - 1, len(self.x), dtype=np.float64)
                        params = np.polyfit(x_old, self.x[:, j].flatten(), deg=1)
                        k = params[0]
                        b = params[1]
                        x_new = np.linspace(0, patch_size - 1, patch_size, dtype=np.float64)
                        data[:, j] = (x_new * len(self.x) / patch_size * k + b)
                else:
                    data = self.x
                X = data[-patch_size:, 2:]
                X_.append(X)
                y_cur = y_label[i - 1]
                if y_cur >= 125:
                    y_.append(125)
                else:
                    y_.append(y_cur)

                i += 1

            self.y = torch.tensor(y_).float()
            self.X = torch.tensor(X_).float()
            self.toggle = 0


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):

        return self.X[i], self.y[i]

def add_remaining_useful_life(df):
    # Get the total number of cycles for each unit
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].size()

    # Merge the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)

    # Calculate remaining useful life for each row (piece-wise Linear)
    remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]

    result_frame["RUL"] = remaining_useful_life
    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)

    return result_frame


def add_operating_condition(df):
    df_op_cond = df.copy()

    df_op_cond['setting_1'] = abs(df_op_cond['setting_1'].round())
    df_op_cond['setting_2'] = abs(df_op_cond['setting_2'].round(decimals=2))

    # converting settings to string and concatanating makes the operating condition into a categorical variable
    df_op_cond['op_cond'] = df_op_cond['setting_1'].astype(str) + '_' + \
                            df_op_cond['setting_2'].astype(str) + '_' + \
                            df_op_cond['setting_3'].astype(str)

    return df_op_cond


def condition_scaler(df_train, df_test, sensor_names):
    # apply operating condition specific scaling
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    print(df_train['op_cond'].unique())
    for condition in df_train['op_cond'].unique():
        scaler.fit(df_train.loc[df_train['op_cond'] == condition, sensor_names])
        df_train.loc[df_train['op_cond'] == condition, sensor_names] = scaler.transform(
            df_train.loc[df_train['op_cond'] == condition, sensor_names])
        df_test.loc[df_test['op_cond'] == condition, sensor_names] = scaler.transform(
            df_test.loc[df_test['op_cond'] == condition, sensor_names])
    return df_train, df_test


def exponential_smoothing(df, sensors, n_samples, alpha):
    df = df.copy()
    # first, take the exponential weighted mean
    df[sensors] = df.groupby('unit_nr')[sensors].apply(lambda x: x.ewm(alpha=1-alpha).mean()).reset_index(level=0,
                                                                                                        drop=True)
    # second, drop first n_samples of each unit_nr to reduce filter delay
    def create_mask(data, samples):
        result = np.ones_like(data)
        result[0:samples] = 0
        return result

    mask = df.groupby('unit_nr')['unit_nr'].transform(create_mask, samples=n_samples).astype(bool)
    df = df[mask]

    return df


def add_remaining_useful_life_test(df, y_test):
    # Get the total number of cycles for each unit
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()

    y_test.index = y_test.index + 1
    # Merge the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)

    result_frame = result_frame.merge(y_test, left_on='unit_nr', right_index=True)
    # Calculate remaining useful life for each row (piece-wise Linear)

    remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"] + result_frame["RUL"]
    result_frame["RUL"] = remaining_useful_life

    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)

    return result_frame

def add_remaining_useful_life(df):
    # Get the total number of cycles for each unit
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()

    # Merge the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)

    # Calculate remaining useful life for each row (piece-wise Linear)
    remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]
    result_frame["RUL"] = remaining_useful_life

    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)

    return result_frame
