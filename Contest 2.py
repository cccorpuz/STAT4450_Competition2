import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import gc
import scipy.ndimage

# train = pd.read_csv("./train.csv")
# test = pd.read_csv("./test.csv")

train = pd.read_csv("./train.csv/train.csv")
test = pd.read_csv("./test.csv/test.csv")

# 1. Removing fake results from test dataframe
test_codes = test["ID_code"].values
test.drop(["ID_code"], axis=1, inplace=True)
test = test.values

unique_count = np.zeros_like(test)  # same col & row size
for i in tqdm(range(test.shape[1])):
    _, index_, count_ = np.unique(test[:, i], return_counts=True, return_index=True)
    unique_count[index_[count_ == 1], i] += 1

real_idx = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
synth_idx = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]

test_length = test.shape[0]
test_synth = test[synth_idx, :]
test = test[real_idx, :]

# 2. Append real test to training
target = np.array(list(train["target"].values))
train_length = train.shape[0]
train.drop(["ID_code"], axis=1, inplace=True)
train.drop(["target"], axis=1, inplace=True)
full = pd.DataFrame(np.concatenate([train.values, test]), columns=train.columns)
synth = pd.DataFrame(test_synth, columns=train.columns)
print(full.shape)

# Reverse features with negative correlation
features = [x for x in full.columns if x.startswith("var")]
for var in features:
    if np.corrcoef(target, train[var])[1][0] < 0:
        full[var] = full[var] * -1

# Getting Counts, Density, Deviation
sigma_fac = 0.001
sigma_base = 4
eps = 0.00000001


def get_features(data):
    # Get counts
    features_count = np.zeros((data.shape[0], len(features)))
    features_density = np.zeros((data.shape[0], len(features)))
    features_deviation = np.zeros((data.shape[0], len(features)))

    sigmas = []
    for i, var in enumerate(tqdm(features)):
        data_int = (data[var].values * 10000).round().astype(int)
        low = data_int.min()
        data_int -= low
        high = data_int.max() + 1
        counts_data = np.bincount(data_int, minlength=high).astype(float)

        # Geometric mean of twice sigma_base and a sigma_scaled which is scaled to the length of array
        sigma_scaled = counts_data.shape[0] * sigma_fac
        sigma = np.power(sigma_base * sigma_base * sigma_scaled, 1 / 3)
        sigmas.append(sigma)
        counts_data_smooth = scipy.ndimage.filters.gaussian_filter1d(counts_data, sigma)
        deviation = counts_data / (counts_data_smooth + eps)
        indices = data_int
        features_count[:, i] = counts_data[indices]
        features_density[:, i] = counts_data_smooth[indices]
        features_deviation[:, i] = deviation[indices]

    features_count_names = [var + "_count" for var in features]
    features_density_names = [var + "_density" for var in features]
    features_deviation_names = [var + "_deviation" for var in features]

    data_count = pd.DataFrame(columns=features_count_names, data=features_count)
    data_count.index = data.index
    data_density = pd.DataFrame(columns=features_density_names, data=features_density)
    data_density.index = data.index
    data_deviation = pd.DataFrame(
        columns=features_deviation_names, data=features_deviation
    )
    data_deviation.index = data.index
    data = pd.concat([data, data_count, data_density, data_deviation], axis=1)

    features_count = features_count_names
    features_density = features_density_names
    features_deviation = features_deviation_names

    return data, features_count, features_density, features_deviation


full, features_count, features_density, features_deviation = get_features(full)
(
    synth,
    fake_features_count,
    fake_features_density,
    fake_features_deviation,
) = get_features(synth)
print(full.shape)
print(f"test_synth: {synth.shape}")

# Standardizing the features
features_to_scale = [features, features_count]

scaler = StandardScaler()
features_to_scale_flatten = [var for sublist in features_to_scale for var in sublist]
scaler.fit(full[features_to_scale_flatten])
features_scaled = scaler.transform(full[features_to_scale_flatten])
full[features_to_scale_flatten] = features_scaled
print(full.shape)

# Split back into train and test
train = full.iloc[:train_length, :]
test = full.iloc[train_length:, :]
del full

gc.collect()
print(train.shape, test.shape)

# Prediction with LightGBM model
# First check accuracy with 70/30 split
x_train, x_val, y_train, y_val = train_test_split(
    train, target, test_size=0.3, random_state=123
)

train_data = lgb.Dataset(x_train, label=y_train)
val_data = lgb.Dataset(x_val, label=y_val)

# params = {
#     "boost_from_average": "false",
#     "boost": "gbdt",
#     "feature_fraction": 1,
#     "learning_rate": 0.08,
#     "max_depth": -1,
#     "metric": "binary_logloss",
#     "num_leaves": 4,
#     "num_threads": 8,
#     "tree_learner": "serial",
#     "objective": "binary",
#     "reg_alpha": 2,
#     "reg_lambda": 0,
#     "verbosity": 1,
#     "max_bin": 256,
# }

params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 31,
    "learning_rate": 0.05,
}

# train the model
num_rounds = 100
model = lgb.train(params, train_data, num_rounds, valid_sets=[val_data])

# make predictions on the validation data
# y_pred = model.predict(x_val)

# print the accuracy
# accuracy = accuracy_score(y_val, y_pred.round().astype(int))
# print(accuracy)

# Make final prediction for test
sub = pd.DataFrame({"ID_code": test_codes})

pred_real = model.predict(test)
pred_synth = model.predict(synth)

preds_all = np.zeros(test_length)
preds_all[real_idx] = pred_real
preds_all[synth_idx] = pred_synth
sub["target"] = preds_all
sub.to_csv("submission.csv", index=False)
print(sub.head(20))
