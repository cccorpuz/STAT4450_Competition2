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
test.drop(["ID_code"], axis=1, inplace=True)
test = test.values

unique_count = np.zeros_like(test)  # same col & row size
for i in tqdm(range(test.shape[1])):
    _, index_, count_ = np.unique(test[:, i], return_counts=True, return_index=True)
    unique_count[index_[count_ == 1], i] += 1

real_idx = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
synth_idx = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]

test = test[real_idx, :]

# 2. Append real test to training
target = np.array(list(train["target"].values))
train_length = train.shape[0]
train.drop(["ID_code"], axis=1, inplace=True)
train.drop(["target"], axis=1, inplace=True)
full = pd.DataFrame(np.concatenate([train.values, test]), columns=train.columns)
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

# Get counts
features_count = np.zeros((full.shape[0], len(features)))
features_density = np.zeros((full.shape[0], len(features)))
features_deviation = np.zeros((full.shape[0], len(features)))

sigmas = []
for i, var in enumerate(tqdm(features)):
    full_int = (full[var].values * 10000).round().astype(int)
    low = full_int.min()
    full_int -= low
    high = full_int.max() + 1
    counts_full = np.bincount(full_int, minlength=high).astype(float)

    # Geometric mean of twice sigma_base and a sigma_scaled which is scaled to the length of array
    sigma_scaled = counts_full.shape[0] * sigma_fac
    sigma = np.power(sigma_base * sigma_base * sigma_scaled, 1 / 3)
    sigmas.append(sigma)
    counts_full_smooth = scipy.ndimage.filters.gaussian_filter1d(counts_full, sigma)
    deviation = counts_full / (counts_full_smooth + eps)
    indices = full_int
    features_count[:, i] = counts_full[indices]
    features_density[:, i] = counts_full_smooth[indices]
    features_deviation[:, i] = deviation[indices]

features_count_names = [var + "_count" for var in features]
features_density_names = [var + "_density" for var in features]
features_deviation_names = [var + "_deviation" for var in features]

full_count = pd.DataFrame(columns=features_count_names, data=features_count)
full_count.index = full.index
full_density = pd.DataFrame(columns=features_density_names, data=features_density)
full_density.index = full.index
full_deviation = pd.DataFrame(columns=features_deviation_names, data=features_deviation)
full_deviation.index = full.index
full = pd.concat([full, full_count, full_density, full_deviation], axis=1)

features_count = features_count_names
features_density = features_density_names
features_deviation = features_deviation_names

print(full.shape)

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
y_pred = model.predict(x_val)

# print the accuracy
accuracy = accuracy_score(y_val, y_pred.round().astype(int))
print(accuracy)
