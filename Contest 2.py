import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from tqdm import tqdm

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

# 1. Removing fake results from test dataframe
test.drop(["ID_code"], axis=1, inplace=True)
test = test.values

unique_count = np.zeros_like(test)  # same col & row size
for i in tqdm(range(test.shape[1])):
    _, index_, count_ = np.unique(test[:, i], return_counts=True, return_index=True)
    unique_count[index_[count_ == 1], i] += 1

real_idx = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
synth_idx = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]

test=test[real_idx,:]

# 2. Append real test to training
train.drop(["ID_code"], axis=1, inplace=True)
train.drop(["target"], axis=1, inplace=True)
train = train.values
print(train.shape)
full = np.concatenate([train, test])
print(full.shape)
