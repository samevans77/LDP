from pandas import DataFrame as df
from ucimlrepo import fetch_ucirepo  # see https://www.archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
# see also https://www.cdc.gov/brfss/annual_data/annual_2014.html
# https://stackoverflow.com/questions/35076223/how-to-randomly-shuffle-data-and-target-in-python
from utils import *
import numpy as np


def unison_shuffle_dataset(a, b):  # https://stackoverflow.com/questions/35076223/how-to-randomly-shuffle-data-and-target-in-python
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def fetch_features_targets_arrays():  # Returns the dataset as a 2D array. Split into features and target
    print("Fetching Dataset... ", end="")
    diabetes = fetch_ucirepo(id=891)
    features = df.to_numpy(diabetes.data.features)
    targets = df.to_numpy(diabetes.data.targets)
    new_features = []
    new_targets = []

    for i in range(0, len(features)):
        if targets[i] == 1:
            new_features.append(features[i])
            new_targets.append(targets[i])

    for j in range(0, np.count_nonzero(np.array(new_targets))):
        if targets[j] == 0:
            new_features.append(features[j])
            new_targets.append(targets[j])
    print("Done!")
    return unison_shuffle_dataset(np.array(new_features), np.array(new_targets))


def unique_features(features, targets, columns_not_to_include):  # Assumes rectangular features
    print("Creating Unique Features... ", end="")
    new_features = np.array([])
    for i in range(0, len(features[0])):  # For each feature (array)
        if i not in columns_not_to_include:
            column = fetch_column(features, i)
            # print(column)
            new_column = []
            for j in range(0, len(column)):  # For each value in the feature (array)
                if targets[j] == 0:
                    new_column.append(column[j] + (i+1)*10)  # Effectively, make the data distinct for each column by
                    # adding a 10 * column count to the value, this makes it unique and discrete. practically would
                    # have to be a LONG random number instead. Makes sure there is no zero value.
                elif targets[j] == 1:
                    new_column.append(column[j] + ((i+1)*10)+5)  # Ensuring uniqueness for both sets of likelihoods.
            new_column = np.asarray(new_column)  # for performance reasons, keeping numpy away until the last second.
            if i == 0:  # first iteration
                new_features = np.vstack(new_column)  # Set it to a vertical array, being vertical features
            else:
                new_features = np.column_stack((new_features, new_column))  # Vertical concatenation
    print("Done!")
    return new_features
    # 100, 101, 150, 151 for: Class A - feature 0, Class A - feature 1. Class B - feature 0, Class B - feature 1


def binary_targets(targets):
    unique = np.unique(targets)
    return_arr = []
    for target in targets:
        return_arr.append(np.where(target == unique)[0][0])
    return return_arr


def collect_unique_to_search(features):  # Assumes rectangular features
    list_of_unique_values = np.unique(features)
    return list_of_unique_values
