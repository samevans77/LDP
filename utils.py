from entropy_calc import *
from pure_ldp.frequency_oracles import *  # see https://github.com/Samuel-Maddock/pure-LDP
from prepare_dataset import *
import math


def fetch_column(data, i):  # Fetches the i-th column
    return [row[i] for row in data]


def calculate_overall_privacy_score(score_array):  # Accepts a flattened array as an input, finds the average of values
    return np.average(score_array)


def calculate_tradeoff_scores(privacy_scores, utility_scores, weight_p, weight_u):  # Returns an array of tradeoff
    # scores for each feature in the dataset, where weight_u + weight_p = 1., Assume w_u = w_p = 0.5
    return [weight_p * p + weight_u * u for p, u in zip(privacy_scores, utility_scores)]


def generalise_feature(dataset, feature_index, set_value_to):  # Takes a feature index and turns all values of that
    # feature to set_value_to.
    print("Generalising feature ", feature_index, "... ", end="")
    for i in range(0, len(dataset)):
        dataset[i][feature_index] = set_value_to
    print("Done!")
    return dataset


def estimates_to_entropy(estimates):
    print("Calculating entropy scores... ", end="")
    output = []
    for estimate in estimates:
        value1 = estimate[0]  # Class A, Feature 0
        value2 = estimate[1]
        value3 = estimate[2]
        value4 = estimate[3]
        if value1 != 0 and value2 == 0 and value3 == 0 and value4 == 0:
            print("Found a generalised feature!")
            output.append(1)  # This should be set to the max entropy of the dataset.
        elif value4 is not None:
            outa = entropy_count(np.array([[value1+value3], [value2+value4]]), sum([value1, value2, value3, value4]))
            output.append(outa)
    print("Done!")
    return output


def information_gain(table):  # Accepts a "table"
    total = sum(sum(table))  # Calculates the overall sum of the tables (the sum of the sum of arrays)
    p_classes = sum(table)
    p_features = np.sum(table, axis=1)
    H_c = 0
    H_ca = 0
    for j in range(table.shape[1]):
        prob = p_classes[j] / total
        H_c = H_c - (prob * np.log2(prob))

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            prob = table[i, j] / p_features[i]
            H_ca = H_ca - (p_features[i] / total * prob * np.log2(prob))

    return H_c - H_ca


def calc_information_gain(counts):  # Accepts a table of tables of counts
    info_gain = np.zeros((len(counts)))  # Creates an output array of length of the input array for the output
    for i in range(len(counts)):  # For each of the input array points
        info_gain[i] = information_gain(counts[i])  # Complete an information gain calculation
    return info_gain  # Returns the output array.


def estimates_to_gain(estimates):
    print("Calculating gain scores... ", end="")
    output = []
    for estimate in estimates:
        value1 = estimate[0]  # Class A, Feature 0
        value2 = estimate[1]
        value3 = estimate[2]
        value4 = estimate[3]
        if value1 != 0 and value2 == 0 and value3 == 0 and value4 == 0:
            print("Found a generalised feature!")
            output.append(1)  # This should be set to the max entropy of the dataset.
        elif value4 is not None:
            gain = calc_information_gain(np.array([[[value1, value2], [value3, value4]]]))
            output.extend(gain)
    print("Done!")
    return output


def find_min_index(tradeoff_scores):  # Finds the index of the minimum value in the array
    array = np.array(tradeoff_scores)
    min_value = np.min(array)
    return np.where(array == min_value)[0]


def rappor(epsilon, data, search_features, data_size):
    # It seems this code can handle a maximum of 48 (tested successfully) features to search for. Beyond that it gives
    # garbage outputs.
    print("Rappor starting... ")
    f = round(1 / (0.5 * math.exp(epsilon / 2) + 0.5), 2)

    final_estimates = []

    # Modify the data so that it's processable by putting it all into one big array. Not necessary to maintain
    # structure of the data since it will all be processed for frequency estimation anyway.
    data = data[:data_size]

    # We will take the data one feature at a time. Starting with the first column and moving along like that.
    for i in range(0, len(data[0])):  # For each feature (assuming rectangular data)
        column = fetch_column(data, i)
        new_data = []
        for datapoint in column:
            new_data.append(datapoint)  # Combining the data together
        search_features = np.ndarray.tolist(np.unique(np.array(new_data)))

        server_rappor = RAPPORServer(f, 1024, 64, max(search_features)+1)  # D being the highest number to check for
        # Must be 1000 or lower to not throw an error. +1 because that's what rappor requires (max searched plus 1).
        client_rappor = RAPPORClient(f, 1024, server_rappor.get_hash_funcs())

        private_record = list(map(client_rappor.privatise, new_data))
        server_rappor.aggregate_all(private_record)
        estimates = server_rappor.estimate_all(search_features, suppress_warnings=True)
        # List of values to estimate the frequency of in the data. Should be one lower than d, and values
        # should be less than 1000.
        if len(estimates) < 4:
            for j in range(len(estimates), 4):
                estimates = np.append(estimates, [0])
        final_estimates.append(estimates)
        print("Round", i, "Done - Estimates:", estimates)
    return final_estimates


def modify_initial_dataset(features, generalise_array_indexes, columns_not_include, size):
    print("Generalising initial (non-unique) dataset... ", end="")
    for rem_index in sorted(columns_not_include, reverse=True):  # Remove columns largest to smallest, preserve index
        features = remove_column(features, rem_index)
    if len(generalise_array_indexes) > 0:
        for gen_index in generalise_array_indexes[0]:  # Generalise all values that have been found to be required.
            features = generalise_feature(features, gen_index, 1)
    print("Done!")
    return features[:size]  # return only first <size> values.


def remove_column(matrix, column_index):
    arr = np.array(matrix)
    arr = np.delete(arr, column_index, 1)
    return np.ndarray.tolist(arr)