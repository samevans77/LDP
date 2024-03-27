from prepare_dataset import *
from utils import *
from model import *
# https://github.com/hnavidan/LDPFeatureSelection
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning


def determine_score_of_current_dataset(features, target, data_size):
    epsilon = 5
    target_privacy_score = 0.9  # or something

    feature_set = collect_unique_to_search(features)  # Determining which of the features are unique.

    estimates = rappor(epsilon, np.ndarray.tolist(features), np.ndarray.tolist(feature_set), data_size)

    privacy_scores = estimates_to_entropy(estimates)  # Getting entropy calculations
    print(privacy_scores)

    overall_privacy = calculate_overall_privacy_score(privacy_scores)  # Overall privacy
    # print(overall_privacy)

    if overall_privacy > target_privacy_score:
        print("Privacy score of: ", overall_privacy, " does exceed target: ", target_privacy_score)
        return estimates, features, targets, privacy_scores, True
    else:
        print("Privacy score of: ", overall_privacy, " does not exceed target: ", target_privacy_score)
        return estimates, features, targets, privacy_scores, False


def generalise_dataset(estimates, privacy_scores, features):
    gain_scores = estimates_to_gain(estimates)  # getting information gain

    tradeoff_scores = calculate_tradeoff_scores(privacy_scores, gain_scores, 0.5, 0.5)  # tradeoffs

    index_to_be_generalised = find_min_index(tradeoff_scores)  # Finding the minimum to be generalised

    generalised_dataset = generalise_feature(features, index_to_be_generalised, 1)  # generalising

    return generalised_dataset, index_to_be_generalised


if __name__ == "__main__":
    data_size = 30000  # Number of datapoints to account for.
    columns_not_to_include = [3, 13, 14, 15, 18, 19, 20]  # Columns which contain nonbinary data

    simplefilter("ignore", category=ConvergenceWarning)  # Disabling warnings for Convergence.

    initial_features, targets = fetch_features_targets_arrays()  # Getting the features.

    features = unique_features(initial_features, targets, columns_not_to_include)  # Discrete unique features.
    generalised_indexes = []

    print("Starting main loop...")
    for i in range(0,20):  # compete max 20 times (currently)
        estimates, features, targets, privacy_scores, complete = determine_score_of_current_dataset(features, targets, data_size)
        if not complete:
            features, index_that_was_generalised = generalise_dataset(estimates, privacy_scores, features)
            generalised_indexes.append(index_that_was_generalised)
        if complete:
            print("Done!")
            features = modify_initial_dataset(initial_features, generalised_indexes, columns_not_to_include, data_size)
            run_models(np.array(features), np.array(targets[:data_size]))

            # ---------- running on the ungeneralised data.
            print("-------------------------------Ungeneralised data-----------------------")
            features = modify_initial_dataset(initial_features, [], columns_not_to_include, data_size)
            run_models(np.array(features), np.array(targets[:data_size]))

            break  # Stop execution once the privacy requirements fulfilled, and models run.

