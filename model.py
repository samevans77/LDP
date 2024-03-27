# Basic AI classification model testing. Alter fetch_features_targets() if you want to change the dataset inputted.
# Should be used to gain classification and performance metrics, compare performance of AI with normal data to
#   performance of AI with modified data.
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def run_models(features, targets):  # https://github.com/alexortner/teaching/blob/master/binary_classification/Top_10_Binary_Classification_BreastCancer.ipynb
    print("Running AI models...")
    x_train, x_test, y_train, y_test = train_test_split(features, targets, random_state=99)

    # Diagnosing problems relating to the dataset splitting
    print(x_test.shape)
    print(x_train.shape)
    # print(y_train.shape)
    # print(y_test.shape)

    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    print(x_test)
    print(x_train)
    print(y_test)
    print(y_train)

    print("Random Forest")
    randomforest(x_train, x_test, y_train, y_test)

    print("Decision Tree")
    decisiontree(x_train, x_test, y_train, y_test)

    print("KNN")
    knn(x_train, x_test, y_train, y_test)


def randomforest(x_train, x_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=64, max_depth=2)
    print("Fitting...")
    rf.fit(x_train, y_train)

    print("Scoring...")
    # print("score on test: " + str(rf.score(x_test, y_test)))
    print("score on train: " + str(rf.score(x_train, y_train)))

    y_pred = rf.predict(x_test)

    # Print accuracy score
    print("Accuracy on test: " + str(accuracy_score(y_test, y_pred)))

    # Print confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Print classification report with additional metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # If you want to extract specific metrics (e.g., sensitivity, true positive count), you can access them from the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    true_positive_count = tp
    true_negative_count = tn

    print("Sensitivity: {:.2f}".format(sensitivity))
    print("Specificity: {:.2f}".format(specificity))
    print("True Positive Count: {}".format(true_positive_count))
    print("True Negative Count: {}".format(true_negative_count))

def decisiontree(x_train, x_test, y_train, y_test):
    dt = DecisionTreeClassifier(min_samples_split=10, max_depth=3)
    print("Fitting...")
    dt.fit(x_train, y_train)

    print("Scoring...")
    # print("score on test: " + str(dt.score(x_test, y_test)))
    print("score on train: " + str(dt.score(x_train, y_train)))

    y_pred = dt.predict(x_test)

    # Print accuracy score
    print("Accuracy on test: " + str(accuracy_score(y_test, y_pred)))

    # Print confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Print classification report with additional metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # If you want to extract specific metrics (e.g., sensitivity, true positive count), you can access them from the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    true_positive_count = tp
    true_negative_count = tn

    print("Sensitivity: {:.2f}".format(sensitivity))
    print("Specificity: {:.2f}".format(specificity))
    print("True Positive Count: {}".format(true_positive_count))
    print("True Negative Count: {}".format(true_negative_count))

def knn(x_train, x_test, y_train, y_test):
    knn = KNeighborsClassifier(algorithm='brute', n_jobs=-1)
    print("Fitting...")
    knn.fit(x_train, y_train)

    print("Scoring...")
    # print("score on test: " + str(knn.score(x_test, y_test)))
    print("score on train: " + str(knn.score(x_train, y_train)))

    y_pred = knn.predict(x_test)

    # Print accuracy score
    print("Accuracy on test: " + str(accuracy_score(y_test, y_pred)))

    # Print confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Print classification report with additional metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # If you want to extract specific metrics (e.g., sensitivity, true positive count), you can access them from the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    true_positive_count = tp
    true_negative_count = tn

    print("Sensitivity: {:.2f}".format(sensitivity))
    print("Specificity: {:.2f}".format(specificity))
    print("True Positive Count: {}".format(true_positive_count))
    print("True Negative Count: {}".format(true_negative_count))