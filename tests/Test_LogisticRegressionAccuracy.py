import pytest

import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split

#Load data
data_path = "data/clean_data.csv"
accuracies_file_path = "tests/tests_pkl/logistic_regression/LogisticRegression_accuracies.pkl"
actual_accuracies_file_path = "current_accuracies/logistic_regression/LogisticRegression_accuracies.pkl"

data = np.loadtxt(data_path, delimiter=",",dtype=float, skiprows=1)
col_names = np.genfromtxt(data_path , delimiter=',', names=True, dtype=float).dtype.names[1:31]

#Split inputs and labels
x_col_names = col_names[0:22]
y_col_names = col_names[22:30]

X = data[:, 1:23]
Y = data[:, 29:30]

#Split into test and train dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.4, random_state=42)

#Define the maximum depth
DEPTH = 5

class TestLogisticRegressionClass:

    def test_compare_accuracies(self):
        accuracies_file = open(accuracies_file_path, "rb")
        new_values = pickle.load(accuracies_file)
        accuracies_file.close()

        accuracies_file = open(actual_accuracies_file_path, "rb")
        actual_values = pickle.load(accuracies_file)
        accuracies_file.close()

        keys = list(actual_values)

        lower_accuracies = []
        no_lower = True
        upper = False

        print(actual_values, new_values)
        for key in keys:
            is_upper = actual_values[key] <= new_values[key]
            comparaison = actual_values[key] < new_values[key]

            if not is_upper:
                no_lower = False
            if not comparaison:
                upper = True

            comparaison_str = str(actual_values[key]) + " <= " + str(new_values[key])
            lower_accuracies.append(key + " : " + str(is_upper) + ' ' + comparaison_str)

        assert no_lower, "key comparaison (actual vs new):\n{}".format("\n".join(lower_accuracies))
        assert upper, "Better accuracy please save weights and save accuracies before the commit.\nkey comparaison (actual vs new):\n{}".format("\n".join(lower_accuracies))

