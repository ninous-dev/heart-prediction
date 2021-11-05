import pytest

import numpy as np
from sklearn.model_selection import train_test_split

from DecisionTree import DecisionTree

#Load data
data_path = "data/clean_data.csv"

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

class TestDecisionTreeClass:
    def test_init(self):
        decision_tree = DecisionTree()
        assert decision_tree.min_participant == 2
        assert decision_tree.max_depth == 2

    def test_init_with_values(self):
        decision_tree = DecisionTree(3, 4)
        assert decision_tree.min_participant == 3
        assert decision_tree.max_depth == 4

    def test_split_input_label(self):
        decision_tree = DecisionTree()
        decision_tree.x_col_names = x_col_names
        inputs, labels = decision_tree.split_input_label(np.concatenate((X, Y), axis=1))
        assert inputs.shape == (11627, 22)
        assert labels.shape == (11627, 1)

    def test_split_input_label(self):
        decision_tree = DecisionTree()
        split = decision_tree.get_split_value(np.concatenate((X, Y), axis=1))
        assert 'information_gain' in split
        assert 'threshold' in split
        assert 'left' in split
        assert 'right' in split
        assert 'column_name' in split
        assert 'column_index' in split
        assert 'operator' in split

    def test_split_input_label(self):
        decision_tree = DecisionTree()
        left, right = decision_tree.split(np.concatenate((X, Y), axis=1), 120, 3, False)
        assert len(left) == 3118
        assert len(right) == 8509

    def test_compute_information_gain(self):
        decision_tree = DecisionTree()
        decision_tree.x_col_names = x_col_names
        left, right = decision_tree.split(np.concatenate((X, Y), axis=1), 120, 3, False)
        information_gain = decision_tree.compute_information_gain(np.concatenate((X, Y), axis=1), left, right)
        assert information_gain == 0.018877327134255717

    def test_entropy(self):
        decision_tree = DecisionTree()
        entropy = decision_tree.entropy(Y)
        assert entropy == 0.810219954261622

    def test_compute_leaf_value(self):
        decision_tree = DecisionTree()
        decision_tree.x_col_names = x_col_names
        entropy = decision_tree.compute_leaf_value(np.concatenate((X, Y), axis=1))
        assert entropy == 0.0

    def test_compute_fit(self):
        decision_tree = DecisionTree()
        decision_tree.fit(X, Y, x_col_names)
        assert decision_tree.x_col_names == x_col_names
        assert decision_tree.root != None

    def test_compute_predict(self):
        decision_tree = DecisionTree()
        decision_tree.fit(X, Y, x_col_names)
        predictions = decision_tree.predict(X)
        assert len(predictions) == 11627
        predictions = np.array(predictions)
        assert len(np.unique(predictions)) == 2
        assert 1 in np.unique(predictions)
        assert 0 in np.unique(predictions)

    def test_compute_evaluate(self):
        decision_tree = DecisionTree()
        decision_tree.fit(X, Y, x_col_names)
        prediction = decision_tree.evaluate(X[0], decision_tree.root)
        assert prediction == 1.0

    def test_compute_evaluate_with_list(self, capsys):
        decision_tree = DecisionTree()
        decision_tree.fit(X, Y, x_col_names)
        prediction = decision_tree.evaluate_with_list(X[0], decision_tree.root)
        output = capsys.readouterr()
        print(output.out)
        assert output.out == "PREVMI : 0.0 = 1.0\nLDLC : 191.0 <= 197.0\n\x1b[6;30;42mValue :\x1b[0m 1.0\n"
