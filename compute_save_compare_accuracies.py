import pickle
import pandas as pd
import subprocess
import os
from statistics import mean

from sklearn.model_selection import train_test_split

from LogisticRegression import *
from DecisionTree import DecisionTree

TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.21
TRAIN_SPLIT = 1 - TEST_SPLIT - VALIDATION_SPLIT

SPLIT_PROPORTIONS = [TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT]


def compute_and_save_accuracies_logisticRegression(data_path, save_directory, accuracies_file_path, loss='BCElogits'):
    df = pd.read_csv(data_path)
    trainers = []
    all_labels_name = df.iloc[:, 23: 31].columns

    #Load and evaluate each trainer
    for i, column in enumerate(all_labels_name) :
        model = LogisticRegression(22,2)
        dataset = HeartDiseaseDataset(data_path, any_disease=False, label_indexes=[23 + i, 24 + i])

        trainer = Trainer(model, dataset.class_weights, save_directory, loss=loss, label_name=column)
        trainer.load_weights(os.path.join(save_directory, f'logistic_regression_{column}.pt'))
        trainers.append(trainer)

    #Compute accuracies
    accuracies = prediction_analyse_labels(all_labels_name, trainers, all_labels_name, data_path, SPLIT_PROPORTIONS, display_confusion=False)

    accuracies['mean'] = mean(accuracies.values())

    save_accuracies_pkl(accuracies_file_path, accuracies)


#Calculation of current recorded weights
data_path = "data/clean_data.csv"
save_directory = "best_weights/logistic_regression_weights"
accuracies_file_path = "best_weights/LogisticRegression_accuracies.pkl"

compute_and_save_accuracies_logisticRegression(data_path, save_directory, accuracies_file_path)


#Calculation of current recorded weights
data_path = "data/clean_data.csv"
save_directory = "current_accuracies/logistic_regression/logistic_regression_weights"
accuracies_file_path = "current_accuracies/logistic_regression/LogisticRegression_accuracies.pkl"

compute_and_save_accuracies_logisticRegression(data_path, save_directory, accuracies_file_path)

print("Compute accuracy based on weights finish")

#Launch of the testsuite
def compute_accuracies_with_saves(data_path, save_directory):
    accuracies = {}
    data = np.loadtxt(data_path, delimiter=",",dtype=float, skiprows=1)
    col_names = np.genfromtxt(data_path , delimiter=',', names=True, dtype=float).dtype.names[1:31]

    #Split inputs and labels
    y_col_names = col_names[22:30]

    X = data[:, 1:23]
    Y = data[:, 29:30]

    files = next(os.walk(save_directory), (None, None, []))[2]

    for i, column in enumerate(y_col_names):
        print(f'==== Evaluate {column} ====')
        file_index = i
        i = i + 23
        Y = data[:, i:(i+1)]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.4, random_state=42)
        decision_tree = DecisionTree()
        decision_tree.import_tree(os.path.join(save_directory, files[file_index]))
        #decision_tree.pretty_print()

        accuracy = decision_tree.prediction_analyse(X_test, Y_test, False, False)
        accuracies[column] = accuracy

    return accuracies


data_path = "data/clean_data.csv"
save_directory = "current_accuracies/decision_tree/decision_tree_saves"
accuracies_file_path = "current_accuracies/decision_tree/decisionTree_accuracies.pkl"

accuracies = compute_accuracies_with_saves(data_path, save_directory)
accuracies['mean'] = mean(accuracies.values())

save_accuracies_pkl(accuracies_file_path, accuracies)

data_path = "data/clean_data.csv"
save_directory = "best_weights/decision_tree_saves"
accuracies_file_path = "best_weights/decisionTree_accuracies.pkl"

accuracies = compute_accuracies_with_saves(data_path, save_directory)
accuracies['mean'] = mean(accuracies.values())

save_accuracies_pkl(accuracies_file_path, accuracies)

print("Compute accuracy based on weights finish -- DecisionTree")

subprocess.run(["python", "-m", "pytest", "tests/Test_LogisticRegressionAccuracy.py"])

subprocess.run(["python", "-m", "pytest", "tests/Test_DecisionTreeAccuracy.py"])
