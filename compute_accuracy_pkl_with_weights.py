import pickle
import pandas as pd
import subprocess
import os
from statistics import mean

from LogisticRegression import *

TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.21
TRAIN_SPLIT = 1 - TEST_SPLIT - VALIDATION_SPLIT
split_proportions = [TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT]

data_path = "data/clean_data.csv"
save_directory = "best_weights/logistic_regression_weights"
accuracies_file_path = "best_weights/LogisticRegression_accuracies.pkl"

df = pd.read_csv(data_path)
trainers = []
#Load and evaluate
for i, column in enumerate(df.iloc[:, 23: 31].columns) :
    model = LogisticRegression(22,2)
    dataset = HeartDiseaseDataset(data_path, any_disease=False, label_indexes=[23 + i, 24 + i])
        
    trainer = Trainer(model, dataset.class_weights, save_directory, loss='BCElogits', lr=0.05, label_name=column)
    trainer.load_weights(os.path.join(save_directory, f'logistic_regression_{column}.pt'))
    trainers.append(trainer)

accuracies = prediction_analyse_labels(df.iloc[:, 23: 31].columns, trainers, df.iloc[:, 23: 31].columns, data_path, split_proportions, display_confusion=False)

accuracies['mean'] = mean(accuracies.values())

save_accuracies_pkl(accuracies_file_path, accuracies)


data_path = "data/clean_data.csv"
save_directory = "current_accuracies/logistic_regression/logistic_regression_weights"
accuracies_file_path = "current_accuracies/logistic_regression/LogisticRegression_accuracies.pkl"

df = pd.read_csv(data_path)
trainers = []
#Load and evaluate
for i, column in enumerate(df.iloc[:, 23: 31].columns) :
    model = LogisticRegression(22,2)
    dataset = HeartDiseaseDataset(data_path, any_disease=False, label_indexes=[23 + i, 24 + i])
        
    trainer = Trainer(model, dataset.class_weights, save_directory, loss='BCElogits', lr=0.05, label_name=column)
    trainer.load_weights(os.path.join(save_directory, f'logistic_regression_{column}.pt'))
    trainers.append(trainer)

accuracies = prediction_analyse_labels(df.iloc[:, 23: 31].columns, trainers, df.iloc[:, 23: 31].columns, data_path, split_proportions, display_confusion=False)

accuracies['mean'] = mean(accuracies.values())

save_accuracies_pkl(accuracies_file_path, accuracies)

print("Compute accuracy based on weights finish")

subprocess.run(["python", "-m", "pytest", "tests/Test_LogisticRegressionAccuracy.py"])
