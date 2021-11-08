import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix

from .LogisticRegression import LogisticRegression
from .HeartDiseaseDataset import HeartDiseaseDataset
from .Trainer import Trainer
from .HeartDiseaseDataset import HeartDiseaseDataset
from .create_dataloaders import create_dataloaders

def save_accuracies_pkl(pkl_filepath, accuracies):
    accuracies_file = open(pkl_filepath, "wb")
    pickle.dump(accuracies, accuracies_file)
    accuracies_file.close()

def prediction_analyse_labels(columns_names, trainers, all_labels, data_path, split_proportions, batch_size=1, display_confusion=True):
    accuracies = {}

    for i, column in enumerate(all_labels):

        if not column in columns_names:
            continue

        for trainer in trainers:
            if column == trainer.label_name:
                dataset = HeartDiseaseDataset(data_path, any_disease=False, label_indexes=[23 + i, 24 + i])
                _, _, test_dataloader = create_dataloaders(dataset, split_proportions, batch_size=batch_size)

                accuracies[column] = prediction_analyse(test_dataloader, trainer, display_confusion=display_confusion)
                print("\n\n")
                break
    return accuracies

def prediction_analyse(dataloader, trainer, display_confusion=True):
    accuracy = trainer.evaluate(dataloader, display=False)
    print("accuracy", accuracy)

    all_predictions = np.array([])
    all_labels = np.array([])

    for i, (inputs, labels) in enumerate(dataloader):
        pred = trainer.model(inputs)
        pred = np.where(pred[0].detach().numpy() < 0., 0, 1)
        pred = np.argmax(pred)

        label = np.argmax(labels[0].detach().numpy())

        all_predictions = np.append(all_predictions, pred)
        all_labels = np.append(all_labels, label)

    cm = confusion_matrix(all_labels, all_predictions)


    if display_confusion:
        display = ConfusionMatrixDisplay(confusion_matrix=cm)
        display.plot()
        plt.show()

    TP = cm[1,1] # true positive 
    TN = cm[0,0] # true negatives
    FP = cm[0,1] # false positives
    FN = cm[1,0] # false negatif

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    #positive predictive value
    PPV = TP / (TP + FP)
    #negative predictive value
    NPV = TN / (TN + FN)


    PPV = TP / (TP + FP) if (TP + FP) != 0 else 'no positives values'
    NPV = TN / (TN + FN) if (TN + FN) != 0 else 'no negatives values'

    print(f'sensitivity : {sensitivity}, specificity : {specificity}, PPV : {PPV}, NPV : {NPV}')
    return accuracy

