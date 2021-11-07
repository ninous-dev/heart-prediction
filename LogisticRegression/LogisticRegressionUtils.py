import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from .LogisticRegression import LogisticRegression
from .Trainer import Trainer

class HeartDiseaseDataset(Dataset): 
    def __init__(self, path, any_disease=False, label_indexes=[30, 31], split_indexes=[1, 23, 23, 31]):
        self.data = np.loadtxt(path, delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(self.data[:, split_indexes[0]:split_indexes[1]])

        if any_disease:
            self.y = torch.from_numpy(np.amax(self.data[:, split_indexes[2]:split_indexes[3]], axis=1))
        else:
            self.y = torch.from_numpy(pd.get_dummies(self.data[:, label_indexes[0]:label_indexes[1]].flatten()).to_numpy(dtype=np.float32))

        self.minMax = torch.from_numpy(compute_min_max(path, split_indexes[0], split_indexes[1]))
        self.len = len(self.data)
        self.class_weights = compute_weight_class(self.y)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        inputs = torch.sub(self.x[idx], self.minMax[0])/ torch.sub(self.minMax[1],self.minMax[0])
        return inputs, self.y[idx]

def create_dataloaders(dataset, split_proportions, batch_size, display_informations=False):
    lengths = [round(len(dataset) * split) for split in split_proportions]

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, lengths=lengths, generator=torch.Generator().manual_seed(42))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        prefetch_factor=1,
        persistent_workers=False,
        pin_memory=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=True
    )

    if display_informations:
        print(f'Total dataset: {len(train_dataloader) + len(val_dataloader) + len(test_dataloader)}, '
            f'train dataset: {len(train_dataloader)}, val dataset: {len(val_dataloader)}, test_dataset: {len(test_dataloader)}')
    return train_dataloader, val_dataloader, test_dataloader

def compute_weight_class(labels):
    unique, counts = np.unique(np.argmax(labels.numpy(), axis=1), return_counts=True)
    class_weights = torch.tensor([(class_counts / labels.shape[0]) for class_counts in counts],dtype=torch.float32)
    return class_weights

def compute_min_max(data_path, first_col_index=1, last_col_index=23):
    data = np.loadtxt(data_path, delimiter=",", dtype=np.float32, skiprows=1)[:, first_col_index:last_col_index]
    return np.stack((data.min(axis=0), data.max(axis=0)))

def save_accuracies_pkl(pkl_filepath, accuracies):
    accuracies_file = open(pkl_filepath, "wb")
    pickle.dump(accuracies, accuracies_file)
    accuracies_file.close()

def train_labels(columns_names, data_path, all_labels, split_proportions, save_directory, nb_epochs=3, batch_size=1, display_history=True):
    trainers = []

    for i, column in enumerate(all_labels):
        if not column in columns_names:
            continue

        dataset = HeartDiseaseDataset(data_path, any_disease=False, label_indexes=[23 + i, 24 + i])
        train_dataloader, val_dataloader, test_dataloader = create_dataloaders(dataset, split_proportions, batch_size)

        model = LogisticRegression(22,2)

        trainer = Trainer(model, dataset.class_weights, save_directory, loss='BCElogits', lr=0.05, label_name=column)
        trainer.fit(train_dataloader, val_dataloader, nb_epochs=nb_epochs)
        trainers.append(trainer)

        if display_history:
            trainer.display_history()

        print("\n\n")
    return trainers

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


    print(f'sensitivity : {sensitivity}, specificity : {specificity}, PPV : {PPV}, NPV : {NPV}')
    return accuracy

