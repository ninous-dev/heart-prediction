from .HeartDiseaseDataset import HeartDiseaseDataset
from .create_dataloaders import create_dataloaders
from .LogisticRegression import LogisticRegression
from .Trainer import Trainer

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

