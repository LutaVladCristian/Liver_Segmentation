"""
Created on Thu Apr  6 14:29:08 2023

@author: vlad_cristian.luta
"""

from monai.utils import first
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from monai.losses import DiceLoss

# Function for visualizing data
def show_patient(data, SLICE_NUMBER=1):
    view_patient = first(data)

    plt.figure("Visualization", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title(f"image {SLICE_NUMBER}")
    plt.imshow(view_patient["image"][0, 0, :, :, SLICE_NUMBER], cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title(f"label {SLICE_NUMBER}")
    plt.imshow(view_patient["label"][0, 0, :, :, SLICE_NUMBER])
    plt.show()

# Metric for evaluating the model (Dice coefficient)
def dice_metric(y_pred, y):
    dice_loss = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
    dice_coeff = 1 - dice_loss(y_pred, y).item()
    return dice_coeff

# Function for training the model
def train(model, data_in, loss_function, optimizer, max_epochs, model_dir, test_interval=1, device=torch.device('cuda:0')):
    best_metric = -1
    best_metric_epoch = -1
    save_loss_train = []
    save_loss_test = []
    save_metric_train = []
    save_metric_test = []
    train_loader, test_loader = data_in

    for epoch in range(max_epochs):
        model.train()
        train_epoch_loss = 0
        train_step = 0
        train_epoch_metric = 0

        for batch_data in train_loader:
            train_step += 1
            volumes = batch_data["image"]
            labels = batch_data["label"]
            labels = labels != 0
            volumes = volumes.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(volumes)

            train_loss = loss_function(outputs, labels)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_metric = dice_metric(outputs, labels)
            train_epoch_metric += train_metric

            print(f"{epoch + 1}/{max_epochs} and {train_step}/{len(train_loader) // train_loader.batch_size} => train_loss: {train_loss.item():.4f} and train_metric: {train_metric:.4f}")

        print('Saving training data after epoch: ' + str(epoch + 1))
        train_epoch_loss /= train_step
        print(f"epoch {epoch + 1} average training loss: {train_epoch_loss:.4f}")
        save_loss_train.append(train_epoch_loss)
        np.save(os.path.join(model_dir, 'train_loss.npy'), save_loss_train)

        train_epoch_metric /= train_step
        print(f"epoch {epoch + 1} average training metric: {train_epoch_metric:.4f}")
        save_metric_train.append(train_epoch_metric)
        np.save(os.path.join(model_dir, 'train_metric.npy'), save_metric_train)

        if (epoch+1) % test_interval == 0:
            model.eval()
            with torch.no_grad():
                test_epoch_loss = 0
                test_metric = 0
                test_step = 0
                test_epoch_metric = 0

                for test_data in test_loader:
                    test_step += 1
                    volumes = test_data["image"]
                    labels = test_data["label"]
                    labels = labels != 0
                    volumes = volumes.to(device)
                    labels = labels.to(device)

                    outputs = model(volumes)

                    test_loss = loss_function(outputs, labels)

                    test_epoch_loss += test_loss.item()
                    test_metric = dice_metric(outputs, labels)
                    test_epoch_metric += test_metric

                    print(f"{epoch + 1}/{max_epochs} and {test_step}/{len(test_loader) // test_loader.batch_size} => test_loss: {test_loss.item():.4f} and test_metric: {test_metric:.4f}")

                print('Saving testing data after epoch: ' + str(epoch + 1))
                test_epoch_loss /= test_step
                print(f"epoch {epoch + 1} average testing loss: {test_epoch_loss:.4f}")
                save_loss_test.append(test_epoch_loss)
                np.save(os.path.join(model_dir, 'test_loss.npy'), save_loss_test)

                test_epoch_metric /= test_step
                print(f"epoch {epoch + 1} average testing metric: {test_epoch_metric:.4f}")
                save_metric_test.append(test_epoch_metric)
                np.save(os.path.join(model_dir, 'test_metric.npy'), save_metric_test)
        
                if test_epoch_metric > best_metric:
                    best_metric = test_epoch_metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(model_dir, "best_metric_model.pth"))

                print(f"current epoch: {epoch + 1} current test Dice coefficient: {test_epoch_metric:.4f}"
                    f"\nbest metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

    print(f"train completed => best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")