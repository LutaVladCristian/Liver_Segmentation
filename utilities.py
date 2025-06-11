from monai.utils import first
import matplotlib.pyplot as plt
import torch
from torch.cuda.amp import autocast, GradScaler
import os
import numpy as np
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

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


# Function for training the model
def train(
    model,
    data_in,
    num_classes,
    loss_function,
    optimizer,
    max_epochs,
    model_dir,
    test_interval=1,
    device=torch.device('cuda:0')
):
    train_loader, test_loader = data_in

    dice_metric = DiceMetric(include_background=False, reduction="none", get_not_nans=False)
    scaler = GradScaler()

    # Tracking
    best_metric = -1
    best_metric_epoch = -1
    save_loss_train, save_loss_test = [], []
    save_metric_train, save_metric_test = [], []

    for epoch in range(max_epochs):
        print(f"\n--- Epoch {epoch + 1}/{max_epochs} ---")
        model.train()
        train_loss_total = 0.0
        train_dice_sum = torch.zeros(num_classes - 1, device=device)  # skip background
        steps = 0

        for step, batch_data in enumerate(train_loader):
            volumes = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            optimizer.zero_grad()
            with autocast(device_type=device.type):
                outputs = model(volumes)
                loss = loss_function(outputs, labels)

            if device.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                preds = torch.softmax(outputs, dim=1)
                labels_onehot = one_hot(labels, num_classes=num_classes)
                dice_scores = dice_metric(y_pred=preds, y=labels_onehot)
                #Average dice scores over the batch dimension before accumulation
                dice_scores = dice_scores.mean(dim=0)
                if dice_scores.ndim > 1:
                    dice_scores = dice_scores.squeeze()
                train_dice_sum += dice_scores
                train_loss_total += loss.item()
                steps += 1

            print(f"Step {step+1}/{len(train_loader)} => "
                  f"Loss: {loss.item():.4f} | "
                  f"Liver Dice: {dice_scores[0].item():.4f} | "
                  f"Tumor Dice: {dice_scores[1].item():.4f}")

        epoch_loss = train_loss_total / steps
        epoch_dice = (train_dice_sum / steps).cpu().numpy()
        avg_dice = epoch_dice.mean()

        save_loss_train.append(epoch_loss)
        save_metric_train.append(avg_dice)
        np.save(os.path.join(model_dir, 'train_loss.npy'), save_loss_train)
        np.save(os.path.join(model_dir, 'train_metric.npy'), save_metric_train)

        print(f"✅ Epoch {epoch+1} Train Avg Loss: {epoch_loss:.4f} | "
              f"Liver Dice: {epoch_dice[0]:.4f}, Tumor Dice: {epoch_dice[1]:.4f}, "
              f"Avg Dice: {avg_dice:.4f}")

        # ---------- TESTING ----------
        if (epoch + 1) % test_interval == 0:
            model.eval()
            test_loss_total = 0.0
            test_dice_sum = torch.zeros(num_classes - 1, device=device)
            steps = 0

            with torch.no_grad():
                for step, test_data in enumerate(test_loader):
                    volumes = test_data["image"].to(device)
                    labels = test_data["label"].to(device)

                    with autocast(device_type=device.type):
                        outputs = model(volumes)
                        loss = loss_function(outputs, labels)

                    preds = torch.softmax(outputs, dim=1)
                    labels_onehot = one_hot(labels, num_classes=num_classes)
                    dice_scores = dice_metric(y_pred=preds, y=labels_onehot)
                    # Average dice scores over the batch dimension before accumulation
                    dice_scores = dice_scores.mean(dim=0)
                    if dice_scores.ndim > 1:
                        dice_scores = dice_scores.squeeze()
                    test_dice_sum += dice_scores
                    test_loss_total += loss.item()
                    steps += 1

                    print(f"Step {step+1}/{len(test_loader)} => "
                          f"Loss: {loss.item():.4f} | "
                          f"Liver Dice: {dice_scores[0].item():.4f} | "
                          f"Tumor Dice: {dice_scores[1].item():.4f}")

            epoch_test_loss = test_loss_total / steps
            epoch_test_dice = (test_dice_sum / steps).cpu().numpy()
            avg_test_dice = epoch_test_dice.mean()

            save_loss_test.append(epoch_test_loss)
            save_metric_test.append(avg_test_dice)
            np.save(os.path.join(model_dir, 'test_loss.npy'), save_loss_test)
            np.save(os.path.join(model_dir, 'test_metric.npy'), save_metric_test)

            print(f"🔍 Epoch {epoch+1} Test Avg Loss: {epoch_test_loss:.4f} | "
                  f"Liver Dice: {epoch_test_dice[0]:.4f}, Tumor Dice: {epoch_test_dice[1]:.4f}, "
                  f"Avg Dice: {avg_test_dice:.4f}")

            if avg_test_dice > best_metric:
                best_metric = avg_test_dice
                best_metric_epoch = epoch + 1
                model_path = os.path.join(model_dir, f"best_model_epoch{epoch+1}_dice{best_metric:.4f}.pth")
                torch.save(model.state_dict(), model_path)
                print(f"💾 Best model saved at epoch {epoch+1} with Avg Dice {best_metric:.4f}")

    print(f"\n🏁 Training complete. Best Avg Dice: {best_metric:.4f} at epoch {best_metric_epoch}")



















