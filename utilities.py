from monai.utils import first
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import seaborn as sns
import torch
from torch.cuda.amp import autocast, GradScaler
import os
import numpy as np
from monai.networks.utils import one_hot
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


def dice_metric(y_pred, y, num_classes=3):
    y = y.long()  # Ensure proper dtype
    y_onehot = one_hot(y, num_classes)  # y is (B, 1, H, W, D)
    y_pred_soft = torch.softmax(y_pred, dim=1)
    y_pred_bin = y_pred_soft.argmax(dim=1, keepdim=True)  # shape: (B, 1, H, W, D)
    y_pred_onehot = one_hot(y_pred_bin, num_classes)

    dice_per_class = []
    for i in range(num_classes):
        intersection = (y_onehot[:, i] * y_pred_onehot[:, i]).sum()
        union = y_onehot[:, i].sum() + y_pred_onehot[:, i].sum()
        dice_score = (2. * intersection) / (union + 1e-5)
        dice_per_class.append(dice_score.item())

    return dice_per_class



# Function for training the model
def train(model, data_in, num_classes, loss_function, optimizer, max_epochs, model_dir, test_interval=1,
          device=torch.device('cuda:0')):
    best_metric = -1
    best_metric_epoch = -1
    save_loss_train, save_loss_test = [], []
    save_metric_train, save_metric_test = [], []
    train_loader, test_loader = data_in

    for epoch in range(max_epochs):
        model.train()
        train_epoch_loss = 0
        train_step = 0
        dice_cumulative = np.zeros(num_classes)

        for batch_data in train_loader:
            train_step += 1
            volumes = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            optimizer.zero_grad()
            outputs = model(volumes)
            train_loss = loss_function(outputs, labels)
            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            dice_scores = dice_metric(outputs, labels, num_classes)
            dice_cumulative += np.array(dice_scores)

            print(f"Epoch {epoch + 1}, Step {train_step}, Loss: {train_loss.item():.4f}, Dice: {dice_scores}")

        avg_train_loss = train_epoch_loss / train_step
        avg_dice_scores = dice_cumulative / train_step
        avg_dice = avg_dice_scores.mean()

        print(
            f"[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.4f}, Dice (mean): {avg_dice:.4f}, Dice (per class): {avg_dice_scores}")

        save_loss_train.append(avg_train_loss)
        save_metric_train.append(avg_dice)
        np.save(os.path.join(model_dir, 'train_loss.npy'), save_loss_train)
        np.save(os.path.join(model_dir, 'train_metric.npy'), save_metric_train)

        if (epoch + 1) % test_interval == 0:
            model.eval()
            with torch.no_grad():
                test_epoch_loss = 0
                dice_cumulative = np.zeros(num_classes)
                all_preds = []
                all_labels = []
                test_step = 0

                for test_data in test_loader:
                    test_step += 1
                    volumes = test_data["image"].to(device)
                    labels = test_data["label"].to(device)

                    outputs = model(volumes)
                    test_loss = loss_function(outputs, labels)
                    test_epoch_loss += test_loss.item()

                    dice_scores = dice_metric(outputs, labels, num_classes)
                    dice_cumulative += np.array(dice_scores)

                    preds = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.uint8).flatten()
                    gts = labels.cpu().numpy().astype(np.uint8).flatten()
                    all_preds.extend(preds)
                    all_labels.extend(gts)

                avg_test_loss = test_epoch_loss / test_step
                avg_dice_scores = dice_cumulative / test_step
                avg_dice = avg_dice_scores.mean()

                cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
                precision = precision_score(all_labels, all_preds, labels=list(range(num_classes)), average=None,
                                            zero_division=0)
                recall = recall_score(all_labels, all_preds, labels=list(range(num_classes)), average=None,
                                      zero_division=0)

                print(f"[Epoch {epoch + 1}] Test Loss: {avg_test_loss:.4f}, Dice (mean): {avg_dice:.4f}")
                print(f"Dice per class: {avg_dice_scores}")
                print(f"Precision per class: {precision}")
                print(f"Recall per class: {recall}")
                print(f"Confusion Matrix:\n{cm}")

                save_loss_test.append(avg_test_loss)
                save_metric_test.append(avg_dice)
                np.save(os.path.join(model_dir, 'test_loss.npy'), save_loss_test)
                np.save(os.path.join(model_dir, 'test_metric.npy'), save_metric_test)

                if avg_dice > best_metric:
                    best_metric = avg_dice
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(model_dir, "best_metric_model.pth"))

                # Optional: visualize confusion matrix
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["BG", "Liver", "Tumor"],
                            yticklabels=["BG", "Liver", "Tumor"])
                plt.title(f'Confusion Matrix (Epoch {epoch + 1})')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.tight_layout()
                plt.savefig(os.path.join(model_dir, f'conf_matrix_epoch_{epoch + 1}.png'))
                plt.close()

    print(f"Training complete. Best Dice: {best_metric:.4f} at epoch {best_metric_epoch}")
















