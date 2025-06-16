from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import torch
from torch import nn
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import numpy as np
import warnings

CUDA = torch.cuda.is_available()
warnings.filterwarnings("ignore")

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_and_evaluate(
    train_loader, test_loader, epoch, lr, model, optimizer, device,
    patience=12,
):
    loss_func = nn.CrossEntropyLoss()
    warmup_epochs = 5
    scheduler = CosineAnnealingLR(optimizer, T_max=epoch - warmup_epochs, eta_min=1e-6)
    best_acc = 0
    counter = 0
    best_model_state = None
    best_preds = None
    best_labels = None

    for ee in tqdm(range(epoch)):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch [{ee+1}/{epoch}] | LR: {current_lr:.6f}")
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        for big, seq, label in train_loader:
            big = big.to(device)        # [B, 1, H, W]
            seq = seq.to(device)        # [B, T, 1, H, W]
            label = label.to(device)

            output = model(big, seq)    # [B, n_classes]
            loss = loss_func(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, pred = torch.max(output, 1)
            total_correct += (pred == label).sum().item()
            total_samples += label.size(0)

        train_acc = 100 * total_correct / total_samples
        avg_loss = total_loss / len(train_loader)

        # ==== 测试 ====
        # warmup + cosine 调度
        if ee < warmup_epochs:
            lr_now = 1e-6 + (lr - 1e-6) * (ee + 1) / warmup_epochs
            adjust_learning_rate(optimizer, lr_now)
        else:
            scheduler.step()
        model.eval()
        correct_test = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for big, seq, label in test_loader:
                big = big.to(device)
                seq = seq.to(device)
                label = label.to(device)

                out = model(big, seq)
                _, pred = torch.max(out, 1)

                correct_test += (pred == label).sum().item()
                all_preds.append(pred.cpu())
                all_labels.append(label.cpu())

        test_acc = 100 * correct_test / len(test_loader.dataset)

        print(f"Epoch [{ee+1}/{epoch}] | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        # ==== Early stopping & 保存 best 结果 ====
        if test_acc > best_acc:
            best_acc = test_acc
            counter = 0
            best_preds = torch.cat(all_preds).numpy()
            best_labels = torch.cat(all_labels).numpy()
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {ee+1}, best test acc: {best_acc:.2f}%")
                break

    # 输出 best confusion matrix
    if best_preds is not None and best_labels is not None:
        cm = confusion_matrix(best_labels, best_preds)
        cm_percent = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)

        print("\nConfusion Matrix ):")
        for row in cm_percent:
            print(["{:.1f}%".format(p * 100) for p in row])
            print([p for p in row])

    # ==== 输出 F1 分数 ====
    if best_preds is not None and best_labels is not None:
        f1 = f1_score(best_labels, best_preds, average='macro')
        print(f"\nMacro F1 Score: {f1:.4f}")
    return best_acc, model

def summarize_fold_accuracies(fold_accuracies):

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    cv_acc = std_acc / mean_acc
    print(f"\n===== Accuracy Summary =====")
    print(f"Mean Accuracy: {mean_acc:.2f}%")
    print(f"Std Deviation: {std_acc:.2f}")
    print(f"Coefficient of Variation (CV): {cv_acc:.4f}")
