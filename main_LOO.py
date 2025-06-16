import torch
from torch import optim
from data_input_LeaveOneOut import prepare_fused_data_loaders
from layers1 import E2EConv
from layers1 import FusedEMGNet
from DATA_training_testing import train_and_evaluate
from DATA_training_testing import summarize_fold_accuracies
def main():
    # ===== 超参数设置 =====
    EPOCH = 40
    LR = 0.00005
    BATCH_SIZE = 64
    n_classes = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 数据加载 =====
    data_folder = r"E:\PyCharm 2024.2.3\pythonProject\HAR\DATA_EMG\data_processing_pearson_fused"
    data_loaders = prepare_fused_data_loaders(data_folder, batch_size=BATCH_SIZE)

    fold_accuracies = []
    for fold, (train_loader, test_loader) in enumerate(data_loaders):
        print(f"\n===== Fold {fold + 1} =====")

        # ===== 构建模型与优化器 =====
        model = FusedEMGNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)

        # ===== 训练与评估 =====
        best_acc, model = train_and_evaluate(
            train_loader, test_loader, EPOCH, LR,
            model, optimizer, device
        )

        print(f"Fold {fold+1} BEST Test Accuracy: {best_acc:.2f}%")
        fold_accuracies.append(best_acc)

    # ===== 汇总结果 =====
    summarize_fold_accuracies(fold_accuracies)

if __name__ == '__main__':
    main()
