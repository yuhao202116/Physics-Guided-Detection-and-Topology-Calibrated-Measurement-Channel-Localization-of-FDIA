import os
import sys
from time import time
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import torch
from torch.utils.data import DataLoader
from sklearn.svm import OneClassSVM
import joblib

from configs.nn_setting import nn_setting
from data.dataset import rnn_dataset
from configs.config import sys_config

np.random.seed(0)
torch.manual_seed(0)


def extract_flattened_data(data_loader):
    """
    将 PyTorch DataLoader 中的 3D 时序数据提取出来并展平为 2D。
    假设原始输入维度为 (B, Length, Features) -> 展平为 (B, Length * Features)
    """
    x_list = []

    print("Extracting and flattening data from DataLoader...")
    for batch_idx, (inputs, _) in enumerate(data_loader):
        batch_size = inputs.shape[0]
        inputs_flat = inputs.view(batch_size, -1).cpu().numpy()
        x_list.append(inputs_flat)


    X_all = np.concatenate(x_list, axis=0)
    print(f"Data extraction complete. Final shape: {X_all.shape}")
    return X_all


if __name__ == '__main__':

    from configs.paths import data_path
    z_noise_path = data_path('z_noise_summary.npy')
    if os.path.exists(z_noise_path):
        z_noise_summary = np.load(z_noise_path)
        print(f'Raw Data shape in disk: {z_noise_summary.shape}')

    print("="*60)
    print("Initializing Datasets for One-Class SVM...")
    print("="*60)

    # 2. 加载数据集
    # 按数据顺序分批提取训练窗口。
    train_set = rnn_dataset(mode='train', istransform=True)
    # 每批提取 512 个窗口。
    train_loader = DataLoader(train_set, batch_size=512, shuffle=False)

    valid_set = rnn_dataset(mode='valid', istransform=True)
    valid_loader = DataLoader(valid_set, batch_size=512, shuffle=False)

    # 3. 将 3D Tensor 转为 2D Numpy Array 以适配 SVM
    print("\n--- Processing Train Data ---")
    X_train = extract_flattened_data(train_loader)

    print("\n--- Processing Valid Data ---")
    X_valid = extract_flattened_data(valid_loader)

    # 4. 初始化 One-Class SVM 模型
    # 超参数说明：
    # nu: 训练误差比例的上界和支持向量比例的下界，取值范围为 (0, 1]。
    # gamma: 核函数参数，'scale' 会根据特征数量自动缩放（1 / (n_features * X.var())）
    nu_value = 0.02
    svm_model = OneClassSVM(kernel='rbf', gamma='scale', nu=nu_value)

    # 5. 训练模型
    print("\n" + "="*60)
    print(f"Start Training One-Class SVM for Case {sys_config.get('case_name', '57')}...")
    print(f"Hyperparameters: kernel='rbf', nu={nu_value}, gamma='scale'")
    print("="*60)

    start_time = time()
    # 使用正常训练窗口拟合 OC-SVM。
    svm_model.fit(X_train)
    end_time = time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    # 6. 验证与推理
    print("\n" + "="*60)
    print("Evaluating Model...")

    # SVM 预测：1 表示正常 (inlier)，-1 表示异常 (outlier)
    train_preds = svm_model.predict(X_train)
    valid_preds = svm_model.predict(X_valid)

    # 统计异常率
    train_anomalies = np.sum(train_preds == -1)
    valid_anomalies = np.sum(valid_preds == -1)

    print(f"[Train Set] Total: {len(train_preds)} | Anomalies detected: {train_anomalies} ({train_anomalies/len(train_preds)*100:.2f}%)")
    print(f"[Valid Set] Total: {len(valid_preds)} | Anomalies detected: {valid_anomalies} ({valid_anomalies/len(valid_preds)*100:.2f}%)")
    print("="*60)

    # 7. 保存模型
    # 使用 joblib 保存 sklearn 模型。
    save_dir = os.path.dirname(nn_setting['model_path'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    svm_save_path = os.path.join(save_dir, 'oc_svm_baseline.pkl')
    joblib.dump(svm_model, svm_save_path)
    print(f"\nModel successfully saved to: {svm_save_path}")
    print("To load the model later, use: model = joblib.load('path_to.pkl')")
