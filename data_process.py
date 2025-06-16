import os
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr


def preprocess_emg(emg, fs=1000):
    # 带通滤波 20–450Hz
    b_band, a_band = butter(4, [20 / (fs/2), 450 / (fs/2)], btype='band')
    emg_filt = filtfilt(b_band, a_band, emg, axis=0)
    emg_abs = np.abs(emg_filt)

    # 低通滤波 10Hz
    b_low, a_low = butter(4, 10 / (fs/2), btype='low')
    emg_final = filtfilt(b_low, a_low, emg_abs, axis=0)
    return emg_final


def extract_windows(emg, labels, win_len=1500, stride=60):
    emg_chl = emg.shape[1]
    windows = []
    window_labels = []
    for i in range(0, len(emg) - win_len + 1, stride):
        win_emg = emg[i:i+win_len]
        mid_label = labels[i + win_len // 2]
        if mid_label in [0, 2, 3, 4, 5, 6]:  # 有效标签
            windows.append(win_emg)
            window_labels.append(mid_label)
    return np.stack(windows), np.array(window_labels)


def calc_pearson_matrix(signal):
    emg_chl = signal.shape[1]
    corr = np.corrcoef(signal.T)
    return corr if corr.shape == (emg_chl, emg_chl) else np.eye(emg_chl)


def extract_small_sequence(big_win, win_len=500, step=50):
    seq = []
    for i in range(0, big_win.shape[0] - win_len + 1, step):  # 2000点 → 31个小窗
        seg = big_win[i:i+win_len]
        seq.append(calc_pearson_matrix(seg))
    return np.stack(seq)  # [31, 11, 11]


def map_labels(raw_labels):
    label_map = {0:0,2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    return np.array([label_map[l] for l in raw_labels])


# ====== 主处理流程 ======
base_path = r'D:\document\data\EMG数据\MYD'
output_path = r'E:\PyCharm 2024.2.3\pythonProject\HAR\DATA_EMG\data_processing_pearson_fused'

os.makedirs(output_path, exist_ok=True)

for i in range(101, 111):
    filename = f"MD{i:02d}.mat"
    filepath = os.path.join(base_path, filename)

    if not os.path.exists(filepath):
        print(f"File not found: {filename}")
        continue

    data = sio.loadmat(filepath)
    emg_raw = data['data'][:, 0:8]
    labels_raw = data['data'][:, 8]

    emg_filtered = preprocess_emg(emg_raw)
    emg_windows, label_windows = extract_windows(emg_filtered, labels_raw)

    N = len(emg_windows)
    big_pearsons = np.zeros((N, 8, 8))
    small_sequences = np.zeros((N, 21, 8, 8))

    for idx in range(N):
        big_pearsons[idx] = calc_pearson_matrix(emg_windows[idx])
        small_sequences[idx] = extract_small_sequence(emg_windows[idx])

    label_mapped = map_labels(label_windows)

    np.save(os.path.join(output_path, f'big_pearsons_AB{i:02d}_8channel.npy'), big_pearsons)
    np.save(os.path.join(output_path, f'small_seq_pearsons_AB{i:02d}_8channel.npy'), small_sequences)
    np.save(os.path.join(output_path, f'labels_AB{i:02d}_8channel.npy'), label_mapped)

    print(f"✅ Processed AB{i:02d}: {N} samples.")
