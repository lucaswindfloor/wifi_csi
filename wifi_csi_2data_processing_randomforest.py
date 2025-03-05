import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import skew, kurtosis
import time
from scipy.fft import fft

# 配置参数
WINDOW_SIZE_SEC = 3.0  # 时间窗口大小（秒）
WINDOW_STRIDE_SEC = 1.0  # 窗口滑动步长（秒）
OUTPUT_DIR = "processed_data"


# 读取CSV文件
def read_csv(file_path):
    """读取CSV文件"""
    try:
        df = pd.read_csv(file_path, skip_blank_lines=True)
        print(f"CSV文件读取成功，形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        return df.dropna()
    except Exception as e:
        print(f"读取CSV文件错误: {e}")
        return None


# 解析CSI数据
def parse_csi_magnitude(csi_text):
    """从csi_magnitude列解析CSI幅度数据"""
    try:
        # 检查输入是否是字符串
        if not isinstance(csi_text, str):
            return None

        # 移除前后的引号(如果有)
        csi_text = csi_text.strip('"\'')

        # 分割并转换为浮点数
        values = [float(x.strip()) for x in csi_text.split(',') if x.strip()]
        return np.array(values)
    except Exception as e:
        # print(f"解析CSI幅度出错: {e}, 文本: {csi_text[:30]}...")
        return None


# 从CSV文件收集CSI数据
def collect_data(file_path):
    """从文件中收集CSI数据"""
    print(f"\n====== 处理文件: {file_path} ======\n")
    print(f"开始处理文件: {file_path}")
    df = read_csv(file_path)

    if df is None:
        return None, None, None

    all_amplitudes = []
    all_timestamps = []

    start_time = time.time()
    successful_samples = 0
    total_samples = len(df)

    for idx, row in df.iterrows():
        try:
            if idx % max(1, total_samples // 10) == 0:
                elapsed = time.time() - start_time
                print(f"进度: {idx}/{total_samples} ({idx / total_samples * 100:.1f}%) - 用时: {elapsed:.1f}秒")

            # 直接从csi_magnitude列获取CSI幅度
            magnitude = parse_csi_magnitude(row['csi_magnitude'])

            if magnitude is not None and len(magnitude) > 0:
                # 存储幅度数据
                all_amplitudes.append(magnitude)

                # 获取时间戳
                timestamp = row['sys_timestamp'] if 'sys_timestamp' in df.columns else idx
                all_timestamps.append(timestamp)

                successful_samples += 1

        except Exception as e:
            # 静默错误处理
            pass

    print(f"成功处理 {successful_samples}/{total_samples} 个样本")

    if successful_samples > 0:
        # 转换为NumPy数组
        all_amplitudes = np.array(all_amplitudes)
        all_timestamps = np.array(all_timestamps)

        # 生成虚拟相位数据(都设为0)
        all_phases = np.zeros_like(all_amplitudes)

        return all_amplitudes, all_phases, all_timestamps
    else:
        print("数据加载失败")
        return None, None, None


# 数据预处理
def preprocess_data(amplitudes, phases=None):
    """CSI数据预处理"""
    if amplitudes is None or len(amplitudes) == 0:
        return None, None

    print(f"开始预处理 {len(amplitudes)} 个样本")

    # 幅度预处理
    preprocessed_amplitudes = []
    for amp in amplitudes:
        # 归一化
        amp_mean = np.mean(amp)
        amp_std = np.std(amp)
        if amp_std > 0:
            amp_normalized = (amp - amp_mean) / amp_std
        else:
            amp_normalized = amp - amp_mean

        # 平滑处理
        try:
            window_length = min(11, len(amp) - 1 if len(amp) % 2 == 0 else len(amp) - 2)
            if window_length > 2:
                amp_smoothed = savgol_filter(amp_normalized, window_length, 2)
            else:
                amp_smoothed = amp_normalized
        except:
            amp_smoothed = amp_normalized

        preprocessed_amplitudes.append(amp_smoothed)

    # 如果没有提供相位，创建虚拟相位(全0)
    if phases is None:
        phases = np.zeros_like(amplitudes)

    # 返回结果
    return np.array(preprocessed_amplitudes), phases


# 将数据分割成时间窗口
def segment_into_windows(data, timestamps, window_size_sec, stride_sec):
    """基于时间分割数据窗口"""
    if data is None or len(data) == 0 or timestamps is None:
        return [], []

    windows = []
    window_times = []

    # 计算数据总时长
    if len(timestamps) > 1:
        total_duration = timestamps[-1] - timestamps[0]
        # 估算采样率
        sampling_rate = len(timestamps) / total_duration
    else:
        # 默认采样率
        sampling_rate = 20  # 假设20Hz
        total_duration = len(data) / sampling_rate

    print(f"数据总时长: {total_duration:.2f}秒, 估算采样率: {sampling_rate:.2f}Hz")

    # 窗口大小和步长(样本数)
    window_samples = int(window_size_sec * sampling_rate)
    stride_samples = int(stride_sec * sampling_rate)

    if window_samples > len(data):
        print(f"警告: 窗口大小({window_samples}样本)大于数据长度({len(data)}样本)")
        # 如果数据长度不足一个窗口，使用整个数据作为一个窗口
        if len(data) > 10:  # 至少要有10个样本
            windows.append(data)
            window_times.append(timestamps[0])
    else:
        # 按样本数分割窗口
        for start_idx in range(0, len(data) - window_samples + 1, stride_samples):
            end_idx = start_idx + window_samples
            window = data[start_idx:end_idx]
            windows.append(window)
            window_times.append(timestamps[start_idx])

    print(f"分割得到 {len(windows)} 个窗口，每个窗口 {window_samples} 个样本")
    return windows, window_times


# 特征提取 - 为随机森林优化的版本
def extract_features(amplitude_windows):
    """从CSI窗口中提取特征 - 针对随机森林算法优化"""
    if not amplitude_windows or len(amplitude_windows) == 0:
        return np.array([]), []

    print(f"从 {len(amplitude_windows)} 个窗口提取特征 (随机森林优化版)")
    features = []
    feature_names = []

    for window_idx, window in enumerate(amplitude_windows):
        # 跳过过小的窗口
        if window.shape[0] < 10:
            continue

        # 每个窗口的特征
        window_features = []

        # ---------- 整体特征 (全部子载波的统计量) ----------
        # 1. 基础统计特征
        mean_all = np.mean(window)
        std_all = np.std(window)

        features_dict = {
            "mean_all": mean_all,
            "std_all": std_all,
            "min_all": np.min(window),
            "max_all": np.max(window),
            "range_all": np.ptp(window),
            "median_all": np.median(window),
            "p25_all": np.percentile(window, 25),
            "p75_all": np.percentile(window, 75),
            "iqr_all": np.percentile(window, 75) - np.percentile(window, 25),
        }

        # 2. 时间序列差分特征
        diff1 = np.diff(window, axis=0)
        mean_diff1 = np.mean(np.abs(diff1))
        std_diff1 = np.std(diff1)
        max_diff1 = np.max(np.abs(diff1))

        # 二阶差分
        diff2 = np.diff(window, n=2, axis=0)
        mean_diff2 = np.mean(np.abs(diff2))

        features_dict.update({
            "mean_diff": mean_diff1,
            "std_diff": std_diff1,
            "max_diff": max_diff1,
            "mean_diff2": mean_diff2,
            "var_ratio": std_diff1 / (std_all + 1e-10),  # 变化率与总体方差比
            "sig_changes": np.sum(np.abs(diff1) > 0.5 * std_all)  # 显著变化的样本数
        })

        # 3. 子载波群组特征 (将子载波分组计算)
        n_subcarriers = window.shape[1]
        group_size = max(1, n_subcarriers // 8)  # 将子载波分成最多8组
        for g in range(0, n_subcarriers, group_size):
            if g + group_size <= n_subcarriers:
                group_data = window[:, g:g + group_size]
                group_mean = np.mean(group_data)
                group_std = np.std(group_data)
                group_diff = np.mean(np.abs(np.diff(group_data, axis=0)))

                features_dict.update({
                    f"group{g // group_size}_mean": group_mean,
                    f"group{g // group_size}_std": group_std,
                    f"group{g // group_size}_diff": group_diff
                })

        # 4. 频域特征 - 使用FFT提取频率特征
        if window.shape[0] >= 30:
            # 对时间维度进行FFT
            fft_result = np.abs(fft(window - np.mean(window, axis=0), axis=0))
            # 仅使用前半部分(由于对称性)
            half_len = window.shape[0] // 2
            fft_half = fft_result[:half_len]

            # 计算不同频段的能量
            low_freq = np.mean(np.sum(fft_half[1:3], axis=0))  # 低频(1-3Hz)
            mid_freq = np.mean(np.sum(fft_half[3:10], axis=0))  # 中频(3-10Hz)
            high_freq = np.mean(np.sum(fft_half[10:], axis=0))  # 高频(>10Hz)

            # 计算峰值频率
            peak_freq_idx = np.argmax(fft_half[1:], axis=0) + 1  # 跳过DC分量
            peak_freq = np.mean(peak_freq_idx) / window.shape[0]

            features_dict.update({
                "low_freq_energy": low_freq,
                "mid_freq_energy": mid_freq,
                "high_freq_energy": high_freq,
                "peak_freq": peak_freq,
                "freq_ratio_low_high": low_freq / (high_freq + 1e-10)
            })

        # 5. 分位数传播特征 - 捕获波形分布变化
        q_vals = [0.1, 0.25, 0.5, 0.75, 0.9]
        for i, q in enumerate(q_vals):
            percentile_val = np.percentile(window, q * 100)
            features_dict[f"percentile_{int(q * 100)}"] = percentile_val

        # 6. 动态特征 - 窗口前半部分和后半部分的比较
        half_idx = window.shape[0] // 2
        first_half = window[:half_idx]
        second_half = window[half_idx:]

        mean_first = np.mean(first_half)
        mean_second = np.mean(second_half)
        std_first = np.std(first_half)
        std_second = np.std(second_half)

        features_dict.update({
            "mean_change": mean_second - mean_first,
            "std_change": std_second - std_first,
            "dynamic_ratio": (mean_second - mean_first) / (mean_all + 1e-10)
        })

        # 7. 抑制极端值，计算非极端值的统计量
        # 找出在均值±2倍标准差范围内的值
        valid_range = (window > (mean_all - 2 * std_all)) & (window < (mean_all + 2 * std_all))
        valid_data = window[valid_range]

        if len(valid_data) > 0:
            features_dict.update({
                "valid_mean": np.mean(valid_data),
                "valid_std": np.std(valid_data),
                "outlier_ratio": 1.0 - (len(valid_data) / window.size)
            })

        # 只为第一个窗口创建特征名称
        if window_idx == 0:
            feature_names = list(features_dict.keys())

        # 添加特征值
        window_features = list(features_dict.values())
        features.append(window_features)

    if len(features) == 0:
        print("警告: 未能提取任何特征")
        return np.array([]), []

    features = np.array(features)
    print(f"提取了 {len(features)} 个样本的特征，每个样本 {len(feature_names)} 个特征")

    return features, feature_names


# 处理单个文件
def process_file(file_path, label):
    """处理单个CSI文件并提取特征"""
    # 1. 加载数据
    amplitudes, phases, timestamps = collect_data(file_path)

    if amplitudes is None or len(amplitudes) == 0:
        return None

    # 2. 数据预处理
    preprocessed_amplitudes, preprocessed_phases = preprocess_data(amplitudes, phases)

    if preprocessed_amplitudes is None:
        print("预处理失败")
        return None

    # 3. 分割时间窗口
    amplitude_windows, window_times = segment_into_windows(
        preprocessed_amplitudes, timestamps,
        WINDOW_SIZE_SEC, WINDOW_STRIDE_SEC
    )

    if len(amplitude_windows) == 0:
        print("窗口分割失败")
        return None

    # 4. 特征提取 - 使用针对随机森林优化的特征提取
    features, feature_names = extract_features(amplitude_windows)

    if len(features) == 0:
        print("特征提取失败")
        return None

    # 5. 准备结果数据
    result = {
        'file_path': file_path,
        'label': label,
        'features': features,
        'feature_names': feature_names,
        'window_times': window_times,
        'processing_time': time.strftime("%Y-%m-%d %H:%M:%S")
    }

    print(f"文件处理完成: 提取了 {len(features)} 个窗口的特征，每个窗口 {features.shape[1]} 个特征")
    return result


# 批处理多个文件
def batch_process(file_paths, labels, output_dir=OUTPUT_DIR):
    """批处理多个CSI文件并保存结果"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_results = []

    for i, (file_path, label) in enumerate(zip(file_paths, labels)):
        print(f"\n处理文件 {i + 1}/{len(file_paths)}: {os.path.basename(file_path)} (标签: {label})")

        # 处理文件
        result = process_file(file_path, label)

        if result is not None:
            # 将结果添加到列表
            all_results.append(result)

            # 保存单个文件的处理结果
            filename = os.path.basename(file_path).replace('.csv', '.pkl')
            output_file = os.path.join(output_dir, f"processed_{filename}")

            with open(output_file, 'wb') as f:
                pickle.dump(result, f)

            print(f"已保存结果到: {output_file}")

    # 保存所有处理结果
    if all_results:
        summary_file = os.path.join(output_dir, "all_processed_data.pkl")
        with open(summary_file, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"\n所有处理结果已保存到: {summary_file}")
        return summary_file
    else:
        print("\n没有成功处理任何文件")
        return None


# 可视化特征重要性（在训练随机森林后使用）
def visualize_feature_importance(model, feature_names, output_file=None):
    """可视化随机森林特征重要性"""
    if not hasattr(model, 'feature_importances_'):
        print("模型没有feature_importances_属性")
        return

    # 获取特征重要性
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # 取前20个最重要的特征
    n_top_features = min(20, len(feature_names))

    plt.figure(figsize=(10, 6))
    plt.title('CSI特征重要性排名 (随机森林)')
    plt.bar(range(n_top_features), importances[indices[:n_top_features]], align='center')
    plt.xticks(range(n_top_features), [feature_names[i] for i in indices[:n_top_features]], rotation=90)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
        print(f"特征重要性图表已保存到: {output_file}")

    plt.show()


# 主函数
if __name__ == "__main__":
    # 指定文件路径和标签
    file_paths = [
        'csi_data/csi_data_20250304_091236empty.csv',  # 无人场景1
        'csi_data/csi_data_20250304_091736empty.csv',  # 无人场景2
        'csi_data/csi_data_20250304_120949empty.csv',  # 无人场景3
        'csi_data/csi_data_20250304_121449empty.csv',  # 无人场景4
        'csi_data/csi_data_20250304_121949empty.csv',  # 无人场景5
        'csi_data/csi_data_20250304_122449empty.csv',  # 无人场景6
        'csi_data/csi_data_20250304_122949empty.csv',  # 无人场景7
        'csi_data/csi_data_20250304_133708empty.csv',  # 无人场景8
        'csi_data/csi_data_20250304_093218sit.csv',  # 有人场景1
        'csi_data/csi_data_20250304_094718person.csv',  # 有人场景2
        'csi_data/csi_data_20250304_134746walk.csv',  # 有人场景3
        'csi_data/csi_data_20250304_135247sit.csv'  # 有人场景4

    ]

    labels = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  # 0=无人, 1=有人

    # 执行批处理
    print("开始CSI数据批处理 (随机森林优化版)...")
    result_file = batch_process(file_paths, labels)

    if result_file:
        print(f"批处理成功完成，结果保存在: {result_file}")
        print("接下来可以使用这些特征训练随机森林模型")
        print("示例: from sklearn.ensemble import RandomForestClassifier")
        print("      model = RandomForestClassifier(n_estimators=50, max_depth=8)")
        print("      model.fit(X_train, y_train)")
    else:
        print("批处理失败")