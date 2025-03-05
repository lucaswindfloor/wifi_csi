import socket
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import math
import numpy as np
from collections import deque
from prettytable import PrettyTable
import time
import os
import csv
import datetime
import joblib
from scipy.signal import savgol_filter
from scipy.fft import fft

# ========================
# 配置参数
# ========================
UDP_IP = "192.168.99.55"
UDP_PORT = 4444
MAX_POINTS = 64
REFRESH_INTERVAL = 100
TABLE_COLS = [
    'mac', 'rate', 'mcs', 'channel',
    'rssi', 'noise_floor', 'bandwidth', 'sampling_frequency'
]

CSI_COLUMNS = [
    "type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth",
    "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding",
    "sgi", "noise_floor", "ampdu_cnt", "channel", "secondary_channel",
    "local_timestamp", "ant", "sig_len", "rx_state", "len", "first_word", "data"
]

# 数据保存参数
DATA_SAVE_DIR = "csi_data"  # 保存数据的目录
SAVE_INTERVAL = 300  # 每300秒(5分钟)保存一个新文件

# 模型和推理参数
MODEL_PATH = "rf_model.joblib"  # 随机森林模型路径
BASELINE_PATH = "baseline_features.npz"  # 基线特征路径
WINDOW_SIZE_SEC = 3.0  # 时间窗口大小（秒）
WINDOW_STRIDE_SEC = 1.0  # 窗口滑动步长（秒）
INFERENCE_INTERVAL = 2  # 每隔几秒进行一次推理

# ========================
# 全局数据结构
# ========================
global_data = {
    'csi_mag': deque(maxlen=MAX_POINTS),
    'rssi': deque(maxlen=MAX_POINTS),
    'sys_timestamps': deque(maxlen=MAX_POINTS),  # 系统时间队列
    'params': {col: 'N/A' for col in TABLE_COLS},
    'lock': threading.Lock(),
    'table_initialized': False,
    'table_cells': {},
    'print_counter': 0,
    # 新增数据保存相关字段
    'current_csv_file': None,
    'current_csv_writer': None,
    'current_file_time': 0,
    'save_data_buffer': [],  # 缓存要保存的数据
    # 新增推理相关字段
    'inference_buffer': {
        'amplitudes': [],
        'timestamps': []
    },
    'inference_result': {
        'has_person': False,
        'probability': 0.0,
        'last_update': 0,
        'window_predictions': []
    },
    'model': None,
    'baseline': None
}

# 确保数据保存目录存在
os.makedirs(DATA_SAVE_DIR, exist_ok=True)


# ========================
# 数据保存函数
# ========================
def get_new_csv_filename():
    """生成带有当前时间戳的CSV文件名"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(DATA_SAVE_DIR, f"csi_data_{timestamp}.csv")


def create_new_csv_file():
    """创建新的CSV文件并返回文件和写入器"""
    filename = get_new_csv_filename()
    file = open(filename, 'w', newline='')

    # CSV文件头部:系统时间、所有CSI列以及CSI幅度数据
    fieldnames = ['sys_timestamp'] + CSI_COLUMNS + ['csi_magnitude']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    print(f"Creating new data file: {filename}")
    return file, writer, time.time()


def save_data_to_csv(packet, sys_timestamp, magnitudes):
    """将数据保存到CSV文件"""
    with global_data['lock']:
        # 检查是否需要创建新文件
        if (global_data['current_csv_file'] is None or
                time.time() - global_data['current_file_time'] >= SAVE_INTERVAL):

            # 如果有旧文件，先关闭
            if global_data['current_csv_file'] is not None:
                global_data['current_csv_file'].close()
                print(f"File closed, duration: {time.time() - global_data['current_file_time']:.1f} seconds")

            # 创建新文件
            file, writer, file_time = create_new_csv_file()
            global_data['current_csv_file'] = file
            global_data['current_csv_writer'] = writer
            global_data['current_file_time'] = file_time

        # 准备数据行
        row_data = {'sys_timestamp': sys_timestamp}
        row_data.update(packet)
        row_data['csi_magnitude'] = ','.join(map(str, magnitudes))

        # 写入数据
        global_data['current_csv_writer'].writerow(row_data)
        global_data['current_csv_file'].flush()  # 确保数据及时写入磁盘


# ========================
# 模型加载和推理函数
# ========================
def load_model_and_baseline():
    """加载训练好的随机森林模型和基线特征"""
    print(f"Loading Random Forest model: {MODEL_PATH}")
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully: {type(model).__name__} with {model.n_estimators} trees")

        # 加载基线特征
        print(f"Loading baseline features: {BASELINE_PATH}")
        baseline = np.load(BASELINE_PATH)
        print(f"Baseline features loaded, contains {baseline['sample_count']} empty samples")

        return model, baseline
    except Exception as e:
        print(f"Failed to load model or baseline: {e}")
        return None, None


def preprocess_data(amplitudes):
    """CSI数据预处理"""
    if amplitudes is None or len(amplitudes) == 0:
        return None

    # 转换为numpy数组
    amplitudes = np.array(amplitudes)

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

    return np.array(preprocessed_amplitudes)


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

    # 窗口大小和步长(样本数)
    window_samples = int(window_size_sec * sampling_rate)
    stride_samples = int(stride_sec * sampling_rate)

    if window_samples > len(data):
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

    return windows, window_times


def extract_features(amplitude_windows):
    """从CSI窗口中提取特征 - 针对随机森林算法优化"""
    if not amplitude_windows or len(amplitude_windows) == 0:
        return np.array([]), []

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
        return np.array([]), []

    features = np.array(features)
    return features, feature_names


def run_inference():
    """执行推理线程"""
    print("Starting inference thread...")

    # 加载模型
    model, baseline = load_model_and_baseline()
    with global_data['lock']:
        global_data['model'] = model
        global_data['baseline'] = baseline

    if model is None:
        print("Model loading failed, inference thread exiting")
        return

    while True:
        try:
            # 每隔一段时间执行推理
            time.sleep(INFERENCE_INTERVAL)

            # 当前时间
            current_time = time.time()

            # 从全局数据中复制到推理缓冲区
            with global_data['lock']:
                inference_buffer = {
                    'amplitudes': list(global_data['inference_buffer']['amplitudes']),
                    'timestamps': list(global_data['inference_buffer']['timestamps'])
                }

            # 检查数据量是否足够
            if len(inference_buffer['amplitudes']) < 10:
                continue

            # 预处理数据
            preprocessed_amplitudes = preprocess_data(inference_buffer['amplitudes'])
            if preprocessed_amplitudes is None:
                continue

            # 分割窗口
            amplitude_windows, window_times = segment_into_windows(
                preprocessed_amplitudes,
                inference_buffer['timestamps'],
                WINDOW_SIZE_SEC,
                WINDOW_STRIDE_SEC
            )

            if len(amplitude_windows) == 0:
                continue

            # 提取特征
            features, feature_names = extract_features(amplitude_windows)
            if len(features) == 0:
                continue

            # 确保特征维度匹配
            expected_features = model.n_features_in_
            if features.shape[1] != expected_features:
                if features.shape[1] < expected_features:
                    # 填充额外特征
                    pad_width = expected_features - features.shape[1]
                    features = np.pad(features, ((0, 0), (0, pad_width)), 'constant')
                else:
                    # 截断多余特征
                    features = features[:, :expected_features]

            # 执行预测
            predictions = model.predict(features)
            probabilities = model.predict_proba(features)[:, 1]  # 取类别1(有人)的概率

            # 计算结果
            person_count = np.sum(predictions == 1)
            no_person_count = np.sum(predictions == 0)
            avg_probability = np.mean(probabilities)

            # 最终结论
            final_prediction = person_count > no_person_count
            confidence = max(person_count, no_person_count) / len(features)

            # 更新全局结果
            with global_data['lock']:
                global_data['inference_result'] = {
                    'has_person': final_prediction,
                    'probability': avg_probability,
                    'confidence': confidence,
                    'last_update': current_time,
                    'window_predictions': list(zip(predictions, probabilities))
                }

            # 输出结果
            status = "Person detected" if final_prediction else "No person"
            print(f"\n[Inference Result] {status} (Prob: {avg_probability:.4f}, Conf: {confidence:.2f})")
            print(f"Windows: {len(features)}, Positive: {person_count}, Negative: {no_person_count}")

        except Exception as e:
            print(f"Inference error: {str(e)}")
            time.sleep(5)  # 出错后暂停一段时间


# ========================
# UDP数据接收线程
# ========================
def udp_receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    sock.settimeout(0.1)

    print(f"UDP receiver started, listening on port: {UDP_PORT}")
    print(f"Data will be saved to: {os.path.abspath(DATA_SAVE_DIR)} directory")

    while True:
        try:
            data, addr = sock.recvfrom(1024)
            decoded = data.decode().strip().replace('"', '')
            if not decoded.startswith("CSI_DATA"):
                continue

            parts = decoded.split(',', len(CSI_COLUMNS) - 1)
            if len(parts) != len(CSI_COLUMNS):
                continue
            packet = dict(zip(CSI_COLUMNS, parts))

            try:
                # 记录系统时间（精确到毫秒）
                sys_timestamp = time.time()

                # 解析CSI数据
                csi_str = packet['data'].strip('[]')
                csi_pairs = [float(x) for x in csi_str.split(',')]
                magnitudes = []
                # 确保偶数长度
                csi_pairs = csi_pairs[:len(csi_pairs) // 2 * 2]
                for i in range(0, len(csi_pairs), 2):
                    magnitudes.append(math.hypot(csi_pairs[i], csi_pairs[i + 1]))

                # 构建参数字典
                new_params = {
                    'mac': packet['mac'][:17],
                    'rate': f"{packet['rate']} Mbps",
                    'mcs': packet['mcs'],
                    'channel': packet['channel'],
                    'rssi': f"{packet['rssi']} dBm",
                    'noise_floor': f"{packet['noise_floor']} dBm",
                    'bandwidth': f"{packet['bandwidth']} MHz" if packet['bandwidth'].isdigit() else "N/A",
                    'sampling_frequency': 'N/A'
                }

                with global_data['lock']:
                    if magnitudes:
                        global_data['csi_mag'].extend(magnitudes)
                        # 为推理添加到缓冲区
                        global_data['inference_buffer']['amplitudes'].append(magnitudes)
                        global_data['inference_buffer']['timestamps'].append(sys_timestamp)

                        # 限制推理缓冲区大小（保留最近30秒的数据）
                        max_buffer_size = int(30 / WINDOW_STRIDE_SEC * 20)  # 假设20Hz采样率
                        if len(global_data['inference_buffer']['amplitudes']) > max_buffer_size:
                            global_data['inference_buffer']['amplitudes'] = \
                                global_data['inference_buffer']['amplitudes'][-max_buffer_size:]
                            global_data['inference_buffer']['timestamps'] = \
                                global_data['inference_buffer']['timestamps'][-max_buffer_size:]

                    global_data['rssi'].append(float(packet['rssi']))
                    global_data['sys_timestamps'].append(sys_timestamp)  # 存储系统时间
                    global_data['params'].update(new_params)

                # 保存数据到CSV文件
                save_data_to_csv(packet, sys_timestamp, magnitudes)

            except Exception as e:
                print(f"Parse error: {str(e)}")
                continue

        except socket.timeout:
            pass
        except Exception as e:
            print(f"Network error: {str(e)}")


# ========================
# 程序退出清理
# ========================
def cleanup():
    """程序退出时执行清理操作"""
    if global_data['current_csv_file'] is not None:
        try:
            global_data['current_csv_file'].close()
            print("CSV data file closed")
        except Exception as e:
            print(f"Error closing data file: {str(e)}")


# ========================
# 可视化系统
# ========================
def init_plots():
    fig = plt.figure(figsize=(16, 9), dpi=100)
    gs = GridSpec(4, 1, height_ratios=[2, 2, 1, 1.5])

    # CSI幅度图
    ax1 = fig.add_subplot(gs[0])
    csi_line, = ax1.plot([], [], 'deepskyblue', lw=1)
    ax1.set_title("CSI Amplitude", fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.7)

    # RSSI曲线图
    ax2 = fig.add_subplot(gs[1])
    rssi_line, = ax2.plot([], [], 'tomato', lw=1.2)
    ax2.set_title("RSSI Variation", fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.7)

    # 推理结果
    ax3 = fig.add_subplot(gs[2])
    ax3.set_title("Human Detection", fontsize=12)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_xticks([])
    ax3.set_yticks([])
    inference_text = ax3.text(0.5, 0.5, "Waiting for inference...",
                              horizontalalignment='center',
                              verticalalignment='center',
                              fontsize=16, color='black',
                              bbox=dict(facecolor='white', alpha=0.8))

    # 参数表格
    ax4 = fig.add_subplot(gs[3])
    ax4.axis('off')
    table = ax4.table(
        cellText=[[global_data['params'][col] for col in TABLE_COLS]],
        colLabels=TABLE_COLS,
        colColours=['#F0F0F0'] * len(TABLE_COLS),
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)

    # 存储单元格引用
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='black')
            cell.set_facecolor('#E0E0E0')
        else:
            global_data['table_cells'][TABLE_COLS[col]] = cell

    global_data['table_initialized'] = True
    plt.subplots_adjust(hspace=0.4, left=0.05, right=0.95)
    return fig, ax1, ax2, ax3, ax4, csi_line, rssi_line, inference_text, table


# ========================
# 可视化更新
# ========================
def update_plots(frame):
    with global_data['lock']:
        csi_data = list(global_data['csi_mag'])
        rssi_data = list(global_data['rssi'])
        sys_timestamps = list(global_data['sys_timestamps'])
        params = global_data['params'].copy()
        inference_result = global_data['inference_result'].copy()

    # 计算采样频率（基于系统时间）
    if len(sys_timestamps) > 1:
        intervals = []
        for i in range(len(sys_timestamps) - 1):
            delta = sys_timestamps[i + 1] - sys_timestamps[i]
            if 0 < delta < 10:  # 过滤异常间隔
                intervals.append(delta)
        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            params['sampling_frequency'] = f"{1 / avg_interval:.2f} Hz"
        else:
            params['sampling_frequency'] = "N/A"
    else:
        params['sampling_frequency'] = "N/A"

    # 更新图表
    if csi_data:
        ax1.set_xlim(0, len(csi_data))
        ax1.set_ylim(min(csi_data) * 0.8, max(csi_data) * 1.2)
        csi_line.set_data(range(len(csi_data)), csi_data)
    if rssi_data:
        ax2.set_xlim(0, len(rssi_data))
        ax2.set_ylim(min(rssi_data) - 5, max(rssi_data) + 5)
        rssi_line.set_data(range(len(rssi_data)), rssi_data)

    # 更新表格
    if global_data['table_initialized']:
        for col in TABLE_COLS:
            cell = global_data['table_cells'].get(col)
            if cell:
                cell.get_text().set_text(params.get(col, 'N/A'))

    # 更新推理结果
    if inference_result.get('last_update', 0) > 0:
        time_since_update = time.time() - inference_result['last_update']
        if time_since_update < 10:  # 最近10秒内的结果
            status = "Present" if inference_result['has_person'] else "Empty"
            confidence = inference_result.get('confidence', 0)
            prob = inference_result.get('probability', 0)

            # 根据结果设置颜色
            if inference_result['has_person']:
                color = 'green'
                bg_color = 'lightgreen'
            else:
                color = 'red'
                bg_color = 'mistyrose'

            inference_text.set_text(f"{status}\nProb: {prob:.3f}, Conf: {confidence:.2f}")
            inference_text.set_color(color)
            inference_text.set_bbox(dict(facecolor=bg_color, alpha=0.8, edgecolor=color))
        else:
            # 推理结果太旧
            inference_text.set_text("Waiting for new inference...")
            inference_text.set_color('gray')
            inference_text.set_bbox(dict(facecolor='white', alpha=0.8))

    # 控制台输出（每5次更新一次）
    global_data['print_counter'] += 1
    if global_data['print_counter'] % 5 == 0:
        table = PrettyTable(TABLE_COLS)
        table.add_row([params[col] for col in TABLE_COLS])
        print("\n" + "=" * 120)
        print(f"System time: {time.time():.3f}")
        print(table)
        print("=" * 120 + "\n")

    return csi_line, rssi_line, inference_text


# ========================
# 主程序
# ========================
if __name__ == "__main__":
    try:
        # 显示数据保存信息
        print(f"WiFi CSI Data Receiver and Human Detection System")
        print(f"Data will be saved every {SAVE_INTERVAL} seconds")
        print(f"Save directory: {os.path.abspath(DATA_SAVE_DIR)}")
        print(f"Inference interval: {INFERENCE_INTERVAL} seconds")

        # 启动UDP接收线程
        recv_thread = threading.Thread(target=udp_receiver, daemon=True)
        recv_thread.start()

        # 启动推理线程
        infer_thread = threading.Thread(target=run_inference, daemon=True)
        infer_thread.start()

        # 初始化可视化
        fig, ax1, ax2, ax3, ax4, csi_line, rssi_line, inference_text, table = init_plots()

        ani = animation.FuncAnimation(
            fig, update_plots,
            interval=REFRESH_INTERVAL,
            blit=True,
            cache_frame_data=False
        )

        plt.show()
    except KeyboardInterrupt:
        print("Program interrupted by user")
    finally:
        # 执行清理操作
        cleanup()
        print("Program exited")