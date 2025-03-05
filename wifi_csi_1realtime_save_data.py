import socket
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import math
from collections import deque
from prettytable import PrettyTable
import time
import os
import csv
import datetime

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
    'save_data_buffer': []  # 缓存要保存的数据
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

    print(f"创建新数据文件: {filename}")
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
                print(f"已关闭数据文件，持续时间: {time.time() - global_data['current_file_time']:.1f}秒")

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
# UDP数据接收线程
# ========================
def udp_receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    sock.settimeout(0.1)

    print(f"UDP接收器已启动，监听端口: {UDP_PORT}")
    print(f"数据将保存到: {os.path.abspath(DATA_SAVE_DIR)} 目录")

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
                    global_data['rssi'].append(float(packet['rssi']))
                    global_data['sys_timestamps'].append(sys_timestamp)  # 存储系统时间
                    global_data['params'].update(new_params)

                # 保存数据到CSV文件
                save_data_to_csv(packet, sys_timestamp, magnitudes)

                # 调试输出
                print(f"接收到数据包: RSSI={packet['rssi']}, 系统时间={sys_timestamp:.3f}")

            except Exception as e:
                print(f"解析错误: {str(e)}")
                continue

        except socket.timeout:
            pass
        except Exception as e:
            print(f"网络错误: {str(e)}")


# ========================
# 程序退出清理
# ========================
def cleanup():
    """程序退出时执行清理操作"""
    if global_data['current_csv_file'] is not None:
        try:
            global_data['current_csv_file'].close()
            print("已关闭CSV数据文件")
        except Exception as e:
            print(f"关闭数据文件出错: {str(e)}")


# ========================
# 可视化系统
# ========================
def init_plots():
    fig = plt.figure(figsize=(16, 9), dpi=100)
    gs = GridSpec(3, 1, height_ratios=[2, 2, 1.5])

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

    # 参数表格
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')
    table = ax3.table(
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
    plt.subplots_adjust(hspace=0.35, left=0.05, right=0.95)
    return fig, ax1, ax2, ax3, csi_line, rssi_line, table


# ========================
# 可视化更新
# ========================
def update_plots(frame):
    with global_data['lock']:
        csi_data = list(global_data['csi_mag'])
        rssi_data = list(global_data['rssi'])
        sys_timestamps = list(global_data['sys_timestamps'])
        params = global_data['params'].copy()

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
        ax1.set_ylim(min(csi_data) * 1.2, max(csi_data) * 1.2)
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

    # 控制台输出（每5次更新一次）
    global_data['print_counter'] += 1
    if global_data['print_counter'] % 5 == 0:
        table = PrettyTable(TABLE_COLS)
        table.add_row([params[col] for col in TABLE_COLS])
        print("\n" + "=" * 120)
        print(f"系统时间: {time.time():.3f}")
        print(table)
        print("=" * 120 + "\n")

    return csi_line, rssi_line


# ========================
# 主程序
# ========================
if __name__ == "__main__":
    try:
        # 显示数据保存信息
        print(f"WiFi CSI 数据接收与可视化")
        print(f"数据将每 {SAVE_INTERVAL} 秒(5分钟)保存到新文件")
        print(f"保存目录: {os.path.abspath(DATA_SAVE_DIR)}")

        recv_thread = threading.Thread(target=udp_receiver, daemon=True)
        recv_thread.start()

        fig, ax1, ax2, ax3, csi_line, rssi_line, table = init_plots()

        ani = animation.FuncAnimation(
            fig, update_plots,
            interval=REFRESH_INTERVAL,
            blit=True,
            cache_frame_data=False
        )

        plt.show()
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        # 执行清理操作
        cleanup()
        print("程序已退出")