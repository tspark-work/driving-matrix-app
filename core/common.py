import platform
import numpy as np
import matplotlib.pyplot as plt

FLOAT32_COLUMNS = [
    "speed", "accXG", "accYG", "accZG",
    "yawDps", "pitchDps", "rollDps",
]

def set_korean_font():
    sys_plat = platform.system()
    if sys_plat == "Windows":
        plt.rc("font", family="Malgun Gothic")
    elif sys_plat == "Darwin":
        plt.rc("font", family="AppleGothic")
    else:
        plt.rc("font", family="NanumGothic")
    plt.rcParams["axes.unicode_minus"] = False

def get_speed_weight(speed):
    if speed <= 30:
        return 1.0
    if speed <= 80:
        return 1.2
    if speed <= 120:
        return 1.5
    return 2.0

def downsample_for_plot(data, max_points=15000):
    """분석 원본은 유지하고 화면 표시용 데이터만 균등 간격으로 축소합니다."""
    if data is None or len(data) <= max_points:
        return data
    step = max(1, int(np.ceil(len(data) / max_points)))
    return data.iloc[::step]
