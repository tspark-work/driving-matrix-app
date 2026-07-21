#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import platform
import plotly.graph_objects as go
import plotly.express as px
# from score_driving import score_driving_style, ScoreConfig
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import medfilt
import gc
from ui_components import render_tire_gain_inputs, render_wear_comparison_chart
import logging
import altair as alt

plt.close('all')
gc.collect()

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- 0. 환경 설정 ---
def set_korean_font():
    sys_plat = platform.system()
    if sys_plat == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif sys_plat == 'Darwin':
        plt.rc('font', family='AppleGothic')
    else:
        plt.rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()
st.set_page_config(page_title="차량 IMU 통합 분석", layout="wide", page_icon="🏎️")

# CSS 최적화 (Style 가독성)
st.markdown("""
    <style>
    .stApp h1 { font-size: 2rem !important; font-weight: 700; }
    .stApp h2 { font-size: 1.6rem !important; }
    .stApp h3 { font-size: 1.3rem !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚗 차량 IMU 데이터 통합 분석 시스템")
st.markdown("""주행 데이터 파일을 통해 차량의 IMU 데이터를 분석하고, 다양한 시각화 및 통계 정보를 제공합니다.""")
st.write("---")

case_options = {
    "Case 1": "⬆️ 정방향",
    "Case 2": "⬇️ 180도 회전"
}

selected_case = st.radio(
    "**무선통신기 설치 방향**",
    options=list(case_options.keys()),
    format_func=lambda x: case_options[x],
    horizontal=True,
)

if selected_case == "Case 1":
    st.caption("✅ **정방향 상태**: 안테나가 차량 후면을 향하고 있습니다. (IMU 데이터 보정 없음)")
else:
    st.caption("🔄 **180도 회전 상태**: 안테나가 전면 유리를 향하고 있습니다. (IMU 데이터 반전 적용)")

def get_speed_weight(speed):
    if speed <= 30: return 1.0
    elif speed <= 80: return 1.2
    elif speed <= 120: return 1.5
    else: return 2.0

# --- 1. 데이터 로드 및 전처리 (캐싱 최적화) ---
def load_data(file):
    try:
        target_columns = ['packetBodyDrivingId', 'dataTime', 'speed', 'accXG', 'accYG', 'accZG', 'yawDps', 'pitchDps', 'rollDps']

        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file, sheet_name='PacketBodyDriving', usecols=lambda c: c in target_columns, engine='calamine')
        else:
            df = pd.read_csv(file, engine='pyarrow', usecols=lambda c: c in target_columns)

        if df.empty:
            st.warning(f"⚠️ {file.name}: 파일에 데이터가 없습니다.")
            return None

        if 'dataTime' not in df.columns:
            st.error(f"❌ {file.name}: 'dataTime' 컬럼을 찾을 수 없습니다.")
            return None

        dt_col = pd.to_datetime(df['dataTime'], errors='coerce', format='%Y-%m-%d %H:%M:%S') # 잘못된 형식은 NaT로 변환

        if dt_col.isna().all():
            st.error(f"❌ {file.name}: 'dataTime' 형식이 올바르지 않습니다.")
            return None

        df['dataTime'] = dt_col
        df['date'] = dt_col.dt.date
        # df['month'] = dt_col.dt.to_period('M').astype(str)
        # df['week'] = "Week " + dt_col.dt.isocalendar().week.astype(str)
        # df['day_name'] = dt_col.dt.day_name()
        # df['overall'] = "전체 기간 (Overall)"
        df['month'] = dt_col.dt.to_period('M').astype(str).astype('category')
        df['week'] = ("Week " + dt_col.dt.isocalendar().week.astype(str)).astype('category')
        df['day_name'] = dt_col.dt.day_name().astype('category')
        df['overall'] = pd.Series(["전체 기간 (Overall)"] * len(df), dtype='category')

        # float64 타입을 float32로 변환하여 메모리 반토막 내기
        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype('float32')

        return df

    except Exception as e:
        st.error(f"🔥 파일 로드 중 오류 발생: {e}")
        return None

# def get_driving_scores_vectorized(data, g_limit):
#     """
#     DSN(Driving Severity Number) 이론을 적용한 벡터화 점수 계산
#     원리: 에너지는 가속도의 제곱(G^2)에 비례하며, 속도가 높을수록 마찰 에너지가 증폭됨
#     """
#     if data.empty:
#         return [0, 0, 0, 0, 0, 0]

#     # 1. 속도 보정 계수 (논문의 속도-에너지 상관관계 반영)
#     # 고속 주행 시 동일 가속도에서도 마찰 에너지 전달률이 높아짐
#     v_ref = 100 # 기준 속도 (km/h)
#     speed_factor = (data['speed'] / v_ref).clip(lower=1.0)

#     # 2. DSN 산출 (에너지 관점: G^2 * Speed_Factor)
#     # 단순 G값이 아닌 G의 제곱을 사용하여 급격한 조작에 높은 페널티 부여
#     dsn_x = (data['accXG']**2) * speed_factor
#     dsn_y = (data['accYG']**2) * speed_factor

#     def calculate_dsn_score(dsn_series, limit):
#         if dsn_series.empty: return 0
#         # 논문에 따라 에너지 총합 또는 상위 구간의 에너지 밀도를 분석
#         # g_limit 역시 제곱하여 에너지 단위(G^2)로 비교
#         energy_score = (dsn_series.quantile(0.9) / (limit**2)) * 100
#         # 시각적 분별력을 위해 비선형 보정 적용
#         return np.clip(np.sqrt(energy_score / 100) * 100, 0, 100)

#     # 3. 항목별 가혹도 산출 (특허 및 논문의 가중치 적용)
#     # 횡방향(Lateral)은 종방향보다 마모 유발 효율이 높으므로 sens_limit을 낮게 설정
#     sens_limit_x = g_limit * 0.6
#     sens_limit_y = g_limit * 0.4 # 횡방향에 더 민감하게 반응

#     res = {
#         'accel': calculate_dsn_score(dsn_x[data['accXG'] > 0], sens_limit_x),
#         'brake': calculate_dsn_score(dsn_x[data['accXG'] < 0], sens_limit_x),
#         'right': calculate_dsn_score(dsn_y[data['accYG'] > 0], sens_limit_y),
#         'left': calculate_dsn_score(dsn_y[data['accYG'] < 0], sens_limit_y),
#         'stability': np.clip((data[['accXG', 'accYG']].var().mean() / (sens_limit_x**2 * 0.1)) * 100, 0, 100),
#         'speeding': np.clip(((data['speed'].quantile(0.95) - 110) / 30) * 100, 0, 100)
#     }

#     # 거미줄 차트 순환 순서: 급가속 - 과속 - 우회전 - 안정성 - 좌회전 - 급제동
#     return [np.nan_to_num(res[k]) for k in ['accel', 'speeding', 'right', 'stability', 'left', 'brake']]

# def get_driving_scores_vectorized(data, g_limit):
#     if data.empty:
#         return [0, 0, 0, 0, 0, 0]

#     # 민감도 보정 계수 (낮은 점수대를 넓게 폄)
#     sens_limit = g_limit * 0.5

#     def boost_score(val):
#         return np.clip(np.sqrt(val / 100) * 100, 0, 100)

#     # 과속 점수 (110km/h 초과 시 점수 부여)
#     speed_limit = 110
#     over_speed = data.loc[data['speed'] > speed_limit, 'speed']
#     speeding_score = 0
#     if not over_speed.empty:
#         # 110~140km/h 구간을 0~100점으로 환산
#         raw_speed_score = ((over_speed.quantile(0.9) - speed_limit) / (140 - speed_limit)) * 100
#         speeding_score = boost_score(raw_speed_score)

#     res = {
#         'accel': boost_score((data.loc[data['accXG'] > 0, 'accXG'].quantile(0.9) / sens_limit) * 100),
#         'brake': boost_score((data.loc[data['accXG'] < 0, 'accXG'].abs().quantile(0.9) / sens_limit) * 100),
#         'right': boost_score((data.loc[data['accYG'] > 0, 'accYG'].quantile(0.9) / sens_limit) * 100),
#         'left' : boost_score((data.loc[data['accYG'] < 0, 'accYG'].abs().quantile(0.9) / sens_limit) * 100),
#         'stability': boost_score((data[['accXG', 'accYG']].std().mean() / (sens_limit * 0.3)) * 100),
#         'speeding': speeding_score
#     }

#     # 6각형 배치를 위한 순서 정렬
#     categories = ['급가속', '과속', '우회전', '불안정성', '좌회전', '급제동']
#     return [np.nan_to_num(res[k]) for k in ['accel', 'speeding', 'right', 'stability', 'left', 'brake']]

def calc_rms(series):
    x = series.to_numpy(dtype=float)
    x = x[~np.isnan(x)] # NaN 제거
    if len(x) == 0: return 0.0
    return np.sqrt(np.mean(x**2))

def get_speed_distribution(df):
    if "speed" not in df.columns or df.empty:
        return 0.0, 0.0, 0.0
    s = df["speed"].dropna()
    n = len(s)
    if n == 0: return 0.0, 0.0, 0.0
    p0_40 = (s < 40).sum() / n * 100
    p40_80 = ((s >= 40) & (s < 80)).sum() / n * 100
    p80_up = (s >= 80).sum() / n * 100
    return p0_40, p40_80, p80_up

# --- 3. 사이드바 구성 ---
if 'all_data' not in st.session_state:
    st.session_state.all_data = pd.DataFrame()
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

def clear_all_data():
    st.cache_data.clear()
    st.cache_resource.clear()  # 리소스 캐시까지 함께 청소

    analysis_keys = [
        'all_data', 'tab4_result', 'tab3_raw_rms', 'tab3_rms_dict',
        'tab3_dist', 'eval_data', 'last_analysis_trigger',
        'df', 'summary'
    ]

    for key in analysis_keys:
        if key in st.session_state:
            st.session_state[key] = pd.DataFrame() # 빈 객체로 대체하여 기존 대용량 메모리 링크 파괴
            del st.session_state[key]

    st.session_state.uploader_key += 1
    gc.collect()
    gc.collect(0)
    gc.collect(1)
    gc.collect(2) # 가장 오래 살아남은 메모리(Generation 2) 영역까지 싹 청소

with st.sidebar:
    st.header("⚙️ 데이터 관리")
    if st.button("🗑️ 모든 데이터 초기화", help="업로드된 모든 주행 데이터를 삭제합니다."):
        clear_all_data()
        st.rerun() # 화면 즉시 갱신
    uploaded_files = st.sidebar.file_uploader(
        "주행 데이터 업로드",
        type=['csv', 'xlsx'],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )

st.sidebar.header("⚙️ 분석 설정")

@st.cache_data(show_spinner=False, max_entries=1)
def get_integrated_data(_files, file_keys):  # _files: 실제 파일 객체 (캐시 키 제외), file_keys: 캐시 키용 튜플
    df_list = []
    total_files = len(_files)
    progress_bar = st.progress(0, text="데이터 통합을 준비 중입니다...")

    for i, file in enumerate(_files):          # ← _files 사용
        percent_complete = int(((i + 1) / total_files) * 100)
        progress_bar.progress(
            percent_complete,
            text=f"데이터 추출 중... ⏳ ({i+1}/{total_files} 파일 완료 - {percent_complete}%)"
        )

        df_temp = load_data(file)
        if df_temp is not None and 'packetBodyDrivingId' in df_temp.columns:
            df_list.append(df_temp)

        del df_temp
        gc.collect()

    if not df_list:
        progress_bar.empty()
        return None

    progress_bar.progress(100, text="✅ 파일 읽기 완료! 데이터를 하나로 병합하고 있습니다...")

    raw_df = pd.concat(df_list, ignore_index=True)
    del df_list
    gc.collect()

    target_col = 'packetBodyDrivingId'
    raw_df = raw_df.dropna(subset=[target_col])
    raw_df = raw_df.drop_duplicates(subset=[target_col]).sort_values(target_col).reset_index(drop=True)

    progress_bar.empty()
    return raw_df


if uploaded_files:
    with st.status("데이터 통합 및 전처리 중...", expanded=True) as status:
        # 캐싱된 통합 함수 호출
        file_keys = tuple((f.name, f.size) for f in uploaded_files)
        raw_df = get_integrated_data(uploaded_files, file_keys)

        if raw_df is None or raw_df.empty:
            status.update(label="통합 실패: 유효한 파일이 없습니다.", state="error")
            st.error("❌ 분석할 수 있는 정상적인 데이터가 없습니다.")
            st.stop()

        status.update(label=f"통합 완료! (총 {len(uploaded_files)}개 파일 성공)", state="complete")

    # 설정값 (UI)
    analysis_unit = st.sidebar.selectbox("분석 단위", ["전체 단위 (Overall)", "일 단위 (Daily)", "주 단위 (Weekly)", "요일 단위 (Day of Week)", "월 단위 (Monthly)"])
    # speed_min = st.sidebar.slider("최소 속도 (km/h)", 0, 100, 1)
    speed_range = st.sidebar.slider(
        "분석 속도 범위 설정 (km/h)",
        0, 200,            # 슬라이더의 전체 범위 (Min, Max)
        (1, 200),          # 초기 선택값
        step=1
    )
    speed_min, speed_max = speed_range
    st.sidebar.caption(f"📊 설정된 범위: {speed_min}km/h ~ {speed_max}km/h")
    r_max = st.sidebar.slider("Dps 최대값", 10.0, 1000.0, 500.0, step=10.0)
    g_max = st.sidebar.slider("G 최대값", 0.1, 2.0, 1.0)
    g_limit = st.sidebar.slider("G-Limit 범위", 0.1, 2.0, 0.5)

    grid_val = st.sidebar.select_slider(
        "Select Matrix Resolution",
        options=[3, 5, 7, 9],
        value=5
    )

    # 마모 임계값
    h_acc, h_brk, h_trn = st.sidebar.columns(3)
    hard_accel_threshold = h_acc.number_input("급가속 G", 0.0, 1.0, 0.05)
    hard_brake_threshold = h_brk.number_input("급제동 G", 0.0, 1.0, 0.05)
    hard_turn_threshold = h_trn.number_input("급선회 G", 0.0, 1.0, 0.05)

    unit_map = {"일 단위 (Daily)": 'date', "주 단위 (Weekly)": 'week', "요일 단위 (Day of Week)": 'day_name', "월 단위 (Monthly)": 'month', "전체 단위 (Overall)": 'overall'}
    group_col = unit_map[analysis_unit]
    group_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'] if group_col == 'day_name' else sorted(raw_df[group_col].unique())

    # 필터링
    @st.cache_data(show_spinner=False, max_entries=1)
    def process_global_dataframe(raw_data, s_min, s_max, g_lim, r_lim, case_sel, g_col, h_acc, h_brk, h_trn):
        mask = ((raw_data['speed'] >= s_min) & (raw_data['speed'] <= s_max) &
                (raw_data['accXG'].abs() <= g_lim) & (raw_data['accYG'].abs() <= g_lim) &
                (raw_data['yawDps'].abs() <= r_lim) & (raw_data['pitchDps'].abs() <= r_lim) & (raw_data['rollDps'].abs() <= r_lim))
        filtered = raw_data.loc[mask]

        # 2. Case 2 반전 처리
        if case_sel == "Case 2":
            cols_to_flip = ['accXG', 'accYG', 'yawDps', 'pitchDps', 'rollDps']
            filtered = filtered.copy()
            filtered[cols_to_flip] *= -1
        else:
            filtered = filtered.copy()

        filtered['is_accel'] = filtered['accXG'] > h_acc
        filtered['is_brake'] = filtered['accXG'] < -h_brk
        filtered['is_turn_L'] = filtered['accYG'] > h_trn
        filtered['is_turn_R'] = filtered['accYG'] < -h_trn
        filtered['is_turn_any'] = filtered['accYG'].abs() > h_trn

        # 4. 그룹바이 요약 (median 사용)
        sum_df = filtered.groupby(g_col, observed=True).agg(
            평균속도=('speed', 'mean'),
            최대분포속도=('speed', 'median'),
            최대속도=('speed', 'max'),
            급가속횟수=('is_accel', 'sum'),
            급제동횟수=('is_brake', 'sum'),
            급선회횟수=('is_turn_any', 'sum'),
            데이터수=('dataTime', 'count')
        )

        # 5. 비율 계산
        sum_df['가속_비율'] = (filtered.groupby(g_col, observed=True)['is_accel'].mean() * 100).round(1)
        sum_df['감속_비율'] = (filtered.groupby(g_col, observed=True)['is_brake'].mean() * 100).round(1)
        sum_df['정속_비율'] = (100 - sum_df['가속_비율'] - sum_df['감속_비율']).round(1)

        sum_df['좌선회_비율'] = (filtered.groupby(g_col, observed=True)['is_turn_L'].mean() * 100).round(1)
        sum_df['우선회_비율'] = (filtered.groupby(g_col, observed=True)['is_turn_R'].mean() * 100).round(1)
        sum_df['직진_비율'] = (100 - sum_df['좌선회_비율'] - sum_df['우선회_비율']).round(1)

        # 마모지수 산출 (가중치 적용)
        sum_df['마모지수'] = (
            (sum_df['급가속횟수'] * 0.5 + sum_df['급제동횟수'] * 0.7 + sum_df['급선회횟수'] * 1.0) /
            sum_df['데이터수'].replace(0, 1) * 1000
        ).round(2)

        return filtered, sum_df

    df, summary = process_global_dataframe(
        raw_df, speed_min, speed_max, g_max, r_max, selected_case,
        group_col, hard_accel_threshold, hard_brake_threshold, hard_turn_threshold
    )

    avg_wear_idx = float(summary['마모지수'].mean()) if not summary.empty else 1.0

    # 🚨 탭2 마크다운용 변수 복구
    acc_threshold = hard_accel_threshold
    brk_threshold = hard_brake_threshold
    turn_threshold = hard_turn_threshold

    # --- 4. 메인 분석 화면 ---
    # tab1, tab2, tab3, tab4 = st.tabs(["📈 시각화 분석", "🔢 데이터 통계", "🛞 마모 인자 분석", "💥 운전 가혹도 분석"])
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 시각화 분석", "🔢 데이터 통계", "🛞 마모 인자 분석", "💥 운전 가혹도 분석", "타이어 마모 예측"])

    with tab1:
        for item in group_list:
            plot_data = df[df[group_col] == item]
            if plot_data.empty: continue

            with st.expander(f"📍 {item} 리포트 ({len(plot_data):,} 샘플)", expanded=True):
                c1, c2, c3 = st.columns(3)
                # c1, c2, c3, c4= st.columns(4)

                # 속도 히스토그램
                with c1:
                    fig, ax = plt.subplots(figsize=(4, 4))
                    st.markdown("<h3 style='text-align: center; margin-bottom: 0px;'>🎢 주행 속도 분포</h3>", unsafe_allow_html=True)
                    sns.histplot(plot_data['speed'], bins=20, kde=True, color='skyblue', ax=ax)
                    avg_speed = plot_data['speed'].mean()

                    if not plot_data['speed'].dropna().empty:
                        max_dist_speed = plot_data['speed'].round(1).mode().iloc[0]
                    else:
                        max_dist_speed = 0

                    ax.axvline(avg_speed, color='red', linestyle='--', linewidth=1.5)
                    ax.text(avg_speed + 2, ax.get_ylim()[1] * 0.9, f'평균: {avg_speed:.1f}',
                            color='red', fontweight='bold', fontsize=9)

                    ax.axvline(max_dist_speed, color='purple', linestyle='-', linewidth=2)
                    ax.text(max_dist_speed + 2, ax.get_ylim()[1] * 0.8, f'최대 분포: {max_dist_speed:.1f}',
                            color='purple', fontweight='bold', fontsize=9)

                    ax.set_xlabel("Speed (km/h)", fontsize=10)
                    ax.set_ylabel("Count", fontsize=10)
                    st.pyplot(fig, width="stretch")
                    plt.close(fig)

                # G-G Map (성능 최적화: 히스토그램2D 기반)
                with c2:
                    boundaries = np.linspace(-g_limit, g_limit, 6)
                    colors = ["#0000FF", "#00FFFF", "#00FF00", "#FFFF00", "#FF0000"]
                    cm = LinearSegmentedColormap.from_list("custom_gg", colors, N=256)
                    fig_g, ax_g = plt.subplots(figsize=(4, 4))
                    st.markdown("<h3 style='text-align: center; margin-bottom: 0px;'>🎯 주행 패턴 맵</h3>", unsafe_allow_html=True)
                    # G-G 다이어그램 히트맵 그리기
                    counts, xedges, yedges = np.histogram2d(
                        plot_data['accYG'], plot_data['accXG'],
                        bins=100, range=[[-g_limit, g_limit], [-g_limit, g_limit]]
                    )
                    counts_smoothed = gaussian_filter(counts, sigma=1.2)
                    counts_smoothed[counts_smoothed < 0.1] = np.nan
                    ax_g.imshow(
                        counts_smoothed.T, origin='lower',
                        extent=[-g_limit, g_limit, -g_limit, g_limit],
                        cmap=cm, aspect='equal', interpolation='bilinear'
                    )

                    ax_g.set_xticks(boundaries)
                    ax_g.set_yticks(boundaries)

                    tick_labels = [f"{b:.1f}" for b in boundaries]
                    ax_g.set_xticklabels(tick_labels, fontsize=9)
                    ax_g.set_yticklabels(tick_labels, fontsize=9)

                    ax_g.set_facecolor('#000033')
                    ax_g.axhline(0, color='white', alpha=0.5, linewidth=0.8)
                    ax_g.axvline(0, color='white', alpha=0.5, linewidth=0.8)

                    for b in boundaries:
                        ax_g.axhline(b, color='white', alpha=0.1, linewidth=0.5)
                        ax_g.axvline(b, color='white', alpha=0.1, linewidth=0.5)

                    ax_g.set_xlabel("Lateral Acceleration (accYG)", fontsize=10)
                    ax_g.set_ylabel("Longitudinal Acceleration (accXG)", fontsize=10)
                    st.pyplot(fig_g, width="stretch")
                    plt.close(fig_g)

                with c3:
                    st.markdown("<h3 style='text-align: center; margin-bottom: 0px;'>🔳 주행 패턴 분포</h3>", unsafe_allow_html=True)
                    boundaries = np.linspace(-g_limit, g_limit, grid_val + 1)
                    bins = [-np.inf] + list(boundaries[1:-1]) + [np.inf]
                    bin_labels = [f"B{i}" for i in range(grid_val)]

                    matrix = pd.crosstab(
                        pd.cut(plot_data['accXG'], bins=bins, labels=bin_labels, duplicates='drop'),
                        pd.cut(plot_data['accYG'], bins=bins, labels=bin_labels, duplicates='drop')
                    ).reindex(index=bin_labels[::-1], columns=bin_labels).fillna(0).astype(int)

                    total_count = matrix.values.sum()
                    fig, ax = plt.subplots(figsize=(4, 4))
                    im = ax.imshow(matrix.values, cmap='RdYlGn_r',
                                extent=[-g_limit, g_limit, -g_limit, g_limit],
                                norm=LogNorm(vmin=1, vmax=matrix.values.max() if matrix.values.max() > 0 else 1))

                    ax.set_xticks(boundaries)
                    ax.set_yticks(boundaries)

                    tick_labels = [f"{b:.1f}" for b in boundaries]
                    ax.set_xticklabels(tick_labels, fontsize=9)
                    ax.set_yticklabels(tick_labels, fontsize=9)

                    bin_centers = (boundaries[:-1] + boundaries[1:]) / 2
                    y_centers = bin_centers[::-1] # Y축 반전 대응
                    offset = (g_limit / grid_val) * 0.7
                    base_font = 10 if grid_val <= 5 else 8

                    for i in range(len(y_centers)):
                        for j in range(len(bin_centers)):
                            val = matrix.iloc[i, j]
                            if val > 0:
                                percentage = (val / total_count) * 100
                                center_idx = grid_val // 2
                                txt_color = "white" if (i == center_idx and j == center_idx) else "black"

                                ax.text(bin_centers[j], y_centers[i] + (offset * 0.2), f"{val:,}",
                                        ha="center", va="center", color=txt_color, fontsize=base_font)

                                ax.text(bin_centers[j], y_centers[i] - (offset * 0.35), f"({percentage:.1f}%)",
                                        ha="center", va="center", color=txt_color, fontsize=base_font-2, alpha=0.8, fontweight='bold')

                    ax.set_aspect('equal')
                    ax.grid(True, which='major', color='white', linestyle='-', linewidth=1.5 if grid_val <= 5 else 0.8, alpha=0.4)
                    ax.set_xlabel("Lateral Acceleration (accYG)", fontsize=10)
                    ax.set_ylabel("Longitudinal Acceleration (accXG)", fontsize=10)

                    st.pyplot(fig, width="stretch")
                    plt.close(fig)

                # with c4:
                #     st.markdown("<h3 style='text-align: center; margin-bottom: 0px;'>⚖️ 자세 거동 분포 (Roll-Pitch)</h3>", unsafe_allow_html=True)

                #     # 그래프 설정
                #     fig_rp, ax_rp = plt.subplots(figsize=(4, 4))

                #     # +-50 범위 설정
                #     limit_val = 50

                #     # 2D Histogram (Heatmap)
                #     # 데이터의 rollDps와 pitchDps를 사용합니다.
                #     counts, xedges, yedges = np.histogram2d(
                #         plot_data['rollDps'], plot_data['pitchDps'],
                #         bins=60, range=[[-limit_val, limit_val], [-limit_val, limit_val]]
                #     )

                #     # 부드러운 시각화를 위해 가우시안 필터 적용
                #     # from scipy.ndlocal.filters import gaussian_filter
                #     # counts_smoothed = gaussian_filter(counts, sigma=1.0)
                #     # counts_smoothed[counts_smoothed < 0.1] = np.nan # 데이터 없는 구간 투명화

                #     # 히트맵 출력
                #     ax_rp.imshow(
                #         counts_smoothed.T, origin='lower',
                #         extent=[-limit_val, limit_val, -limit_val, limit_val],
                #         cmap='magma', aspect='equal', interpolation='bilinear'
                #     )

                #     # 가이드라인 (십자선 및 그리드)
                #     ax_rp.axhline(0, color='white', alpha=0.5, linewidth=1)
                #     ax_rp.axvline(0, color='white', alpha=0.5, linewidth=1)
                #     ax_rp.set_facecolor('#000022') # 어두운 배경으로 대비 강화

                #     # 라벨 및 틱 설정
                #     ax_rp.set_xlabel("Roll Rate (dps)", fontsize=9)
                #     ax_rp.set_ylabel("Pitch Rate (dps)", fontsize=9)
                #     ax_rp.set_xticks([-50, -25, 0, 25, 50])
                #     ax_rp.set_yticks([-50, -25, 0, 25, 50])
                #     ax_rp.tick_params(labelsize=8)

                #     st.pyplot(fig_rp, width="stretch")
                #     plt.close(fig_rp)

                del plot_data  # 루프 끝에 추가
                gc.collect()

    with tab2:
        st.markdown("### 🔢 주행 그룹별 가혹도 및 거동 비율")
        display_cols = [
            '평균속도', '최대분포속도', '최대속도',
            '가속_비율', '정속_비율', '감속_비율',
            '좌선회_비율', '직진_비율', '우선회_비율',
            '데이터수'
        ]

        if group_col == 'day_name':
            display_summary = summary.reindex(group_list)[display_cols]
        else:
            display_summary = summary[display_cols]

        st.dataframe(
            display_summary.style
            .format("{:.1f} km/h", subset=['평균속도', '최대분포속도', '최대속도'])
            .format("{:,}", subset=['데이터수'])
            .format("{:.1f}%", subset=[c for c in display_summary.columns if '비율' in c])
        )
        with st.expander("ℹ️ 분석 기준 및 비율 계산 로직 안내", expanded=False):
            st.markdown(f"""
            본 대시보드의 거동 비율은 사이드바에서 설정하신 임계값(급가속G, 급제동G, 급선회G)을 기준으로 산출됩니다.

            **1. 종방향 거동 (가속/정속/감속)**
            * **가속**: 앞뒤 가속도(`accXG`) > `{acc_threshold}G` (설정된 급가속 기준)
            * **감속**: 앞뒤 가속도(`accXG`) < `-{brk_threshold}G` (설정된 급제동 기준)
            * **정속**: 가속과 감속 사이의 모든 구간 (미세한 속도 변화 포함)

            **2. 횡방향 거동 (좌선회/직진/우선회)**
            * **좌선회**: 좌우 가속도(`accYG`) > `{turn_threshold}G` (설정된 급선회 기준)
            * **우선회**: 좌우 가속도(`accYG`) < `-{turn_threshold}G`
            * **직진**: 좌우 흔들림이 `{turn_threshold}G` 이내인 구간

            **3. 비율 계산 공식**
            * 각 항목의 비율(%) = (해당 거동 데이터 수 / 전체 데이터 수) × 100
            """)
        st.divider()
        if not display_summary.empty:
            radar_data = display_summary[['가속_비율', '정속_비율', '감속_비율',
                                        '좌선회_비율', '직진_비율', '우선회_비율']].mean()

            c_long, c_lat = st.columns(2)

            with c_long:
                fig_donut_long = go.Figure(data=[go.Pie(
                    labels=['가속', '정속', '감속'],
                    values=[radar_data['가속_비율'], radar_data['정속_비율'], radar_data['감속_비율']],
                    hole=.5,
                    marker_colors=['#EA4335', '#34A853', '#FBBC04'],
                    textinfo='label+percent',
                    hoverinfo='label+value'
                )])
                fig_donut_long.update_layout(
                    # 제목에 사이드바에서 선택한 분석 단위(예: 요일별 분석)를 표시
                    title=dict(text=f"전체 {analysis_unit} 종방향 비중", x=0.5, y=0.95),
                    showlegend=False,
                    margin=dict(t=50, b=20, l=20, r=20),
                    width=400,
                    height=350
                )
                st.plotly_chart(fig_donut_long)

            with c_lat:
                fig_donut_lat = go.Figure(data=[go.Pie(
                    labels=['좌선회', '직진', '우선회'],
                    values=[radar_data['좌선회_비율'], radar_data['직진_비율'], radar_data['우선회_비율']],
                    hole=.5,
                    marker_colors=['#4285F4', '#BDC1C6', '#8AB4F8'],
                    textinfo='label+percent',
                    hoverinfo='label+value'
                )])
                fig_donut_lat.update_layout(
                    title=dict(text=f"전체 {analysis_unit} 횡방향 비중", x=0.5, y=0.95),
                    showlegend=False,
                    margin=dict(t=50, b=20, l=20, r=20),
                    width=400,
                    height=350
                )
                st.plotly_chart(fig_donut_lat)

    with tab3:
        st.header("🔬 타이어 마모 인자 분석")

        with st.expander("📝 데이터 전처리 기준", expanded=False):
            st.markdown(f"""
            주행 데이터 분석의 정확도를 높이기 위해 다음과 같은 필터링 조건이 적용되었습니다.

            **1. 최소 속도 필터링 (Speed Threshold)**
            * **현재 기준**: `{speed_min} km/h` 이상인 데이터만 분석에 포함
            * 본 분석에서는 속도가 **`{speed_min} km/h` 미만**인 모든 구간을 '정차' 또는 '유효하지 않은 주행' 구간으로 간주합니다.
            * 신호 대기, 예열, 혹은 차량 정지 상태에서 발생하는 IMU 센서 데이터는 거동 비율 통계(가속/선회 등)에서 제외됩니다.

            **2. 속도 구간의 분류 기준**
            * **시내 (Urban)**: `{speed_min} km/h` ~ `40 km/h` 구간
            * **일반 (Suburban)**: `40 km/h` ~ `80 km/h` 구간
            * **고속 (Highway)**: `80 km/h` 초과 구간

            **3. 구간별 비중(%) 계산 방식**
            * 최소 속도(`{speed_min} km/h`) 필터를 통과한 **전체 유효 주행 데이터 수**를 기준으로 합니다.
            * **공식**: `(해당 구간 데이터 수 / 유효 주행 전체 데이터 수) × 100`
            """)

        st.info("💡 분석할 기간의 시작 날짜와 종료 날짜를 입력하세요. (예: 2025-10-01)")
        mileage_cols = ["시작", "종료"]

        if 'df' in locals() and not df.empty:
            auto_start = df['dataTime'].min().strftime('%Y-%m-%d')
            auto_end = df['dataTime'].max().strftime('%Y-%m-%d')
        else:
            auto_start = ""
            auto_end = ""

        if 'schedule_data' not in st.session_state:
            st.session_state.schedule_data = {"시작": auto_start, "종료": auto_end}
        elif auto_start and (st.session_state.schedule_data["시작"] == "" or st.session_state.get("last_data_count", 0) != len(df)):
            st.session_state.schedule_data = {"시작": auto_start, "종료": auto_end}
            st.session_state.last_data_count = len(df)

        input_df = pd.DataFrame([{
            "시작": st.session_state.schedule_data["시작"],
            "종료": st.session_state.schedule_data["종료"]
        }])

        edited_schedule = st.data_editor(
            input_df,
            column_config={m: st.column_config.TextColumn(m, width="medium") for m in mileage_cols},
            num_rows="fixed",
            width="stretch",
            hide_index=True,
            key="schedule_editor"
        )

        if not edited_schedule.empty:
            start_val = edited_schedule.iloc[0]["시작"]
            end_val = edited_schedule.iloc[0]["종료"]

            if start_val and end_val:
                try:
                    current_trigger = f"{start_val}_{end_val}_{len(df)}"

                    if st.session_state.get("last_analysis_trigger") != current_trigger:
                        if 'dataTime' in df.columns:
                            start_dt = pd.to_datetime(start_val)
                            end_dt = pd.to_datetime(end_val) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

                            mask = (df['dataTime'] >= start_dt) & (df['dataTime'] <= end_dt)
                            filtered_df = df.loc[mask]

                            if not filtered_df.empty:
                                df_run = filtered_df[filtered_df['speed'] > speed_min].copy() if 'speed' in filtered_df.columns else filtered_df.copy()

                                if not df_run.empty:
                                    summary_cols = ["speed", "accXG", "accYG", "accZG", "yawDps", "rollDps", "pitchDps"]
                                    rms_dict = {}
                                    raw_rms_values = {}
                                    for col in summary_cols:
                                        if col in df_run.columns:
                                            val = calc_rms(df_run[col])
                                            raw_rms_values[col] = val
                                            unit = "G" if "acc" in col else "dps" if "Dps" in col else "km/h"
                                            rms_dict[col] = f"{val:.4f} {unit}" if "acc" in col else f"{val:.2f} {unit}"

                                    p0, p40, p80 = get_speed_distribution(df_run)

                                    # 계산된 무거운 결과들을 세션 상태에 모두 저장
                                    st.session_state.tab3_rms_dict = rms_dict
                                    st.session_state.tab3_raw_rms = raw_rms_values
                                    st.session_state.tab3_dist = [{"시내 (0-40)": f"{p0:.1f}%", "일반 (40-80)": f"{p40:.1f}%", "고속 (80+)": f"{p80:.1f}%"}]
                                    st.session_state.tab3_filtered_len = len(filtered_df)
                                    st.session_state.last_analysis_trigger = current_trigger

                    if "tab3_rms_dict" in st.session_state:
                        st.success(f"✅ 구간 분석 완료 (데이터: {st.session_state.tab3_filtered_len}건)")
                        st.divider()

                        # 대시보드 테이블 출력 (단순 출력이라 0초 걸림)
                        rms_horiz_df = pd.DataFrame([st.session_state.tab3_rms_dict])
                        st.markdown("**📍 선택 기간 IMU RMS**")
                        st.dataframe(rms_horiz_df.style.set_properties(**{'text-align': 'center', 'background-color': '#e1f5fe'}), width="stretch", hide_index=True)

                        dist_df = pd.DataFrame(st.session_state.tab3_dist)
                        st.markdown("**🛣️ 선택 기간 주행 속도 비율**")
                        st.dataframe(dist_df.style.set_properties(**{'text-align': 'center'}), width="stretch", hide_index=True)

                        # 예측 모델 영역
                        st.divider()
                        st.markdown("### 🤖 마모량(WI) 예측 모델")

                        g_weights = render_tire_gain_inputs()

                        loc = ["FL", "FR", "RL", "RR"]
                        raw_rms = st.session_state.tab3_raw_rms

                        if 'eval_data' not in st.session_state:
                            st.session_state['eval_data'] = pd.DataFrame({
                                "위치": loc,
                                "실측 마모값": [0.0] * len(loc),
                                "speed": [raw_rms.get('speed', 0)] * len(loc),
                                "accXG": [raw_rms.get('accXG', 0)] * len(loc),
                                "accYG": [raw_rms.get('accYG', 0)] * len(loc),
                                "accZG": [raw_rms.get('accZG', 0)] * len(loc),
                                "yawDps": [raw_rms.get('yawDps', 0)] * len(loc),
                                "rollDps": [raw_rms.get('rollDps', 0)] * len(loc),
                                "pitchDps": [raw_rms.get('pitchDps', 0)] * len(loc)
                            })

                        df_calc = st.session_state['eval_data'].copy()

                        # ---------------------------------------------------------
                        # 🛞 타이어 위치별 독립 가중치(Gain) 매핑 연산
                        # ---------------------------------------------------------
                        w_int   = df_calc['위치'].map(lambda p: g_weights[p]['int'])
                        w_accX  = df_calc['위치'].map(lambda p: g_weights[p]['x'])
                        w_accY  = df_calc['위치'].map(lambda p: g_weights[p]['y'])
                        w_accZ  = df_calc['위치'].map(lambda p: g_weights[p]['z'])
                        w_yaw   = df_calc['위치'].map(lambda p: g_weights[p]['yaw'])
                        w_roll  = df_calc['위치'].map(lambda p: g_weights[p]['roll'])
                        w_pitch = df_calc['위치'].map(lambda p: g_weights[p]['pitch'])
                        w_speed = df_calc['위치'].map(lambda p: g_weights[p]['speed'])

                        # 위치별 고유 계수가 적용된 다중 선형 회귀 예측 수식 연산
                        df_calc['예측 마모값'] = (
                            w_int
                            + w_accX  * df_calc['accXG']
                            + w_accY  * df_calc['accYG']
                            + w_accZ  * df_calc['accZG']
                            + w_yaw   * df_calc['yawDps']
                            + w_roll  * df_calc['rollDps']
                            + w_pitch * df_calc['pitchDps']
                            + w_speed * df_calc['speed']
                        )

                        df_calc['오차율(%)'] = df_calc.apply(
                            lambda row: round((abs(row['실측 마모값'] - row['예측 마모값']) / row['실측 마모값'] * 100), 1) if row['실측 마모값'] != 0 else 0.0, axis=1
                        )
                        df_calc['예측율(%)'] = df_calc['오차율(%)'].apply(lambda x: max(0.0, round(100 - x, 1)))

                        st.markdown("#### 📋 최종 신뢰도 평가 결과 레포트")

                        display_cols = ['위치', '실측 마모값', '예측 마모값', '오차율(%)', '예측율(%)']

                        # 양식 입력 격리
                        with st.form(key="wear_input_form_fixed"):
                            edited_display = st.data_editor(
                                df_calc[display_cols],
                                column_config={
                                    "위치": st.column_config.TextColumn("위치", disabled=True),
                                    "실측 마모값": st.column_config.NumberColumn("실측 마모값 (입력/수정)", format="%.1f"),
                                    "예측 마모값": st.column_config.NumberColumn("예측 마모값 (자동계산)", disabled=True, format="%.1f"),
                                    "오차율(%)": st.column_config.NumberColumn("오차율(%)", disabled=True, format="%.1f"),
                                    "예측율(%)": st.column_config.NumberColumn("예측율(%)", disabled=True, format="%.1f")
                                },
                                hide_index=True,
                            )
                            submit_wear = st.form_submit_button("💾 실측값 반영 및 그래프 갱신")

                        if submit_wear:
                            if not edited_display['실측 마모값'].equals(st.session_state['eval_data']['실측 마모값']):
                                st.session_state['eval_data']['실측 마모값'] = edited_display['실측 마모값'].values
                                st.user_triggered_trend = True # 플래그 설정
                                st.rerun()

                        # ---------------------------------------------------------
                        # 5. 비교 그래프 렌더링
                        # ---------------------------------------------------------
                        render_wear_comparison_chart(edited_display)
                    else:
                        st.warning("⚠️ 주행 데이터 기간 필터링 연산에 실패했거나 데이터가 비어있습니다.")
                except Exception as e:
                    st.error(f"날짜 처리 오류: {e}")

                except Exception as e:
                    st.error(f"날짜 처리 오류: {e}")
            else:
                st.warning("⚠️ 주행 데이터를 먼저 업로드해 주세요.")
        else:
            st.write("---")
            st.info("ℹ️ 시작/종료 날짜를 입력하면 기간 분석이 시작됩니다.")

    with tab4:
        st.write("### 🚗 운전 가혹도 분석")

        if df.empty:
            st.warning("분석할 데이터가 없습니다. 먼저 파일을 업로드해주세요.")
        else:
            dt = 0.5  # 2Hz 고정
            if 'dataTimeDate' not in df.columns:
                df['dataTimeDate'] = pd.to_datetime(df['dataTime']).dt.date

            with st.form("severity_analysis_form"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    median_window = st.number_input("Median 필터 윈도우 (홀수)", min_value=1, max_value=51, value=9, step=2)
                with c2:
                    step_size = st.number_input("분석 간격 (Step Size)", min_value=1, max_value=100, value=2)
                with c3:
                    window_size = st.number_input("통계 계산 윈도우 크기", min_value=1, max_value=100, value=9)

                with st.expander("⚙️ 가중치 및 최종 점수 세부 파라미터 설정 (클릭하여 열기)", expanded=True):
                    st.markdown("##### 1️⃣ 지표별 가중치 (합계 = 1.0 권장)")
                    wx_col, wy_col, wyaw_col = st.columns(3)
                    with wx_col:
                        w_x = st.number_input("X축 (가감속) 가중치", min_value=0.0, max_value=1.0, value=0.50, step=0.05)
                    with wy_col:
                        w_y = st.number_input("Y축 (선회) 가중치", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
                    with wyaw_col:
                        w_yaw = st.number_input("Yaw축 (회전각속도) 가중치", min_value=0.0, max_value=1.0, value=0.20, step=0.05)

                    st.markdown("##### 2️⃣ 최종 100점 변환 파라미터")
                    penalty_factor = st.number_input("최종 감점 계수 (Scale Factor)", min_value=0.00001, max_value=0.01000, value=0.00038, step=0.00001, format="%.5f")

                    submitted = st.form_submit_button("🚀 분석 실행", type="primary")

            if submitted:
                with st.spinner("가혹도 통계량을 초고속 계산 중입니다..."):
                    # --- 1. 데이터 추출 및 필터 적용 (기존 유지) ---
                    analysis_df = df[['accXG', 'accYG', 'yawDps']].copy()
                    analysis_df['accXG(Filter)'] = medfilt(analysis_df['accXG'].to_numpy(), kernel_size=median_window)
                    analysis_df['accYG(Filter)'] = medfilt(analysis_df['accYG'].to_numpy(), kernel_size=median_window)
                    analysis_df['yawDps(Filter)'] = medfilt(analysis_df['yawDps'].to_numpy(), kernel_size=median_window)

                    # --- 2. 윈도우별 통계량 계산 (★for 루프 제거, 100% 벡터화 연산) ---
                    roller = analysis_df.rolling(window=window_size, center=False)

                    res_dict = {}
                    for name in ['accXG', 'accYG', 'yawDps']:
                        f_col = f'{name}(Filter)'
                        res_dict[name] = analysis_df[name]
                        res_dict[f_col] = analysis_df[f_col]

                        # RMS, STD, Jerk 연산 (X, Y, Yaw 3개 축 모두 생성됨)
                        res_dict[f'{name}(rms)'] = np.sqrt((analysis_df[f_col]**2).rolling(window=window_size).mean())
                        res_dict[f'{name}(STD)'] = roller[f_col].std(ddof=0)
                        res_dict[f'{name}(jerk)'] = analysis_df[f_col].diff().abs().rolling(window=window_size - 1).max() / dt

                    # 딕셔너리를 데이터프레임으로 변환
                    res_df = pd.DataFrame(res_dict)
                    res_df['speed'] = df['speed'].loc[res_df.index]
                    res_df['date'] = df['dataTimeDate'].loc[res_df.index] # [추가] 일별 점수 집계를 위해 날짜 매핑

                    metrics_cols = [
                        'accXG(rms)', 'accXG(STD)', 'accXG(jerk)',
                        'accYG(rms)', 'accYG(STD)', 'accYG(jerk)',
                        'yawDps(rms)', 'yawDps(STD)', 'yawDps(jerk)'
                    ]
                    Q_MAX = {col: max(res_df[col].quantile(0.995), 0.001) for col in metrics_cols}

                    if 'speed' in df.columns:
                        res_df['speed_weight'] = res_df['speed'].apply(get_speed_weight)
                    else:
                        res_df['speed_weight'] = 1.0

                    norm = {}
                    for col, max_val in Q_MAX.items():
                        norm[col] = (res_df[col] / max_val).clip(0, 1)

                    # 축별 가혹도 (RMS 15%, STD 35%, JERK 50%) * 100
                    res_df['score_X'] = (norm['accXG(rms)'] * 0.15 + norm['accXG(STD)'] * 0.35 + norm['accXG(jerk)'] * 0.50) * 100
                    res_df['score_Y'] = (norm['accYG(rms)'] * 0.15 + norm['accYG(STD)'] * 0.35 + norm['accYG(jerk)'] * 0.50) * 100
                    res_df['score_Yaw'] = (norm['yawDps(rms)'] * 0.15 + norm['yawDps(STD)'] * 0.35 + norm['yawDps(jerk)'] * 0.50) * 100

                    # 기존 가중치 체계를 새 수식(X: 35%, Y: 35%, Yaw: 30%)으로 완전 대체
                    res_df['ISI_Base'] = res_df['score_X'] * w_x + res_df['score_Y'] * w_y + res_df['score_Yaw'] * w_yaw
                    res_df['가혹도(ISI)'] = res_df['ISI_Base'] * res_df['speed_weight']

                    # [접목 단계 3] 일별 시간당 가혹 데미지 밀도 및 최종 100점 만점 점수 산출
                    daily_scores = []
                    for date_val, group in res_df.groupby('date'):
                        hours = len(group) / 7220.0 # 2Hz 데이터 기준 시간(Hour) 변환 계수
                        harsh_events = group['가혹도(ISI)'][group['가혹도(ISI)'] > 30]

                        if len(harsh_events) == 0:
                            hourly_rate = 0
                        else:
                            hourly_rate = (harsh_events - 30).sum() / hours

                        # 최종 100점 만점 변환 (계수 적용)
                        final_score = max(0, min(100, 100 - (hourly_rate * penalty_factor)))
                        daily_scores.append({
                            'date': date_val,
                            '최종운전점수': round(final_score, 1)
                        })

                    # 일별 점수 데이터프레임 생성
                    daily_score_df = pd.DataFrame(daily_scores)

                    # step_size 간격으로 자르기 전 원본 날짜를 이미 매핑했으므로 iloc 연산이 간소화됩니다.
                    result_df = res_df.iloc[window_size - 1 : : step_size].dropna().reset_index(drop=True)

                    # 일별 계산된 최종 100점 점수를 테이블에 결합 (대시보드 표출용)
                    result_df = result_df.merge(daily_score_df, on='date', how='left')
                    st.session_state.tab4_result = result_df

                    # Streamlit 화면에 일별 최종 점수를 요약 브리핑해줍니다.
                    st.write("#### 📅 일별 최종 운전 스코어 (100점 만점)")
                    st.dataframe(daily_score_df)

                    del analysis_df, res_dict, daily_scores
                    gc.collect()

                st.success(f"✅ 분석 완료! (총 {len(result_df)}개의 데이터 포인트)")

            # --- 3. 출력 및 시각화 ---
            if 'tab4_result' in st.session_state:
                res = st.session_state.tab4_result

                st.subheader("📈 가혹도 분석 결과 테이블")
                st.dataframe(res, width="stretch", height=400)

                daily_summary = res[['date', '최종운전점수']].drop_duplicates().sort_values('date')

                # 종합 점수 계산 (v2 일별 최종 점수들의 평균값 산출)
                monthly_score = daily_summary['최종운전점수'].mean()

                # 리포트 메트릭 화면 표시
                st.subheader("🏆 운전 점수 리포트")
                c1, c2 = st.columns(2)
                c1.metric("종합 점수", f"{monthly_score:.1f}점")
                c2.metric("최근 점수", f"{daily_summary['최종운전점수'].iloc[-1]:.1f}점")

                # 시각화 데이터 생성 및 라인 차트 렌더링
                summary_plot = daily_summary.copy()
                summary_plot['date_str'] = pd.to_datetime(summary_plot['date']).dt.strftime('%Y-%m-%d')

                chart = alt.Chart(summary_plot).mark_line(
                    point=True
                ).encode(
                    x=alt.X('date_str:O', title='날짜'),
                    y=alt.Y('최종운전점수:Q', scale=alt.Scale(domain=[0, 100]), title='운전 점수'),
                    tooltip=['date_str', '최종운전점수']
                ).properties(
                    width='container',
                    height=300
                )

                # 3. Streamlit에 출력
                st.altair_chart(chart)
    with tab5:
        st.write("### 🛞 타이어 상태 분석 (온도 및 공기압)")
        try:
            if os.path.exists('device_list.csv'):
                device_df = pd.read_csv('device_list.csv', sep='\t')
                device_df.columns = device_df.columns.str.strip()
                pos_map = {1: 'FL', 2: 'FR', 3: 'RL', 4: 'RR'}
                device_df['Position'] = device_df['타이어 위치'].map(pos_map)

                # [최적화 1] lambda 대신 사용할 1차원 딕셔너리 사전 생성 (속도 10배 이상 향상)
                sensor_to_info = device_df.set_index('SENSOR_ID').to_dict('index')
                sn_map = {k: v['무선통신기_SN'] for k, v in sensor_to_info.items()}
                pos_map_dict = {k: v['Position'] for k, v in sensor_to_info.items()}
                valid_sensors = set(sensor_to_info.keys()) # isin 검색용 set (매우 빠름)
            else:
                st.error("device_list.csv 파일을 찾을 수 없습니다.")
                st.stop()
        except Exception as e:
            st.error(f"장비 리스트 로드 오류: {e}")
            st.stop()

        if not uploaded_files:
            st.warning("분석할 데이터가 없습니다.")
        else:
            with st.spinner("데이터 처리 중..."):
                try:
                    # 1. 파일별 타이어 데이터 읽기
                    tire_dfs = []
                    target_cols = ['dataTime', 'sensorId', 'pressure', 'temperature', 'loadEstimation', 'wearEstimation']

                    for file in uploaded_files:
                        file.seek(0)
                        df_t = pd.read_excel(file, sheet_name='PacketBodyTire', engine='calamine', usecols=lambda c: c in target_cols)
                        df_t = df_t[(df_t['loadEstimation'] > 0) & (df_t['wearEstimation'] > 0)]
                        tire_dfs.append(df_t)

                    tire_df = pd.concat(tire_dfs, ignore_index=True)

                    tire_df = tire_df[tire_df['sensorId'].isin(valid_sensors)].copy()
                    tire_df['SN'] = tire_df['sensorId'].map(sn_map)
                    tire_df['Position'] = tire_df['sensorId'].map(pos_map_dict)

                    tire_df['dataTime'] = pd.to_datetime(tire_df['dataTime'])
                    tire_df = tire_df.sort_values('dataTime')

                    # 2. 주행 데이터도 정렬
                    df_speed = raw_df[['dataTime', 'speed']].copy()
                    df_speed['dataTime'] = pd.to_datetime(df_speed['dataTime'])
                    df_speed = df_speed.sort_values('dataTime')

                    # 3. merge_asof 사용 (가장 가까운 과거의 속도 매칭)
                    matched_df = pd.merge_asof(tire_df, df_speed, on='dataTime', direction='nearest')

                    fz_apatch_gain = 1
                    fz_speed_gain = 1.97
                    fz_gain_1 = 9.1497
                    fz_gain_2 = -1520.6

                    # matched_df['calculated_load'] = (((matched_df['loadEstimation'] * 1000) ** fz_apatch_gain) * (matched_df['pressure']) / (matched_df['speed'] ** fz_speed_gain)) * fz_gain_1 + fz_gain_2
                    matched_df['Apatch_Z'] = (matched_df['loadEstimation'] * 1000)
                    matched_df['Apatch_X'] = (matched_df['wearEstimation'] * 1000)

                    st.dataframe(matched_df.head(100))
                    st.subheader("📊 타이어 위치별 APATCH")

                    melted_df = matched_df.melt(
                        id_vars=['dataTime', 'Position'],
                        value_vars=['Apatch_X', 'Apatch_Z'],
                        var_name='Metric',
                        value_name='Value'
                    )

                    positions = ['FL', 'FR', 'RL', 'RR']
                    cols = st.columns(2)

                    color_scale = alt.Scale(
                        domain=['Apatch_X', 'Apatch_Z'],
                        range=['#72BCEE', "#3962E9"]
                    )

                    for i, pos in enumerate(positions):
                        # 해당 위치의 데이터만 필터링
                        pos_data = melted_df[melted_df['Position'] == pos]

                        if not pos_data.empty:
                            # 2열 레이아웃을 위해 index 계산
                            # daily_avg = pos_data.resample('D', on='dataTime')[['Apatch_X', 'Apatch_Z']].mean().reset_index()
                            # 결측치 보간 (선 연결)
                            # daily_avg = daily_avg.interpolate(method='linear').reset_index()
                            col = cols[i % 2]

                            with col:
                                st.write(f"#### 타이어 위치: {pos}")

                                points = alt.Chart(pos_data).mark_circle(opacity=1, size=50, clip=False).encode(
                                    x=alt.X('dataTime:T', axis=alt.Axis(format='%m-%d', labelAngle=0),
                                            scale=alt.Scale(padding=10),
                                            title='날짜'),
                                    y=alt.Y('Value:Q', scale=alt.Scale(domain=[0, 50000])),
                                    color=alt.Color('Metric:N', scale=color_scale),
                                    tooltip=['dataTime', 'Metric', 'Value']
                                )

                                trendline = points.transform_regression(
                                    'dataTime', 'Value', groupby=['Metric']
                                ).mark_line(strokeDash=[5, 5], clip=False)

                                chart = (points + trendline).properties(height=300).configure_view(strokeWidth=0).interactive()
                                st.altair_chart(chart, width='stretch')

                except Exception as e:
                    st.error(f"데이터 처리 오류: {e}")


    #     # [내부 함수 정의]
    #     def predict_wear_by_imu(mileage, pos, wear_idx, start_depth=8.0, start_mileage=0):
    #         # 기본 마모율 (실제 데이터에 맞춰 조금 하향 조정: 0.00004)
    #         if 'F' in pos:
    #             base_rate = 0.00007  # 전륜 기준 마모율
    #             if 'L' in pos:
    #                 side_weight = 0.9  # 기준
    #             elif 'R' in pos:
    #                 side_weight = 1.2 # 우측 타이어에 5% 가중치 (환경에 따라 조정)
    #         else:
    #             base_rate = 0.000018  # 후륜 기준 마모율 (전륜의 약 50~60%)
    #             side_weight = 1.0

    #         slope = base_rate * (1 + (wear_idx / 10)) * side_weight
    #         # (입력한 마일리지 - 시작 마일리지) 만큼의 마모량 계산
    #         relative_mileage = max(0, mileage - start_mileage)
    #         return start_depth - (slope * relative_mileage)

    #     # 1. 실측 데이터 입력 테이블
    #     st.markdown("#### 📝 타이어 그루브별 마모 실측치 입력 (단위: mm)")
    #     mileages = [0, 5000, 10000, 15000, 20000]
    #     positions = ['FL', 'FR', 'RL', 'RR']

    #     if 'measure_df' not in st.session_state:
    #         init_data = []
    #         for m in mileages:
    #             for p in positions:
    #                 init_data.append({'Mileage': m, 'Position': p, 'S1': np.nan, 'C1': np.nan, 'C2': np.nan, 'S2': np.nan})
    #         st.session_state.measure_df = pd.DataFrame(init_data)

    #     edited_df = st.data_editor(
    #         st.session_state.measure_df,
    #         column_config={
    #             "Mileage": st.column_config.NumberColumn("Mileage (km)", format="%d", disabled=True),
    #             "Position": st.column_config.TextColumn("Position", disabled=True),
    #             "S1": st.column_config.NumberColumn("S1", format="%.2f"),
    #             "C1": st.column_config.NumberColumn("C1", format="%.2f"),
    #             "C2": st.column_config.NumberColumn("C2", format="%.2f"),
    #             "S2": st.column_config.NumberColumn("S2", format="%.2f"),
    #         },
    #         hide_index=True # 인덱스 번호 숨기기
    #     )
    #     st.session_state.measure_df = edited_df

    #     # 2. 비교 그래프 렌더링
    #     st.divider()
    #     st.write("### 📉 실측 추세와 IMU 알고리즘 비교")

    #     valid_input_df = edited_df.dropna(subset=['S1', 'C1', 'C2', 'S2'], how='all')

    #     if valid_input_df.empty:
    #         st.info("💡 위의 테이블에 타이어 실측 데이터를 입력하면 비교 그래프가 생성됩니다.")
    #     else:
    #         limit_depth = 1.6
    #         x_future = np.linspace(0, 80000, 100)
    #         fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    #         axes = axes.flatten()

    #         for i, pos in enumerate(positions):
    #             ax = axes[i]

    #             # 해당 위치(Position) 데이터 중 숫자가 입력된 데이터만 추출
    #             pos_data = edited_df[
    #                 (edited_df['Position'] == pos) &
    #                 (edited_df[['S1', 'C1', 'C2', 'S2']].notnull().any(axis=1))
    #             ].sort_values('Mileage')

    #             if not pos_data.empty:
    #                 # 1. 개별 포인트와 평균값 계산 (None 제외)
    #                 x_measured = pos_data['Mileage'].values
    #                 y_points = {
    #                     'S1': pos_data['S1'].astype(float).values,
    #                     'C1': pos_data['C1'].astype(float).values,
    #                     'C2': pos_data['C2'].astype(float).values,
    #                     'S2': pos_data['S2'].astype(float).values
    #                 }
    #                 # 평균 계산 시 NaN 무시
    #                 y_measured_avg = pos_data[['S1', 'C1', 'C2', 'S2']].mean(axis=1).values

    #                 # 시작점 설정 (첫 번째 입력된 유효 데이터 기준)
    #                 start_m = x_measured[0]
    #                 start_d = y_measured_avg[0]

    #                 # [A] 실측 데이터 시각화
    #                 for label, values in y_points.items():
    #                     # 값이 있는 데이터만 산점도로 표시
    #                     valid_mask = ~np.isnan(values)
    #                     ax.scatter(x_measured[valid_mask], values[valid_mask], label=f'Actual {label}', s=60, alpha=0.7)

    #                 # [B] 평균 추세선 (데이터가 2개 이상일 때만 기울기 계산)
    #                 if len(x_measured) > 1:
    #                     z_act = np.polyfit(x_measured, y_measured_avg, 1)
    #                     ax.plot(x_future, np.poly1d(z_act)(x_future), color='gray', linestyle=':', alpha=0.5, label='Actual Trend (Avg)')

    #                 # [C] IMU 알고리즘 예측선
    #                 # predict_wear_by_imu 함수가 정의되어 있다고 가정합니다.
    #                 y_imu_pred = [predict_wear_by_imu(x, pos, avg_wear_idx, start_d, start_m) for x in x_future]
    #                 ax.plot(x_future, y_imu_pred, color='blue', linestyle='--', lw=2.5, label='IMU Prediction')

    #                 # [D] 정확도 계산
    #                 if len(x_measured) > 1:
    #                     actual_slope = (y_measured_avg[0] - y_measured_avg[-1]) / (x_measured[-1] - x_measured[0] + 1e-9)
    #                     pred_slope = (y_imu_pred[0] - y_imu_pred[-1]) / (x_future[-1] - x_future[0])
    #                     slope_error = abs(actual_slope - pred_slope) / (max(actual_slope, 1e-12))
    #                     accuracy = max(0.0, 100 - (slope_error * 100))
    #                 else:
    #                     accuracy = 100.0

    #                 ax.text(0.05, 0.15, f"Start Depth: {start_d:.2f}mm", transform=ax.transAxes, fontsize=10)
    #                 ax.text(0.05, 0.08, f"Slope Accuracy: {accuracy:.1f}%", transform=ax.transAxes,
    #                         fontweight='bold', color='blue', bbox=dict(facecolor='white', alpha=0.8))

    #             else:
    #                 # 해당 포지션에 데이터가 없을 경우 가이드 메시지
    #                 ax.text(0.5, 0.5, f"No data for {pos}", ha='center', va='center', transform=ax.transAxes)

    #             ax.axhline(limit_depth, color='red', linestyle=':', alpha=0.7)
    #             ax.set_title(f"Tire Comparison: {pos}", fontsize=14)
    #             ax.set_ylim(0, 8.5)
    #             ax.legend(loc='upper right')
    #             ax.grid(True, alpha=0.3)

    #         st.pyplot(fig)
    #         plt.close(fig)
else:
    st.info("👈 데이터를 업로드해주세요.")


# In[ ]:




