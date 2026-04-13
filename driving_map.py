#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from score_driving import score_driving_style, ScoreConfig
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

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
st.markdown("""
주행 데이터 파일을 통해 차량의 IMU 데이터를 분석하고, 다양한 시각화 및 통계 정보를 제공합니다.
""")

# --- 1. 데이터 로드 및 전처리 (캐싱 최적화) ---
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file, sheet_name='PacketBodyDriving')
        else:
            df = pd.read_csv(file, sep=None, engine='python')

        if df.empty:
            st.warning(f"⚠️ {file.name}: 파일에 데이터가 없습니다.")
            return None

        if 'dataTime' not in df.columns:
            st.error(f"❌ {file.name}: 'dataTime' 컬럼을 찾을 수 없습니다.")
            return None

        dt_col = pd.to_datetime(df['dataTime'], errors='coerce') # 잘못된 형식은 NaT로 변환

        if dt_col.isna().all():
            st.error(f"❌ {file.name}: 'dataTime' 형식이 올바르지 않습니다.")
            return None

        df['dataTime'] = dt_col
        df['date'] = dt_col.dt.date
        df['month'] = dt_col.dt.to_period('M').astype(str)
        df['week'] = "Week " + dt_col.dt.isocalendar().week.astype(str)
        df['day_name'] = dt_col.dt.day_name()
        df['overall'] = "전체 기간 (Overall)"

        return df

    except Exception as e:
        st.error(f"🔥 파일 로드 중 오류 발생: {e}")
        return None

def get_driving_scores_vectorized(data, g_limit):
    """
    DSN(Driving Severity Number) 이론을 적용한 벡터화 점수 계산
    원리: 에너지는 가속도의 제곱(G^2)에 비례하며, 속도가 높을수록 마찰 에너지가 증폭됨
    """
    if data.empty:
        return [0, 0, 0, 0, 0, 0]

    # 1. 속도 보정 계수 (논문의 속도-에너지 상관관계 반영)
    # 고속 주행 시 동일 가속도에서도 마찰 에너지 전달률이 높아짐
    v_ref = 100 # 기준 속도 (km/h)
    speed_factor = (data['speed'] / v_ref).clip(lower=1.0)

    # 2. DSN 산출 (에너지 관점: G^2 * Speed_Factor)
    # 단순 G값이 아닌 G의 제곱을 사용하여 급격한 조작에 높은 페널티 부여
    dsn_x = (data['accXG']**2) * speed_factor
    dsn_y = (data['accYG']**2) * speed_factor

    def calculate_dsn_score(dsn_series, limit):
        if dsn_series.empty: return 0
        # 논문에 따라 에너지 총합 또는 상위 구간의 에너지 밀도를 분석
        # g_limit 역시 제곱하여 에너지 단위(G^2)로 비교
        energy_score = (dsn_series.quantile(0.9) / (limit**2)) * 100
        # 시각적 분별력을 위해 비선형 보정 적용
        return np.clip(np.sqrt(energy_score / 100) * 100, 0, 100)

    # 3. 항목별 가혹도 산출 (특허 및 논문의 가중치 적용)
    # 횡방향(Lateral)은 종방향보다 마모 유발 효율이 높으므로 sens_limit을 낮게 설정
    sens_limit_x = g_limit * 0.6
    sens_limit_y = g_limit * 0.4 # 횡방향에 더 민감하게 반응

    res = {
        'accel': calculate_dsn_score(dsn_x[data['accXG'] > 0], sens_limit_x),
        'brake': calculate_dsn_score(dsn_x[data['accXG'] < 0], sens_limit_x),
        'right': calculate_dsn_score(dsn_y[data['accYG'] > 0], sens_limit_y),
        'left': calculate_dsn_score(dsn_y[data['accYG'] < 0], sens_limit_y),
        'stability': np.clip((data[['accXG', 'accYG']].var().mean() / (sens_limit_x**2 * 0.1)) * 100, 0, 100),
        'speeding': np.clip(((data['speed'].quantile(0.95) - 110) / 30) * 100, 0, 100)
    }

    # 거미줄 차트 순환 순서: 급가속 - 과속 - 우회전 - 안정성 - 좌회전 - 급제동
    return [np.nan_to_num(res[k]) for k in ['accel', 'speeding', 'right', 'stability', 'left', 'brake']]


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
    st.session_state.all_data = pd.DataFrame()
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0
    st.session_state.uploader_key += 1
    if 'uploaded_file' in st.session_state:
        del st.session_state.uploaded_file
    st.rerun()

with st.sidebar:
    st.header("⚙️ 데이터 관리")
    if st.button("🗑️ 모든 데이터 초기화", help="업로드된 모든 주행 데이터를 삭제합니다."):
        clear_all_data()
        st.rerun() # 화면 즉시 갱신

st.sidebar.header("⚙️ 분석 설정")
uploaded_files = st.sidebar.file_uploader("주행 데이터 업로드", type=['csv', 'xlsx'], accept_multiple_files=True)

if uploaded_files:
    df_list = []

    with st.status("데이터 통합 및 전처리 중...", expanded=False) as status:
        for file in uploaded_files:
            st.write(f"파일 읽는 중: {file.name}")
            df_temp = load_data(file)
            if df_temp is not None:
                if 'packetBodyDrivingId' in df_temp.columns:
                    df_list.append(df_temp)
                else:
                    st.warning(f"⚠️ {file.name}: 'packetBodyDrivingId' 컬럼이 없어 제외되었습니다.")
            else:
                continue

        if not df_list:
            status.update(label="통합 실패: 유효한 파일이 없습니다.", state="error")
            st.error("❌ 분석할 수 있는 정상적인 데이터가 없습니다. 파일을 확인해주세요.")
            st.stop()

        raw_df = pd.concat(df_list, ignore_index=True)
        target_col = 'packetBodyDrivingId'

        initial_count = len(raw_df)
        raw_df = raw_df.dropna(subset=[target_col])
        raw_df = raw_df.drop_duplicates(subset=[target_col]).sort_values(target_col).reset_index(drop=True)

        final_count = len(raw_df)
        status.update(label=f"통합 완료! (총 {len(df_list)}개 파일 성공)", state="complete")

    # 중복 제거 알림 (중복이 있었을 경우만 표시)
    if initial_count > final_count:
        st.caption(f"ℹ️ 중복된 데이터 {initial_count - final_count:,}건을 제거했습니다.")

    # 설정값 (UI)
    analysis_unit = st.sidebar.selectbox("분석 단위", ["전체 단위 (Overall)", "일 단위 (Daily)", "주 단위 (Weekly)", "요일 단위 (Day of Week)", "월 단위 (Monthly)"])
    speed_min = st.sidebar.slider("최소 속도 (km/h)", 0, 100, 1)
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

    # 필터링
    df = raw_df[(raw_df['speed'] >= speed_min) & (raw_df['accXG'].abs() <= g_max) & (raw_df['accYG'].abs() <= g_max)].copy()

    unit_map = {"일 단위 (Daily)": 'date', "주 단위 (Weekly)": 'week', "요일 단위 (Day of Week)": 'day_name', "월 단위 (Monthly)": 'month', "전체 단위 (Overall)": 'overall'}
    group_col = unit_map[analysis_unit]

    # 정렬 설정
    group_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'] if group_col == 'day_name' else sorted(df[group_col].unique())

    summary = df.groupby(group_col).agg(
        평균속도=('speed', 'mean'),
        최대속도=('speed', 'max'),
        급가속횟수=('accXG', lambda x: (x > hard_accel_threshold).sum()),
        급제동횟수=('accXG', lambda x: (x < -hard_brake_threshold).sum()),
        급선회횟수=('accYG', lambda x: (x.abs() > hard_turn_threshold).sum()),
        데이터수=('dataTime', 'count')
    )

    # 마모지수 산출 (가중치 적용)
    summary['마모지수'] = (
        (summary['급가속횟수'] * 0.5 +
         summary['급제동횟수'] * 0.7 +
         summary['급선회횟수'] * 1.0) /
        summary['데이터수'].replace(0, 1) * 1000
    ).round(2)

    avg_wear_idx = float(summary['마모지수'].mean()) if not summary.empty else 1.0

    # --- 4. 메인 분석 화면 ---
    tab1, tab2, tab3 = st.tabs(["📈 시각화 분석", "🔢 데이터 통계", "🛞 마모 인자 분석"])

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

    def get_mode_speed(x):
        if x.dropna().empty: return 0.0
        return x.round(1).mode().iloc[0]

    summary = df.groupby(group_col).agg(
        평균속도=('speed', 'mean'),
        최대속도=('speed', 'max'),
        최대분포속도=('speed', get_mode_speed), # 최빈값 로직 적용
        급가속횟수=('accXG', lambda x: (x > hard_accel_threshold).sum()),
        급제동횟수=('accXG', lambda x: (x < -hard_brake_threshold).sum()),
        급선회횟수=('accYG', lambda x: (x.abs() > hard_turn_threshold).sum()),
        데이터수=('dataTime', 'count')
    )

    acc_threshold = hard_accel_threshold # 또는 hard_accel_threshold * 0.5 (유연한 판정 시)
    brk_threshold = hard_brake_threshold

    summary['가속_비율'] = df.groupby(group_col)['accXG'].apply(
        lambda x: (x > acc_threshold).sum() / len(x) * 100).round(1)
    summary['감속_비율'] = df.groupby(group_col)['accXG'].apply(
        lambda x: (x < -brk_threshold).sum() / len(x) * 100).round(1)
    summary['정속_비율'] = (100 - summary['가속_비율'] - summary['감속_비율']).round(1)

    # 2. 횡방향 거동 분류 (사용자가 설정한 급선회 기준 활용)
    turn_threshold = hard_turn_threshold

    summary['좌선회_비율'] = df.groupby(group_col)['accYG'].apply(
        lambda x: (x > turn_threshold).sum() / len(x) * 100).round(1)
    summary['우선회_비율'] = df.groupby(group_col)['accYG'].apply(
        lambda x: (x < -turn_threshold).sum() / len(x) * 100).round(1)
    summary['직진_비율'] = (100 - summary['좌선회_비율'] - summary['우선회_비율']).round(1)

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
        with st.expander("ℹ️ 분석 기준 및 비율 계산 로직 안내", expanded=True):
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

        with st.expander("📝 데이터 전처리 기준", expanded=True):
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

        if 'schedule_data' not in st.session_state:
            st.session_state.schedule_data = {m: "" for m in mileage_cols}

        input_df = pd.DataFrame([st.session_state.schedule_data])

        edited_schedule = st.data_editor(
            input_df,
            column_config={m: st.column_config.TextColumn(m, width="medium") for m in mileage_cols},
            num_rows="fixed",
            width="stretch",
            hide_index=True,
            key="schedule_editor"
        )

        is_input_complete = not (edited_schedule.values == "").any()

        if is_input_complete:
            if not df.empty:
                try:
                    analysis_df = df.copy()
                    analysis_df.columns = [c.strip() for c in analysis_df.columns] # 공백 제거
                    target_col = 'dataTimeDate'
                    if target_col not in analysis_df.columns:
                        st.error(f"❌ 파일에 '{target_col}' 컬럼이 없습니다. (현재 컬럼: {list(analysis_df.columns)})")
                    else:
                        analysis_df['datetime'] = pd.to_datetime(analysis_df[target_col])
                        start_dt = pd.to_datetime(edited_schedule.iloc[0]["시작"])
                        end_dt = pd.to_datetime(edited_schedule.iloc[0]["종료"])

                        mask = (analysis_df['datetime'] >= start_dt) & (analysis_df['datetime'] <= end_dt)
                        filtered_df = analysis_df.loc[mask]

                        if filtered_df.empty:
                            st.error(f"❌ {start_dt.date()} ~ {end_dt.date()} 기간 내에 데이터가 없습니다.")
                        else:
                            st.success(f"✅ {start_dt.date()} ~ {end_dt.date()} 구간 분석 완료 (데이터: {len(filtered_df)}건)")
                            st.divider()

                            df_run = filtered_df[filtered_df['speed'] > speed_min].copy() if 'speed' in filtered_df.columns else filtered_df.copy()

                            if df_run.empty:
                                st.warning(f"⚠️ 해당 기간 내 주행 데이터({speed_min}km/h 이상)가 없습니다.")
                            else:
                                summary_cols = ["speed", "accXG", "accYG", "accZG", "yawDps", "rollDps", "pitchDps"]
                                rms_dict = {}
                                for col in summary_cols:
                                    if col in df_run.columns:
                                        val = calc_rms(df_run[col])
                                        unit = "G" if "acc" in col else "dps" if "Dps" in col else "km/h"
                                        rms_dict[col] = f"{val:.4f} {unit}" if "acc" in col else f"{val:.2f} {unit}"

                                rms_horiz_df = pd.DataFrame([rms_dict])

                                st.markdown("**📍 선택 기간 IMU RMS**")
                                st.dataframe(
                                    rms_horiz_df.style.set_properties(**{'text-align': 'center', 'background-color': '#e1f5fe'}),
                                    width="stretch",
                                    hide_index=True
                                )
                                p0, p40, p80 = get_speed_distribution(df_run)
                                dist_df = pd.DataFrame([{"시내 (0-40)": f"{p0:.1f}%", "일반 (40-80)": f"{p40:.1f}%", "고속 (80+)": f"{p80:.1f}%"}])
                                st.markdown("**🛣️ 선택 기간 주행 속도 비율**")
                                st.dataframe(dist_df.style.set_properties(**{'text-align': 'center'}), width="stretch", hide_index=True)

                except Exception as e:
                    st.error(f"날짜 처리 오류: {e}")
            else:
                st.warning("⚠️ 주행 데이터를 먼저 업로드해 주세요.")
        else:
            st.write("---")
            st.info("ℹ️ 시작/종료 날짜를 입력하면 기간 분석이 시작됩니다.")
else:
    st.info("👈 데이터를 업로드해주세요.")


# In[ ]:




