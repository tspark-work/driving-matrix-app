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
from matplotlib.colors import LogNorm

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
    if file.name.endswith('.xlsx'):
        df = pd.read_excel(file, sheet_name='PacketBodyDriving')
    else:
        df = pd.read_csv(file, sep=None, engine='python')

    # 벡터화된 시간 변수 생성
    dt_col = pd.to_datetime(df['dataTime'])
    df['dataTime'] = dt_col
    df['date'] = dt_col.dt.date
    df['month'] = dt_col.dt.to_period('M').astype(str)
    df['week'] = "Week " + dt_col.dt.isocalendar().week.astype(str)
    df['day_name'] = dt_col.dt.day_name()
    df['overall'] = "전체 기간 (Overall)"
    return df

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

# --- 3. 사이드바 구성 ---
st.sidebar.header("⚙️ 분석 설정")
uploaded_files = st.sidebar.file_uploader("주행 데이터 업로드", type=['csv', 'xlsx'], accept_multiple_files=True)

if uploaded_files:
    # raw_df = load_data(uploaded_file)
    # 1. 여러 파일을 담을 리스트 생성
    df_list = []

    with st.status("데이터 통합 및 전처리 중...", expanded=False) as status:
        for file in uploaded_files:
            st.write(f"파일 읽는 중: {file.name}")
            df_temp = load_data(file)
            df_list.append(df_temp)

        # 2. 데이터 통합
        raw_df = pd.concat(df_list, ignore_index=True)
        target_col = 'packetBodyDrivingId'

        # 3. 중복 제거 (데이터 시간 기준)
        initial_count = len(raw_df)
        raw_df = raw_df.drop_duplicates(subset=[target_col]).sort_values(target_col).reset_index(drop=True)
        final_count = len(raw_df)
        status.update(label=f"통합 완료! (총 {len(uploaded_files)}개 파일)", state="complete")

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
    hard_accel_threshold = h_acc.number_input("급가속 G", 0.0, 1.0, 0.3)
    hard_brake_threshold = h_brk.number_input("급제동 G", 0.0, 1.0, 0.3)
    hard_turn_threshold = h_trn.number_input("급선회 G", 0.0, 1.0, 0.3)

    # 필터링
    df = raw_df[(raw_df['speed'] >= speed_min) & (raw_df['accXG'].abs() <= g_max) & (raw_df['accYG'].abs() <= g_max)].copy()

    unit_map = {"일 단위 (Daily)": 'date', "주 단위 (Weekly)": 'week', "요일 단위 (Day of Week)": 'day_name', "월 단위 (Monthly)": 'month', "전체 단위 (Overall)": 'overall'}
    group_col = unit_map[analysis_unit]

    # 정렬 설정
    group_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'] if group_col == 'day_name' else sorted(df[group_col].unique())

    # --- 4. 메인 분석 화면 ---
    tab1, tab2, tab3 = st.tabs(["📈 시각화 분석", "🔢 데이터 통계", "🛞 타이어 마모 예측"])

    with tab1:
        for item in group_list:
            plot_data = df[df[group_col] == item]
            if plot_data.empty: continue

            with st.expander(f"📍 {item} 리포트 ({len(plot_data):,} 샘플)", expanded=True):
                c1, c2, c3 = st.columns(3)

                # 속도 히스토그램
                with c1:
                    fig, ax = plt.subplots(figsize=(4, 4))
                    st.markdown("<h3 style='text-align: center; margin-bottom: 0px;'>🎢 주행 속도 분포</h3>", unsafe_allow_html=True)
                    sns.histplot(plot_data['speed'], bins=20, kde=True, color='skyblue', ax=ax)
                    avg_speed = plot_data['speed'].mean()
                    ax.axvline(avg_speed, color='red', linestyle='--', linewidth=1.5, label=f'Avg: {avg_speed:.1f}')
                    ax.text(avg_speed + 2, ax.get_ylim()[1] * 0.9, f'{avg_speed:.1f} km/h',
                            color='red', fontweight='bold', fontsize=9)
                    ax.set_xlabel("Speed (km/h)", fontsize=10)
                    ax.set_ylabel("Data Frequency (Count)", fontsize=10)
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

                # Radar Chart
                # with c3:
                #     st.markdown("<h4 style='text-align: center;'>🕸️ 주행 성향 분석</h4>", unsafe_allow_html=True)
                #     scores = get_driving_scores_vectorized(plot_data, g_limit)
                #     # cfg = ScoreConfig(
                #     #     fs=100.0,
                #     #     speed_unit="kmh"
                #     # )
                #     # result = score_driving_style(plot_data, cfg)
                #     # scores = [
                #     #     result["harsh_accel_score"],
                #     #     result["harsh_brake_score"],
                #     #     result["left_turn_score"],
                #     #     result["right_turn_score"],
                #     #     result["instability_score"],
                #     #     result["overspeed_score"],
                #     # ]
                #     categories = ['급가속', '급제동', '좌회전', '우회전', '불안정성', '과속']
                #     fig_radar = go.Figure(go.Scatterpolar(r=scores+[scores[0]], theta=categories+[categories[0]], fill='toself', line_color='#FF4B4B'))
                #     fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=400, margin=dict(l=30, r=30, t=30, b=30))
                #     st.plotly_chart(fig_radar, width="stretch", config={'displayModeBar': True})
                #     # st.json(result["debug"])

    with tab2:
        # 그룹별 통계 (한 번에 계산)
        summary = df.groupby(group_col).agg(
            평균속도=('speed', 'mean'),
            최대속도=('speed', 'max'),
            급가속횟수=('accXG', lambda x: (x > hard_accel_threshold).sum()),
            급제동횟수=('accXG', lambda x: (x < -hard_brake_threshold).sum()),
            급선회횟수=('accYG', lambda x: (x.abs() > hard_turn_threshold).sum()),
            데이터수=('dataTime', 'count')
        )
        if group_col == 'day_name': summary = summary.reindex(group_list)
        st.dataframe(summary.style.highlight_max(axis=0, color="#fffb18"))

    # with tab3:
    #     st.write("### 📉 타이어 위치별 상세 마모 프로파일 (Center / Shoulder / Total)")

    #     # 1. 마모지수 및 기본 설정
    #     if '마모지수' not in summary.columns:
    #         summary['마모지수'] = ((summary['급가속횟수']*0.5 + summary['급제동횟수']*0.7 + summary['급선회횟수']*1.0) /
    #                             summary['데이터수'].replace(0, 1) * 1000).round(2)

    #     avg_idx = summary['마모지수'].mean()
    #     base_wear_rate = 0.0001  # 기준 마모율
    #     x_range = np.linspace(0, 80000, 100)
    #     limit_depth = 1.6

    #     # 타이어 위치 및 마모 부위별 가중치 설정
    #     tire_positions = ['FL', 'FR', 'RL', 'RR']
    #     # 부위별 가중치: Shoulder는 코너링(횡G)에, Center는 가감속(종G)에 더 민감하다고 가정
    #     part_configs = {
    #         'Center': {'color': 'red', 'weight_adj': 1.0},
    #         'Shoulder': {'color': 'green', 'weight_adj': 1.2}, # 횡G 영향으로 보통 더 빨리 마모
    #         'Total': {'color': 'blue', 'weight_adj': 1.1}
    #     }

    #     # 2. 2x2 그래프 레이아웃 생성
    #     fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    #     axes = axes.flatten() # 2D 배열을 1D로 변환하여 반복문 사용 용이하게 함

    #     for i, pos in enumerate(tire_positions):
    #         ax = axes[i]
    #         # 위치별 기본 가중치 (전륜 1.2, 후륜 0.8)
    #         pos_weight = 1.2 if 'F' in pos else 0.8

    #         for part, config in part_configs.items():
    #             # 최종 마모율 = 기본 * 주행지수 * 위치가중치 * 부위가중치
    #             slope = base_wear_rate * (1 + (avg_idx / 10)) * pos_weight * config['weight_adj']
    #             y_vals = 8.0 - (slope * x_range)
    #             y_vals = np.clip(y_vals, 0, 8.0)

    #             # 추세선 그리기
    #             ax.plot(x_range, y_vals, label=part, color=config['color'], lw=2)

    #             # 마모 한계선 교차점 계산 및 표시
    #             intercept_km = (8.0 - limit_depth) / slope
    #             if intercept_km < 80000:
    #                 ax.scatter(intercept_km, limit_depth, color=config['color'], s=30)
    #                 ax.text(intercept_km, limit_depth + 0.2, f"{int(intercept_km):,}km",
    #                         color=config['color'], fontsize=8, ha='center')

    #         # 그래프 세부 설정
    #         ax.axhline(limit_depth, color='grey', linestyle='--', alpha=0.5)
    #         ax.set_title(f"Tire Profile: {pos}", fontsize=14)
    #         ax.set_xlabel("Mileage (km)")
    #         ax.set_ylabel("Depth (mm)")
    #         ax.set_ylim(0, 8.5)
    #         ax.grid(True, linestyle=':', alpha=0.6)
    #         ax.legend(loc='upper right')

    #     st.pyplot(fig)
    #     plt.close(fig)

    #     # 3. 요약 리포트
    #     st.divider()
    #     cols = st.columns(4)
    #     for i, pos in enumerate(tire_positions):
    #         # 종합(Total) 기준으로 남은 수명 표시
    #         pos_weight = 1.2 if 'F' in pos else 0.8
    #         total_slope = base_wear_rate * (1 + (avg_idx / 10)) * pos_weight * 1.1
    #         life_km = (8.0 - limit_depth) / total_slope
    #         cols[i].metric(f"{pos} 예상 수명", f"{int(life_km):,} km")
else:
    st.info("👈 데이터를 업로드해주세요.")


# In[ ]:





