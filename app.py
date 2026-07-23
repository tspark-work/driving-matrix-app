#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gc
import logging
import pandas as pd
import streamlit as st

from core.common import set_korean_font
from core.data_processing import get_integrated_data, process_global_dataframe
from tabs.visualization_tab import render_visualization_tab
from tabs.statistics_tab import render_statistics_tab
from tabs.wear_factor_tab import render_wear_factor_tab
from tabs.severity_tab import render_severity_tab
from tabs.tire_tab import render_tire_tab

gc.collect()

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- 0. 환경 설정 ---
set_korean_font()
st.set_page_config(page_title="차량 IMU 통합 분석", layout="wide", page_icon="🏎️")

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

# --- 3. 사이드바 구성 ---
if 'all_data' not in st.session_state:
    st.session_state.all_data = pd.DataFrame()
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

def clear_all_data():
    st.cache_data.clear()
    st.cache_resource.clear()

    analysis_keys = [
        'all_data', 'tab4_result', 'tab3_raw_rms', 'tab3_rms_dict',
        'tab3_dist', 'eval_data', 'last_analysis_trigger',
        'df', 'summary'
    ]

    for key in analysis_keys:
        if key in st.session_state:
            st.session_state[key] = pd.DataFrame()
            del st.session_state[key]

    st.session_state.uploader_key += 1
    gc.collect()
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)

with st.sidebar:
    st.header("⚙️ 데이터 관리")
    if st.button("🗑️ 모든 데이터 초기화", help="업로드된 모든 주행 데이터를 삭제합니다."):
        clear_all_data()
        st.rerun()
    uploaded_files = st.sidebar.file_uploader(
        "주행 데이터 업로드",
        type=['csv', 'xlsx'],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )

st.sidebar.header("⚙️ 분석 설정")

if uploaded_files:
    with st.status("데이터 통합 및 전처리 중...", expanded=True) as status:
        file_keys = tuple((f.name, f.size) for f in uploaded_files)
        raw_df = get_integrated_data(uploaded_files, file_keys)

        if raw_df is None or raw_df.empty:
            status.update(label="통합 실패: 유효한 파일이 없습니다.", state="error")
            st.error("❌ 분석할 수 있는 정상적인 데이터가 없습니다.")
            st.stop()

        status.update(label=f"통합 완료! (총 {len(uploaded_files)}개 파일 성공)", state="complete")

    analysis_unit = st.sidebar.selectbox("분석 단위", ["전체 단위 (Overall)", "일 단위 (Daily)", "주 단위 (Weekly)", "요일 단위 (Day of Week)", "월 단위 (Monthly)"])
    speed_range = st.sidebar.slider(
        "분석 속도 범위 설정 (km/h)",
        0, 200,
        (1, 200),
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

    h_acc, h_brk, h_trn = st.sidebar.columns(3)
    hard_accel_threshold = h_acc.number_input("급가속 G", 0.0, 1.0, 0.05)
    hard_brake_threshold = h_brk.number_input("급제동 G", 0.0, 1.0, 0.05)
    hard_turn_threshold = h_trn.number_input("급선회 G", 0.0, 1.0, 0.05)

    unit_map = {"일 단위 (Daily)": 'date', "주 단위 (Weekly)": 'week', "요일 단위 (Day of Week)": 'day_name', "월 단위 (Monthly)": 'month', "전체 단위 (Overall)": 'overall'}
    group_col = unit_map[analysis_unit]
    group_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'] if group_col == 'day_name' else sorted(raw_df[group_col].unique())

    df, summary = process_global_dataframe(
        raw_df, speed_min, speed_max, g_max, r_max, selected_case,
        group_col, hard_accel_threshold, hard_brake_threshold, hard_turn_threshold
    )

    avg_wear_idx = float(summary['마모지수'].mean()) if not summary.empty else 1.0

    acc_threshold = hard_accel_threshold
    brk_threshold = hard_brake_threshold
    turn_threshold = hard_turn_threshold

    # --- 4. 메인 분석 화면 ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 시각화 분석",
        "🔢 데이터 통계",
        "🛞 마모 인자 분석",
        "💥 운전 가혹도 분석",
        "타이어 마모 예측",
    ])

    with tab1:
        render_visualization_tab(df, group_col, group_list, g_limit, grid_val)

    with tab2:
        render_statistics_tab(
            summary, group_col, group_list, analysis_unit,
            acc_threshold, brk_threshold, turn_threshold
        )

    with tab3:
        render_wear_factor_tab(df, speed_min)

    with tab4:
        render_severity_tab(df)

    with tab5:
        render_tire_tab(uploaded_files, raw_df)

else:
    st.info("👈 데이터를 업로드해주세요.")


# In[ ]:




