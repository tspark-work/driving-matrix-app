import streamlit as st
import plotly.graph_objects as go

def render_statistics_tab(summary, group_col, group_list, analysis_unit, acc_threshold, brk_threshold, turn_threshold):
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

