# ui_components.py
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def render_tire_gain_inputs():
    """
    타이어 위치별(FL, FR, RL, RR) 고유 초기값을 적용하여
    예측 수식 계수 조절 UI를 렌더링하고 설정된 계수 딕셔너리를 반환하는 함수
    """
    default_vals_by_pos = {
        "FL": {
            "int": 42343.0099, "x": -791.5682, "y": 34776.3646, "z": -42656.5477,
            "yaw": -6508.1880, "roll": 310.2177, "pitch": -6122.7138, "speed": 421.3498
        },
        "FR": {
            "int": 23882.1489, "x": 1389.4141, "y": 23452.6922, "z": -29459.2530,
            "yaw": -4555.4972, "roll": 1887.8173, "pitch": -4364.6197, "speed": 375.4558
        },
        "RL": {
            "int": 30240.6238, "x": 56900.9284, "y": -86024.7478, "z": -391.0638,
            "yaw": 6257.6668, "roll": 9555.7524, "pitch": -20200.7290, "speed": 179.5255
        },
        "RR": {
            "int": 25317.3273, "x": 126646.1154, "y": -132976.9669, "z": -45215.8267,
            "yaw": 3616.8330, "roll": 11273.9471, "pitch": -29302.9341, "speed": 826.9955
        }
    }

    positions = {
        "FL": "🛞 FL (전륜 좌측)",
        "FR": "🛞 FR (전륜 우측)",
        "RL": "🛞 RL (후륜 좌측)",
        "RR": "🛞 RR (후륜 우측)"
    }

    g_weights = {}

    with st.expander("⚙️ 타이어 위치별 예측 수식 가중치(Gain) 실시간 조정", expanded=True):
        tabs = st.tabs(list(positions.values()))

        for (pos_code, pos_name), tab in zip(positions.items(), tabs):
            with tab:
                # 💡 2. 해당 바퀴 전용 초기값 딕셔너리를 쏙 뺍니다.
                defaults = default_vals_by_pos[pos_code]

                col_g1, col_g2, col_g3, col_g4 = st.columns(4)
                with col_g1:
                    val_int = st.number_input(f"{pos_code} Intercept (상수항)", value=defaults["int"], format="%.4f", key=f"{pos_code}_int")
                    val_x   = st.number_input(f"{pos_code} accXG 계수", value=defaults["x"], format="%.4f", key=f"{pos_code}_x")
                with col_g2:
                    val_y   = st.number_input(f"{pos_code} accYG 계수", value=defaults["y"], format="%.4f", key=f"{pos_code}_y")
                    val_z   = st.number_input(f"{pos_code} accZG 계수", value=defaults["z"], format="%.4f", key=f"{pos_code}_z")
                with col_g3:
                    val_yaw = st.number_input(f"{pos_code} yawDps 계수", value=defaults["yaw"], format="%.4f", key=f"{pos_code}_yaw")
                    val_roll= st.number_input(f"{pos_code} rollDps 계수", value=defaults["roll"], format="%.4f", key=f"{pos_code}_roll")
                with col_g4:
                    val_pitch = st.number_input(f"{pos_code} pitchDps 계수", value=defaults["pitch"], format="%.4f", key=f"{pos_code}_pitch")
                    val_spd   = st.number_input(f"{pos_code} speed 계수", value=defaults["speed"], format="%.4f", key=f"{pos_code}_spd")

                # 결과 취합
                g_weights[pos_code] = {
                    "int": val_int, "x": val_x, "y": val_y, "z": val_z,
                    "yaw": val_yaw, "roll": val_roll, "pitch": val_pitch, "speed": val_spd
                }

    return g_weights

def render_wear_comparison_chart(edited_display):
    """
    현재 탭3의 Plotly 비교 그래프를 사양 변경 없이 그대로 렌더링하는 함수
    """
    # 1. 종합 평균 예측율 계산 및 출력
    avg_accuracy = edited_display['예측율(%)'].mean()
    st.markdown(f"**종합 평균 예측율: {avg_accuracy:.1f}%**")

    # 2. 피규어 객체 생성 및 축 범위 산출
    fig_compare = go.Figure()
    min_val = min(edited_display['예측 마모값'].min(), edited_display['실측 마모값'].min()) * 0.9
    max_val = max(edited_display['예측 마모값'].max(), edited_display['실측 마모값'].max()) * 1.1

    if pd.isna(min_val) or np.isinf(min_val): min_val = 0
    if pd.isna(max_val) or np.isinf(max_val): max_val = 40000

    # 3. Ideal Line (Y = X) 트레이스 추가
    fig_compare.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines', name='Ideal Line (Y = X)',
        line=dict(color='rgba(214, 39, 40, 0.6)', width=2, dash='dash'),
        hovertemplate="기준선 (추정 = 실측)<extra></extra>"
    ))

    # 4. 위치별 데이터 마커 추가
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
    for i, row in edited_display.iterrows():
        fig_compare.add_trace(go.Scatter(
            x=[row['예측 마모값']], y=[row['실측 마모값']],
            mode='markers+text', name=f"{row['위치']} 위치",
            marker=dict(size=14, color=colors[i % len(colors)], line=dict(width=2, color='White'), opacity=0.9),
            hovertemplate=f"<b>{row['위치']} 위치</b><br>추정 마모량: %{{x:,.1f}}<br>실측 마모량: %{{y:,.1f}}<br>예측율: {row['예측율(%)']}%<extra></extra>"
        ))

    # 5. 기존 레이아웃 속성 100% 동일하게 반영
    fig_compare.update_layout(
        title="🔮 마모 지수(WI) 추정치 vs 실측치 신뢰도 평가",
        xaxis_title="수식 추정 마모량 (Predicted WI)", yaxis_title="실제 계측 마모량 (Actual WI)",
        xaxis=dict(range=[min_val, max_val], gridcolor='rgba(200,200,200,0.15)', zeroline=False),
        yaxis=dict(range=[min_val, max_val], gridcolor='rgba(200,200,200,0.15)', zeroline=False),
        legend=dict(title="분석 위치", x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)"),
        margin=dict(l=50, r=40, t=60, b=50), hovermode="closest", width=700, height=500
    )

    # 6. 차트 출력
    st.plotly_chart(fig_compare, width="stretch")
