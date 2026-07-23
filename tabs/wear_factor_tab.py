import pandas as pd
import streamlit as st

from core.data_processing import calc_rms, get_speed_distribution
from ui_components import render_tire_gain_inputs, render_wear_comparison_chart

def render_wear_factor_tab(df, speed_min):
    st.header("🔬 타이어 마모 인자 분석")

    with st.expander("📝 데이터 전처리 기준", expanded=False):
        st.markdown(f"""
        주행 데이터 분석의 정확도를 높이기 위해 다음과 같은 필터링 조건이 적용되었습니다.

        **1. 최소 속도 필터링 (Speed Threshold)**
        * **현재 기준**: `{speed_min} km/h` 이상인 데이터만 분석에 포함
        * 본 분석에서는 속도가 **`{speed_min} km/h` 미만**인 모든 구간을 '정차' 또는 '유효하지 않은 주행' 구간으로 간주합니다.

        **2. 속도 구간의 분류 기준**
        * **시내 (Urban)**: `{speed_min} km/h` ~ `40 km/h` 구간
        * **일반 (Suburban)**: `40 km/h` ~ `80 km/h` 구간
        * **고속 (Highway)**: `80 km/h` 초과 구간
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

                                st.session_state.tab3_rms_dict = rms_dict
                                st.session_state.tab3_raw_rms = raw_rms_values
                                st.session_state.tab3_dist = [{"시내 (0-40)": f"{p0:.1f}%", "일반 (40-80)": f"{p40:.1f}%", "고속 (80+)": f"{p80:.1f}%"}]
                                st.session_state.tab3_filtered_len = len(filtered_df)
                                st.session_state.last_analysis_trigger = current_trigger

                if "tab3_rms_dict" in st.session_state:
                    st.success(f"✅ 구간 분석 완료 (데이터: {st.session_state.tab3_filtered_len}건)")
                    st.divider()

                    rms_horiz_df = pd.DataFrame([st.session_state.tab3_rms_dict])
                    st.markdown("**📍 선택 기간 IMU RMS**")
                    st.dataframe(rms_horiz_df.style.set_properties(**{'text-align': 'center', 'background-color': '#e1f5fe'}), width="stretch", hide_index=True)

                    dist_df = pd.DataFrame(st.session_state.tab3_dist)
                    st.markdown("**🛣️ 선택 기간 주행 속도 비율**")
                    st.dataframe(dist_df.style.set_properties(**{'text-align': 'center'}), width="stretch", hide_index=True)

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

                    w_int   = df_calc['위치'].map(lambda p: g_weights[p]['int'])
                    w_accX  = df_calc['위치'].map(lambda p: g_weights[p]['x'])
                    w_accY  = df_calc['위치'].map(lambda p: g_weights[p]['y'])
                    w_accZ  = df_calc['위치'].map(lambda p: g_weights[p]['z'])
                    w_yaw   = df_calc['위치'].map(lambda p: g_weights[p]['yaw'])
                    w_roll  = df_calc['위치'].map(lambda p: g_weights[p]['roll'])
                    w_pitch = df_calc['위치'].map(lambda p: g_weights[p]['pitch'])
                    w_speed = df_calc['위치'].map(lambda p: g_weights[p]['speed'])

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
                            st.user_triggered_trend = True
                            st.rerun()

                    render_wear_comparison_chart(edited_display)
                else:
                    st.warning("⚠️ 주행 데이터 기간 필터링 연산에 실패했거나 데이터가 비어있습니다.")
            except Exception as e:
                st.error(f"날짜 처리 오류: {e}")
        else:
            st.warning("⚠️ 주행 데이터를 먼저 업로드해 주세요.")
    else:
        st.write("---")
        st.info("ℹ️ 시작/종료 날짜를 입력하면 기간 분석이 시작됩니다.")

