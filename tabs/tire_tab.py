import os
import gc
import re

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

def normalize_id(value) -> str:
    if pd.isna(value):
        return ""

    value_str = str(value).strip()

    if value_str.endswith(".0"):
        value_str = value_str[:-2]

    return value_str

def extract_gateway_serial(filename: str) -> str | None:
    basename = os.path.basename(filename)

    # export.<통신기SN>. 형태
    match = re.search(
        r"^export\.([^.]+)\.",
        basename,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return normalize_id(match.group(1))

def render_tire_tab(uploaded_files, raw_df):
    st.write("### 🛞 타이어 상태 분석 (온도 및 공기압)")
    try:
        if os.path.exists('device_list.csv'):
            device_df = pd.read_csv('device_list.csv', sep='\t', dtype=str)
            device_df.columns = device_df.columns.str.strip()

            device_df['SENSOR_ID'] = device_df['SENSOR_ID'].apply(normalize_id)
            device_df['무선통신기_SN'] = device_df['무선통신기_SN'].apply(normalize_id)

            pos_map = {'1': 'FL', '2': 'FR', '3': 'RL', '4': 'RR', 1: 'FL', 2: 'FR', 3: 'RL', 4: 'RR'}
            device_df['Position'] = device_df['타이어 위치'].map(pos_map)

            sensor_to_info = device_df.set_index('SENSOR_ID').to_dict('index')
            sn_map = {k: v['무선통신기_SN'] for k, v in sensor_to_info.items()}
            pos_map_dict = {k: v['Position'] for k, v in sensor_to_info.items()}
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
                tire_dfs = []
                target_cols = ['dataTime', 'sensorId', 'pressure', 'temperature', 'loadEstimation', 'wearEstimation']

                for file in uploaded_files:
                    gateway_sn = extract_gateway_serial(file.name)

                    if not gateway_sn:
                        st.warning(f"⚠️ 파일명에서 통신기 SN을 추출할 수 없습니다: {file.name}")
                        continue

                    allowed_sensors = set(device_df[device_df['무선통신기_SN'] == gateway_sn]['SENSOR_ID'])

                    if not allowed_sensors:
                        st.warning(f"⚠️ 센서 데이터가 없습니다 : {file.name}")
                        continue

                    file.seek(0)
                    df_t = pd.read_excel(file, sheet_name='PacketBodyTire', engine='calamine', usecols=lambda c: c in target_cols)

                    if df_t.empty or 'sensorId' not in df_t.columns:
                        continue

                    df_t['sensorId'] = df_t['sensorId'].apply(normalize_id)
                    df_t = df_t[df_t['sensorId'].isin(allowed_sensors)].copy()

                    if df_t.empty:
                        continue

                    for col in ['pressure', 'temperature', 'loadEstimation', 'wearEstimation']:
                        if col in df_t.columns:
                            df_t[col] = pd.to_numeric(df_t[col], errors='coerce').astype('float32')

                    df_t = df_t[(df_t['loadEstimation'] > 0) & (df_t['wearEstimation'] > 0)]

                    if not df_t.empty:
                        tire_dfs.append(df_t)

                if not tire_dfs:
                    st.error("❌ 조건에 일치하는 유효한 타이어 센서 데이터가 없습니다.")
                    st.write(f"📁 파일에서 추출한 SN: `{gateway_sn}`")
                    st.write(f"🎯 매핑된 센서 목록: `{allowed_sensors}`")
                    return

                tire_df = pd.concat(tire_dfs, ignore_index=True)

                tire_df['SN'] = tire_df['sensorId'].map(sn_map)
                tire_df['Position'] = tire_df['sensorId'].map(pos_map_dict)

                tire_df['dataTime'] = pd.to_datetime(tire_df['dataTime']).astype('datetime64[us]')
                tire_df = tire_df.sort_values('dataTime')

                df_speed = raw_df[['dataTime', 'speed']].copy()
                df_speed['dataTime'] = pd.to_datetime(df_speed['dataTime']).astype('datetime64[us]')
                df_speed = df_speed.sort_values('dataTime')

                matched_df = pd.merge_asof(tire_df, df_speed, on='dataTime', direction='nearest')

                fz_apatch_gain = 1
                fz_speed_gain = 1.97
                fz_gain_1 = 9.1497
                fz_gain_2 = -1520.6

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

                # [최적화 4] 범주형 데이터로 변환하여 메모리 소비 대폭 절감
                melted_df['Metric'] = melted_df['Metric'].astype('category')
                melted_df['Position'] = melted_df['Position'].astype('category')

                positions = ['FL', 'FR', 'RL', 'RR']
                cols = st.columns(2)

                color_scale = alt.Scale(
                    domain=['Apatch_X', 'Apatch_Z'],
                    range=['#72BCEE', "#3962E9"]
                )

                for i, pos in enumerate(positions):
                    pos_data = melted_df[melted_df['Position'] == pos]

                    if not pos_data.empty:
                        col = cols[i % 2]

                        with col:
                            st.write(f"#### 타이어 위치: {pos}")

                            if len(pos_data) > 3000:
                                step = len(pos_data) // 3000
                                plot_data = pos_data.iloc[::step]
                            else:
                                plot_data = pos_data

                            points = alt.Chart(plot_data).mark_circle(opacity=1, size=50, clip=False).encode(
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

                            del plot_data
                            gc.collect()

            except Exception as e:
                st.error(f"데이터 처리 오류: {e}")

