import gc
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from scipy.signal import medfilt

@st.cache_data(show_spinner=False, max_entries=4, ttl=3600)
def calculate_severity_cached(
    data_key,
    _source_df,
    median_window,
    step_size,
    window_size,
    dt,
    w_x,
    w_y,
    w_yaw,
    w_rms,
    w_std,
    w_jerk,
    penalty_factor,
):
    """운전 가혹도 결과만 캐시합니다. _source_df는 해시 대상에서 제외됩니다."""
    base_cols = ['accXG', 'accYG', 'yawDps', 'speed', 'dataTimeDate']
    analysis_df = _source_df.loc[:, base_cols].copy()

    for col in ['accXG', 'accYG', 'yawDps', 'speed']:
        analysis_df[col] = pd.to_numeric(analysis_df[col], errors='coerce').astype('float32')

    for name in ['accXG', 'accYG', 'yawDps']:
        values = analysis_df[name].to_numpy(dtype=np.float32, copy=False)
        analysis_df[f'{name}(Filter)'] = medfilt(
            values,
            kernel_size=int(median_window)
        ).astype(np.float32, copy=False)

    res_dict = {}
    for name in ['accXG', 'accYG', 'yawDps']:
        f_col = f'{name}(Filter)'
        filtered = analysis_df[f_col]
        res_dict[name] = analysis_df[name]
        res_dict[f_col] = filtered
        res_dict[f'{name}(rms)'] = np.sqrt(
            filtered.pow(2).rolling(window=int(window_size)).mean()
        )
        res_dict[f'{name}(STD)'] = filtered.rolling(
            window=int(window_size)
        ).std(ddof=0)
        res_dict[f'{name}(jerk)'] = (
            filtered.diff().abs().rolling(
                window=max(1, int(window_size) - 1)
            ).max() / np.float32(dt)
        )

    res_df = pd.DataFrame(res_dict)
    res_df['speed'] = analysis_df['speed']
    res_df['date'] = analysis_df['dataTimeDate']

    metrics_cols = [
        'accXG(rms)', 'accXG(STD)', 'accXG(jerk)',
        'accYG(rms)', 'accYG(STD)', 'accYG(jerk)',
        'yawDps(rms)', 'yawDps(STD)', 'yawDps(jerk)'
    ]
    q_max = {
        col: max(float(res_df[col].quantile(0.995)), 0.001)
        for col in metrics_cols
    }

    speed_values = res_df['speed'].to_numpy(dtype=np.float32, copy=False)
    res_df['speed_weight'] = np.select(
        [speed_values <= 30, speed_values <= 80, speed_values <= 120],
        [1.0, 1.2, 1.5],
        default=2.0
    ).astype(np.float32)

    norm = {
        col: (res_df[col] / np.float32(max_val)).clip(0, 1)
        for col, max_val in q_max.items()
    }

    res_df['score_X'] = (
        norm['accXG(rms)'] * w_rms
        + norm['accXG(STD)'] * w_std
        + norm['accXG(jerk)'] * w_jerk
    ) * 100
    res_df['score_Y'] = (
        norm['accYG(rms)'] * w_rms
        + norm['accYG(STD)'] * w_std
        + norm['accYG(jerk)'] * w_jerk
    ) * 100
    res_df['score_Yaw'] = (
        norm['yawDps(rms)'] * w_rms
        + norm['yawDps(STD)'] * w_std
        + norm['yawDps(jerk)'] * w_jerk
    ) * 100

    res_df['ISI_Base'] = (
        res_df['score_X'] * w_x
        + res_df['score_Y'] * w_y
        + res_df['score_Yaw'] * w_yaw
    )
    res_df['가혹도(ISI)'] = res_df['ISI_Base'] * res_df['speed_weight']

    daily_scores = []
    for date_val, group in res_df.groupby('date', observed=True):
        hours = max(len(group) / 7220.0, 1e-9)
        harsh_events = group.loc[group['가혹도(ISI)'] > 25, '가혹도(ISI)'] - 25
        hourly_rate = 0.0 if harsh_events.empty else float((harsh_events ** 1.3).sum() / hours)
        final_score = max(0.0, min(100.0, 100.0 - hourly_rate * penalty_factor))
        daily_scores.append({
            'date': date_val,
            '최종운전점수': round(final_score, 1)
        })

    daily_score_df = pd.DataFrame(daily_scores)
    result_df = (
        res_df.iloc[int(window_size) - 1::int(step_size)]
        .dropna()
        .reset_index(drop=True)
        .merge(daily_score_df, on='date', how='left')
    )

    numeric_cols = result_df.select_dtypes(include=['float64']).columns
    if len(numeric_cols) > 0:
        result_df[numeric_cols] = result_df[numeric_cols].astype('float32')

    return result_df, daily_score_df

def render_severity_tab(df):
    st.write("### 🚗 운전 가혹도 분석")

    if df.empty:
        st.warning("분석할 데이터가 없습니다. 먼저 파일을 업로드해주세요.")
    else:
        dt = 0.5
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
                st.markdown("##### 1️⃣ 축별 가중치 (합계 = 1.0 권장)")
                wx_col, wy_col, wyaw_col = st.columns(3)
                with wx_col:
                    w_x = st.number_input("X축 (가감속) 가중치", min_value=0.0, max_value=1.0, value=0.50, step=0.05)
                with wy_col:
                    w_y = st.number_input("Y축 (선회) 가중치", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
                with wyaw_col:
                    w_yaw = st.number_input("Yaw축 (회전각속도) 가중치", min_value=0.0, max_value=1.0, value=0.20, step=0.05)

                st.markdown("##### 2️⃣ 지표별 가중치 (RMS / STD / Jerk)")
                wrms_col, wstd_col, wjerk_col = st.columns(3)
                with wrms_col:
                    w_rms = st.number_input("RMS 가중치", min_value=0.0, max_value=1.0, value=0.10, step=0.05)
                with wstd_col:
                    w_std = st.number_input("STD 가중치", min_value=0.0, max_value=1.0, value=0.40, step=0.05)
                with wjerk_col:
                    w_jerk = st.number_input("Jerk 가중치", min_value=0.0, max_value=1.0, value=0.50, step=0.05)

                st.markdown("##### 3️⃣ 최종 100점 변환 파라미터")
                penalty_factor = st.number_input("최종 감점 계수 (Scale Factor)", min_value=0.00001, max_value=0.01000, value=0.00038, step=0.00001, format="%.5f")

                submitted = st.form_submit_button("🚀 분석 실행", type="primary")

        if submitted:
            with st.spinner("가혹도 통계량을 초고속 계산 중입니다..."):
                severity_data_key = (
                    len(df),
                    str(df['dataTime'].min()),
                    str(df['dataTime'].max()),
                    float(df['speed'].min()) if 'speed' in df.columns else None,
                    float(df['speed'].max()) if 'speed' in df.columns else None,
                )

                result_df, daily_score_df = calculate_severity_cached(
                    data_key=severity_data_key,
                    _source_df=df,
                    median_window=median_window,
                    step_size=step_size,
                    window_size=window_size,
                    dt=dt,
                    w_x=w_x,
                    w_y=w_y,
                    w_yaw=w_yaw,
                    w_rms=w_rms,
                    w_std=w_std,
                    w_jerk=w_jerk,
                    penalty_factor=penalty_factor,
                )

                st.session_state.tab4_result = result_df
                st.session_state.daily_score_df = daily_score_df
                gc.collect()

            st.success(f"✅ 분석 완료! (총 {len(result_df)}개의 데이터 포인트)")

        if 'tab4_result' in st.session_state:
            res = st.session_state.tab4_result

            if 'daily_score_df' in st.session_state:
                st.write("#### 📅 일별 최종 운전 스코어 (100점 만점)")
                st.dataframe(st.session_state.daily_score_df)

            st.subheader("📈 가혹도 분석 결과 테이블")
            st.dataframe(res, width="stretch", height=400)

            daily_summary = res[['date', '최종운전점수']].drop_duplicates().sort_values('date')
            monthly_score = daily_summary['최종운전점수'].mean()

            st.subheader("🏆 운전 점수 리포트")
            c1, c2 = st.columns(2)
            c1.metric("종합 점수", f"{monthly_score:.1f}점")
            c2.metric("최근 점수", f"{daily_summary['최종운전점수'].iloc[-1]:.1f}점")

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
            st.altair_chart(chart)


