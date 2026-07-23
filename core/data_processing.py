import gc
import streamlit as st
import pandas as pd
import numpy as np

from core.common import FLOAT32_COLUMNS

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

        dt_col = pd.to_datetime(df['dataTime'], errors='coerce', format='%Y-%m-%d %H:%M:%S')

        if dt_col.isna().all():
            st.error(f"❌ {file.name}: 'dataTime' 형식이 올바르지 않습니다.")
            return None

        df['dataTime'] = dt_col
        df['date'] = dt_col.dt.date
        df['month'] = dt_col.dt.to_period('M').astype(str).astype('category')
        df['week'] = ("Week " + dt_col.dt.isocalendar().week.astype(str)).astype('category')
        df['day_name'] = dt_col.dt.day_name().astype('category')
        df['overall'] = pd.Series(["전체 기간 (Overall)"] * len(df), dtype='category')

        # 센서 숫자 컬럼은 로딩 직후 float32로 통일하여 메모리 사용량을 절반 수준으로 절감
        for col in FLOAT32_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')

        float_cols = df.select_dtypes(include=['float64']).columns
        if len(float_cols) > 0:
            df[float_cols] = df[float_cols].astype('float32')

        return df

    except Exception as e:
        st.error(f"🔥 파일 로드 중 오류 발생: {e}")
        return None

def calc_rms(series):
    x = series.to_numpy(dtype=np.float32, copy=False)
    x = x[~np.isnan(x)]
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

@st.cache_data(show_spinner=False, max_entries=1)
def get_integrated_data(_files, file_keys):
    df_list = []
    total_files = len(_files)
    progress_bar = st.progress(0, text="데이터 통합을 준비 중입니다...")

    for i, file in enumerate(_files):
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

@st.cache_data(show_spinner=False, max_entries=1)
def process_global_dataframe(raw_data, s_min, s_max, g_lim, r_lim, case_sel, g_col, h_acc, h_brk, h_trn):
    mask = ((raw_data['speed'] >= s_min) & (raw_data['speed'] <= s_max) &
            (raw_data['accXG'].abs() <= g_lim) & (raw_data['accYG'].abs() <= g_lim) &
            (raw_data['yawDps'].abs() <= r_lim) & (raw_data['pitchDps'].abs() <= r_lim) & (raw_data['rollDps'].abs() <= r_lim))
    filtered = raw_data.loc[mask]

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

    sum_df = filtered.groupby(g_col, observed=True).agg(
        평균속도=('speed', 'mean'),
        최대분포속도=('speed', 'median'),
        최대속도=('speed', 'max'),
        급가속횟수=('is_accel', 'sum'),
        급제동횟수=('is_brake', 'sum'),
        좌선회횟수=('is_turn_L', 'sum'),
        우선회횟수=('is_turn_R', 'sum'),
        급선회횟수=('is_turn_any', 'sum'),
        데이터수=('dataTime', 'count')
    )

    # 기 산출된 값 재사용으로 O(1) 비율 연산
    sum_df['가속_비율'] = (sum_df['급가속횟수'] / sum_df['데이터수'] * 100).fillna(0).round(1)
    sum_df['감속_비율'] = (sum_df['급제동횟수'] / sum_df['데이터수'] * 100).fillna(0).round(1)
    sum_df['정속_비율'] = (100.0 - sum_df['가속_비율'] - sum_df['감속_비율']).round(1)

    sum_df['좌선회_비율'] = (sum_df['좌선회횟수'] / sum_df['데이터수'] * 100).fillna(0).round(1)
    sum_df['우선회_비율'] = (sum_df['우선회횟수'] / sum_df['데이터수'] * 100).fillna(0).round(1)
    sum_df['직진_비율'] = (100.0 - sum_df['좌선회_비율'] - sum_df['우선회_비율']).round(1)

    sum_df['마모지수'] = (
        (sum_df['급가속횟수'] * 0.5 + sum_df['급제동횟수'] * 0.7 + sum_df['급선회횟수'] * 1.0) /
        sum_df['데이터수'].replace(0, 1) * 1000
    ).round(2)

    return filtered, sum_df
