# 차량 IMU 통합 분석 - 5개 탭 모듈 분리 버전

## 구조

```text
driving_map_tabs5/
├─ app.py
├─ ui_components.py        # 기존 파일을 동일 위치에 배치
├─ device_list.csv         # 타이어 상태 분석 시 필요
├─ requirements.txt
├─ core/
│  ├─ __init__.py
│  ├─ common.py
│  └─ data_processing.py
└─ tabs/
   ├─ __init__.py
   ├─ visualization_tab.py  # 탭1 시각화 분석
   ├─ statistics_tab.py     # 탭2 데이터 통계
   ├─ wear_factor_tab.py    # 탭3 마모 인자 분석
   ├─ severity_tab.py       # 탭4 운전 가혹도 분석
   └─ tire_tab.py           # 탭5 타이어 상태/마모 예측
```

## 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

기존 `ui_components.py`와 `device_list.csv`는 프로젝트 루트에 복사해야 합니다.
