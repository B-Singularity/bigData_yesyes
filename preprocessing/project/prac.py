import pandas as pd
import numpy as np

try:
    # 1. 원본 마스터 데이터 파일을 읽어옵니다.
    # 이 스크립트가 'simulation_master_data_final.csv'와 같은 폴더에 있어야 합니다.
    file_path = 'simulation_data_with_noise.csv'
    df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    print(f"성공: '{file_path}' 파일을 읽었습니다.")

    # 2. 특성 공학 변수 생성
    print("특성 공학 변수 생성을 시작합니다...")

    # 2-1. 이동평균 (추세 파악)
    # 중기 BSI 추세
    df['BSI_6M_MA'] = df['BSI_Composite'].rolling(window=6).mean()
    # 장기 GDP 추세
    df['GDP_12M_MA'] = df['GDP'].rolling(window=12).mean()

    # 2-2. 변동성 (리스크 측정)
    # 12개월 환율 변동성(표준편차)
    df['Exchange_Rate_12M_Std'] = df['Exchange_Rate'].rolling(window=12).std()

    # 2-3. 모멘텀 (변화의 가속도)
    # 3개월 소비자심리지수 변화량
    df['CCSI_3M_Momentum'] = df['CCSI'].diff(periods=3)
    # 6개월 ICT 생산량 변화량
    df['ICT_Prod_6M_Momentum'] = df['ICT_Production'].diff(periods=6)

    # 2-4. 시차 변수 (지연 효과)
    # 3개월 전 기업 대출 금리
    df['Loan_Rate_Lag_3M'] = df['Corporate_Loan_Rate'].shift(3)
    print("성공: 6개의 새로운 특성 공학 변수를 생성했습니다.")

    # 3. 계산 과정에서 생긴 앞부분의 빈 값(NaN) 채우기
    # bfill을 사용하여 가장 가까운 미래의 값으로 채웁니다.
    df.bfill(inplace=True)
    print("성공: 모든 빈 값을 채웠습니다.")

    # 4. 새로운 파일로 저장
    output_path = 'simulation_data_with_features.csv'
    df.to_csv(output_path)

    print(f"\n모든 작업 완료! 최종 데이터가 '{output_path}' 파일로 저장되었습니다.")

except FileNotFoundError:
    print(f"\n[오류] 입력 파일 '{file_path}'을 찾을 수 없습니다.")
    print("이 파이썬 스크립트와 'simulation_master_data_final.csv' 파일이 같은 폴더에 있는지 확인해주세요.")
except Exception as e:
    print(f"\n[오류] 데이터 처리 중 문제가 발생했습니다: {e}")