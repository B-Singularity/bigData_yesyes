
from preprocessing.hiring.time_aligner import TimeAligner


def main():
    """
    데이터 전처리 파이프라인을 실행하는 메인 함수
    """
    print("===== 데이터 전처리 파이프라인 시작 =====")

    # 1. TimeAligner 객체 생성 (시뮬레이션 기간: 1985년 1월 ~ 2025년 12월)
    aligner = TimeAligner(start_date='1985-01-01', end_date='2025-12-31')

    # 2. 처리할 데이터 파일 목록 정의
    # (파일 경로, 날짜 컬럼명, 데이터 컬럼명)
    # ★★★ 이 부분을 자신의 실제 파일에 맞게 수정해야 합니다 ★★★
    files_to_process = [
        {
            'name': 'unemployment_rate',
            'path': 'unemployment_data.csv',
            'date_col': '시점',  # 실제 파일의 날짜 컬럼 이름
            'data_col': '실업률 (%)'  # 실제 파일의 데이터 컬럼 이름
        },
        {
            'name': 'gdp',
            'path': 'gdp_data.xlsx',
            'date_col': '분기',
            'data_col': '국내총생산'
        },
        {
            'name': 'population',
            'path': 'population_data.csv',
            'date_col': '연도',
            'data_col': '총인구 (명)'
        }
        # ... 여기에 처리하고 싶은 모든 파일을 추가 ...
    ]

    # 3. 각 파일을 순서대로 로드하고 aligner에 추가
    print("\n--- 1. 개별 데이터 파일 로딩 및 추가 ---")
    for file_info in files_to_process:
        aligner.load_and_add_dataset(
            dataset_name=file_info['name'],
            file_path=file_info['path'],
            date_col=file_info['date_col'],
            data_col=file_info['data_col']
        )
        print("-" * 20)

    # 4. 모든 데이터셋에 대한 최종 처리 및 저장 실행
    print("\n--- 2. 전체 데이터 병합, 보간 및 저장 ---")
    final_df = aligner.process_and_save('master_data.parquet')

    # 5. 결과 확인
    if final_df is not None:
        print("\n--- 3. 최종 결과 확인 ---")
        print("최종 마스터 데이터프레임 (상위 5개):")
        print(final_df.head())
        print("\n최종 마스터 데이터프레임 (하위 5개):")
        print(final_df.tail())
        print(f"\n최종 Shape: {final_df.shape}")
        print(f"결측치 없음 확인: {final_df.isnull().sum().sum() == 0}")

    print("\n===== 데이터 전처리 파이프라인 종료 =====")


if __name__ == '__main__':
    main()