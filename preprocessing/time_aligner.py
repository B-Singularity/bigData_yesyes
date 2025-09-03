import pandas as pd
import numpy as np
import os


class TimeAligner:
    """
    데이터 모델링에 필요한 통계 데이터의 년도, 주기 등을 맞춥니다.
    1985 - 2025년 까지 월별로 데이터 정제를 합니다.
    년도가 초과할 경우 자르고 부족할 경우 통계적인 보간법을 이용해 채웁니다.
    """

    def __init__(self, start_date: str = '1985-01-01', end_date: str = '2025-12-31', freq: str = 'MS'):
        """
        TimeAligner를 초기화하고 마스터 시간 축을 설정합니다.
        """
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        self.master_index = pd.date_range(start=self.start_date, end=self.end_date, freq=self.freq)
        self.datasets = {}  # 처리할 데이터셋을 저장할 딕셔너리

    def _load_single_file(self, file_path: str) -> pd.DataFrame:
        """
        [Private] 단일 파일을 확장자에 따라 읽어 데이터프레임으로 반환합니다.
        """
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

        print(f"'{file_path}' 파일을 읽습니다... (확장자: {file_extension})")

        try:
            if file_extension == '.csv':
                try:
                    df = pd.read_csv(file_path, encoding='cp949')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='euc-kr')
            elif file_extension == '.xlsx':
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"지원하지 않는 파일 형식입니다: {file_extension}")

            print("파일 로딩 성공!")
            return df

        except FileNotFoundError:
            print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
            return None
        except Exception as e:
            print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
            return None

    def load_and_add_dataset(self, dataset_name: str, file_path: str, date_col: str, data_col: str):
        """
        [Public] 파일을 로드하고, 날짜 인덱스를 설정한 뒤 처리 목록에 추가합니다.
        """
        df = self._load_single_file(file_path)
        if df is not None:
            # 원본 데이터프레임 복사
            temp_df = df[[date_col, data_col]].copy()
            # 컬럼 이름 표준화
            temp_df.rename(columns={date_col: 'date', data_col: dataset_name}, inplace=True)
            # 날짜 형식 변환 및 인덱스 설정
            temp_df['date'] = pd.to_datetime(temp_df['date'])
            temp_df.set_index('date', inplace=True)

            self.datasets[dataset_name] = temp_df
            print(f"'{dataset_name}' 데이터셋이 처리 목록에 추가되었습니다.")

    def _merge_and_impute(self) -> pd.DataFrame:
        """
        [Private] 저장된 모든 데이터셋을 병합하고 보간합니다.
        """
        master_df = pd.DataFrame(index=self.master_index)

        for name, df in self.datasets.items():
            master_df = master_df.join(df)

        master_df = master_df.resample(self.freq).last()

        master_df.interpolate(method='linear', inplace=True)
        master_df.bfill(inplace=True)
        master_df.ffill(inplace=True)

        return master_df

    def _save_to_parquet(self, df: pd.DataFrame, file_path: str):
        """
        [Private] 데이터프레임을 Parquet 형식으로 저장합니다.
        """
        print(f"데이터를 '{file_path}' 경로에 Parquet 파일로 저장합니다.")
        df.to_parquet(file_path)

    def process_and_save(self, output_path: str) -> pd.DataFrame:
        """
        [Public] 저장된 모든 데이터셋으로 전체 전처리 파이프라인을 실행합니다.
        """
        if not self.datasets:
            print("오류: 처리할 데이터셋이 없습니다. 먼저 load_and_add_dataset()을 실행하세요.")
            return None

        print("\n1. 데이터 병합 및 보간 시작...")
        final_data = self._merge_and_impute()
        print("완료.")

        self._save_to_parquet(final_data, output_path)
        print("\n모든 전처리 과정이 완료되었습니다.")

        return final_data