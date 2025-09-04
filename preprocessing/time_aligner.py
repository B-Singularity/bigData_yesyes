import pandas as pd
import numpy as np
import os


class TimeAligner:
    def __init__(self, start_date: str = '1985-01', end_date: str = '2025-12', freq: str = 'MS'):
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        self.master_index = pd.date_range(start=self.start_date, end=self.end_date, freq=self.freq)
        self.datasets = {}

    def _load_single_file(self, file_path: str) -> pd.DataFrame:
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

        print(f"'{file_path}' 파일을 읽습니다...")

        try:
            if file_extension == '.csv':
                df = pd.read_csv(file_path, encoding='utf-8-sig')
            elif file_extension == '.xlsx':
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"지원하지 않는 파일 형식입니다: {file_extension}")

            print("파일 로딩 성공!")
            print(f"로드된 컬럼: {df.columns.tolist()}")
            return df
        except Exception as e:
            print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
            try:
                print("euc-kr 인코딩으로 다시 시도합니다...")
                df = pd.read_csv(file_path, encoding='euc-kr')
                print("파일 로딩 성공!")
                print(f"로드된 컬럼: {df.columns.tolist()}")
                return df
            except Exception as e2:
                print(f"euc-kr 인코딩 읽기도 실패했습니다: {e2}")
                return None

    def load_and_add_dataset(self, dataset_name: str, file_path: str, date_col: str, data_col: str,
                             is_preprocessed=False, df=None):
        if not is_preprocessed:
            df = self._load_single_file(file_path)

        if df is not None:
            df.columns = df.columns.str.strip()
            temp_df = df[[date_col, data_col]].copy()
            temp_df.rename(columns={date_col: 'date', data_col: dataset_name}, inplace=True)
            temp_df['date'] = pd.to_datetime(temp_df['date'])

            if self.freq == 'MS':
                temp_df['date'] = temp_df['date'].dt.to_period('M').dt.to_timestamp()

            temp_df.set_index('date', inplace=True)
            self.datasets[dataset_name] = temp_df
            print(f"'{dataset_name}' 데이터셋이 처리 목록에 추가되었습니다.")

    def _merge_and_impute(self) -> pd.DataFrame:
        master_df = pd.DataFrame(index=self.master_index)

        for name, df in self.datasets.items():
            master_df = master_df.join(df, how='outer')

        master_df = master_df.loc[self.start_date:self.end_date]
        master_df = master_df.resample(self.freq).mean()

        master_df.interpolate(method='linear', inplace=True)

        for col in master_df.columns:
            if pd.isnull(master_df[col].iloc[0]):
                first_valid_idx = master_df[col].first_valid_index()
                if first_valid_idx is None:
                    continue

                mean = master_df[col].mean()
                std_dev = master_df[col].std()
                if pd.isnull(std_dev) or std_dev == 0:
                    std_dev = 0

                missing_indices = master_df.index[:master_df.index.get_loc(first_valid_idx)]
                num_missing = len(missing_indices)

                # 1. 기울기 예측 시도
                trend_window = master_df[col].loc[first_valid_idx:].iloc[:4]
                avg_slope = trend_window.diff().mean()
                if pd.isnull(avg_slope):
                    avg_slope = 0

                first_valid_val = master_df[col].loc[first_valid_idx]
                num_steps_back = np.arange(len(missing_indices), 0, -1)
                extrapolated_values = first_valid_val - num_steps_back * avg_slope

                # 2. 합리성 검사
                min_real = master_df[col].min()
                max_real = master_df[col].max()
                is_unreasonable = any(extrapolated_values < min_real * 0.5) or any(
                    extrapolated_values > max_real * 1.5) or any(extrapolated_values < 0)

                # 3. Plan B 적용
                if is_unreasonable:
                    print(f"'{col}' 컬럼의 기울기 예측이 비합리적이라 판단하여 '평균+노이즈' 방식으로 대체합니다.")
                    noise = np.random.normal(loc=0, scale=std_dev, size=num_missing)
                    imputed_values = mean + noise
                else:
                    print(f"'{col}' 컬럼에 '기울기+노이즈' 방식을 적용합니다.")
                    noise = np.random.normal(loc=0, scale=std_dev, size=num_missing)
                    imputed_values = extrapolated_values + noise

                master_df.loc[missing_indices, col] = imputed_values

        master_df.ffill(inplace=True)

        return master_df

    def _save_to_csv(self, df: pd.DataFrame, file_path: str):
        print(f"데이터를 '{file_path}' 경로에 CSV 파일로 저장합니다.")
        df.to_csv(file_path)

    def process_and_save(self, output_path: str) -> pd.DataFrame:
        if not self.datasets:
            print("오류: 처리할 데이터셋이 없습니다.")
            return None

        print("\n1. 데이터 병합 및 보간 시작...")
        final_data = self._merge_and_impute()
        print("완료.")

        self._save_to_csv(final_data, output_path)
        return final_data


if __name__ == '__main__':
    wage_df = pd.read_csv('월별_임금_총액(1993-2019).csv', encoding='utf-8')
    wage_df['date'] = pd.to_datetime(wage_df['Year'].astype(str) + '-' + wage_df['Month'].astype(str), format='%Y-%m')
    print("'월별_임금_총액' 데이터 전처리 완료.")

    aligner = TimeAligner()

    aligner.load_and_add_dataset('population', 'population_long.csv', 'date', 'population')
    aligner.load_and_add_dataset('unemployment_rate', 'unemployment_long.csv', 'date', 'unemployment_rate')
    aligner.load_and_add_dataset('productivity_index', 'productivity_all_long.csv', 'date', 'productivity_index')
    aligner.load_and_add_dataset('total_wage', None, 'date', 'Total_Wage', is_preprocessed=True, df=wage_df)

    temp_output_path = 'aligned_data.csv'
    final_df = aligner.process_and_save(temp_output_path)

    if final_df is not None:
        print(f"\n데이터를 성공적으로 '{temp_output_path}'에 저장했습니다.")
        print("날짜 형식을 'YYYY-MM'으로 변경합니다...")

        final_df.index = final_df.index.to_period('M')

        final_output_path = 'aligned_data_year_month.csv'
        final_df.to_csv(final_output_path)

        print(f"최종 데이터가 '{final_output_path}'에 저장되었습니다.")
        print("\n'년-월' 형식으로 수정된 데이터 (상위 5개):")
        print(final_df.head())
    else:
        print("데이터 처리 중 오류가 발생했습니다.")