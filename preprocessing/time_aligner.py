import pandas as pd
import numpy as np
import os
from statsforecast.models import AutoARIMA
from statsmodels.tsa.arima.model import ARIMA


class TimeAligner:
    def __init__(self, start_date: str = '1995-01', end_date: str = '2025-12', freq: str = 'MS'):
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
        except Exception:
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

    def _backcast_with_arima(self, series_to_forecast: pd.Series) -> pd.Series:
        print(f"\n'{series_to_forecast.name}'에 대한 StatsForecast AutoARIMA 최적 모델 탐색...")
        auto_model = AutoARIMA(season_length=12, trace=False, stepwise=True)
        sf = auto_model.fit(y=series_to_forecast.values)

        best_order = sf.model_.get('order', (1, 1, 1))
        best_seasonal_order = sf.model_.get('seasonal_order', (0, 0, 0, 0))
        print(f"최적 파라미터: {best_order}, 계절성: {best_seasonal_order}")

        model = ARIMA(series_to_forecast, order=best_order, seasonal_order=best_seasonal_order)
        fitted_model = model.fit()

        backcast_start = self.master_index[0]
        backcast_end = series_to_forecast.index[0] - pd.DateOffset(months=1)

        if backcast_start > backcast_end:
            return None

        print(f"'{series_to_forecast.name}' 후방예측 수행: {backcast_start.date()} ~ {backcast_end.date()}")
        backcast_result = fitted_model.predict(start=backcast_start, end=backcast_end)
        return backcast_result

    def _fallback_impute_with_hybrid_model(self, master_df: pd.DataFrame, col: str, series: pd.Series):
        print(f"'{col}' ARIMA 예측 실패. 하이브리드 보간법(Plan B)을 적용합니다.")
        mean = series.mean()
        std_dev = series.std()
        if pd.isnull(std_dev) or std_dev == 0:
            std_dev = 0

        missing_indices = master_df.index[:master_df.index.get_loc(series.index[0])]
        num_missing = len(missing_indices)

        trend_window = series.iloc[:4]
        avg_slope = trend_window.diff().mean()
        if pd.isnull(avg_slope):
            avg_slope = 0

        first_valid_val = series.iloc[0]
        num_steps_back = np.arange(num_missing, 0, -1)
        extrapolated_values = first_valid_val - num_steps_back * avg_slope

        min_real, max_real = series.min(), series.max()
        is_unreasonable = any(extrapolated_values < min_real * 0.5) or any(extrapolated_values > max_real * 1.5) or any(
            extrapolated_values < 0)

        if is_unreasonable:
            print(f"'{col}'의 기울기 예측이 비합리적이므로 '평균+노이즈'로 대체합니다.")
            noise = np.random.normal(loc=0, scale=std_dev, size=num_missing)
            imputed_values = mean + noise
        else:
            print(f"'{col}'에 '기울기+노이즈' 방식을 적용합니다.")
            noise = np.random.normal(loc=0, scale=std_dev, size=num_missing)
            imputed_values = extrapolated_values + noise

        master_df.loc[missing_indices, col] = imputed_values

    def _merge_and_impute(self) -> pd.DataFrame:
        master_df = pd.DataFrame(index=self.master_index)
        for name, df in self.datasets.items():
            master_df = master_df.join(df, how='outer')

        master_df = master_df.loc[self.start_date:self.end_date]
        master_df = master_df.resample(self.freq).mean()
        master_df.interpolate(method='linear', inplace=True)

        for col in master_df.columns:
            if pd.isnull(master_df[col].iloc[0]):
                series = master_df[col].dropna()
                if len(series) > 24:  # ARIMA를 시도하기에 충분한 데이터가 있는지 확인
                    try:
                        backcast = self._backcast_with_arima(series)
                        if backcast is not None:
                            master_df[col].update(backcast)
                            master_df.loc[master_df[col] < 0, col] = 0
                    except Exception as e:
                        # ARIMA 실패 시 하이브리드 모델 호출
                        self._fallback_impute_with_hybrid_model(master_df, col, series)
                else:
                    # 데이터가 너무 적으면 바로 하이브리드 모델 호출
                    self._fallback_impute_with_hybrid_model(master_df, col, series)

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
    aligner.load_and_add_dataset('change_qoq', 'productivity_all_long.csv', 'date', 'change_qoq')
    aligner.load_and_add_dataset('growth_rate_qoq', 'productivity_all_long.csv', 'date', 'growth_rate_qoq')
    aligner.load_and_add_dataset('total_wage', None, 'date', 'Total_Wage', is_preprocessed=True, df=wage_df)
    aligner.load_and_add_dataset('BSI', 'BSI_long.csv', 'Date', 'BSI_Composite')

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