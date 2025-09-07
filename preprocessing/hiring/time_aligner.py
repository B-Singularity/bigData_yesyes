import pandas as pd
import numpy as np
import os
from statsforecast.models import AutoARIMA
from statsmodels.tsa.arima.model import ARIMA


class TimeAligner:
    """
    다양한 주기의 시계열 데이터를 월간으로 정렬하고 보간하는 클래스.
    """

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
                df = pd.read_csv(file_path, encoding='utf-8-sig', header=0)
            elif file_extension == '.xlsx':
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"지원하지 않는 파일 형식입니다: {file_extension}")
            print("파일 로딩 성공!")
            return df
        except Exception:
            try:
                print("euc-kr 인코딩으로 다시 시도합니다...")
                df = pd.read_csv(file_path, encoding='euc-kr', header=0)
                print("파일 로딩 성공!")
                return df
            except Exception as e2:
                print(f"euc-kr 인코딩 읽기도 실패했습니다: {e2}")
                return None

    def add_yearly_dataset_as_monthly(self, dataset_name: str, file_path: str, category_col: str, category_name: str):
        """
        연간(Yearly) 데이터를 월간(Monthly)으로 보간하여 데이터셋에 추가합니다.
        """
        df_yearly = self._load_single_file(file_path)
        if df_yearly is None: return

        df_yearly.rename(columns={category_col: 'Category'}, inplace=True)
        df_yearly['Category'] = df_yearly['Category'].str.replace('"', '').str.strip()
        df_yearly_filtered = df_yearly[df_yearly['Category'] == category_name].copy()

        if df_yearly_filtered.empty:
            print(f"경고: {file_path}에서 '{category_name}' 카테고리를 찾지 못했습니다.")
            return

        df_long = df_yearly_filtered.melt(id_vars=['Category'], var_name='date', value_name=dataset_name)
        df_long = df_long[['date', dataset_name]].copy()
        df_long['date'] = pd.to_datetime(df_long['date'], format='%Y')
        df_long.set_index('date', inplace=True)

        # 연간 데이터를 월간으로 리샘플링하고, 비어있는 월들은 선형 보간으로 채웁니다.
        df_monthly = df_long.resample('MS').interpolate(method='linear')

        # 보간된 데이터는 월별 흐름을 나타내므로, 연간 총합을 12로 나눈 값으로 스케일 조정
        df_monthly[dataset_name] = df_monthly[dataset_name] / 12

        self.datasets[dataset_name] = df_monthly
        print(f"-> '{dataset_name}' 연간 데이터가 월간으로 변환되어 목록에 추가되었습니다.")

    def load_and_add_dataset(self, dataset_name: str, file_path: str = None, date_col: str = None, data_col: str = None,
                             is_preprocessed=False, df=None):
        """월간 데이터를 데이터셋 목록에 추가합니다."""
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
            print(f"-> '{dataset_name}' 월간 데이터가 처리 목록에 추가되었습니다.")

    def _backcast_with_arima(self, series_to_forecast: pd.Series) -> pd.Series:
        print(f"\n'{series_to_forecast.name}'에 대한 AutoARIMA 최적 모델 탐색...")
        auto_model = AutoARIMA(season_length=12, trace=False, stepwise=True)
        sf = auto_model.fit(y=series_to_forecast.values)
        best_order = sf.model_.get('order', (1, 1, 1))
        best_seasonal_order = sf.model_.get('seasonal_order', (0, 0, 0, 0))
        model = ARIMA(series_to_forecast, order=best_order, seasonal_order=best_seasonal_order)
        fitted_model = model.fit()
        backcast_start = self.master_index[0]
        backcast_end = series_to_forecast.index[0] - pd.DateOffset(months=1)
        if backcast_start > backcast_end: return None
        print(f"'{series_to_forecast.name}' 후방예측 수행: {backcast_start.date()} ~ {backcast_end.date()}")
        return fitted_model.predict(start=backcast_start, end=backcast_end)

    def _fallback_impute_with_hybrid_model(self, master_df: pd.DataFrame, col: str, series: pd.Series):
        print(f"'{col}' ARIMA 예측 실패. 하이브리드 보간법(Plan B)을 적용합니다.")
        mean, std_dev = series.mean(), series.std()
        if pd.isnull(std_dev) or std_dev == 0: std_dev = 0
        missing_indices = master_df.index[:master_df.index.get_loc(series.index[0])]
        num_missing = len(missing_indices)
        avg_slope = series.iloc[:4].diff().mean()
        if pd.isnull(avg_slope): avg_slope = 0
        first_valid_val = series.iloc[0]
        extrapolated_values = first_valid_val - np.arange(num_missing, 0, -1) * avg_slope
        noise = np.random.normal(loc=0, scale=std_dev, size=num_missing)
        master_df.loc[missing_indices, col] = extrapolated_values + noise

    def _merge_and_impute(self) -> pd.DataFrame:
        master_df = pd.DataFrame(index=self.master_index)
        for name, df in self.datasets.items():
            master_df = master_df.join(df, how='outer')
        master_df = master_df.loc[self.start_date:self.end_date].resample(self.freq).mean()
        master_df.interpolate(method='linear', inplace=True)
        for col in master_df.columns:
            if pd.isnull(master_df[col].iloc[0]):
                series = master_df[col].dropna()
                if len(series) > 24:
                    try:
                        backcast = self._backcast_with_arima(series)
                        if backcast is not None: master_df[col].update(backcast)
                    except Exception:
                        self._fallback_impute_with_hybrid_model(master_df, col, series)
                else:
                    self._fallback_impute_with_hybrid_model(master_df, col, series)
        master_df.ffill(inplace=True)
        return master_df

    def process_and_save(self, output_path: str) -> pd.DataFrame:
        if not self.datasets:
            print("오류: 처리할 데이터셋이 없습니다.")
            return None
        print("\n데이터 병합 및 보간 시작...")
        final_data = self._merge_and_impute()
        final_data.to_csv(output_path)
        print(f"데이터를 '{output_path}' 경로에 CSV 파일로 저장했습니다.")
        return final_data


if __name__ == '__main__':
    try:
        print("--- 1. 개별 데이터 전처리 시작 ---")
        # (BSI 전처리)
        bsi_df_raw = pd.read_csv('../../hire_data_file/기업경영판단_BSI_20250905140637.csv', header=0)
        bsi_df_raw.rename(columns={'기업경영판단별(1)': 'Category'}, inplace=True)
        bsi_df_raw['Category'] = bsi_df_raw['Category'].str.replace('"', '').str.strip()
        bsi_long = bsi_df_raw.melt(id_vars=['Category'], var_name='Date_Raw', value_name='Value')
        bsi_long['Value'] = pd.to_numeric(bsi_long['Value'], errors='coerce')
        bsi_long['Date_Raw'] = bsi_long['Date_Raw'].astype(str).apply(lambda x: '.'.join(x.split('.')[:2]))
        bsi_long['Type_Num'] = bsi_long.groupby(['Category', 'Date_Raw']).cumcount()
        bsi_long['Type'] = bsi_long['Type_Num'].map({0: '전망', 1: '실적'})
        bsi_long['Date'] = pd.to_datetime(bsi_long['Date_Raw'], format='%Y.%m')
        bsi_actuals = bsi_long[bsi_long['Type'] == '실적'].copy()
        bsi_pivot = bsi_actuals.pivot_table(index='Date', columns='Category', values='Value')
        target_cats = ['생산증가율', '내수판매', '수출']
        existing_cats = [col for col in target_cats if col in bsi_pivot.columns]
        bsi_pivot['BSI_Composite'] = bsi_pivot[existing_cats].mean(axis=1)
        bsi_df = bsi_pivot[['BSI_Composite']].reset_index()
        print("BSI 데이터 전처리 완료.")

        print("\n--- 2. 데이터 정렬 및 통합 시작 ---")
        aligner = TimeAligner()

        # (연간 데이터를 월간으로 추가)
        aligner.add_yearly_dataset_as_monthly(
            'GFCF_ICT_Real',
            'hire_data_file/경제활동별_총고정자본형성_실질__연간__20250907155231.csv',
            '계정항목별',
            '정보통신업'
        )

        # (기존 월간 데이터 추가)
        aligner.load_and_add_dataset('population', 'aligned_data_year_month.csv', 'Unnamed: 0', 'population')
        aligner.load_and_add_dataset('unemployment_rate', 'aligned_data_year_month.csv', 'Unnamed: 0',
                                     'unemployment_rate')
        aligner.load_and_add_dataset('productivity_index', 'aligned_data_year_month.csv', 'Unnamed: 0',
                                     'productivity_index')
        aligner.load_and_add_dataset('growth_rate_qoq', 'aligned_data_year_month.csv', 'Unnamed: 0', 'growth_rate_qoq')
        aligner.load_and_add_dataset('total_wage', 'aligned_data_year_month.csv', 'Unnamed: 0', 'total_wage')
        aligner.load_and_add_dataset('BSI_Composite', df=bsi_df, is_preprocessed=True, date_col='Date',
                                     data_col='BSI_Composite')

        # (3. 모든 데이터를 통합하고 저장)
        final_output_path = 'aligned_data_with_GFCF.csv'
        final_df = aligner.process_and_save(final_output_path)

        if final_df is not None:
            print(f"\n모든 데이터가 성공적으로 '{final_output_path}'에 저장되었습니다.")
            print("최종 통합 데이터 (상위 5개):")
            print(final_df.head())

    except FileNotFoundError as e:
        print(f"\n[오류] 파일 없음: '{e.filename}' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"\n[오류] 분석 중 예외가 발생했습니다: {e}")