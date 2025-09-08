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

    def add_preprocessed_df(self, df: pd.DataFrame, dataset_name: str, date_col: str, data_col: str):
        if df is None or df.empty:
            print(f"경고: '{dataset_name}'에 대한 데이터프레임이 비어있어 건너뜁니다.")
            return
        if date_col not in df.columns or data_col not in df.columns:
            print(f"경고: '{dataset_name}' 처리 중 필요한 컬럼('{date_col}' 또는 '{data_col}')을 찾을 수 없어 건너뜁니다.")
            return

        df.columns = df.columns.str.strip()
        temp_df = df[[date_col, data_col]].copy()
        temp_df.rename(columns={date_col: 'date', data_col: dataset_name}, inplace=True)
        temp_df['date'] = pd.to_datetime(temp_df['date'])
        temp_df['date'] = temp_df['date'].dt.to_period('M').dt.to_timestamp()

        temp_df.set_index('date', inplace=True)

        # 중복된 날짜 인덱스는 평균값으로 처리
        if not temp_df.index.is_unique:
            temp_df = temp_df.groupby('date').mean()

        self.datasets[dataset_name] = temp_df
        print(f"-> '{dataset_name}' 데이터가 처리 목록에 추가되었습니다.")

    def add_yearly_df_as_monthly(self, df: pd.DataFrame, dataset_name: str, date_col: str, data_col: str):
        if df is None or df.empty:
            print(f"경고: '{dataset_name}'에 대한 데이터프레임이 비어있어 건너뜁니다.")
            return
        if date_col not in df.columns or data_col not in df.columns:
            print(f"경고: '{dataset_name}' 처리 중 필요한 컬럼('{date_col}' 또는 '{data_col}')을 찾을 수 없어 건너뜁니다.")
            return

        temp_df = df[[date_col, data_col]].copy()
        temp_df.rename(columns={date_col: 'date', data_col: dataset_name}, inplace=True)
        temp_df['date'] = pd.to_datetime(temp_df['date'], errors='coerce').dt.to_period('A').dt.to_timestamp()
        temp_df.dropna(subset=['date'], inplace=True)

        temp_df.set_index('date', inplace=True)

        # 중복된 날짜 인덱스는 평균값으로 처리
        if not temp_df.index.is_unique:
            temp_df = temp_df.groupby('date').mean()

        df_monthly = temp_df.resample('MS').asfreq().interpolate(method='linear')
        df_monthly[dataset_name] = df_monthly[dataset_name] / 12

        self.datasets[dataset_name] = df_monthly
        print(f"-> '{dataset_name}' 연간 데이터가 월간으로 변환되어 목록에 추가되었습니다.")

    def _backcast_with_arima(self, series_to_forecast: pd.Series) -> pd.Series:
        series_to_forecast = series_to_forecast.asfreq('MS').interpolate()
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
        print(f"'{col}' ARIMA 예측 실패 또는 데이터 부족. 하이브리드 보간법(Plan B)을 적용합니다.")
        mean, std_dev = series.mean(), series.std()
        if pd.isnull(std_dev) or std_dev == 0: std_dev = 0
        missing_indices = master_df.index[master_df.index < series.index[0]]
        num_missing = len(missing_indices)
        if num_missing == 0: return

        avg_slope = series.iloc[:4].diff().mean()
        if pd.isnull(avg_slope): avg_slope = 0
        first_valid_val = series.iloc[0]
        extrapolated_values = first_valid_val - np.arange(num_missing, 0, -1) * avg_slope
        noise = np.random.normal(loc=0, scale=std_dev / 10, size=num_missing)
        master_df.loc[missing_indices, col] = extrapolated_values + noise

    def _merge_and_impute(self) -> pd.DataFrame:
        master_df = pd.DataFrame(index=self.master_index)
        for name, df in self.datasets.items():
            if not df.index.is_unique:
                df = df[~df.index.duplicated(keep='first')]
            master_df = master_df.join(df, how='outer')

        master_df = master_df.loc[self.start_date:self.end_date].resample(self.freq).mean()

        master_df.interpolate(method='linear', limit_direction='both', inplace=True)

        for col in master_df.columns:
            if pd.isnull(master_df[col].iloc[0]):
                series = master_df[col].dropna()
                if not series.empty and len(series) > 24:
                    try:
                        backcast = self._backcast_with_arima(series)
                        if backcast is not None: master_df[col].update(backcast)
                    except Exception as e:
                        print(f"ARIMA 오류 발생 ({col}): {e}")
                        self._fallback_impute_with_hybrid_model(master_df, col, series)
                elif not series.empty:
                    self._fallback_impute_with_hybrid_model(master_df, col, series)

        master_df.ffill(inplace=True)
        master_df.bfill(inplace=True)
        return master_df

    def process_and_save(self, output_path: str) -> pd.DataFrame:
        if not self.datasets:
            print("오류: 처리할 데이터셋이 없습니다.")
            return None
        print("\n--- 3. 데이터 병합 및 보간 시작 ---")
        final_data = self._merge_and_impute()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_data.to_csv(output_path)
        print(f"\n데이터를 '{output_path}' 경로에 CSV 파일로 저장했습니다.")
        return final_data


def preprocess_ict_export_import(file_path):
    print(f"전처리: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path)
    df_pivot = df.pivot_table(index='연도', columns='구분', values='금액(백만US$)').reset_index()
    df_pivot.columns = ['연도', 'ICT_Import', 'ICT_Export']
    return df_pivot


def preprocess_info_comm(file_path):
    print(f"전처리: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path)
    df['날짜'] = pd.to_datetime(df['기간'].str.split('/').str[0], format='%Y.%m')
    return df


def preprocess_investment(file_path):
    print(f"전처리: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path)
    df['지수종류'] = df['지수종류'].str.strip()
    df_pivot = df.pivot_table(index='날짜', columns=['도메인', '지수종류'], values='돈')
    df_pivot.columns = ['_'.join(col).strip().replace(' (2020=100)', '') for col in df_pivot.columns.values]
    df_pivot.reset_index(inplace=True)
    return df_pivot


def preprocess_software(file_path):
    print(f"전처리: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path)
    df = df[df['비목별(2)'] == '컴퓨터 소프트웨어'].copy()
    df['금액'] = pd.to_numeric(df['금액'], errors='coerce')
    df.dropna(subset=['금액'], inplace=True)
    return df


def safe_read_csv(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"경고: 파일 없음 '{os.path.basename(file_path)}', 건너뜁니다.")
        return None


if __name__ == '__main__':
    DATA_DIRECTORY = './'
    file_paths = {
        "ict_export_import": os.path.join(DATA_DIRECTORY, 'ict_export_import_total_long.csv'),
        "info_comm": os.path.join(DATA_DIRECTORY, 'info_comm_long.csv'),
        "investment_raw": os.path.join(DATA_DIRECTORY, 'investment_final_data_all_categories.csv'),
        "software": os.path.join(DATA_DIRECTORY, 'software_long.csv'),
        "exchange_rate_raw": os.path.join(DATA_DIRECTORY, 'korea_exchange_rate_long.csv'),
        "gfcf_total_raw": os.path.join(DATA_DIRECTORY, 'GFCF_ICT_Real_long.csv'),
        "corporate_loans_raw": os.path.join(DATA_DIRECTORY, 'corporate_loans_long.csv'),
    }

    try:
        print("--- 1. 개별 데이터 전처리 시작 ---")
        df_ict_exp_imp = preprocess_ict_export_import(file_paths["ict_export_import"])
        df_info_comm = preprocess_info_comm(file_paths["info_comm"])
        df_investment = preprocess_investment(file_paths["investment_raw"])
        df_software = preprocess_software(file_paths["software"])

        print("\n--- 2. 데이터 정렬 및 통합 시작 ---")
        aligner = TimeAligner()

        aligner.add_yearly_df_as_monthly(df_ict_exp_imp, 'ICT_Export', '연도', 'ICT_Export')
        aligner.add_yearly_df_as_monthly(df_ict_exp_imp, 'ICT_Import', '연도', 'ICT_Import')
        aligner.add_yearly_df_as_monthly(safe_read_csv(file_paths["exchange_rate_raw"]), 'Exchange_Rate', '연도', '환율')
        aligner.add_yearly_df_as_monthly(safe_read_csv(file_paths["gfcf_total_raw"]), 'GFCF_Total_Real', 'Year',
                                         'GFCF_Real')
        aligner.add_yearly_df_as_monthly(df_software, 'Software_Investment', '기간', '금액')

        aligner.add_preprocessed_df(safe_read_csv(file_paths["corporate_loans_raw"]), 'Corporate_Loan_Rate', '날짜',
                                    '금리(연리%)')
        aligner.add_preprocessed_df(df_info_comm, 'GFCF_InfoComm_Quarterly', '날짜', '값')

        investment_cols = [col for col in df_investment.columns if col != '날짜']
        for col in investment_cols:
            aligner.add_preprocessed_df(df_investment, col, '날짜', col)

        final_output_path = 'outputs/aligned_final_dataset_originals.csv'
        final_df = aligner.process_and_save(final_output_path)

        if final_df is not None:
            print("\n모든 원본 데이터가 성공적으로 통합되었습니다.")
            print("최종 통합 데이터 (상위 5개):")
            print(final_df.head())
            print(f"\n총 {len(final_df.columns)}개의 컬럼이 생성되었습니다.")

    except FileNotFoundError as e:
        print(f"\n[오류] 파일 없음: '{e.filename}' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"\n[오류] 분석 중 예외가 발생했습니다: {e}")