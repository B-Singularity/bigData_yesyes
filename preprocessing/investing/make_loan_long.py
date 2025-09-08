import pandas as pd
import os

# --- 설정 ---
source_file = "../../investing_data_file/예금은행_대출금리_신규취급액_기준__20250908004729.csv"
output_folder = "outputs"
output_file = os.path.join(output_folder, "corporate_loans_long.csv")

try:
    # 1. CSV 파일 불러오기 (첫 번째 줄을 헤더로 사용)
    df = pd.read_csv(source_file, header=0)

    # 2. '계정항목별' 컬럼에서 '기업대출 (연리%)'만 필터링
    filtered_df = df[df['계정항목별'].str.strip() == '기업대출 (연리%)'].copy()

    # 3. Wide 포맷 -> Long 포맷으로 변환 (melt)
    #    - (수정) id_vars에서 존재하지 않는 '단위' 컬럼 제거
    long_df = filtered_df.melt(
        id_vars=['계정항목별'],
        var_name='날짜',
        value_name='금리(연리%)'
    )

    # 4. 최종 데이터 정리
    long_df['날짜'] = pd.to_datetime(long_df['날짜'], format='%Y.%m')
    long_df['금리(연리%)'] = pd.to_numeric(long_df['금리(연리%)'], errors='coerce')
    long_df.dropna(subset=['금리(연리%)'], inplace=True)

    #    - 불필요한 '계정항목별' 컬럼은 제거하고 날짜순으로 정렬
    final_df = long_df.drop(columns=['계정항목별']).sort_values(by='날짜').reset_index(drop=True)

    # 5. CSV 파일로 저장
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    final_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"기업대출 데이터 추출 및 변환 완료: {output_file}")
    print("\n### 최종 데이터 샘플 ###")
    print(final_df.head())

except FileNotFoundError:
    print(f"오류: '{source_file}' 파일을 찾을 수 없습니다.")
except KeyError:
    print("오류: '계정항목별' 컬럼을 찾을 수 없습니다. 파일의 첫 번째 줄이 컬럼 이름인지 확인해주세요.")
except Exception as e:
    print(f"알 수 없는 오류가 발생했습니다: {e}")