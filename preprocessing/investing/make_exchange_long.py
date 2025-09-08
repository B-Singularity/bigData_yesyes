import pandas as pd
import os

# --- 설정 ---
source_file = "../../investing_data_file/환율_20250908003618.csv"
output_folder = "outputs"
output_file = os.path.join(output_folder, "korea_exchange_rate_long.csv")

try:
    # 1. CSV 파일 불러오기 (첫 번째 줄을 헤더로 사용)
    df = pd.read_csv(source_file, header=0)

    # 2. '국가별' 컬럼에서 '대한민국'만 필터링
    filtered_df = df[df['국가(통화단위)별'].str.strip() == '대한민국(Won)'].copy()

    # 3. Wide 포맷 -> Long 포맷으로 변환 (melt)
    #    - id_vars: 기준이 될 '국가별' 컬럼
    #    - var_name: 변환 후 연도가 들어갈 컬럼 이름 ('연도')
    #    - value_name: 변환 후 환율값이 들어갈 컬럼 이름 ('환율')
    long_df = filtered_df.melt(
        id_vars=['국가(통화단위)별'],
        var_name='연도',
        value_name='환율'
    )

    # 4. 최종 데이터 정리
    #    - '환율' 컬럼을 숫자 형식으로 변환, 변환 안되는 값('-')은 삭제
    long_df['환율'] = pd.to_numeric(long_df['환율'], errors='coerce')
    long_df.dropna(subset=['환율'], inplace=True)

    #    - '국가별' 컬럼은 이제 필요 없으므로 제거하고 연도순으로 정렬
    final_df = long_df.drop(columns=['국가(통화단위)별']).sort_values(by='연도').reset_index(drop=True)

    # 5. CSV 파일로 저장
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    final_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"대한민국 환율 데이터 추출 및 변환 완료: {output_file}")
    print("\n### 최종 데이터 샘플 ###")
    print(final_df.head())

except FileNotFoundError:
    print(f"오류: '{source_file}' 파일을 찾을 수 없습니다.")
except KeyError:
    print("오류: '국가별' 컬럼을 찾을 수 없습니다. 파일의 첫 번째 줄이 컬럼 이름인지 확인해주세요.")
except Exception as e:
    print(f"알 수 없는 오류가 발생했습니다: {e}")