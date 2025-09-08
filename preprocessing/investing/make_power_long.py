import pandas as pd

# 다중 헤더를 가진 CSV 파일 불러오기
df = pd.read_csv("../../investing_data_file/산업별_표준산업코드_중분류__월별_124시_전력소비계수_20250908011030.csv", header=[0, 1])

# '컴퓨터 프로그래밍, 시스템 통합 및 관리업' 관련 열과 식별자 열 선택
id_vars = [('월별', '월별'), ('시간별', '시간별')]
value_vars = [col for col in df.columns if '컴퓨터 프로그래밍' in col[1]]

# 필요한 열만 추출하여 새로운 DataFrame 생성
programming_df = df[id_vars + value_vars]

# DataFrame을 장형식으로 변환
long_df = pd.melt(
    programming_df,
    id_vars=id_vars,
    value_vars=value_vars,
    var_name='기간',
    value_name='전력소비계수'
)

# '기간' 열에서 연도만 남기기
long_df['기간'] = long_df['기간'].apply(lambda x: x[0])

# 열 이름 재정의
long_df.columns = ['월별', '시간별', '기간', '전력소비계수']

# 결과를 CSV 파일로 저장
long_df.to_csv("computer_programming_long.csv", index=False)

print("데이터가 'computer_programming_long.csv' 파일로 성공적으로 변환 및 저장되었습니다.")
print("\n변환된 데이터의 일부:")
print(long_df.head())