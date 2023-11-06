import pandas as pd

# CSV 파일 불러오기
input_file = 'data/현대_senti_2.csv'
output_file = 'data/현대_senti_3.csv'

# CSV 파일을 읽어오고, 날짜 열을 날짜 데이터로 처리하여 불러옵니다.
df = pd.read_csv(input_file, index_col=None, header=None)

# 날짜 형식 변경
df.iloc[1:, 0] = pd.to_datetime(df.iloc[1:, 0], format='%Y%m%d').dt.strftime('%Y-%m-%d')

# 변경된 데이터프레임을 CSV 파일로 저장
df.to_csv(output_file, index=False, header=False)