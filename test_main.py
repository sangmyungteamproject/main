import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import FinanceDataReader as fdr
import tensorflow as tf
import openpyxl
import time
import ast
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Lambda, MaxPooling1D, Dropout
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.summary import create_file_writer, scalar
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l1, l2

from define import *

# 학습 시간 측정
start_time = time.time()

# region 데이터 가져오기
corp_list = []

samsung = load_corp_data('005930', '2003-01-01', '2020-12-31', 'data/삼성전자_senti.csv')
corp_list.append(samsung)

kakao = load_corp_data('005930', '2006-08-01', '2023-10-01', 'data/kakao_senti.csv')
corp_list.append(kakao)

sk = load_corp_data('034730', '2014-02-01', '2023-10-01', 'data/sk_senti.csv')
corp_list.append(sk)

hyundai = load_corp_data('005380', '2016-01-01', '2023-10-01', 'data/현대_senti.csv')
corp_list.append(hyundai)

cj = load_corp_data('001040', '2003-01-01', '2023-10-01', 'data/cj_senti.csv')
corp_list.append(cj)

#nongshim = load_corp_data('004370', '2003-01-01', '2023-10-01', 'data/농심_senti.csv')
#corp_list.append(nongshim)

stock = pd.concat(corp_list, ignore_index=True)
#stock = stock.sort_values(by='Date')

#eco = load_corp_data('247540','2022-12-01','2023-10-01','data/에코프로비엠_senti.csv')

samsung2 = load_corp_data('005930', '2022-09-01', '2023-06-01', 'data/삼성전자_senti.csv')

test_stock = samsung2

# 감성점수 없는 행 삭제
stock = stock.dropna(subset=['senti_val'])
# endregion

# region 입력변수 리스트 추가
# 기본 입력변수 리스트
scale_ft_cols = ['Open', 'High', 'Low', 'Volume']
# 감성점수
if senti:
    scale_ft_cols.append('senti_val')
# 환율
if ex_rate:
    scale_ft_cols.append('USD/KRW')
    scale_ft_cols.append('JPY/KRW')
# 미 국채 10년
if bonds:
    scale_ft_cols.append('DGS10')
# 봉 색깔
if color:
    scale_ft_cols.append('Color')
# 봉 길이
if bar_len:
    scale_ft_cols.append('Bar_len')
# 양봉 길이
if red_bar:
    scale_ft_cols.append('Red_bar')
# 음봉 길이
if blue_bar:
    scale_ft_cols.append('Blue_bar')
# 최근 5일간 양,음봉의 추세
if trend:
    scale_ft_cols.append('Trend')
# 20일 이동평균
if ma20:
    scale_ft_cols.append('MA_20')
# 종가 대비 20일 이평의 상대 위치
if ma_20_1:
    scale_ft_cols.append('MA_20_1')
# 종가가 20일 이평보다 큰지, 작은지 여부
if ma_20_2:
    scale_ft_cols.append('MA_20_2')
# 종가 전고점 대비 현재 종가의 값
if cummax:
    scale_ft_cols.append('Cummax')
# 윗꼬리 길이 = 종가대비 고가와 종가차이
if uptail:
    scale_ft_cols.append('Uptail')
# 최근 5일 최저가 대비 가격 차이
if five_days_change:
    scale_ft_cols.append('5days_change')
# 최근 4일 최저가 대비 가격 차이
if four_days_change:
    scale_ft_cols.append('4days_change')
# 최근 3일 최저 거래량 대비 거래량 차이
if three_vol_change:
    scale_ft_cols.append('3vol_change')
# 거래량 10일 이동평균
if vol_ma10:
    scale_ft_cols.append('Volume_MA10')
# 거래량 10일 이평대비 당일 거래량 값의 위치
if pos_vol10ma:
    scale_ft_cols.append('Pos_Vol10MA')
# endregion

# region 데이터 히트맵 그리기
# correlation_matrix = stock[scale_ft_cols].corr()
#
# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
# plt.title('Correlation Heatmap')
# plt.show()
# endregion

# region 스케일링
ft_scaler = MinMaxScaler()
scaled_ft = ft_scaler.fit_transform(stock[scale_ft_cols])
df = pd.DataFrame(scaled_ft, columns=scale_ft_cols)

tg_scaler = MinMaxScaler()
scaled_tg = tg_scaler.fit_transform(stock[TARGET_DATA].values.reshape(-1, 1))
df[TARGET_DATA] = scaled_tg

test_ft_scaler = MinMaxScaler()
scaled_test_ft = test_ft_scaler.fit_transform(test_stock[scale_ft_cols])
test_df = pd.DataFrame(scaled_test_ft, columns=scale_ft_cols)

test_tg_scaler = MinMaxScaler()
test_scaled_tg = test_tg_scaler.fit_transform(test_stock[TARGET_DATA].values.reshape(-1, 1))
test_df[TARGET_DATA] = test_scaled_tg
# endregion

# region 데이터셋 분할
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test \
    = train_test_split(
    df.drop(labels=TARGET_DATA, axis=1),
    df[TARGET_DATA],
    test_size=TEST_SIZE,
    random_state=0,
    shuffle=False)

z_test = test_df[TARGET_DATA]
z_train = test_df.drop(TARGET_DATA, axis=1)

# train_data는 학습용 데이터셋, test_data는 검증용 데이터셋 입니다.
train_data = windowed_dataset(y_train, WINDOW_SIZE, BATCH_SIZE, True)
test_data = windowed_dataset(y_test, WINDOW_SIZE, BATCH_SIZE, False)

for_train = windowed_dataset(z_train, WINDOW_SIZE, BATCH_SIZE, False)
for_test = windowed_dataset(z_test, WINDOW_SIZE, BATCH_SIZE, False)
# endregion

# region 모델학습
for i in range(0, REP_SIZE):
    # 모델 구조 : 특성추출 레이어(padding = casual -> 현재 위치 이전 정보만 사용하도록 제한), LSTM, Dense
    globals()['test_' + str(i)] = Sequential([
        Conv1D(filters=32, kernel_size=5, padding="causal", activation="relu", input_shape=[WINDOW_SIZE, 1]),
        LSTM(16, activation='tanh'),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dropout(0.2),
        Dense(1),
    ])

    # 최적화 함수
    optimizer = Adam(LEARNING_RATE)
    # 손실함수 = Sequence 학습에 비교적 좋은 퍼포먼스를 내는 Huber()를 사용합니다.
    loss = Huber()

    # 모델 컴파일, 손실함수 = huber, 최적화함수 = adam, 평가지표 = MSE
    globals()['test_' + str(i)].compile(loss=loss, optimizer=optimizer, metrics=['mse'])

    # earlystopping은 10번 epoch통안 val_loss 개선이 없다면 학습을 멈춥니다.
    globals()['e_stop_' + str(i)] = EarlyStopping(monitor='val_loss', patience=PATIENCE)

    # val_loss 기준 체크포인터도 생성합니다.
    globals()['filename_' + str(i)] = os.path.join('tmp', 'checkpointer_' + str(i) + '.ckpt')
    globals()['checkpoint_' + str(i)] = ModelCheckpoint(globals()['filename_' + str(i)],
                                                        save_weights_only=True,
                                                        save_best_only=True,
                                                        monitor='val_loss',
                                                        verbose=1)

    # 모델 훈련
    globals()['test_' + str(i)].fit(train_data, validation_data=test_data, epochs=EPOCH,
                                    callbacks=[globals()['checkpoint_' + str(i)], globals()['e_stop_' + str(i)]])

    globals()['test_' + str(i)].load_weights(globals()['filename_' + str(i)])

    globals()['pred_' + str(i)] = globals()['test_' + str(i)].predict(for_test)
# endregion

# region 리스케일링
idx_count = y_test.count()

if rescale:
    for i in range(0, REP_SIZE):
        globals()['rescaled_pred_' + str(i)] = test_tg_scaler.inverse_transform(
            np.array(globals()['pred_' + str(i)]).reshape(-1, 1))
        tmp = globals()['rescaled_pred_' + str(i)]
        globals()['rescaled_pred_' + str(i)] = pd.DataFrame(tmp, index=test_stock['Date'][WINDOW_SIZE:])

    rescaled_actual = test_tg_scaler.inverse_transform(z_test[WINDOW_SIZE:].values.reshape(-1, 1))
    rescaled_actual = pd.DataFrame(rescaled_actual, index=test_stock.index[WINDOW_SIZE:])
    rescaled_actual.index = test_stock['Date'][WINDOW_SIZE:]

else:
    for i in range(0, REP_SIZE):
        globals()['rescaled_pred_' + str(i)] = np.array(globals()['pred_' + str(i)])
        tmp = globals()['rescaled_pred_' + str(i)]
        globals()['rescaled_pred_' + str(i)] = pd.DataFrame(tmp, index=test_stock['Date'][WINDOW_SIZE:])

    rescaled_actual = z_test[WINDOW_SIZE:]
    rescaled_actual = pd.DataFrame(rescaled_actual)
    rescaled_actual.index = test_stock['Date'][WINDOW_SIZE:]

rep_pred = globals()['rescaled_pred_' + '0']

actual_change_data = test_stock['Close'][WINDOW_SIZE:]
actual_change_data = pd.Series(actual_change_data)
actual_change_data = actual_change_data.diff()
actual_change_data = actual_change_data.iloc[1:]  # 첫 번째 요소 제거

actual_change_data_binary = actual_change_data.apply(lambda x: 1 if x >= 0 else 0)

for i in range(REP_SIZE):
    globals()['rescaled_pred_binary_' + str(i)] = change_binary(globals()['rescaled_pred_' + str(i)])

pred_change_data_binary = globals()['rescaled_pred_binary_' + '0']
# endregion

# 종료 시간 기록
end_time = time.time()
# 실행 시간 계산
execution_time = end_time - start_time
# 결과 출력
print(f"실행 시간: {execution_time:.2f}초")

# region 데이터 시각화
plt.figure(figsize=(12, 9))
plt.plot(rescaled_actual, label='actual')
plt.plot(rep_pred, label='prediction')
plt.legend()
plt.show()
# endregion

# region 평가 데이터 저장
res = confirm_result(rescaled_actual, rep_pred, actual_change_data_binary, pred_change_data_binary, USE_CHANGE_DATA)
print(res)

try:
    # 기존 파일 열기
    res_df = pd.read_excel(FILE_PATH, sheet_name=TARGET_SHEET, index_col=0)
    print("기존 파일 불러오기 성공")

except FileNotFoundError:
    # 새로운 파일 생성
    res_df = pd.DataFrame()
    print("새로운 파일 생성 성공")

# 변경할 작업 수행
for i in range(REP_SIZE):
    res = confirm_result(rescaled_actual, globals()['rescaled_pred_' + str(i)], actual_change_data_binary,
                         globals()['rescaled_pred_binary_' + str(i)], USE_CHANGE_DATA)
    # noinspection PyProtectedMember
    res_df = res_df._append(res, ignore_index=True)

res_df.to_excel(FILE_PATH, sheet_name='Data')
# endregion

# 마지막 예측값
_date = rep_pred.index[-1].strftime('%Y-%m-%d')
_value = int(rep_pred.iloc[-1, 0])
print(f"예측 날짜 : {_date} => 예측 가격 : {_value}")
