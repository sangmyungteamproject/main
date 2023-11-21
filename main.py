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
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Lambda, MaxPooling1D, Dropout
from tensorflow.keras.losses import Huber, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.summary import create_file_writer, scalar
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l1, l2
from define import *

# 학습 시간 측정
start_time = time.time()

# region 데이터 가져오기
hyundai = load_corp_data('005380', '2016-01-04', '2018-02-08', 'data/현대_senti_3.csv')
stock = hyundai

# 마지막 날 더미 데이터 삽입
last_row = stock.iloc[-1].copy()
current_index = last_row.name
new_date = pd.to_datetime(last_row['Date']) + timedelta(days=1)
while new_date.weekday() >= 5:
    new_date += timedelta(days=1)
last_row['Date'] = new_date
last_row['datetime'] = new_date
new_index = new_date
# last_row.name = new_index
stock = stock._append(last_row, ignore_index=False)


# endregion

# region 입력변수 리스트 추가
scale_ft_cols = []
if open_price:
    scale_ft_cols.append('Open')
if high_price:
    scale_ft_cols.append('High')
if low_price:
    scale_ft_cols.append('Low')
scale_ft_cols.append('Close')
if volume:
    scale_ft_cols.append('Volume')

if pos_count:
    scale_ft_cols.append('positive_count')
if pos_score:
    scale_ft_cols.append('positive_score')
if neg_count:
    scale_ft_cols.append('negative_count')
if neg_score:
    scale_ft_cols.append('negative_score')
if total_count:
    scale_ft_cols.append('total_count')
if total_score:
    scale_ft_cols.append('total_score')
if senti:
    scale_ft_cols.append('senti_val')

if rsi:
    scale_ft_cols.append('RSI')
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

# region 스케일링

# 스케일 후 columns
ft_scaler = MinMaxScaler()
scaled_ft = ft_scaler.fit_transform(stock[scale_ft_cols])
df = pd.DataFrame(scaled_ft, columns=scale_ft_cols)

tg_scaler = MinMaxScaler()
scaled_tg = tg_scaler.fit_transform(stock[TARGET_DATA].values.reshape(-1, 1))
df[TARGET_DATA] = scaled_tg
# endregion

# region 데이터셋 분할
train = df.head(int((1 - TEST_SIZE) * len(df)))
test = df.tail(int(TEST_SIZE * len(df)))

train_ft = train[scale_ft_cols]
train_tg = train[TARGET_DATA]
train_ft, train_tg = make_dataset(train_ft, train_tg, WINDOW_SIZE)

test_ft = test[scale_ft_cols]
test_tg = test[TARGET_DATA]
test_ft, test_tg = make_dataset(test_ft, test_tg, WINDOW_SIZE)
# endregion

# region 모델학습
for i in range(0, REP_SIZE):
    # 모델 구조 : 특성추출 레이어(padding = casual -> 현재 위치 이전 정보만 사용하도록 제한), LSTM, Dense
    model = Sequential([
        Conv1D(filters=32, kernel_size=5, padding="causal", activation="relu", input_shape=[train_ft.shape[1], train_ft.shape[2]]),
        LSTM(16, activation='tanh'),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    # 최적화 함수
    optimizer = Adam(LEARNING_RATE)
    # 손실 함수
    loss = BinaryCrossentropy()

    # 모델 컴파일
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    # earlystopp
    # earlystopping = EarlyStopping(monitor='val_accuracy', patience=PATIENCE)

    # val_loss 기준 체크포인터도 생성합니다.
    # file_path = os.path.join('tmp', 'checkpointer_' + str(i) + '.ckpt')
    # ckpt = ModelCheckpoint(file_path,
    #                        save_weights_only=True,
    #                        save_best_only=True,
    #                        monitor='val_accuracy',
    #                        verbose=1)

    # 모델 훈련
    history = model.fit(x=train_ft, y=train_tg, validation_split=0.1875, epochs=EPOCH, batch_size=BATCH_SIZE)

    # model.load_weights(file_path)

    globals()['test_' + str(i)] = model
    # globals()['checkpoint_' + str(i)] = ckpt
    globals()['history_' + str(i)] = history

    globals()['pred_' + str(i)] = globals()['test_' + str(i)].predict(test_ft)

# endregion

# region 리스케일링
idx_count = test[TARGET_DATA].count()
if rescale:
    for i in range(0, REP_SIZE):
        globals()['rescaled_pred_' + str(i)] = tg_scaler.inverse_transform(
            np.array(globals()['pred_' + str(i)]).reshape(-1, 1))
        tmp = globals()['rescaled_pred_' + str(i)]
        globals()['rescaled_pred_' + str(i)] = pd.DataFrame(tmp, index=stock['Date'][-idx_count + WINDOW_SIZE:])

    rescaled_actual = tg_scaler.inverse_transform(test[TARGET_DATA][WINDOW_SIZE:].values.reshape(-1, 1))
    rescaled_actual = pd.DataFrame(rescaled_actual, index=stock.index[-idx_count + WINDOW_SIZE:])
    rescaled_actual.index = stock['Date'][-idx_count + WINDOW_SIZE:]

else:
    for i in range(0, REP_SIZE):
        globals()['rescaled_pred_' + str(i)] = np.array(globals()['pred_' + str(i)])
        tmp = globals()['rescaled_pred_' + str(i)]
        globals()['rescaled_pred_' + str(i)] = pd.DataFrame(tmp, index=stock['Date'][-idx_count + WINDOW_SIZE:])

        rescaled_actual = test[TARGET_DATA][WINDOW_SIZE:]
        rescaled_actual = pd.DataFrame(rescaled_actual)
        rescaled_actual.index = stock['Date'][-idx_count + WINDOW_SIZE:]
# endregion

# region 모델 평가
rep_pred = globals()['rescaled_pred_' + '0']

actual_change_data = stock[TARGET_DATA][-idx_count + WINDOW_SIZE:]
actual_change_data = pd.Series(actual_change_data)
if TARGET_DATA != 'Signal':
    actual_change_data = actual_change_data.diff()
    actual_change_data = actual_change_data.iloc[1:]  # 첫 번째 요소 제거
    actual_change_data_binary = actual_change_data.apply(lambda x: 1 if x >= 0 else 0)
else:
    actual_change_data_binary = actual_change_data

for i in range(REP_SIZE):
    globals()['rescaled_pred_binary_' + str(i)] = change_binary(globals()['rescaled_pred_' + str(i)], TARGET_DATA)

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

# region 학습 그래프
rep_history = globals()['history_' + '0']
loss = rep_history.history['loss']
accuracy = rep_history.history['accuracy']
val_loss = rep_history.history['val_loss']
val_accuracy = rep_history.history['val_accuracy']
if DRAW_GRAPH:
    # 손실 그래프
    plt.figure(figsize=(12, 9))
    plt.plot(loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.title('train and val Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    # 정확도 그래프
    plt.figure(figsize=(12, 9))
    plt.plot(accuracy, label='train_acc')
    plt.plot(val_accuracy, label='val_acc')
    plt.title('train and val Acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
# endregion

# 마지막 예측값
_date = rep_pred.index[-1].strftime('%Y-%m-%d')
if TARGET_DATA != 'Signal':
    _value = int(rep_pred.iloc[-1, 0])
    print(f"예측 날짜 : {_date} => 예측 가격 : {_value}")
else:
    _direction = int(rep_pred.iloc[-1, 0])
    if _direction == 1:
        print(f"예측 날짜 : {_date} => 예측 방향 : 상승")
    else:
        print(f"예측 날짜 : {_date} => 예측 방향 : 하락")
