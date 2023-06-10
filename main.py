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

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Lambda, MaxPooling1D, Dropout
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from define import windowed_dataset, confirm_result, STOCK_CODE, EPOCH, \
    LEARNING_RATE, TEST_SIZE, DATA_DATE, WINDOW_SIZE, BATCH_SIZE, REP_SIZE, \
    FILE_PATH, TARGET_SHEET, change_binary, perform_dickey_fuller_test, CONV1D_LSTM, \
    _rescale, positive_negative, chk_red_bar, chk_blue_bar, cal_ma, cal_cummax, \
    cal_days_change

TARGET_DATA = 'Close'
USE_CHANGE_DATA = False

# 시간 측정
start_time = time.time()

# 입력변수 리스트
scale_ft_cols = ['Open', 'High', 'Low', 'Volume']

# region 데이터 가져오기
stock = fdr.DataReader(STOCK_CODE, DATA_DATE)
stock['Year'] = stock.index.year
stock['Month'] = stock.index.month
stock['Day'] = stock.index.day
stock.head()

# 결측값 제거
stock = stock[stock['Open'] != 0]

# endregion

# region 입력 변수 데이터 생성
stock['Diff'] = stock['Close'] - stock['Open']

stock['Color'] = stock['Diff'].apply(positive_negative)  # 봉 색깔
scale_ft_cols.append('Color')

stock['Bar_len'] = stock['Diff'] / stock['Open']  # 봉 길이
scale_ft_cols.append('Bar_len')

stock['Red_bar'] = stock.apply(chk_red_bar, axis=1) # 양봉 길이
stock['Blue_bar'] = stock.apply(chk_blue_bar, axis=1)  # 음봉 길이
scale_ft_cols.append('Red_bar')
scale_ft_cols.append('Blue_bar')

# 최근 5일간 양,음봉의 추세
stock['Trend'] = stock['Color'].shift(4) + stock['Color'].shift(3) \
                 + stock['Color'].shift(2) + stock['Color'].shift(1) + stock['Color']
scale_ft_cols.append('Trend')

# 20일 이동평균
stock['MA_20'] = cal_ma(stock, 'Close', 20)
scale_ft_cols.append('MA_20')

# 종가 대비 20일 이평의 상대 위치
stock['MA_20_1'] = stock['MA_20'] / stock['Close']
scale_ft_cols.append('MA_20_1')

# 종가가 20일 이평보다 큰지, 작은지 여부
stock['MA_20_2'] = stock['MA_20_1'].apply(lambda x: 1 if x >= 1 else 0)
scale_ft_cols.append('MA_20_2')

# 종가 전고점 대비 현재 종가의 값
stock['Cummax'] = stock['Close'] / stock['Close'].expanding().max()
scale_ft_cols.append('Cummax')

# 윗꼬리 길이 = 종가대비 고가와 종가차이
stock['Uptail'] = (stock['High'] - stock['Close']) / stock['Close']
scale_ft_cols.append('Uptail')

# 최근 5일 최저가 대비 가격 차이
stock['5days_change'] = cal_days_change(stock, 5, 'Close')
scale_ft_cols.append('5days_change')

# 최근 4일 최저가 대비 가격 차이
stock['4days_change'] = cal_days_change(stock, 4, 'Close')
scale_ft_cols.append('4days_change')

# 최근 3일 최저 거래량 대비 거래량 차이
stock['3vol_change'] = cal_days_change(stock, 3, 'Volume')
scale_ft_cols.append('3vol_change')

# 거래량 10일 이동평균
stock['Volume_MA10'] = cal_ma(stock, 'Volume', 10)
scale_ft_cols.append('Volume_MA10')

# 거래량 10일 이평대비 당일 거래량 값의 위치
stock['Pos_Vol10MA'] = stock['Volume'] / stock['Volume_MA10']
scale_ft_cols.append('Pos_Vol10MA')

stock = stock.iloc[20:]
# endregion

# region 스케일링
ft_scaler = MinMaxScaler()
scaled_ft = ft_scaler.fit_transform(stock[scale_ft_cols])
df = pd.DataFrame(scaled_ft, columns=scale_ft_cols)

tg_scaler = MinMaxScaler()
scaled_tg = tg_scaler.fit_transform(stock[TARGET_DATA].values.reshape(-1, 1))
df[TARGET_DATA] = scaled_tg

print(df)
# endregion

# region 데이터셋 분할
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test \
    = train_test_split(
    df.drop(labels='Close', axis=1),
    df['Close'],
    test_size=TEST_SIZE,
    random_state=0,
    shuffle=False)

# train_data는 학습용 데이터셋, test_data는 검증용 데이터셋 입니다.
train_data = windowed_dataset(y_train, WINDOW_SIZE, BATCH_SIZE, True)
test_data = windowed_dataset(y_test, WINDOW_SIZE, BATCH_SIZE, False)
# endregion

# region 모델학습
for i in range(0, REP_SIZE):
    # 모델 구조 : 특성추출 레이어(padding = casual -> 현재 위치 이전 정보만 사용하도록 제한), LSTM, Dense
    globals()['test_' + str(i)] = Sequential(
        [Conv1D(filters=32, kernel_size=5, padding="causal", activation="relu", input_shape=[WINDOW_SIZE, 1]),
         LSTM(16, activation='tanh'), Dense(16, activation="relu"), Dense(1), ])

    # 최적화 함수
    optimizer = Adam(LEARNING_RATE)
    # 손실함수 = Sequence 학습에 비교적 좋은 퍼포먼스를 내는 Huber()를 사용합니다.
    loss = Huber()

    # 모델 컴파일, 손실함수 = huber, 최적화함수 = adam, 평가지표 = MSE
    globals()['test_' + str(i)].compile(loss=loss, optimizer=optimizer, metrics=['mse'])

    # earlystopping은 10번 epoch통안 val_loss 개선이 없다면 학습을 멈춥니다.
    # val_loss = mse
    globals()['e_stop_' + str(i)] = EarlyStopping(monitor='val_loss', patience=10)

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

    globals()['pred_' + str(i)] = globals()['test_' + str(i)].predict(test_data)
# endregion

# region 리스케일링
idx_count = y_test.count()

if _rescale:
    for i in range(0, REP_SIZE):
        globals()['rescaled_pred_' + str(i)] = tg_scaler.inverse_transform(
            np.array(globals()['pred_' + str(i)]).reshape(-1, 1))
        tmp = globals()['rescaled_pred_' + str(i)]
        globals()['rescaled_pred_' + str(i)] = pd.DataFrame(tmp, index=stock.index[-idx_count + WINDOW_SIZE:])

    rescaled_actual = tg_scaler.inverse_transform(y_test[WINDOW_SIZE:].values.reshape(-1, 1))
    rescaled_actual = pd.DataFrame(rescaled_actual, index=stock.index[-idx_count + WINDOW_SIZE:])
    rescaled_actual.index = stock.index[-idx_count + WINDOW_SIZE:]

else:
    for i in range(0, REP_SIZE):
        globals()['rescaled_pred_' + str(i)] = np.array(globals()['pred_' + str(i)])
        tmp = globals()['rescaled_pred_' + str(i)]
        globals()['rescaled_pred_' + str(i)] = pd.DataFrame(tmp, index=stock.index[-idx_count + WINDOW_SIZE:])

    rescaled_actual = y_test[WINDOW_SIZE:]
    rescaled_actual = pd.DataFrame(rescaled_actual)
    rescaled_actual.index = stock.index[-idx_count + WINDOW_SIZE:]

rep_pred = globals()['rescaled_pred_' + '0']

actual_change_data = stock['Close'][-idx_count + WINDOW_SIZE:]
actual_change_data = pd.Series(actual_change_data)
actual_change_data = actual_change_data.diff()
actual_change_data = actual_change_data.iloc[1:]  # 첫 번째 요소 제거

actual_change_data_binary = actual_change_data.apply(lambda x: 1 if x >= 0 else 0)

for i in range(REP_SIZE):
    globals()['rescaled_pred_binary_' + str(i)] = change_binary(globals()['rescaled_pred_' + str(i)])

pred_change_data_binary = globals()['rescaled_pred_binary_' + '0']
# endregion

# 마지막 예측값
print(rep_pred.iloc[-1])

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

# stationary_test
perform_dickey_fuller_test(stock[TARGET_DATA])

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
