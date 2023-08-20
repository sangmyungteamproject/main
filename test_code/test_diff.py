# 주가 데이터 변량값을 타겟 데이터로 실험하는 코드

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

from define import windowed_dataset, confirm_result, STOCK_CODE, EPOCH, \
    LEARNING_RATE, TEST_SIZE, DATA_DATE, WINDOW_SIZE, BATCH_SIZE, REP_SIZE, \
    FILE_PATH, TARGET_SHEET, change_binary, perform_dickey_fuller_test

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Lambda
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 시간 측정
start_time = time.time()

# 2018년 1월 1일 데이터 부터 가져옴
stock = fdr.DataReader(STOCK_CODE, DATA_DATE)

stock['Year'] = stock.index.year
stock['Month'] = stock.index.month
stock['Day'] = stock.index.day

stock.head()

stock['Close_Changed'] = stock['Close'].diff()
#stock['Close_Changed'] = stock['Change']
stock = stock.iloc[1:]

TARGET_DATA = 'Close_Changed'
USE_CHANGE_DATA = True

ft_scaler = MinMaxScaler()

# 스케일을 적용할 column을 정의합니다.
scale_ft_cols = ['Open', 'High', 'Low', 'Volume']

scaled_ft = ft_scaler.fit_transform(stock[scale_ft_cols])

df = pd.DataFrame(scaled_ft, columns=scale_ft_cols)

tg_scaler = MinMaxScaler()

scaled_tg = tg_scaler.fit_transform(stock[TARGET_DATA].values.reshape(-1, 1))
df[TARGET_DATA] = scaled_tg

print(df)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test \
    = train_test_split(
    df.drop(labels=TARGET_DATA, axis=1),
    df[TARGET_DATA],
    test_size=TEST_SIZE,
    random_state=0,
    shuffle=False)

# train_data는 학습용 데이터셋, test_data는 검증용 데이터셋 입니다.
train_data = windowed_dataset(y_train, WINDOW_SIZE, BATCH_SIZE, True)
test_data = windowed_dataset(y_test, WINDOW_SIZE, BATCH_SIZE, False)

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
    globals()['filename_' + str(i)] = os.path.join('../tmp', 'checkpointer_' + str(i) + '.ckpt')
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

idx_count = y_test.count()

# 리스케일링

_rescale = False

if (_rescale):
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

actual_change_data = stock[TARGET_DATA][-idx_count + WINDOW_SIZE:]
actual_change_data = pd.Series(actual_change_data)
actual_change_data = actual_change_data.diff()
actual_change_data = actual_change_data.iloc[1:]  # 첫 번째 요소 제거

actual_change_data_binary = actual_change_data.apply(lambda x: 1 if x >= 0 else 0)

for i in range(REP_SIZE):
    globals()['rescaled_pred_binary_' + str(i)] = change_binary(globals()['rescaled_pred_' + str(i)])

pred_change_data_binary = globals()['rescaled_pred_binary_' + '0']

# 마지막 예측값
print(rep_pred.iloc[-1])

# 종료 시간 기록
end_time = time.time()

# 실행 시간 계산
execution_time = end_time - start_time

# 결과 출력
print(f"실행 시간: {execution_time:.2f}초")

# 데이터 시각화
plt.figure(figsize=(12, 9))
plt.plot(rescaled_actual, label='actual')
plt.plot(rep_pred, label='prediction')
plt.legend()
plt.show()

res = confirm_result(rescaled_actual, rep_pred, actual_change_data_binary, pred_change_data_binary, USE_CHANGE_DATA)
print(res)

perform_dickey_fuller_test(stock[TARGET_DATA])

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
