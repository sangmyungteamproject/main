import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import FinanceDataReader as fdr
import tensorflow as tf

# 삼성전자(005930) 전체 (1996-11-05 ~ 현재)
STOCK_CODE = '005930'

stock = fdr.DataReader(STOCK_CODE)

stock['Year'] = stock.index.year
stock['Month'] = stock.index.month
stock['Day'] = stock.index.day

# 스케일링 이전의 원본 데이터 저장
original_data = stock['Close'].values

stock.head()

plt.figure(figsize=(16, 9))
sns.lineplot(y=stock['Close'], x=stock.index)
plt.xlabel('time')
plt.ylabel('price')

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
# 스케일을 적용할 column을 정의합니다.
scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

scale_ft_cols = ['Open','High','Low','Volume']

# 스케일 후 columns
#scaled = scaler.fit_transform(stock[scale_cols])

scaled_ft = scaler.fit_transform(stock[scale_ft_cols])

#df = pd.DataFrame(scaled, columns=scale_cols)
df = pd.DataFrame(scaled_ft, columns = scale_ft_cols)


scaler1 = MinMaxScaler()

scaled_tg = scaler1.fit_transform(stock['Close'].values.reshape(-1, 1))
df['Close'] = scaled_tg
df.index = stock.index

print(df)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.drop(labels = 'Close', axis = 1), df['Close'], test_size=0.3, random_state=0, shuffle=False)

# 테스트 데이터에 대한 스케일링 이전의 원본 데이터 추출
WINDOW_SIZE=20
BATCH_SIZE=32

def windowed_dataset(series, window_size, batch_size, shuffle):
    #데이터 차원 확장
    series = tf.expand_dims(series, axis=-1)
    #텐서 슬라이싱
    ds = tf.data.Dataset.from_tensor_slices(series)
    #윈도우 생성
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    #데이터 평탄화 (윈도우 -> 배치)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    #셔플
    if shuffle:
        ds = ds.shuffle(1000)
    #입력, 출력 변환
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    #데이터 셋 배치 단위로 나누기, 버퍼크기 설정 = 1
    return ds.batch(batch_size).prefetch(1)

# train_data는 학습용 데이터셋, test_data는 검증용 데이터셋 입니다.
train_data = windowed_dataset(y_train, WINDOW_SIZE, BATCH_SIZE, True)
test_data = windowed_dataset(y_test, WINDOW_SIZE, BATCH_SIZE, False)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Lambda
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential([
    # 1차원 feature map 생성
    # 특성 추출 레이어
    Conv1D
    (
        #필터 수, 필터의 크기
        filters=32, kernel_size=5,
        #현재위치 이전 정보만 사용하도록 제한
        padding="causal",
        #활성화 함수 = relu
        activation="relu",
        #입력 데이터 형태
        input_shape=[WINDOW_SIZE, 1]
    ),
    # LSTM, 활성화 함수 tanh, 유닛 16개
    LSTM(16, activation='tanh'),
    #완전 연결 레이어, 최종 예측
    Dense(16, activation="relu"),
    #출력 레이어
    Dense(1),
])

# Sequence 학습에 비교적 좋은 퍼포먼스를 내는 Huber()를 사용합니다.
# 손실함수
loss = Huber()
# 최적화 함수
optimizer = Adam(0.0003)
# 모델 컴파일, 손실함수 = huber, 최적화함수 = adam, 평가지표 = MSE
model.compile(loss=Huber(), optimizer=optimizer, metrics=['mse'])

# earlystopping은 10번 epoch통안 val_loss 개선이 없다면 학습을 멈춥니다.
# val_loss = mse
earlystopping = EarlyStopping(monitor='val_loss', patience=10)

# val_loss 기준 체크포인터도 생성합니다.
filename = os.path.join('tmp', 'ckeckpointer.ckpt')
checkpoint = ModelCheckpoint(filename,
                             save_weights_only=True,
                             save_best_only=True,
                             monitor='val_loss',
                             verbose=1)
# 모델 훈련

history = model.fit\
    (
    train_data,
    validation_data=(test_data),
    epochs=50,
    callbacks=[checkpoint, earlystopping]
    )

model.load_weights(filename)

pred = model.predict(test_data)

rescaled_actual = scaler1.inverse_transform(y_test[WINDOW_SIZE:].values.reshape(-1,1))
rescaled_actual = pd.DataFrame(rescaled_actual, index=y_test.index[WINDOW_SIZE:])
rescaled_actual.index = y_test.index[WINDOW_SIZE:]

rescaled_pred = scaler1.inverse_transform(np.array(pred).reshape(-1,1))
rescaled_pred = pd.DataFrame(rescaled_pred, index=y_test.index[WINDOW_SIZE:])


print(rescaled_pred.iloc[-1])

from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_squared_log_error,r2_score

def confirm_result(actual, pred):
    MAE = mean_absolute_error(actual, pred)
    RMSE = np.sqrt(mean_squared_error(actual, pred))
    MSLE = mean_squared_log_error(actual, pred)
    RMSLE = np.sqrt(mean_squared_log_error(actual, pred))
    R2 = r2_score(actual, pred)

    pd.options.display.float_format = '{:.5f}'.format
    Result = pd.DataFrame(data=[MAE,RMSE,RMSLE,R2],
                          index = ['MAE','RMSE','RMSLE','R2'],
                          columns = ['Results'])

    return Result

plt.figure(figsize=(12, 9))
plt.plot(rescaled_actual, label='actual')
plt.plot(rescaled_pred, label='prediction')
plt.legend()
plt.show()

print('Rescale X')
res1 = confirm_result(y_test[WINDOW_SIZE:].values.reshape(-1,1), np.array(pred).reshape(-1,1))
print(res1)
print('Rescale O')
res2 = confirm_result(rescaled_actual,rescaled_pred)
print(res2)

