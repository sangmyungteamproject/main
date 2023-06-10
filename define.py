import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, \
    accuracy_score
from statsmodels.tsa.stattools import adfuller


# region 함수
def windowed_dataset(series, window_size, batch_size, shuffle):
    # 데이터 차원 확장
    series = tf.expand_dims(series, axis=-1)
    # 텐서 슬라이싱
    ds = tf.data.Dataset.from_tensor_slices(series)
    # 윈도우 생성
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    # 데이터 평탄화 (윈도우 -> 배치)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    # 셔플
    if shuffle:
        ds = ds.shuffle(1000)
    # 입력, 출력 변환
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    # 데이터 셋 배치 단위로 나누기, 버퍼크기 설정 = 1
    return ds.batch(batch_size).prefetch(1)


def change_binary(data):
    pred_change_data = data.diff()
    pred_change_data = pred_change_data.iloc[1:]

    pred_change_data_binary = np.where(pred_change_data > 0, 1, 0)

    return pred_change_data_binary


def confirm_result(actual, pred, actual_bin, pred_bin, use_change_data):
    global rmsle
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred)
    if not use_change_data:
        rmsle = np.sqrt(mean_squared_log_error(actual, pred))
        msle = mean_squared_log_error(actual, pred)

    # 정확도 계산
    accuracy = accuracy_score(actual_bin, pred_bin)
    # 정밀도 계산
    precision = precision_score(actual_bin, pred_bin)
    # 재현율 계산
    recall = recall_score(actual_bin, pred_bin)
    # F1 스코어 계산
    f1 = f1_score(actual_bin, pred_bin)

    pd.options.display.float_format = '{:.5f}'.format

    if not use_change_data:
        table = \
            {
                "MSE": [mae], "RMSE": [rmse], "RMSLE": [rmsle], "R2": [r2],
                "ACCURACY": [accuracy], "PRECISION": [precision], "RECALL": [recall],
                "F1_SCORE": [f1]
            }
        res = pd.DataFrame(table)
    else:
        table = \
            {
                "MSE": [mae], "RMSE": [rmse], "RMSLE": ['0'], "R2": [r2],
                "ACCURACY": [accuracy], "PRECISION": [precision], "RECALL": [recall],
                "F1_SCORE": [f1]
            }
        res = pd.DataFrame(table)

    return res


def perform_dickey_fuller_test(data):
    # Dickey-Fuller 검정 수행
    result = adfuller(data)

    # 결과 출력
    print("Dickey-Fuller Test 결과:")
    print("검정 통계량 (Test Statistic):", result[0])
    print("p-value:", result[1])
    print("사용된 시차 (Lags Used):", result[2])
    print("사용된 관측값 수 (Number of Observations Used):", result[3])
    print("임계값 (Critical Values):")
    for key, value in result[4].items():
        print(f"\t{key}: {value}")


def positive_negative(x):
    if x >= 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def chk_red_bar(data):
    return data['Color'] * data['Bar_len'] if data['Color'] == 1 else 0


def chk_blue_bar(data):
    return data['Color'] * data['Bar_len'] if data['Color'] == -1 else 0


def cal_ma(data, column_name, window_size):
    df = data[column_name].rolling(window=window_size).mean()
    return df


def cal_cummax(data):
    # 특정 시점 이전까지의 전고점 구하기
    cummax = data['Close'].cummax()
    # 종가 전고점 대비 현재 종가의 값
    rtn = data['Close'] / cummax

    return rtn


def cal_days_change(data, day, data_type):
    min_value = data[data_type].rolling(window=day).min()
    rtn = (data[data_type] - min_value) / min_value

    return rtn


# endregion

CONV1D_LSTM = """
Conv1D(filters=32, kernel_size=5, padding="causal", activation="relu", input_shape=[WINDOW_SIZE, 1]),
LSTM(16, activation='tanh'),
Dense(16, activation="relu"),
Dense(1),
"""

# 종목 : 삼성전자(005930)
STOCK_CODE = '005930'
# 데이터 가져오기 시작할 날짜
DATA_DATE = '2018-01-01'
# 학습 횟수
EPOCH = 300
# 옵티마이저 학습률
LEARNING_RATE = 0.0002
# 검증데이터 비율
TEST_SIZE = 0.2

# 리스케일링 여부
_rescale = True

# 테스트 반복 횟수, epoch X
REP_SIZE = 20

WINDOW_SIZE = 20
BATCH_SIZE = 64

FILE_PATH = 'C:/Users/kim/Desktop/res_df.xlsx'
TARGET_SHEET = 'Data'
