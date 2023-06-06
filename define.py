import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, \
    accuracy_score


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


def confirm_result(actual, pred, actual_bin, pred_bin):
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    msle = mean_squared_log_error(actual, pred)
    rmsle = np.sqrt(mean_squared_log_error(actual, pred))
    r2 = r2_score(actual, pred)

    # 정확도 계산
    accuracy = accuracy_score(actual_bin, pred_bin)
    # 정밀도 계산
    precision = precision_score(actual_bin, pred_bin)
    # 재현율 계산
    recall = recall_score(actual_bin, pred_bin)
    # F1 스코어 계산
    f1 = f1_score(actual_bin, pred_bin)

    pd.options.display.float_format = '{:.5f}'.format

    table = \
        {
            "MSE": [mae], "RMSE": [rmse], "RMSLE": [rmsle], "R2": [r2],
            "ACCURACY": [accuracy], "PRECISION": [precision], "RECALL": [recall],
            "F1_SCORE": [f1]
        }
    res = pd.DataFrame(table)

    return res


# 삼성전자(005930) 전체 (1996-11-05 ~ 현재)
STOCK_CODE = '005930'
# 데이터 가져오기 시작할 날짜
DATA_DATE = '2018-01-01'
# 학습 횟수
EPOCH = 300
# 옵티마이저 학습률
LEARNING_RATE = 0.0001
# 검증데이터 비율
TEST_SIZE = 0.2

# 테스트 반복 횟수, epoch X
REP_SIZE = 20

WINDOW_SIZE = 30
BATCH_SIZE = 32

FILE_PATH = 'C:/Users/kim/Desktop/res_df.xlsx'
TARGET_SHEET = 'Data'
