import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, \
    accuracy_score
from statsmodels.tsa.stattools import adfuller

import FinanceDataReader as fdr


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


def change_binary(data, target_data):
    if target_data == 'Signal':
        pred_change_data_binary = np.where(data >= 0.5, 1, 0)
    else:
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
                "MAE": [mae], "RMSE": [rmse], "RMSLE": [rmsle], "R2": [r2],
                "ACCURACY": [accuracy], "PRECISION": [precision], "RECALL": [recall],
                "F1_SCORE": [f1]
            }
        res = pd.DataFrame(table)
    else:
        table = \
            {
                "MAE": [mae], "RMSE": [rmse], "RMSLE": ['0'], "R2": [r2],
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


def cal_rsi(df, rsi_period=14):
    # 종가 변화량
    df['Price_Change'] = df['Close'].diff()
    # 양수 가격 변화와 음수 가격 변화 계산
    df['Gain'] = df['Price_Change'].apply(lambda x: x if x > 0 else 0)
    df['Loss'] = df['Price_Change'].apply(lambda x: abs(x) if x < 0 else 0)

    # 초기값 설정
    avg_gain = df['Gain'][:rsi_period].mean()
    avg_loss = df['Loss'][:rsi_period].mean()

    rsi_values = []
    for i in range(rsi_period, len(df['Close'])):
        au = ((rsi_period - 1) * avg_gain + df['Gain'].iloc[i]) / rsi_period
        ad = ((rsi_period - 1) * avg_loss + df['Loss'].iloc[i]) / rsi_period

        _rsi = au / (au + ad)
        rsi_values.append(_rsi)

    # RSI 열을 데이터프레임에 추가
    df['RSI'] = [None] * rsi_period + rsi_values

    return df


def load_corp_data(stock_code, start_date, end_date, senti_file_path):
    # region 데이터 가져오기
    stock = fdr.DataReader(stock_code, start=start_date, end=end_date)
    # 환율
    usd_krw = fdr.DataReader('USD/KRW', start=start_date, end=end_date)
    jpy_krw = fdr.DataReader('JPY/KRW', start=start_date, end=end_date)
    # 미 국채 금리 10년
    dgs = fdr.DataReader('FRED:DGS10', start=start_date, end=end_date)

    stock['Year'] = stock.index.year
    stock['Month'] = stock.index.month
    stock['Day'] = stock.index.day

    # 감성점수 데이터 불러오기
    senti_df = pd.read_csv(senti_file_path, header=None, index_col=None)
    header = ['date', 'senti_val']
    senti_df.columns = header

    # 중복된 날짜 병합 후 정렬
    senti_df = senti_df.groupby('date').mean().reset_index()
    senti_df = senti_df.sort_values(by='date')

    # 'date' 열을 datetime 형식으로 변환
    senti_df['datetime'] = pd.to_datetime(senti_df['date'])

    # 'datetime' 열에서 금, 토, 일에 해당하는 행만 선택
    weekend_data = senti_df[senti_df['datetime'].dt.dayofweek.isin([4, 5, 6])]
    # 'datetime' 열에서 각 주말 날짜의 금요일 날짜만 추출
    friday_dates = weekend_data['datetime'] - pd.to_timedelta((weekend_data['datetime'].dt.dayofweek - 4) % 7, unit='d')
    # 추출한 금요일 날짜를 기준으로 그룹화하고 평균 계산
    result = weekend_data.groupby(friday_dates)['senti_val'].mean().reset_index()
    # 'datetime' 열에서 토요일(요일 코드 5)과 일요일(요일 코드 6)에 해당하는 행을 삭제
    senti_df = senti_df[~senti_df['datetime'].dt.dayofweek.isin([5, 6])]
    # 'senti_df' 데이터프레임의 'datetime' 값이 'result' 데이터프레임의 'datetime' 값과 일치하는 행 선택
    merge_condition = senti_df['datetime'].isin(result['datetime'])
    # 'senti_df'에서 선택한 행의 'senti_val' 열 값을 'result' 데이터프레임의 'senti_val' 열 값으로 대체
    senti_df.loc[merge_condition, 'senti_val'] = result.loc[
        result['datetime'].isin(senti_df['datetime']), 'senti_val'].values
    # 'result' 데이터프레임의 'datetime' 열에만 있는 데이터를 'senti_df'에 추가
    result_only_rows = result[~result['datetime'].isin(senti_df['datetime'])]
    senti_df = pd.concat([senti_df, result_only_rows], ignore_index=True)

    # 'date' 열이 NaN인 행 선택
    nan_date_condition = senti_df['date'].isna()
    # 'date' 열이 NaN인 행의 'datetime' 값을 '%Y-%m-%d' 형식으로 변경
    senti_df.loc[nan_date_condition, 'date'] = senti_df.loc[nan_date_condition, 'datetime'].dt.strftime('%Y-%m-%d')
    # 'date' 열의 데이터 타입을 'object'로 변경
    senti_df['date'] = senti_df['date'].astype('object')

    senti_df = senti_df.sort_values(by='date')
    senti_df.reset_index(drop=True, inplace=True)
    stock['date'] = stock.index
    stock.reset_index(drop=False, inplace=True)
    stock['date'] = stock['date'].apply(lambda x: x.strftime("%Y-%m-%d"))
    if senti:
        # 주가 정보 데이터 프레임에 감성점수 데이터를 날짜 열을 기준으로 병합
        stock = pd.merge(stock, senti_df, on='date', how='left')
        # 감성 점수 없는 날 데이터 보간
        stock['senti_val'] = stock['senti_val'].interpolate(method='linear')

    # 결측값 제거
    stock = stock[stock['Open'] != 0]
    # endregion

    # region 입력 변수 데이터 생성
    # 변량 (입력변수 아님)
    stock['Diff'] = stock['Close'] - stock['Open']
    # 방향
    stock['Signal'] = stock['Diff'].apply(lambda x: 1 if x > 0 else 0)
    # rsi
    if rsi:
        cal_rsi(stock, RSI_PERIOD)
    # 환율
    if ex_rate:
        stock['USD/KRW'] = usd_krw['Close']
        stock['JPY/KRW'] = jpy_krw['Close']
    # 미 국채 10년
    if bonds:
        stock['DGS10'] = dgs
    # 봉 색깔
    if color:
        stock['Color'] = stock['Diff'].apply(positive_negative)
    # 봉 길이
    if bar_len:
        stock['Bar_len'] = stock['Diff'] / stock['Open']
    # 양봉 길이
    if red_bar:
        stock['Red_bar'] = stock.apply(chk_red_bar, axis=1)
    # 음봉 길이
    if blue_bar:
        stock['Blue_bar'] = stock.apply(chk_blue_bar, axis=1)
    # 최근 5일간 양,음봉의 추세
    if trend:
        stock['Trend'] = stock['Color'].shift(4) + stock['Color'].shift(3) \
                         + stock['Color'].shift(2) + stock['Color'].shift(1) + stock['Color']
    # 20일 이동평균
    if ma20:
        stock['MA_20'] = cal_ma(stock, 'Close', 20)
    # 종가 대비 20일 이평의 상대 위치
    if ma_20_1:
        stock['MA_20_1'] = stock['MA_20'] / stock['Close']
    # 종가가 20일 이평보다 큰지, 작은지 여부
    if ma_20_2:
        stock['MA_20_2'] = stock['MA_20_1'].apply(lambda x: 1 if x >= 1 else 0)
    # 종가 전고점 대비 현재 종가의 값
    if cummax:
        stock['Cummax'] = stock['Close'] / stock['Close'].expanding().max()
    # 윗꼬리 길이 = 종가대비 고가와 종가차이
    if uptail:
        stock['Uptail'] = (stock['High'] - stock['Close']) / stock['Close']
    # 최근 5일 최저가 대비 가격 차이
    if five_days_change:
        stock['5days_change'] = cal_days_change(stock, 5, 'Close')
    # 최근 4일 최저가 대비 가격 차이
    if four_days_change:
        stock['4days_change'] = cal_days_change(stock, 4, 'Close')
    # 최근 3일 최저 거래량 대비 거래량 차이
    if three_vol_change:
        stock['3vol_change'] = cal_days_change(stock, 3, 'Volume')
    # 거래량 10일 이동평균
    if vol_ma10:
        stock['Volume_MA10'] = cal_ma(stock, 'Volume', 10)
    # 거래량 10일 이평대비 당일 거래량 값의 위치
    if pos_vol10ma:
        stock['Pos_Vol10MA'] = stock['Volume'] / stock['Volume_MA10']

    # 앞부분 데이터 삭제 (20일 이평 기준)
    stock = stock.iloc[20:]
    # endregion

    return stock


def find_cols_with_inf(df, scale_ft_cols):
    inf_cols = []
    for col_name in scale_ft_cols:
        if df[col_name].isin([float('inf'), float('-inf')]).any():
            inf_cols.append(col_name)
    return inf_cols


# endregion

# region 입력변수 사용여부
open_price = True
high_price = True
low_price = True
volume = True

rsi = False
senti = False
ex_rate = False
bonds = False

color = False
bar_len = color
red_bar = color
blue_bar = color

# color True 필수
trend = False

ma20 = False
# ma20 필수
ma_20_1 = False
ma_20_2 = False

cummax = False
uptail = False

five_days_change = False
four_days_change = five_days_change

three_vol_change = False
vol_ma10 = False
pos_vol10ma = False
# endregion

# 리스케일링 여부
rescale = True
# 테스트 반복 횟수, epoch X
REP_SIZE = 20

TARGET_DATA = 'Signal'
USE_CHANGE_DATA = False

# 학습 횟수
EPOCH = 200
# 옵티마이저 학습률
LEARNING_RATE = 0.0002
# 테스트 데이터 비율
TEST_SIZE = 0.2
WINDOW_SIZE = 20
BATCH_SIZE = 32

RSI_PERIOD = 14

PATIENCE = 10

FILE_PATH = 'C:/Users/kim/Desktop/res_df.xlsx'
TARGET_SHEET = 'Data'
