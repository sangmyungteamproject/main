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

stocks = fdr.DataReader('DJI')
print(stocks)

