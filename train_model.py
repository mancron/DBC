import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

# 전처리 (개별 데이터)
def preprocess(df):
    df = df.copy()  # 안전한 복사
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['price'] > 0].copy()

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['date_int'] = df['date'].astype(int) / 10 ** 9
    df['name'] = df['name'].astype('category').cat.codes

    features = ['name', 'date_int', 'year', 'month', 'day', 'weekofyear']
    y_log = np.log(df['price'])

    return df[features], y_log

# 전처리 (범주 데이터)
def preprocess_avg(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['avg_price'] > 0].copy()

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['date_int'] = df['date'].astype(int) / 10 ** 9
    df['name'] = df['name'].astype('category').cat.codes

    # 평균값과 표준편차 모두 활용
    features = ['name', 'date_int', 'year', 'month', 'day', 'weekofyear', 'std_dev']
    y_log = np.log(df['avg_price'])

    return df[features], y_log

# 모델 학습
def train_xgboost(X, y):
    # 최근 데이터에 더 큰 가중치 부여
    X = X.copy()
    max_time = X['date_int'].max()
    min_time = X['date_int'].min()
    X['weight'] = 1 + 9 * ((X['date_int'] - min_time) / (max_time - min_time))  # 1~10 사이 가중치

    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X.drop(columns=['weight']), y, X['weight'], test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=800,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        #colsample_bytree=0.8,
        gamma=0.8,
        min_child_weight=3,
        random_state=42
    )
    model.set_params(early_stopping_rounds=100)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        sample_weight=w_train,  # 가중치 적용
        verbose=False
    )
    return model
