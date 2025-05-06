import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def make_future_features(start_date, future_dates, name_code=None, std_dev=None):
    df_future = pd.DataFrame({'date': future_dates})
    df_future['date_int'] = (df_future['date'] - start_date).dt.total_seconds()
    df_future['year'] = df_future['date'].dt.year
    df_future['month'] = df_future['date'].dt.month
    df_future['day'] = df_future['date'].dt.day
    df_future['dayofweek'] = df_future['date'].dt.dayofweek
    df_future['weekofyear'] = df_future['date'].dt.isocalendar().week.astype(int)
    if name_code is not None:
        df_future['name'] = name_code
    if std_dev is not None:
        df_future['std_dev'] = std_dev
    return df_future

def draw(df, df_ref, single_model, category_model, product_name, category_name, days=30):
    df_prod = df[(df['name'] == product_name) & (df['price'] > 0)].copy()
    df_prod['date'] = pd.to_datetime(df_prod['date'])
    df_prod = df_prod.sort_values('date')

    if df_prod.empty:
        print("유효한 가격 데이터가 없습니다.")
        return

    # 실제 가격 (파란 선)
    plt.plot(df_prod['date'], df_prod['price'], label="실제 가격", color='blue', linestyle='-')

    # 단일 모델 예측 - 과거 예측 시각화 (빨간 점선)
    from train_model import preprocess
    X_all, _ = preprocess(df_prod)
    y_pred_all_single = np.exp(single_model.predict(X_all))
    plt.plot(df_prod['date'], y_pred_all_single, label="단일 모델 과거 예측", color='red', linestyle='--')

    # 단일 모델 예측 - 미래 예측 (초록색)
    future_dates = pd.date_range(df_prod['date'].max(), periods=days + 1, freq='D')[1:]
    name_code_single = df_prod['name'].astype('category').cat.codes.iloc[0]
    future_features_single = make_future_features(df_prod['date'].min(), future_dates, name_code=name_code_single)
    required_features_single = ['name', 'date_int', 'year', 'month', 'day', 'dayofweek', 'weekofyear']
    y_pred_single_log = single_model.predict(future_features_single[required_features_single])
    y_pred_single = np.exp(y_pred_single_log)
    plt.plot(future_dates, y_pred_single, label="단일 모델 예측", color='green')

    # 범주 모델 예측 (주황색)
    df_ref_prod = df_ref[df_ref['name'] == category_name].copy()
    if not df_ref_prod.empty:
        df_ref_prod['date'] = pd.to_datetime(df_ref_prod['date'])
        df_ref_prod = df_ref_prod.sort_values('date')
        avg_std_dev = df_ref_prod['std_dev'].mean() if 'std_dev' in df_ref_prod.columns else 0.0
        name_code_cat = df_ref_prod['name'].astype('category').cat.codes.iloc[0]
        future_features_category = make_future_features(df_ref_prod['date'].min(), future_dates, name_code=name_code_cat, std_dev=avg_std_dev)
        required_features_cat = ['name', 'date_int', 'year', 'month', 'day', 'dayofweek', 'weekofyear', 'std_dev']
        y_pred_ref_log = category_model.predict(future_features_category[required_features_cat])
        y_pred_ref = np.exp(y_pred_ref_log)
        plt.plot(future_dates, y_pred_ref, label="범주 모델 예측", color='orange')

    plt.xlabel("날짜")
    plt.ylabel("가격")
    plt.title(f"{product_name} 예측 비교")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
