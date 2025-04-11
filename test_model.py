import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from train_model import get_connection
import platform
import pickle
import os

# DB에서 제품 데이터 불러오기
def load_data(table_name):
    conn = get_connection()
    query = f"SELECT name, date, price FROM {table_name}"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# 그래프 그리기: 실제 + 예측
def plot_actual_and_predicted(df, model, product_name, days=30):
    df = df[(df['name'] == product_name) & (df['price'] > 0)].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    if df.empty:
        print(f"{product_name}에 대한 유효한 데이터가 없습니다.")
        return

    start_date = df['date'].min()
    latest_date = df['date'].max()

    # 실제 데이터 전처리
    name_category = df['name'].astype('category')
    product_code = name_category.cat.codes.iloc[0]

    df['date_int'] = df['date'].astype(int) / 10 ** 9
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['name'] = product_code

    features = ['name', 'date_int', 'year', 'month', 'day', 'dayofweek', 'weekofyear']
    X_actual = df[features]
    y_actual = df['price']

    # 미래 날짜 생성
    future_dates = pd.date_range(start=latest_date + timedelta(days=1), periods=days)
    future_df = pd.DataFrame({'date': future_dates})
    future_df['date_int'] = future_df['date'].astype(int) / 10 ** 9
    future_df['year'] = future_df['date'].dt.year
    future_df['month'] = future_df['date'].dt.month
    future_df['day'] = future_df['date'].dt.day
    future_df['dayofweek'] = future_df['date'].dt.dayofweek
    future_df['weekofyear'] = future_df['date'].dt.isocalendar().week.astype(int)
    future_df['name'] = product_code

    X_future = future_df[features]
    log_preds = model.predict(X_future)
    future_preds = np.exp(log_preds)

    # 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(df['date'], y_actual, label="실제 가격", linestyle='-')
    plt.plot(future_dates, future_preds, label="예측 가격", linestyle='--', color='orange')
    plt.title(f"{product_name} 가격 추이 및 향후 {days}일 예측\n(데이터 시작일: {start_date.date()}, 마지막일: {latest_date.date()})")
    plt.xlabel("날짜")
    plt.ylabel("가격")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 실행
if __name__ == "__main__":
    if platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    else:
        plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False

    table_options = {
        "1": "vga_price",
        "2": "cpu_price",
        "3": "mboard_price",
        "4": "power_price"
    }

    print("예측할 제품의 카테고리를 선택하세요:")
    print("1. VGA\n2. CPU\n3. MainBoard\n4. Power")
    choice = input("번호 입력: ").strip()

    if choice not in table_options:
        print("올바른 번호를 선택하세요.")
        exit()

    table_name = table_options[choice]
    df = load_data(table_name)

    product_name = input("\n예측할 제품 이름을 입력하세요: ").strip()

    if product_name not in df['name'].unique():
        print(f"'{product_name}'는 데이터베이스에 존재하지 않습니다.")
        exit()

    model_filename = f"xgb_model_{product_name.replace(' ', '_')}.pkl"
    if not os.path.exists(model_filename):
        print(f"'{model_filename}' 모델 파일이 존재하지 않습니다. 먼저 학습을 진행하세요.")
        exit()

    with open(model_filename, "rb") as f:
        model = pickle.load(f)

    # 사용자에게 예측일 수 입력 받기
    days_input = input("\n예측할 일 수를 입력하세요(기본값=30): ").strip()
    if days_input.isdigit():
        days = int(days_input)
    else:
        print("기본값(30일)으로 설정")
        days = 30

    plot_actual_and_predicted(df, model, product_name, days=days)
