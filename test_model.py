import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from train_model import get_connection
import platform
import pickle
import os

# DB에서 데이터 불러오기
def load_data():
    conn = get_connection()
    query = "SELECT name, date, price FROM product_prices"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# 로그 변환 포함한 전처리
def preprocess(df):
    df = df[df['price'] > 0].copy()  # 로그 변환을 위해 0보다 큰 값만 사용
    df['date'] = pd.to_datetime(df['date'])
    df['date_int'] = df['date'].astype(int) / 10 ** 9
    df['name'] = df['name'].astype('category').cat.codes
    X = df[['name', 'date_int']]
    y = np.log(df['price'])  # 로그 변환
    return X, y

# 실제 가격 시각화
def plot_actual_prices(df, product_name):
    df = df[(df['name'] == product_name) & (df['price'] > 0)].copy()    # 해당하는 상품만 골라냄
    df['date'] = pd.to_datetime(df['date']) # 시계열 데이터로 사용하기 위해 타입 변환
    df = df.sort_values('date') # 오름차순 정렬
    plt.figure(figsize=(10, 5))
    plt.plot(df['date'], df['price'], marker='o', linestyle='-')
    plt.title(f"실제 가격 추이 - {product_name}")
    plt.xlabel("날짜")
    plt.ylabel("가격")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 1개월 예측 함수
def predict_next_month(df, model, product_name, days=30):
    df['date'] = pd.to_datetime(df['date'])
    latest_date = df['date'].max()  # 해당 품목의 가장 최근 데이터 (해당 기간 이후부터 예측)

    # name_code 추출
    name_category = df['name'].astype('category')
    product_code = name_category.cat.codes[df['name'] == product_name].iloc[0]  # 정수로 변환된 품목의 데이터를 가져옴

    # 미래 날짜 생성 및 예측용 DataFrame 구성
    future_dates = pd.date_range(start=latest_date + timedelta(days=1), periods=days)   # 가장 최근 데이터 이후의 날짜 생성
    future_df = pd.DataFrame({
        'name': [product_code] * days,
        'date_int': future_dates.astype(int) / 10 ** 9  # 날짜를 정수로 변환 (형식 일치화)
    })

    # 예측 및 로그 역변환
    log_predictions = model.predict(future_df)
    predictions = np.exp(log_predictions)  # 로그 복원

    # 결과 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(future_dates, predictions, marker='o', linestyle='--', color='orange')
    plt.title(f"{product_name} - 향후 {days}일 가격 예측")
    plt.xlabel("예측 날짜")
    plt.ylabel("예상 가격")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 한글 폰트 설정
    if platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    else:
        plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False

    test_product = input("\n예측할 제품 이름을 입력하세요: ")
    # 선택한 제품만 로드 및 전처리
    df = load_data()
    if test_product not in df['name'].unique():
        print(f"[오류] '{test_product}'는 데이터베이스에 존재하지 않습니다.")
        exit()

    # 로그 적용된 모델 로드
    model_filename = f"xgb_model_{test_product.replace(' ', '_')}.pkl"  # 공백을 _로 변환
    if not os.path.exists(model_filename):
        print(f"[오류] '{model_filename}' 모델 파일이 존재하지 않습니다. 먼저 학습을 진행하세요.")
        exit()

    with open(model_filename, "rb") as f:
        model = pickle.load(f)

    # 실제 가격 그래프
    plot_actual_prices(df, test_product)

    # 향후 30일 예측 그래프
    predict_next_month(df, model, test_product)