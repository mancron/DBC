import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import mysql.connector

# get_connection 함수 직접 정의
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="0000",
        database="danawa_crawler_data"
    )

# DB에서 데이터 불러오기 (전체 데이터 / 개별학습 X)
def load_data():
    conn = get_connection()
    query = "SELECT name, date, price FROM product_prices"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# 전처리
def preprocess(df):
    df['date'] = pd.to_datetime(df['date']) # 컬럼을 날짜 형식으로 변환
    df['date_int'] = df['date'].astype(int) / 10 ** 9 # 학습을 위해 날짜를 수치형으로 다시 변환
    df['name'] = df['name'].astype('category').cat.codes    # 학습을 위해 상품 이름을 정수코드로 변환
    df = df[df['price'] > 0]  # 로그 변환을 위해 0보다 큰 가격만 사용
    y_log = np.log(df['price'])  # 로그 변환된 y (여러 데이터의 변동량 학습을 위해 / 복수 데이터 학습은 아직 구현X)
    return df[['name', 'date_int']], y_log

# 학습
def train_xgboost(X, y):
    model = xgb.XGBRegressor(
        objective='reg:squarederror',   # 손실 최소화 학습 / 일반적 수치 예측에 주로 사용
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6
    )
    model.fit(X, y)
    return model

# 전체 상품이름 출력(확인용)
def get_unique_product_names():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT name FROM product_prices")
    names = [row[0] for row in cursor.fetchall()]
    conn.close()
    return names

# 실행
if __name__ == "__main__":

    # 물품의 이름 리스트 출력
    names = get_unique_product_names()
    print("물품 전체의 수량인", len(names), "개의 고유한 제품명:")
    for name in names:
        print("-", name)

    # 사용자로부터 학습할 제품명을 입력 받음
    selected_name = input("\n학습할 제품명을 정확히 입력하세요: ")

    df = load_data()

    # 선택한 제품만 필터링
    df = df[df['name'] == selected_name]  # 개별 학습

    if df.empty:
        print("\n[오류] 해당 제품에 대한 데이터가 없습니다.")
    else:
        X, y = preprocess(df)  # 전처리 및 로그 변환
        model = train_xgboost(X, y)  # 모델 학습

        # 모델 저장 (제품명을 파일명에 포함)
        filename = f"xgb_model_{selected_name.replace(' ', '_')}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(model, f)

        print(f"\n[완료] 모델 학습 및 저장 완료 → {filename}")