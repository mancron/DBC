import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import mysql.connector
from sklearn.model_selection import train_test_split

# 연결 함수
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="0000",
        database="danawa_vga"
    )

# 선택한 카테고리 테이블에서 데이터 불러오기
def load_data(table_name):
    conn = get_connection()
    query = f"SELECT name, date, price FROM {table_name}"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def load_all_data(table_name):
    conn = get_connection()
    query = f"SELECT name, date, avg_price, std_dev FROM {table_name}"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# 전처리 (개별 데이터 학습)
def preprocess(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['price'] > 0].copy()

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['date_int'] = df['date'].astype(int) / 10 ** 9
    df['name'] = df['name'].astype('category').cat.codes

    features = ['name', 'date_int', 'year', 'month', 'day', 'dayofweek', 'weekofyear']
    y_log = np.log(df['price'])

    return df[features], y_log

# 전처리 (평균값)
def preprocess_avg(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['avg_price'] > 0].copy()

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['date_int'] = df['date'].astype(int) / 10 ** 9
    df['name'] = df['name'].astype('category').cat.codes

    # 평균값과 표준편차 모두 활용
    features = ['name', 'date_int', 'year', 'month', 'day', 'dayofweek', 'weekofyear', 'std_dev']
    y_log = np.log(df['avg_price'])

    return df[features], y_log

# 모델 학습
def train_xgboost(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.set_params(early_stopping_rounds=20)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return model

# 고유한 제품명 가져오기
def get_unique_product_names(table_name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT DISTINCT name FROM {table_name}")
    names = [row[0] for row in cursor.fetchall()]
    conn.close()
    return names

# 실행
if __name__ == "__main__":
    table_options = {
        "1": "vga_price",
        "2": "ref_vga_stats"
    }

    print("카테고리를 선택하세요:")
    print("1. VGA\n2. MainBoard AVG")
    choice = input("번호 입력: ").strip()

    if choice not in table_options:
        print("올바른 번호를 선택하세요.")
        exit()

    table_name = table_options[choice]
    if (choice=='2'):
        df = load_all_data(table_name)
    else:
        df = load_data(table_name)
    product_names = get_unique_product_names(table_name)

    print(f"\n선택된 카테고리 '{table_name}'의 제품 수: {len(product_names)}")
    for name in product_names:
        print(name)

    selected_name = input("\n학습할 제품명을 정확히 입력하세요: ").strip()
    df = df[df['name'] == selected_name]

    if df.empty:
        print("\n해당 제품에 대한 데이터가 없습니다.")
    else:
        if table_name == "ref_vga_stats":
            X, y = preprocess_avg(df)
        else:
            X, y = preprocess(df)
        model = train_xgboost(X, y)
        filename = f"xgb_model_{selected_name.replace(' ', '_')}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(model, f)
        print(f"\n모델 학습 및 저장 완료 : {filename}")
