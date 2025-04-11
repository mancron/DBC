import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from train_model import get_connection
import platform
import pickle
import os
import re

# 학습 모델을 가져오기 위한 제품명 컷팅
def extract_gpu_info(text):
    pattern = re.search(
        r'(지포스|라데온)\s+'
        r'(GTX|GT|RTX|RX|XT|XTX)?\s*'
        r'(\d{3,5})'
        r'(?:\s*(Ti|SUPER))?'
        r'(?:\s*(Ti|SUPER))?'
        r'.*?(\d{1,2}GB)',
        text, re.IGNORECASE
    )
    if pattern:
        brand, prefix, number, suffix1, suffix2, memory = pattern.groups()
        suffixes = []
        for s in (suffix1, suffix2):
            if s and s.upper() not in [x.upper() for x in suffixes]:
                suffixes.append(s.upper())
        model = f"{brand} {prefix or ''}{number}"
        if suffixes:
            model += ' ' + ' '.join(suffixes)
        model += f" {memory}"
        return model.strip()
    return None

# DB에서 제품 데이터 불러오기
def load_data(table_name):
    conn = get_connection()
    query = f"SELECT name, date, price FROM {table_name}"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ref_vga_stats 불러오기
def load_ref_stats():
    conn = get_connection()
    df = pd.read_sql("SELECT name, date, std_dev FROM ref_vga_stats", conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    return df

# 그래프 그리기
def plot_with_avg_model(vga_df, ref_df, model, product_name, category_name, days=30):
    df = vga_df[(vga_df['name'] == product_name) & (vga_df['price'] > 0)].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    if df.empty:
        print(f"{product_name}에 대한 유효한 데이터가 없습니다.")
        return

    start_date = df['date'].min()
    latest_date = df['date'].max()

    # name code 및 std_dev 가져오기
    name_category = ref_df['name'].astype('category')
    if category_name not in name_category.values:
        print(f"{category_name}은 ref_vga_stats 테이블에 존재하지 않습니다.")
        return

    name_code = name_category.cat.codes[name_category == category_name].iloc[0]
    recent_std = ref_df[ref_df['name'] == category_name].sort_values('date').iloc[-1]['std_dev']

    # 미래 날짜 생성
    future_dates = pd.date_range(start=latest_date + timedelta(days=1), periods=days)
    future_df = pd.DataFrame({'date': future_dates})
    future_df['date_int'] = future_df['date'].astype(int) / 10 ** 9
    future_df['year'] = future_df['date'].dt.year
    future_df['month'] = future_df['date'].dt.month
    future_df['day'] = future_df['date'].dt.day
    future_df['dayofweek'] = future_df['date'].dt.dayofweek
    future_df['weekofyear'] = future_df['date'].dt.isocalendar().week.astype(int)
    future_df['name'] = name_code
    future_df['std_dev'] = recent_std

    features = ['name', 'date_int', 'year', 'month', 'day', 'dayofweek', 'weekofyear', 'std_dev']
    log_preds = model.predict(future_df[features])
    future_preds = np.exp(log_preds)

    # 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(df['date'], df['price'], label="실제 가격", linestyle='-')
    plt.plot(future_dates, future_preds, label="예측 가격", linestyle='-', color='orange')
    plt.title(f"{product_name} 가격 추이 및 향후 {days}일 예측\n(시리즈: {category_name})")
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

    print("\n예측할 VGA 제품 이름을 정확히 입력하세요:")
    product_name = input("제품 이름: ").strip()

    # 제품명 기반 시리즈 분류 추출
    category_name = extract_gpu_info(product_name)
    if not category_name:
        print(f"입력된 제품명에서 유효한 GPU 시리즈 범주를 추출할 수 없습니다: '{product_name}'")
        exit()

    # 테이블에서 VGA 데이터 및 통계 데이터 불러오기
    vga_df = load_data("vga_price")
    ref_df = load_ref_stats()

    if product_name not in vga_df['name'].unique():
        print(f"'{product_name}'는 vga_price 테이블에 존재하지 않습니다.")
        exit()

    model_filename = f"xgb_model_{category_name.replace(' ', '_')}.pkl"
    if not os.path.exists(model_filename):
        print(f"시리즈 범주 '{category_name}'에 대한 모델 파일이 존재하지 않습니다: {model_filename}")
        exit()

    with open(model_filename, "rb") as f:
        model = pickle.load(f)

    # 예측 일 수 입력
    days_input = input("\n예측할 일 수를 입력하세요(기본값=30): ").strip()
    days = int(days_input) if days_input.isdigit() else 30

    # 예측 및 시각화 실행
    plot_with_avg_model(vga_df, ref_df, model, product_name, category_name, days)