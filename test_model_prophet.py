
import platform
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from datetime import timedelta
from train_model import get_connection

def load_data(table_name):
    conn = get_connection()
    df = pd.read_sql(f"SELECT name, date, price FROM {table_name}", conn)
    conn.close()
    return df

def plot_prophet(df, model, product_name, days=30):
    df = df[df['name'] == product_name]
    if df.empty:
        print("데이터 없음.")
        return

    df['ds'] = pd.to_datetime(df['date'])
    df['y'] = df['price']

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    plt.figure(figsize=(10, 5))
    plt.plot(df['ds'], df['y'], label="실제 가격")
    plt.plot(forecast['ds'], forecast['yhat'], label="예측 가격", linestyle='--')
    plt.title(f"{product_name} 가격 예측 (Prophet)")
    plt.xlabel("날짜")
    plt.ylabel("가격")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    else:
        plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False

    table_name = "vga_price"  # 예시
    product_name = input("제품 이름 입력: ").strip()

    with open(f"prophet_model_{product_name.replace(' ', '_')}.pkl", "rb") as f:
        model = pickle.load(f)

    df = load_data(table_name)
    plot_prophet(df, model, product_name)
