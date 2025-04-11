import pandas as pd
import pickle
from prophet import Prophet
from train_model import get_connection, get_unique_product_names

def load_data(table_name):
    conn = get_connection()
    df = pd.read_sql(f"SELECT name, date, price FROM {table_name}", conn)
    conn.close()
    return df

def train_prophet(df):
    df = df[df['price'] > 0].copy()
    df = df.rename(columns={'date': 'ds', 'price': 'y'})[['ds', 'y']]
    model = Prophet()
    model.fit(df)
    return model

if __name__ == "__main__":
    table_options = {
        "1": "vga_price",
        "2": "cpu_price",
        "3": "mboard_price",
        "4": "power_price"
    }

    print("카테고리를 선택하세요:")
    print("1. VGA\n2. CPU\n3. MainBoard\n4. Power")
    choice = input("번호 입력: ").strip()

    table_name = table_options.get(choice)
    if not table_name:
        print("올바른 번호를 선택하세요.")
        exit()

    df = load_data(table_name)
    product_names = get_unique_product_names(table_name)

    print(f"\n선택된 카테고리 '{table_name}'의 제품 수: {len(product_names)}")
    for name in product_names:
        print("-", name)

    selected_name = input("\n학습할 제품명을 입력하세요: ").strip()
    df = df[df['name'] == selected_name]

    if df.empty:
        print("데이터 없음.")
    else:
        model = train_prophet(df)
        filename = f"prophet_model_{selected_name.replace(' ', '_')}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(model, f)
        print(f"Prophet 모델 저장 완료: {filename}")
