import pandas as pd
import mysql.connector
import re

# DB 연결
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="0000",
        database="danawa_vga"
    )

# 고유한 제품명 가져오기
def get_unique_product_names(table_name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT DISTINCT name FROM {table_name}")
    names = [row[0] for row in cursor.fetchall()]
    conn.close()
    return names

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

# 개별 테이블에서 데이터 불러오기
def load_data(table_name):
    conn = get_connection()
    query = f"SELECT name, date, price FROM {table_name}"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# 범주 테이블에서 데이터 불러오기
def load_all_data(table_name):
    conn = get_connection()
    query = f"SELECT name, date, avg_price, std_dev FROM {table_name}"
    df = pd.read_sql(query, conn)
    conn.close()
    return df