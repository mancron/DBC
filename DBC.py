import csv
import os
import mysql.connector
import re  # 정규 표현식 사용

# MySQL 연결 설정
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="0000",
    database="danawa_crawler_data"
)
cursor = conn.cursor()

data_folder = "data"  # 데이터가 있는 폴더


def extract_price(price_info):
    """ '정품_'이 포함된 데이터에서 마지막 숫자 가격만 추출하는 함수 """
    if "정품_" in price_info:
        matches = re.findall(r'(\d{1,3}(?:,\d{3})*)', price_info)  # 쉼표 포함된 숫자 찾기
        if matches:
            return matches[-1].replace(",", "")  # 가장 마지막 숫자를 반환
    elif price_info.replace(",", "").isdigit():  # 일반 숫자 가격이면 그대로 사용
        return price_info.replace(",", "")

    return None  # 유효한 가격이 없으면 무시


def insert_csv_to_db(file_path):
    """ 주어진 CSV 파일에서 '정품_'이 붙거나 그냥 숫자로 된 가격만 MySQL에 삽입하는 함수 """
    if not os.path.exists(file_path):
        print(f"파일 없음: {file_path}")
        return

    try:
        with open(file_path, encoding="utf-8") as file:
            reader = csv.reader(file)
            headers = next(reader)  # 헤더 읽기
            dates = headers[2:]  # ID, Name 제외한 날짜 부분

            for row in reader:
                try:
                    name = row[1]  # 두 번째 열 (이름)
                    for i, price_info in enumerate(row[2:], start=2):
                        try:
                            price = extract_price(price_info)  # 가격 추출 함수 사용
                            if price is not None:
                                date = dates[i - 2].split()[0]  # 날짜 추출
                                cursor.execute(
                                    "INSERT INTO product_prices (name, date, price) VALUES (%s, %s, %s)",
                                    (name, date, int(price))
                                )
                        except (IndexError, ValueError) as e:
                            print(f"오류 발생: {price_info}, 이유: {e} -> 무시됨")
                        except mysql.connector.Error as err:
                            print(f"MySQL 삽입 오류: {err}")

                except Exception as e:
                    print(f"행 처리 중 예외 발생: {e}")

    except FileNotFoundError:
        print(f"파일을 찾을 수 없음: {file_path}")

    except Exception as e:
        print(f"예외 발생 ({file_path}): {e}")


# 2020-07 ~ 2025-02 폴더 순회
for year in range(2020, 2026):
    for month in range(1, 13):
        folder_name = f"{year}-{month:02d}"
        folder_path = os.path.join(data_folder, folder_name)
        if not os.path.exists(folder_path):
            continue

        # 처리할 파일 목록
        csv_files = ["VGA.csv", "CPU.csv", "MBoard.csv", "Power.csv"]

        for file_name in csv_files:
            file_path = os.path.join(folder_path, file_name)
            insert_csv_to_db(file_path)

        print(f"{year}-{month} 삽입완료")

# 변경사항 저장 및 연결 종료
conn.commit()
conn.close()
print("모든 '정품' 및 일반 가격 데이터 삽입 완료 및 MySQL 연결 종료!")
