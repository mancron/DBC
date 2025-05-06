from db_conn import load_data, load_all_data, get_unique_product_names, extract_gpu_info
from train_model import preprocess, preprocess_avg, train_xgboost
from test_model import plot_combined_prediction
import matplotlib.pyplot as plt
import platform
import pickle
import os

def detect_tier(df_price, df_stats, selected_name):
    refined = extract_gpu_info(selected_name)
    if not refined:
        return None, None

    df_target = df_price[df_price['name'] == selected_name]
    if df_target.empty:
        return None, None

    avg_target_price = df_target['price'].mean()

    stats_b = df_stats[df_stats['name'] == f"{refined} (보급형)"]
    stats_s = df_stats[df_stats['name'] == f"{refined} (상급형)"]

    if stats_b.empty or stats_s.empty:
        return None, None

    avg_b = stats_b['avg_price'].mean()
    avg_s = stats_s['avg_price'].mean()

    diff_b = abs(avg_target_price - avg_b)
    diff_s = abs(avg_target_price - avg_s)

    if diff_b < diff_s:
        return f"{refined} (보급형)", "보급형"
    else:
        return f"{refined} (상급형)", "상급형"


def no_tier(refined_name, df_stats):
    return not df_stats[df_stats['name'].str.startswith(refined_name + " (")].any()


if __name__ == "__main__":
    if platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    else:
        plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False

    df = load_data("vga_price")
    df_2 = load_all_data("ref_vga_stats")

    product_names = get_unique_product_names("vga_price")
    print(f"\n그래픽 카드 제품 수: {len(product_names)}")
    for name in product_names:
        print(name)

    selected_name = input("\n학습할 제품명을 정확히 입력하세요: ").strip()
    refined_name = extract_gpu_info(selected_name)

    # ✅ 1. 단일 모델 학습
    df_single = df[df['name'] == selected_name]
    if not df_single.empty:
        X, y = preprocess(df_single)
        model = train_xgboost(X, y)
        safe_single_name = selected_name.replace(' ', '_')
        filename = f"xgb_model_single_{safe_single_name}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(model, f)
        print(f"\n단일 모델 저장 완료: {filename}")
    else:
        print("단일 모델 데이터 없음")

    # ✅ 2. 범주 or 통합 모델 학습
    tiered_name, tier = detect_tier(df, df_2, selected_name)
    if tiered_name:
        df_tier = df_2[df_2['name'] == tiered_name]
        if not df_tier.empty:
            X2, y2 = preprocess_avg(df_tier)
            model2 = train_xgboost(X2, y2)
            safe_refined_name = refined_name.replace(' ', '_')
            filename = f"xgb_model_{safe_refined_name}_({tier}).pkl"
            with open(filename, "wb") as f:
                pickle.dump(model2, f)
            print(f"범주 모델 저장 완료: {filename}")
        else:
            print(f"범주 '{tier}' 데이터 없음")
    else:
        # 등급 없음일 경우 → 통합 모델 학습
        if no_tier(refined_name, df_2):
            df_no_tier = df_2[df_2['name'] == refined_name]
            if not df_no_tier.empty:
                X2, y2 = preprocess_avg(df_no_tier)
                model2 = train_xgboost(X2, y2)
                safe_refined_name = refined_name.replace(' ', '_')
                filename = f"xgb_model_{safe_refined_name}.pkl"
                with open(filename, "wb") as f:
                    pickle.dump(model2, f)
                print(f"통합 모델 저장 완료: {filename}")
            else:
                print("통합 모델 데이터 없음")
        else:
            print("등급 판단 실패 또는 데이터 누락")

    days_input = input("\n예측할 일 수를 입력하세요 (기본값=30): ").strip()
    days = int(days_input) if days_input.isdigit() else 30

    model_path_single = f"xgb_model_single_{selected_name.replace(' ', '_')}.pkl"
    if tiered_name:
        model_path_category = f"xgb_model_{refined_name.replace(' ', '_')}_({tier}).pkl"
    else:
        model_path_category = f"xgb_model_{refined_name.replace(' ', '_')}.pkl"

    if os.path.exists(model_path_single) and os.path.exists(model_path_category):
        with open(model_path_single, "rb") as f:
            single_model = pickle.load(f)
        with open(model_path_category, "rb") as f:
            category_model = pickle.load(f)

        print("\n예측 비교 시각화를 시작합니다...")
        plot_combined_prediction(df, df_2, single_model, category_model, selected_name, tiered_name or refined_name,
                                 days=days)
    else:
        print("단일 또는 범주 모델 파일이 누락되어 시각화를 수행할 수 없습니다.")
        if not os.path.exists(model_path_single):
            print(f"단일 모델 파일 없음: {model_path_single}")
        if not os.path.exists(model_path_category):
            print(f"범주 모델 파일 없음: {model_path_category}")
