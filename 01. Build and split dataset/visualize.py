import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_interaction_density(df, user_col='user_id', item_col='entity_id'):
    """
    Phân tích mật độ tương tác của User và Item để chọn k-core phù hợp.
    """
    print("="*50)
    print("🔍 BÁO CÁO MẬT ĐỘ TƯƠNG TÁC TỔNG QUAN")
    print("="*50)
    
    # 1. Đếm số lượng tương tác của từng User và Item
    user_counts = df[user_col].value_counts()
    item_counts = df[item_col].value_counts()
    
    print(f"Tổng số tương tác (dòng): {len(df):,}")
    print(f"Tổng số Users duy nhất: {len(user_counts):,}")
    print(f"Tổng số Items duy nhất: {len(item_counts):,}\n")
    
    # 2. In thống kê mô tả với các mốc phần trăm quan trọng
    percentiles = [0.25, 0.5, 0.75, 0.90, 0.95, 0.99]
    
    print("👤 THỐNG KÊ USER (Mỗi User có bao nhiêu tương tác?):")
    print(user_counts.describe(percentiles=percentiles).to_string())
    print("-" * 30)
    
    print("📦 THỐNG KÊ ITEM (Mỗi Item được tương tác bao nhiêu lần?):")
    print(item_counts.describe(percentiles=percentiles).to_string())
    print("-" * 30)
    
    # 3. Tính toán thử nghiệm sự sụt giảm dữ liệu với các mức K khác nhau
    print("\n📉 MÔ PHỎNG NẾU ÁP DỤNG LỌC K-CORE (Chỉ tính riêng lẻ 1 vòng):")
    test_k_values = [3, 5, 6, 7, 8, 9, 10]
    
    for k in test_k_values:
        users_kept = (user_counts >= k).sum()
        items_kept = (item_counts >= k).sum()
        
        pct_users = (users_kept / len(user_counts)) * 100
        pct_items = (items_kept / len(item_counts)) * 100
        
        print(f"👉 K = {k:<2} | Giữ lại {pct_users:>5.1f}% Users ({users_kept}) và {pct_items:>5.1f}% Items ({items_kept})")

    # 4. Vẽ biểu đồ trực quan
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Cắt đuôi ở mức 95% để biểu đồ không bị méo bởi các outlier quá lớn
    user_cutoff = np.percentile(user_counts, 95)
    item_cutoff = np.percentile(item_counts, 95)
    
    axes[0].hist(user_counts[user_counts <= user_cutoff], bins=50, color='skyblue', edgecolor='black')
    axes[0].set_title(f'Phân phối tương tác của User\n(Bỏ qua 5% Users siêu tích cực)')
    axes[0].set_xlabel('Số lượng tương tác')
    axes[0].set_ylabel('Số lượng User')
    
    axes[1].hist(item_counts[item_counts <= item_cutoff], bins=50, color='lightcoral', edgecolor='black')
    axes[1].set_title(f'Phân phối tương tác của Item\n(Bỏ qua 5% Items siêu phổ biến)')
    axes[1].set_xlabel('Số lượng tương tác')
    axes[1].set_ylabel('Số lượng Item')
    
    plt.tight_layout()
    plt.show()

# Cách dùng: (Giả sử file của bạn tên là data.csv)
name = 'book'
path = '/home/hp/Study/07. Luan Van/03. TPRec/01. Build and split dataset/data/book/book_processed_interactions.csv'
all_dataset = pd.read_csv(path)
analyze_interaction_density(all_dataset, user_col='user_id', item_col='entity_id')