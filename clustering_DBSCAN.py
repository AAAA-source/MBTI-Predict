import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import DBSCAN
from sklearn.model_selection import StratifiedKFold

# 定義資料夾路徑
input_folder = "preprocessing/txt_output"

# 儲存文本和 MBTI 類型
texts = []
labels = []

# 讀取每個 .txt 文件
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(input_folder, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read().strip()
            
            # 假設最後 4 個字是 MBTI 類型
            text_part = content[:-5].strip()  # 文本部分
            mbti_part = content[-4:]         # MBTI 類型
            
            texts.append(text_part)
            labels.append(mbti_part)

# 將 MBTI 類型轉為數值標籤
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# 初始化 TF-IDF 向量化器
vectorizer = TfidfVectorizer( min_df=2 , max_df = 0.9 , stop_words = "english" )

# 初始化 StratifiedKFold
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 儲存每次交叉驗證的結果
adjusted_rand_scores = []
normalized_mutual_infos = []

# 執行交叉驗證
for train_index, test_index in kf.split(texts, labels_encoded):
    # 分割資料
    X_train = np.array(texts)[train_index]
    y_train = np.array(labels_encoded)[train_index]
    X_test = np.array(texts)[test_index]
    y_test = np.array(labels_encoded)[test_index]
    
    # 對訓練集和測試集進行向量化
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 使用 DBSCAN 聚類
    dbscan = DBSCAN(eps=0.8, min_samples=3, metric='cosine')
    cluster_labels = dbscan.fit_predict(X_test_tfidf)
    
    # 評估聚類結果
    ari = adjusted_rand_score(y_test, cluster_labels)
    nmi = normalized_mutual_info_score(y_test, cluster_labels)
    
    adjusted_rand_scores.append(ari)
    normalized_mutual_infos.append(nmi)

# 輸出結果
print(f"Adjusted Rand Index (每次分割): {adjusted_rand_scores}")
print(f"Adjusted Rand Index 平均: {np.mean(adjusted_rand_scores)}")
print(f"Normalized Mutual Information (每次分割): {normalized_mutual_infos}")
print(f"Normalized Mutual Information 平均: {np.mean(normalized_mutual_infos)}")
