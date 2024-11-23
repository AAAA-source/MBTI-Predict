import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold



# 定義資料夾路徑
input_folder = "preprocessing/txt_output"  # 替換為你的檔案資料夾路徑

# 儲存處理後的資料
texts = []
labels = []

# 讀取每個 .txt 檔案並分割內容和 MBTI 標籤
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(input_folder, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read().strip()
            
            # 分割文字和 MBTI 標籤（假設最後 4 個字母是 MBTI 類型）
            text_part = content[:-5].strip()  # 去掉最後 4 個字母（MBTI）和標點符號
            mbti_part = content[-4:]         # 提取最後 4 個字母
            
            texts.append(text_part)
            labels.append(mbti_part)

# 建立模型管道，將向量化和 Naive Bayes 結合
model = make_pipeline(CountVectorizer(binary = True), BernoulliNB())

# 設定K-fold交叉驗證
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 儲存每次交叉驗證的分數
scores = []

# 執行K-fold交叉驗證
for train_index, test_index in kf.split(texts , labels):
    X_train, X_test = np.array(texts)[train_index], np.array(texts)[test_index]
    y_train, y_test = np.array(labels)[train_index], np.array(labels)[test_index]
    
    # 訓練模型
    model.fit(X_train, y_train)
    
    # 評估模型
    score = model.score(X_test, y_test)
    scores.append(score)

# 輸出每次交叉驗證的分數與平均分數
print(f"每次交叉驗證的分數: {scores}")
print(f"平均分數: {np.mean(scores)}")

