import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score

# 定義資料夾路徑
input_folder = "txt_output"  # 替換為存放原始文字檔案的資料夾路徑

# 儲存處理後的資料
texts = []  # 儲存文本
labels = []  # 儲存 MBTI 標籤

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

# 檢查資料是否讀取成功
if len(texts) == 0 or len(labels) == 0:
    raise ValueError("資料讀取失敗，請檢查輸入資料夾結構或內容格式！")

# 初始化 TF-IDF 向量化器
vectorizer = TfidfVectorizer(min_df=2, max_df=0.9, stop_words="english" , max_features = 100)

# 將文字資料轉換為 TF-IDF 特徵矩陣
X = vectorizer.fit_transform(texts)
y = np.array(labels)

# 初始化 SVM 模型，指定 kernel='linear'
model = SVC(kernel='rbf', random_state=42)

# 設定 K-fold 交叉驗證
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 儲存每次交叉驗證的分數
scores = []

print("during k-fold") ;
# 執行 K-fold 交叉驗證
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # 訓練模型
    model.fit(X_train, y_train)
    
    # 預測測試資料
    y_pred = model.predict(X_test)
    
    # 計算準確率
    accuracy = accuracy_score(y_test, y_pred)
    scores.append(accuracy)
    
    # 印出該分折的詳細報告
    print(f"分類報告 for fold:")
    print(classification_report(y_test, y_pred, zero_division=1))

# 輸出每次交叉驗證的分數與平均分數
print(f"每次交叉驗證的準確率: {scores}")
print(f"平均準確率: {np.mean(scores)}")
