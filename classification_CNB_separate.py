import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

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

# 初始化 Count Vectorizer
vectorizer = CountVectorizer(min_df=2, max_df=0.9, stop_words="english", max_features=5000)

# 將文字資料轉換為 Count 特徵矩陣
X = vectorizer.fit_transform(texts)
y = np.array(labels)

# 提取四個字母的維度
letters = np.array([[label[i] for label in y] for i in range(4)]).T

# 設定 K-fold 交叉驗證
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 儲存每個字母的分類準確率
letter_accuracies = [[] for _ in range(4)]
overall_accuracies = []

# 對四個字母分別訓練模型並進行交叉驗證
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    letter_preds = []

    for i in range(4):
        y_train = letters[train_index, i]
        y_test = letters[test_index, i]
        
        # 初始化 CNB 模型
        model = ComplementNB()
        model.fit(X_train, y_train)
        
        # 預測
        y_pred = model.predict(X_test)
        letter_preds.append(y_pred)
        
        # 計算該字母的準確率
        accuracy = accuracy_score(y_test, y_pred)
        letter_accuracies[i].append(accuracy)
    
    # 合併四個字母的預測，計算整體準確率
    y_pred_combined = np.array(letter_preds).T
    y_test_combined = letters[test_index]
    combined_accuracy = np.mean(np.all(y_pred_combined == y_test_combined, axis=1))
    overall_accuracies.append(combined_accuracy)

# 顯示結果
for i, letter in enumerate(["第一字母 (I/E)", "第二字母 (N/S)", "第三字母 (F/T)", "第四字母 (J/P)"]):
    print(f"{letter} 的平均準確率: {np.mean(letter_accuracies[i]):.4f}")

print(f"整體 MBTI 分類平均準確率: {np.mean(overall_accuracies):.4f}")
