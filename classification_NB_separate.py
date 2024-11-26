import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

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

# 初始化 count vector 向量化器
vectorizer = CountVectorizer(min_df=2, max_df=0.9, stop_words="english")

# 將文字資料轉換為 count vector 特徵矩陣
X = vectorizer.fit_transform(texts)
y = np.array(labels)

# 將 MBTI 四個字母分別分割為四組標籤
y_labels = {
    "letter_1": np.array([label[0] for label in y]),  # E/I
    "letter_2": np.array([label[1] for label in y]),  # S/N
    "letter_3": np.array([label[2] for label in y]),  # T/F
    "letter_4": np.array([label[3] for label in y]),  # J/P
}

# 初始化模型
models = {
    "letter_1": MultinomialNB(),
    "letter_2": MultinomialNB(),
    "letter_3": MultinomialNB(),
    "letter_4": MultinomialNB(),
}

# 設定 K-fold 交叉驗證
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 儲存每次交叉驗證的分數
overall_scores = []

# 執行 K-fold 交叉驗證
for train_index, test_index in kf.split(X, y_labels["letter_1"]):  # 使用其中一個標籤來平衡分割
    X_train, X_test = X[train_index], X[test_index]
    y_train_preds, y_test_preds = [], []

    # 對每個字母建立模型並訓練
    for letter, model in models.items():
        y_train = y_labels[letter][train_index]
        y_test = y_labels[letter][test_index]
        
        # 訓練模型
        model.fit(X_train, y_train)
        
        # 預測測試資料
        y_pred = model.predict(X_test)
        
        # 收集每個字母的預測結果
        y_train_preds.append(y_train)
        y_test_preds.append(y_pred)
    
    # 組合四個字母的預測結果成完整的 MBTI 類型
    y_test_combined = np.array(["".join(x) for x in zip(*y_test_preds)])
    y_true_combined = np.array(["".join(x) for x in zip(*[y_labels[letter][test_index] for letter in models.keys()])])
    
    # 計算 MBTI 類型的準確率
    accuracy = accuracy_score(y_true_combined, y_test_combined)
    overall_scores.append(accuracy)

# 輸出結果
print(f"每次交叉驗證的準確率: {overall_scores}")
print(f"平均準確率: {np.mean(overall_scores)}")
