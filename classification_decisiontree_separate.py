import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
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
            
            # 分割文字和 MBTI 標籤
            text_part = content[:-5].strip()  # 去掉最後 4 個字母（MBTI）
            mbti_part = content[-4:]         # 提取最後 4 個字母
            
            texts.append(text_part)
            labels.append(mbti_part)

# 檢查資料是否讀取成功
if len(texts) == 0 or len(labels) == 0:
    raise ValueError("資料讀取失敗，請檢查輸入資料夾結構或內容格式！")

# 初始化 Count Vectorizer
vectorizer = CountVectorizer(min_df=2, max_df=0.9, stop_words="english")
X = vectorizer.fit_transform(texts)

# 將 MBTI 分為四個字母
labels_split = np.array([[label[0], label[1], label[2], label[3]] for label in labels])
y1, y2, y3, y4 = labels_split.T

# 定義模型與交叉驗證
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 初始化結果儲存
results = {}

# 四個字母分別建模
for i, (y, letter) in enumerate(zip([y1, y2, y3, y4], ["I/E", "N/S", "F/T", "J/P"])):
    model = DecisionTreeClassifier(random_state=42)
    scores = []
    
    print(f"\n正在分類第 {i+1} 字母 ({letter}):")
    
    # K-fold 交叉驗證
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # 訓練模型
        model.fit(X_train, y_train)
        
        # 預測測試集
        y_pred = model.predict(X_test)
        
        # 計算準確率
        accuracy = accuracy_score(y_test, y_pred)
        scores.append(accuracy)
        
        # 印出分類報告
        print(classification_report(y_test, y_pred, zero_division=1))
    
    # 儲存結果
    results[letter] = scores
    print(f"{letter} 的平均準確率: {np.mean(scores):.4f}")

# 整體 MBTI 準確率
combined_predictions = []
combined_true = []

# 遍歷 K-fold 結果，組合四個字母
for train_index, test_index in kf.split(X, y1):
    X_test = X[test_index]
    
    pred1 = model.fit(X[train_index], y1[train_index]).predict(X_test)
    pred2 = model.fit(X[train_index], y2[train_index]).predict(X_test)
    pred3 = model.fit(X[train_index], y3[train_index]).predict(X_test)
    pred4 = model.fit(X[train_index], y4[train_index]).predict(X_test)
    
    combined_predictions.extend(["".join(p) for p in zip(pred1, pred2, pred3, pred4)])
    combined_true.extend(["".join(p) for p in zip(y1[test_index], y2[test_index], y3[test_index], y4[test_index])])

overall_accuracy = accuracy_score(combined_true, combined_predictions)
print(f"整體 MBTI 分類平均準確率: {overall_accuracy:.4f}")
