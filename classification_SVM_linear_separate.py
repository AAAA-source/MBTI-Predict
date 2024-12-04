import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

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
            text_part = content[:-5].strip()  # 去掉最後 4 個字母（MBTI）和標點符號
            mbti_part = content[-4:]         # 提取最後 4 個字母
            
            texts.append(text_part)
            labels.append(mbti_part)

# 檢查資料是否讀取成功
if len(texts) == 0 or len(labels) == 0:
    raise ValueError("資料讀取失敗，請檢查輸入資料夾結構或內容格式！")

# 初始化 TF-IDF 向量化器
vectorizer = TfidfVectorizer(min_df=2, max_df=0.9, stop_words="english")

# 將文字資料轉換為 TF-IDF 特徵矩陣
X = vectorizer.fit_transform(texts)
y = np.array(labels)

# 將資料分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# 定義 MBTI 四個字母的位置
mbti_letters = ["I/E", "N/S", "T/F", "J/P"]

# 儲存四個獨立的 SVM 模型
models = []

# 訓練四個模型
for i in range(4):
    # 抽取第 i 個字母作為目標
    y_train_binary = [label[i] for label in y_train]
    model = LinearSVC(random_state=42, class_weight="balanced", C=2)
    model.fit(X_train, y_train_binary)
    models.append(model)

# 測試階段
predictions = []
for i, model in enumerate(models):
    # 針對測試資料預測第 i 個字母
    letter_predictions = model.predict(X_test)
    predictions.append(letter_predictions)

# 組合四個模型的結果為完整的 MBTI 類型
final_predictions = ["".join(letters) for letters in zip(*predictions)]


from collections import Counter

# 預測結果和實際類型的統計
type_counts = Counter(y_test)  # 每種類型在測試集中實際出現的次數
type_correct = Counter()       # 每種類型被正確分類的次數
type_predicted = Counter()     # 每種類型被模型預測為該類型的次數

# 統計正確分類、實際數量和預測數量
for pred, true in zip(final_predictions, y_test):
    if pred == true:
        type_correct[true] += 1  # 正確分類的計數
    type_predicted[pred] += 1    # 被預測為該類型的計數

# 計算精確率（Precision）和召回率（Recall）
type_metrics = {}
for mbti in sorted(type_counts.keys()):
    precision = type_correct[mbti] / type_predicted[mbti] if type_predicted[mbti] > 0 else 0
    recall = type_correct[mbti] / type_counts[mbti] if type_counts[mbti] > 0 else 0
    type_metrics[mbti] = {"precision": precision, "recall": recall}

# 輸出每種類型的精確率和召回率
print("\n各種類型的精確率和召回率:")
for mbti, metrics in type_metrics.items():
    print(f"{mbti}: Precision={metrics['precision']:.2%}, Recall={metrics['recall']:.2%}")

# 額外輸出測試數量、正確數量和預測數量
print("\n類型統計:")
for mbti in sorted(type_counts.keys()):
    print(f"{mbti}: 測試數量={type_counts[mbti]}, 正確數量={type_correct[mbti]}, 預測數量={type_predicted[mbti]}")



