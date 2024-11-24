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
    model = LinearSVC(random_state=42, class_weight="balanced", C=0.75)
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

# 計算分類準確率
correct_count = sum([1 for pred, true in zip(final_predictions, y_test) if pred == true])
accuracy = correct_count / len(y_test)

# 印出每個字母的分類報告
for i, (letter, model) in enumerate(zip(mbti_letters, models)):
    y_test_binary = [label[i] for label in y_test]
    letter_predictions = predictions[i]
    print(f"\n分類報告 for {letter}:")
    print(classification_report(y_test_binary, letter_predictions, zero_division=1))

# 輸出整體結果
print(f"\n測試資料總數: {len(y_test)}")
print(f"正確分類數量: {correct_count}")
print(f"分類準確率: {accuracy:.2%}")
