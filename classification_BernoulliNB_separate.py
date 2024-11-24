import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# 定義資料夾路徑
input_folder = "txt_output"  # 替換為你的檔案資料夾路徑

# 儲存處理後的資料
texts = []
labels = []

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

# 將資料分為訓練和測試集
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.1, random_state=42, stratify=labels)

# 定義每個模型對應的 MBTI 字母索引和模型
mbti_letters = ["I/E", "N/S", "T/F", "J/P"]
models = []

for i in range(4):
    # 每個模型只針對一個 MBTI 字母進行分類
    y_train_binary = [label[i] for label in y_train]  # 只取第 i 個字母
    pipeline = make_pipeline(CountVectorizer(binary=True), BernoulliNB())
    pipeline.fit(X_train, y_train_binary)
    models.append(pipeline)

# 測試階段
predictions = []
for i, model in enumerate(models):
    # 每個模型對測試資料進行預測
    letter_predictions = model.predict(X_test)
    predictions.append(letter_predictions)

# 組合四個模型的結果
final_predictions = ["".join(letters) for letters in zip(*predictions)]

# 計算準確率
correct_count = sum([1 for pred, true in zip(final_predictions, y_test) if pred == true])
accuracy = correct_count / len(y_test)

# 輸出結果
print(f"測試資料總數: {len(y_test)}")
print(f"正確分類數量: {correct_count}")
print(f"分類準確率: {accuracy:.2%}")
