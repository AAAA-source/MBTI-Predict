import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
from sklearn.model_selection import StratifiedKFold

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

# 假設文本和標籤已定義
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.1, random_state=42, stratify=labels
)

# 使用 CountVectorizer 處理 Naive Bayes 的資料
cnb_vectorizer = CountVectorizer()
X_train_cnb = cnb_vectorizer.fit_transform(X_train)
X_test_cnb = cnb_vectorizer.transform(X_test)

# 使用 TfidfVectorizer 處理 SVM 的資料
svm_vectorizer = TfidfVectorizer()
X_train_svm = svm_vectorizer.fit_transform(X_train)
X_test_svm = svm_vectorizer.transform(X_test)

# 訓練 Complement Naive Bayes 模型
cnb_model = ComplementNB()
cnb_model.fit(X_train_cnb, y_train)

# 訓練 SVM 模型
svm_model = LinearSVC(random_state=42, class_weight="balanced", C=5)
svm_model.fit(X_train_svm, y_train)

# 定義每個 label 的權重
label_weights_cnb = { "ENFJ": 0.85 ,"ENFP": 0.88 , "ENTJ":0.88 , "ENTP":0.84 , "ESFJ":1.00 , "ESFP":1.00, "ESTJ":0.63 , "ESTP":0.62,"INFJ":0.67,"INFP":0.68 ,"INTJ":0.84 ,"INTP":0.79, "ISFJ":0.40 , "ISFP":0.89 , "ISTJ":0.97 , "ISTP":0.86}
label_weights_svm = { "ENFJ": 0.75 ,"ENFP": 0.81 , "ENTJ":0.82 , "ENTP":0.84 , "ESFJ":0.69 , "ESFP":0.77, "ESTJ":0.83 , "ESTP":0.92,"INFJ":0.84,"INFP":0.82  ,"INTJ":0.87  ,"INTP":0.88, "ISFJ":0.69 , "ISFP":0.58 , "ISTJ":0.74 , "ISTP":0.81}

# NB 信心分數 (對數處理)
cnb_scores = cnb_model.predict_log_proba(X_test_cnb)
cnb_labels = cnb_model.classes_

# SVM 信心分數
svm_scores = svm_model.decision_function(X_test_svm)

# 加權分數計算
combined_scores = []
final_predictions = []

for i in range(X_test_cnb.shape[0]):
    # 對每個樣本，將 CNB 和 SVM 的信心分數加權處理
    nb_score = {label: score * label_weights_cnb[label] for label, score in zip(cnb_labels, cnb_scores[i])}
    svm_score = {label: score * label_weights_svm[label] for label, score in zip(cnb_labels, svm_scores[i])}

    # 合併 NB 和 SVM 的分數
    total_score = {label: 0.5 * nb_score[label] + 0.5 * svm_score[label] for label in cnb_labels}

    # 找出最高分數的 label
    best_label = max(total_score, key=total_score.get)
    final_predictions.append(best_label)
    combined_scores.append(total_score)

# 印出結果
# 使用 classification_report 顯示每個類別的 precision 和 recall
report = classification_report(y_test, final_predictions, target_names=cnb_labels, output_dict=True)

# 顯示 precision 和 recall
print(" label  precision recall ")
for label in cnb_labels:
    print(f" {label}:    {report[label]['precision']:.4f}  {report[label]['recall']:.4f}" )

# 計算加權平均 (weighted average) 和 宏平均 (macro average)
weighted_precision = report['weighted avg']['precision']
weighted_recall = report['weighted avg']['recall']
macro_precision = report['macro avg']['precision']
macro_recall = report['macro avg']['recall']

# 印出加權平均和宏平均
print("\nWeighted Average Precision: {:.4f}".format(weighted_precision))
print("Weighted Average Recall: {:.4f}".format(weighted_recall))
print("Macro Average Precision: {:.4f}".format(macro_precision))
print("Macro Average Recall: {:.4f}".format(macro_recall))
