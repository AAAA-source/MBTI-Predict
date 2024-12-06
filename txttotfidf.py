import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# 資料夾路徑
input_dir = "txt_output"
output_dir = "tf-idf"

# 確保輸出資料夾存在
os.makedirs(output_dir, exist_ok=True)

# 讀取所有 txt_output 中的檔案內容
file_contents = []
file_names = []

for file_name in os.listdir(input_dir):
    if file_name.endswith(".txt") :
        file_path = os.path.join(input_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            file_contents.append(file.read())
            file_names.append(file_name)

# 使用 TfidfVectorizer 進行轉換
vectorizer = TfidfVectorizer( min_df=2 , max_df = 0.9 , stop_words = "english" )
tfidf_matrix = vectorizer.fit_transform(file_contents)

# 將每個檔案的 TF-IDF vector 儲存到 tf-idf 資料夾中
for idx, file_name in enumerate(file_names):
    tfidf_vector = tfidf_matrix[idx].toarray()[0]  # 獲取稀疏矩陣的密集表示
    output_path = os.path.join(output_dir, f"{file_name}.npy")
    np.save(output_path, tfidf_vector)

print(f"已成功將 TF-IDF vector 儲存到 {output_dir} 資料夾內。")
