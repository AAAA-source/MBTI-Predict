# MBTI-Predict
predict author's MBTI through text

the row file of MBTI text : https://www.kaggle.com/datasets/zeyadkhalid/mbti-personality-types-500-dataset/data
(please download at kaggle)

please put the row file in the same directory with the code , and rename the file to "MBTI.csv"


1. first run the transtotxt.py code , it will split the row data to many small text data
(split by row)
2. you can run any code with suitable environment(with all package we need) , then it will return the result
3. if the file name _separate , it means that the classification/clustering is letter by letter 
    i.e. there are 4 classifier with I/E , S/N ,  F/T , J/P
4. for the result with one classifier , it will return the precision , recall , accuracy  and F1-score ; and for the multiple classifiers , it will at least return the accuracy
