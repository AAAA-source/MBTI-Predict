# MBTI-Predict
predict author's MBTI through text

the row file of MBTI text : https://www.kaggle.com/datasets/zeyadkhalid/mbti-personality-types-500-dataset/data
(please download at kaggle)

please put the row file in the same directory with the code , and rename the file to "MBTI.csv"


1. first run the transtotxt.py code , it will split the row data to many small text data
(split by row) and put them into "txt_output" file.
2. you can run any code with suitable environment(with all package we need) , then it will return the result
3. if the file name _separate , it means that the classification/clustering is letter by letter 
    i.e. there are 4 classifier with I/E , S/N ,  F/T , J/P 
4. if the file name _reduce , it means the data will first be resample before training model.
    which will resample by two ways(oversampling/undersampling)
5. for the result with one classifier , it will return the precision , recall , accuracy  and F1-score ; and for the multiple classifiers , it will at least return the accuracy.

-------------------------------------------------------------------------------------
For our research , we split in 3 steps : 
1. train general models : we choosed the Naive Bayes(NB) , BernoulliNB , Complement NB , KNN , SVM(with linear kernel and rbf kernel) , decision tree methods. and train those models with K-fold validation(k = 10)

After that , we found that out data is unbalanced , some of categories have more the 20000 sample , but some only have less than 1000 sample.
So we try to conquer this problem. Therefore , we try the balanced random forest method and the xgboost way.And resample data making the data set "balanced"

2. We then try the balanced random forest method and the xgboost way. However , the two models performence aren't good enough (in fact , the performance are worse than SVM , NB model).
And every model with resample data preform worse then raw data. This result just like what we thought.(we believe the more data be trained , the better preformence;
but the resample data make the training set small or adding some random data will interfere the raw distribution).

Because step can not efficiently conquer the problem , we try another way -- merge the models.

3. We choose the best two models which we already have. The SVM and CNB have the best f1-score and precision.
Because the prediction of MBTI is try to predict the author's personality , we now focus on precision.

We define the confidence score of SVM model with class c as SVM(c) and the confidence score of CNB model with class c as CNB(c)
Now , we define the final score of merge model is 
    final score(c) = SVM(c) * precSVM(c) + CNB(c) * precCNB(c)
which precSVM(c) is the precision of SVM model in class c (we observed by k-fold validation before , step 1) 
and precCNB is the precision of CNB model in class c.

Last , the merge model will output the class which have the maximum final score.

After we training the merge model , we compare the "macro average" of the model and SVM , CNB model. 
(Choose macro average is because the data is unbalanced and we believe the author's MBTI personality is uniform distribution in real world)

The macro average of the merge model is 0.83 , and SVM is 0.81 , CNB is 0.78
