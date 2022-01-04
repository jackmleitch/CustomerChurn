# Building Interpretable Models on Imbalanced Data
## Predicting customer churn from a telecom provider 

I've always believed that to truely learn data science you need to practice data science and I wanted to do this project to practice working with imbalanced classes in classification problems. This was also a perfect oppurtinity to start working with [mlflow](https://mlflow.org/) to help track my machine learning experiments: it allows me to track the different models i've used, the parameters i've training with, and the metrics i've recorded.

This project was aimed at predicting customer churn using the teleccomunications data found on [Kaggle](https://www.kaggle.com/c/customer-churn-prediction-2020/overview). That is, we want to be able to predict if a given customer is going the leave the telecom provider based on the infromation we have on that customer. Now why is this useful? Well, if we can predict which customers we think are going to leave **before** they actually leave then we can try to do something about it! For example, we could target them with specific offers, and maybe we could even use the model to provide us insight into what to offer them because we will know, or at least have an idea, as to why they are leaving. 

## Performance vs Interpretability 
It's very important to know and understand the problem/task at hand before we start to even think about writing any code. Would it be useful in this case to build a really powerful model like XGBOOST? No, of course not. Our goal isn't to squeeze every drop of performance out of our model. Our goal is to **understand** why people are leaving so we can actually do something about it and try to get them to stay. In an ideal world we would build a very interpretable model, but in reality we may have to find a happy medium between performance and interpretability. As always, logistic regression would be a good start.

## Model Evaluation
We now need to decide how we are going to evaluate our models and what we are going to be happy with. Personally, I think it's important to decide an end goal beforehand as otherwise it's going to be hard to decide when to stop, squeezing out those extra 1%s is not worth it a lot of the time.

Due to the nature of our data it is likely that our classes are going to be highly imbalanced, with the case we are interested in (customers leaving) being the minority class. This makes selecting the right metric super important.

![](Building%20Interpretable%20Models%20on%20Imbalanced%20Data_files/markdown_4_local_image_tag_0.jpeg)
    
The metrics we are going to be interested in are **precision**, **recall**, and other metrics associated with these. Precision is the ratio of correct positive predictions to the overall number of positive predictions. Recall is the ratio of correct positive predictions to the overall number of positive predictions in the dataset. 
In our case we are looking at trying to retain customers by predicting which customers are going to leave: so we aren't too fussed if we miss-classify some customers as 'churn' when they are not (false positives). If anything, these miss-classifications might be customers that would soon become 'churn' if nothing changes as they may lie on the edge of the decision boundary. So, we are looking to **maximize recall** as it will minimize the number of false negatives. 

We are also going to look at the **F-measure** as it provides a way to express both concerns of precision and recall with a single score - we don't just want to forfeit precision to get 100% recall!

### Model Specifications
Once we have built our final model, we can then use a precision-recall curve to optimize our performance of on the positive (minority class). In this case we are going to assume that stakeholders in our imaginary telecoms business want to **achieve a recall of 0.80** (i.e. we identify 80% of the positive samples correctly) while maximizing precision. 


## Data
* train.csv - the training set. Contains 4250 rows with 20 columns. 3652 samples (85.93%) belong to class churn=no and 598 samples (14.07%) belong to class churn=yes.
* test.csv - the test set. Contains 850 rows with 18 columns.

The 20 columns contain the following information:
* **state**, string. 2-letter code of the US state of customer residence
* **account_length**, numerical. Number of months the customer has been with the current telcom provider
* **area_code**, string="area_code_AAA" where AAA = 3 digit area code.
* **international_plan**, (yes/no). The customer has international plan.
* **voice_mail_plan**, (yes/no). The customer has voice mail plan.
* **number_vmail_messages**, numerical. Number of voice-mail messages.
* **total_a_b** with **a = (day, eve, night, intl)** and **b = (minutes, calls, charge)**, numerical. Total (minutes, calls, charge) of (day, eve, night, intl) calls.
* **number_customer_service_calls**, numerical. Number of calls to customer service
* **churn**, (yes/no). Customer churn - target variable.



https://gist.github.com/e3f884dd546724ce7d271d2d2d17f2e8

EDA (along with the all the modeling etc.) was done in different python scripts and I have chosen to not include it here as it is irrelevant to the topic I am writing about. It is nonetheless very important and you can find this whole project on my [Github page](https://github.com/jackmleitch/CustomerChurn). 


Due to the high cardinality of the state columns we need to be careful when encoding them otherwise we will end up with 50 different features!


```
df.head()
```




![png](Building%20Interpretable%20Models%20on%20Imbalanced%20Data_files/output_8_0.png)



We can look at a correlation matrix to see any initial promising features.



https://gist.github.com/9a199460f06955d7aff7d9a307863989




    <AxesSubplot:>




![png](Building%20Interpretable%20Models%20on%20Imbalanced%20Data_files/output_10_1.png)


Ok, this doesn't look great but we can see that churn is somewhat correlated with total_day_minutes, total_day_charge, and number_customer_service_calls. Let's build a simple **baseline model** with total_day_minutes and number_customer_service_calls (we omit total_day_charge because it's strongly correlated with total_day_minutes).

### Feature Engineering
We can generate a few features to encapsulate daily totals. We also map the state feature to the regions the state belongs to as it massively reduces the feature dimensionality. Finally, we can map churn target value to binary as this is required for a lot of models. 



https://gist.github.com/9abb064b59e6b10e63dceb8747506858




![png](Building%20Interpretable%20Models%20on%20Imbalanced%20Data_files/output_13_0.png)



## Modeling   
The first thing that always needs to be done is to split the data into train and validation sets as it is a vital step to avoid overfitting, improve generalizability, and it helps us compare potential models. In this case we use stratified K-fold cross-validation as our dataset is highly imbalanced and we want to ensure the class distribution is consistent across folds. 



https://gist.github.com/beaafeb775ce17e2bbdb4db2ceaace72

### Baseline Model
We start by building a simple baseline model so that we have something to compare our later models to. In a regression scenario we could simply use the average of the target variable at every prediction, in our classification case however we are going to use a logistic regression model trained on our two most correlated features. 

Before we start lets initialize Mlflow and write a general scoring function to evaluate our model.



https://gist.github.com/07c50b48105641b77725e57c0c7ac7d4

Now let's build our baseline model and see how it does on each validation set!



https://gist.github.com/3303c33ae9f9df2b179ee3a92d46567e

    Average F1 = 0.08725760427444854, Recall = 0.04847338935574229, Precision = 0.440932400932401


So yea, the results aren't great (actually they are terrible) but that only means we are going to get better. This was only a baseline! We can try a few things to improve our model: 
* We can balance out our classes by over and under-sampling as the imbalance is causing bias towards the majority class in our model.
* We can train on more features.

### SMOTE: To Over-Sample the Minority Class
One problem we have with imbalanced classification is that there are too few examples of the minority class for a model to effectively learn the decision boundary. One way to solve this problem would be to over-sample the examples in the minority class. This could be achieved by simply duplicating examples from the minority class in the training dataset, although this does not provide any additional information to the model. An improvement on duplicating examples from the minority class is to synthesize new examples from the minority class. A common technique for this, introduced in [this paper](https://arxiv.org/abs/1106.1813), is SMOTE. It's worth noting that over sampling isn't our only option, we could for example  under-sample (or combine a mix of the two) by using a technique such as [TOMEK-links](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis#Tomek_links) (SMOTE-TOMEK helps do both under and over-sampling in one go). In our case however, the best performance boost came from SMOTE alone. 

Before we do this however, lets write a general feature processing pipeline to get our data ready for modeling. Our function returns a Sklearn pipeline object that we can use to fit and transform our data. It first splits the data into numeric, categorical features, and binary features as we process each of these differently. The categorical features are encoded using one-hot encoding, while the binary features are left alone. Finally, the numeric features have their missing values imputed. Scaling the numeric features was also tried but it didn't lead to a performance increase. It's also in our best interest to not scale these as it makes interpreting the results harder.



https://gist.github.com/78bd3cbdb9cab83825f33bc1bbecd8fa

A general training function is written below, notice we can choose whether we want to use SMOTE or not. The sampling strategy in SMOTE controls how much we resample the minority class and it's something we can tune later.



https://gist.github.com/90e593cdffb80c45f4859bccabb29109

Before we use SMOTE, let's train a logistic regression model on all of the features to try and get a new baseline.  



https://gist.github.com/b268cb417b2519182b72bf4df2f11d8b

    Average F1 = 0.3066722566642678, Recall = 0.2107563025210084, Precision = 0.5720696669209255


The results are definitely better than before but still not great. We've waited long enough, let's try using SMOTE! We are going to use SMOTE to over-sample our churn datapoints so that we end up with equal class distributions. 



https://gist.github.com/b2e221c0ce602c0df09d42f38bf126b4

An important side note: when using SMOTE we need to evaluate performance on a validation set which has **not** been over-sampled. Otherwise, we will not be getting a true performance measure.


```
train_and_eval(df, model_name="lr_all_features_smote")
```

    Average F1 = 0.49789052753778434, Recall = 0.6906862745098039, Precision = 0.3894759793369957





    0.49789052753778434



Wow! We have boosted the F1 score from 0.31 to 0.50, and the recall has gone from 0.21 to 0.69! An important note is that the precision has practically stayed the same. Why is that? Well, what is precision measuring? Mathematically, precision the number of true positives divided by the number of true positives plus the number of false positives. It tells us that our model is correct 47% of the time when trying to predict positive samples. So by over-sampling we have decreased the number of false negatives but we have also increased the number of false positives. This is OK as we decided we will favor false positives over false negatives. An intuitive way to see this change is by looking at a **confusion matrix**. 



https://gist.github.com/ec8ef03b83cac6eaac2e8dd91d186dc3


![png](Building%20Interpretable%20Models%20on%20Imbalanced%20Data_files/output_33_0.png)


We can see that the TP number goes from 33 to 90 and the FN number goes from 86 to 29, great! However, as a consequence of this we see the FP number goes from 21 to 158. But, as mentioned earlier, we are ok with that as we care more about finding out which customers are going to leave. 
A little side note: the FP and FN rates can be tuned using the probability threshold and the easiest way to compare the two models is to compare F1 scores.

## Feature Selection 
We can train a more complicated model and then use this to select features. Specifically, we train a random forest classifier and then use SHAP values to select the most promising features. Narrowing down the feature space helps reduce dimensionality and generalizability while also making interpreting results easier. It's a win win!



https://gist.github.com/e5a15327beb7cdc82660da4d59093bf7



https://gist.github.com/e3b810e2cc57567ccfcf95caa85ac275


```
shap.summary_plot(shap_values, features=X, feature_names=feature_names, plot_type='bar')
```


![png](Building%20Interpretable%20Models%20on%20Imbalanced%20Data_files/output_38_0.png)



```
features_keep = [
    'total_charge', 'number_customer_service_calls', 'international_plan', 'total_day_minutes', 'total_day_charge', 
    'total_minutes', 'total_intl_calls', 'voice_mail_plan', 'total_intl_minutes', 'number_vmail_messages', 'total_intl_charge'
]
```

We can now define a new pipeline to preprocess and only select our needed features. 



https://gist.github.com/2946cd814c80bf28901158607772f527

Let's see how our logistic regression model does with our narrowed down features!


```
train_and_eval(df, model_name="lr_selected_features_smote")
```

    Average F1 = 0.5025520964212148, Recall = 0.6940336134453782, Precision = 0.3940789241235847





    0.5025520964212148



Awesome, we've removed features and our performance increases slightly. This confirms to us that the other features weren't important.

### But can we do better than our logistic regression?
It would be easy here to go all guns blazing and train an XGBOOST model, but remember that is **not** our goal. Our goal is to build an **interpretable** model that we can use to try and keep customers from leaving. 
As well as logistic regression, decision tree classifiers are very interpretable. Let's see how it gets on.


```
from sklearn.tree import DecisionTreeClassifier
train_and_eval(df, model=DecisionTreeClassifier(), model_name="dt_selected_features_smote")
```

    Average F1 = 0.8152920749388362, Recall = 0.8461484593837536, Precision = 0.786946006985776





    0.8152920749388362



It does well! We need to be **very** careful though as decision trees overfit like there is not tomorrow. Let's go ahead and tune hyperparameters on both models to see if we can optimize things a little more. We will use [Optuna](https://optuna.org/) as I just love how easy and fast it is. 

### Hyperparameter Tuning

We need a be super careful here, decision trees are very prone to overfitting and this is why random forest models are usually preferred. The random forest can generalize over the data in a better way as the randomized feature selection acts as a form of regularization. As discussed earlier though, in our case we care more about interpretability than performance. Now, although cross-validation is great for seeing how the model is generalizing, it doesn't necessarily prevent overfitting as we will just end up overfitting the validation sets. 

One measure of overfitting is when the training score is much higher than the testing score. I initially tried setting the objective function in the Optuna trial to the cross-validated validation scores but this still lead to overfitting as DTs don't have much regularization. 

Another possibility, that is this case worked superbly, is weighting the difference between cross-validated training scores and validation scores vs the validation score itself. For example, for F1 scores, a possible objective function is

$$ \sqrt{(\overline{F1}_{\mathrm{valid}} - \overline{F1}_{\mathrm{train}})^2} + 4(1 - \overline{F1}_{\mathrm{valid}})$$
In this case the RMSE of the difference between validation and training is weighted four times less than the validation F1 score.
Optimizing this function forces the train and valid score to stay close, while also maximizing the validation score.


```
def train_valid_f1_score(mean_train, mean_test):
    '''
    RMSE of the difference between testing and training is weighted four times less than the test accuracy 
    '''
    return np.sqrt((mean_test - mean_train)**2) + 4 * (1 - mean_test)
```



https://gist.github.com/5f66f1efcc961b87de377e1793bb3147



https://gist.github.com/2e43b479a42128bef589b4802b41ce31

    [32m[I 2021-12-28 15:03:39,797][0m A new study created in memory with name: no-name-eeeda589-2aad-4745-9c2f-ceca0dde2dce[0m


This gives us the following results:
* **Logistic Regression** Best Hyperparameters: {'solver' : 'liblinear', 'penalty' : 'l2', 'C' : 1.14, 'tol' : 0.0002, max_iter : 150}
* **Decision Tree** Best Hyperparameters: {'max_depth': 7, 'min_samples_leaf': 4, 'min_samples_split': 10, 'criterion': 'gini'},
* **SMOTE SAMPLING STRATEGY** 0.70 for LR and 0.56 for DT

Let's train some model with these parameters and see what we get!



https://gist.github.com/36e28f7be4bcf7ca5c8d4b3a771bee06


```
params = {'solver' : 'liblinear', 'penalty' : 'l2', 'C' : 1.14, 'tol' : 0.0002, 'max_iter' : 150}
train(df, LogisticRegression, params, 0.70)
```

    Average training F1 score 0.5009518013999269
    Average validation F1 score 0.5034031228088905
    Overfit score 1.9888388301734015



```
params = {'max_depth': 7, 'min_samples_leaf': 4, 'min_samples_split': 10, 'criterion': 'gini'}
train(df, DecisionTreeClassifier, params, 0.70)
```

    Average training F1 score 0.9153416502877135
    Average validation F1 score 0.8928098706544405
    Overfit score 0.451292297015511


Although this model looks great and doesn't appear to be overfitting we are going to go with the model below that has been tuned with a lower maximum depth. Our goal is interpretability and a depth of 7 doesn't really give us that. So we are sacrificing a little bit of accuracy for interpretability.


```
params = {'max_depth': 4, 'min_samples_leaf': 18, 'min_samples_split': 17, 'criterion': 'entropy'}
train(df, DecisionTreeClassifier, params, 0.78)
```

    Average training F1 score 0.8217456929425225
    Average validation F1 score 0.816103270284813
    Overfit score 0.7412293415184575



```
with mlflow.start_run(run_name="dt_tuned") as mlops_run:
        mlflow.log_metric("F1", 0.8161)
```

Great, so we now have a few potential models. We are going to move forward with the decision tree model as the logistic regression model isn't quite up to scratch. 

### Pruning the Decision Tree
Although our model doesn't appear to be overfitting, we are still going to prune the decision tree as it helps us get rid of sub-nodes that don't have much predictive power. We do this with the hope that this helps our model generalize better. An added bonus, that tends to come with most regularization, is that it also helps improve the interpretability of the model. 

We can prune our tree by picking the right cost complexity parameter. We will start by training a model on the whole dataset, with our chosen hyperparams, to find our space of $\alpha$'s - the cost complexity parameter.



https://gist.github.com/3d75470cfeddcce4d1ad481ebbc5c88a


```
# train full model 
params = {'max_depth': 4, 'min_samples_leaf': 18, 'min_samples_split': 17, 'criterion': 'entropy'}
clf, features = train_full(df=df, model=DecisionTreeClassifier, params=params)
```


```
# get cost complexity alphas from model
ccp_alphas = clf.cost_complexity_pruning_path(x_train, y_train)['ccp_alphas']
ccp_alphas
```




    array([0.        , 0.00057893, 0.00059309, 0.00098679, 0.00117254,
           0.0019927 , 0.00322822, 0.00418948, 0.00719436, 0.00811411,
           0.02386822, 0.06419409])



We can then score each of these $\alpha$'s in a cross-validated way to find the best complexity to choose.



https://gist.github.com/877473c526c0b736b2dd6a1c47389106

    {'0.0': 0.816103270284813,
     '0.0005789325156238256': 0.816103270284813,
     '0.0005930920192462885': 0.816103270284813,
     '0.0009867859833913376': 0.816103270284813,
     '0.0011725425508943704': 0.816103270284813,
     '0.0019927001338321468': 0.816103270284813,
     '0.003228215942898363': 0.816103270284813,
     '0.0041894808849862325': 0.816103270284813,
     '0.007194356040110275': 0.8142671700825213,
     '0.00811411007422369': 0.8147206576154481,
     '0.023868223077443323': 0.7667293021998818,
     '0.06419408989742087': 0.6540410407393117}


We can see that the value $\alpha = 0.00811411$ is the best complexity to choose. In general, as $\alpha$ increases the number of nodes and depth decreases. So we pick the highest $\alpha$ value that still has a good average F1 score.

We can now train our final model!


```
params = {'max_depth': 4, 'min_samples_leaf': 18, 'min_samples_split': 17, 'criterion': 'entropy', 'ccp_alpha': 0.00811411}
clf, features = train_full(df=df, model=DecisionTreeClassifier, params=params)
```

### How Does It Perform on Test Data



https://gist.github.com/99146469f80054e7b635472bb5ed4aac



https://gist.github.com/ba17efdc286e818f803e65399e53e540

    Average F1 = 0.7772925764192139, Recall = 0.6592592592592592, Precision = 0.9468085106382979



```
with mlflow.start_run(run_name="dt_final_test_data") as mlops_run:
        mlflow.log_metric("F1", 0.777)
        mlflow.log_metric("Recall", 0.659)
        mlflow.log_metric("Preision", 0.947)
```

The model does well on the test data! We can see that the precision is a lot higher than the recall however but this is something that can be tuned by changing the prediction probability threshold. In our case we are trying to get to 80% recall while maximizing precision.

### Picking the Optimal Probability Threshold

We can now tune the probability threshold to try and optimize our precision recall trade-off. Let's plot a precision recall curve and find the optimal threshold to achieve 80% recall. 



https://gist.github.com/1c8ac2e9891ea7e63bc514426ad7f2f6


![png](Building%20Interpretable%20Models%20on%20Imbalanced%20Data_files/output_75_0.png)




https://gist.github.com/9ab7a0aaed2c647fe18bc34438055d98

    0.4263565891472868


If we predict on our test set again then hopefully we will see something closer to our desired recall! And yes, I know i've committed the cardinal sin of using the test set twice but this was for demonstration purposes. The test set is normally a 'one and done' situation. 



https://gist.github.com/227faf7dda266c151caef06e3ed259e0

    Average F1 = 0.7958477508650519, Recall = 0.8518518518518519, Precision = 0.7467532467532467


Awesome! We have done what we wanted to do, and things work well! It's also great to see test scores so similar to our earlier scores as it shows we haven't overfitted our model. Let's log this model on MLFLOW.



https://gist.github.com/68a6d7db34198e31b05c7dcf9e6423bc



https://gist.github.com/001e13a6be6db4310be733101e6b9fca

### MLFLOW Experiment Session
![](Building%20Interpretable%20Models%20on%20Imbalanced%20Data_files/markdown_82_local_image_tag_0.png)

Oops, I've miss-spelled precision... oh well.

## Model Interpreting 

Tree based models split the data multiple times according to certain cutoff values in the features. By splitting, different subsets of the dataset are created, where each instance belonging to a certain subset. To predict the outcome in each leaf node, the average outcome of the training data in this node is used. 

The interpretation for decision trees are very easy: Starting from the root node, you go to the next nodes and the edges tell you which subsets you are looking at. Once you reach the leaf node, the node tells you the predicted outcome. All the edges are connected by ‘AND’.

So the general way we can predict is: If feature x is **smaller** (/bigger) than threshold c **AND** ... then the predicted outcome is the most common value of the target of the instances in that node. 

Let's plot out tree and see what we can infer! 



https://gist.github.com/fd60e430350e4fe6cad0b17d25076727




![svg](Building%20Interpretable%20Models%20on%20Imbalanced%20Data_files/output_85_0.svg)



![](Building%20Interpretable%20Models%20on%20Imbalanced%20Data_files/markdown_86_local_image_tag_0.png)

Each node in the tree will give a condition and the left node below is True and the right node is False.
The first split was performed with the total day minutes feature, which counts the total minutes of all calls made in the day. We can see for example that if the total minutes is less than 71 we follow the tree left and if the minutes is greater than 71 we go right. 

Each prediction from the tree is made by following the tree down until a root node is hit. For example, if a customer has less than 71 total day minutes and there total charge is between 0.04 and 1 then we would predict them to churn. 

We can see that the charge from the telecom provider seems to be a big distinguishing factor between customers and this is confirmed by the SHAP feature importance plot earlier. By following the tree left we can see that customers with a high day charge but low day minutes tend to churn more than stay. If the day charge is less than 3 however, the customers tend to stay no matter what the minutes are! One possible explanation to this could be that the customers churning are on mobile plans that don't correctly suit their needs, this would need to be investigated further though. 

Another interesting observation is that if a customer has a high total day minutes ($\geq 71$) and they do not speak to customer service ($\leq  0.965$ calls i.e. no calls) they are more likely to churn than customers that do speak to customer service. Again, this would need further investigation to draw conclusions as to why this is true. 

As with most data problems, it quite often leads to more questions to be answered!

### Conclusion 
We have built an interpretable machine learning model that can identify customers that are likely to churn with our desired recall of 80% (we actually achieved 85% on the test set) and a precision of 75%. That is we identify 85% of the churned customers correctly and 75% of our churn predictions are accurate. Using this model we can then understand the key factors driving customers to leave and hopefully we can use this to try and keep more customers in the long run.  

