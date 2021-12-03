import pandas as pd
from src.data.load_data import read_params
# load in training data 
config = read_params("params.yaml")
external_data_path = config["external_data_config"]["external_data_csv"]
df = pd.read_csv(external_data_path, sep=",", encoding="utf-8")
# check we have our 20 cols
assert len(df.columns) == 20


df = df.drop(columns=["area_code", "state"], axis=1)
df.head()


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="white")
# Compute the correlation matrix
corr = df.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


from sklearn.model_selection import StratifiedKFold

def stratKFold(df, n_splits=5):
    """
    Perform stratified K fold cross validation on training set
    :param df: pd dataframe to split
    :param n_splits: number of folds 
    :return: df with kfold column
    """
    # create new column 'kfold' with val -1
    df["kfold"] = -1
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # target values
    y = df['churn'].values
    # initialise kfold class
    kf = StratifiedKFold(n_splits=n_splits)
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = f
    return df
        
df = stratKFold(df)


# target mapping
target_mapping = {"no": 0, "yes": 1}
df.loc[:, 'churn'] = df['churn'].map(target_mapping)


import mlflow 
from sklearn.metrics import f1_score, recall_score, precision_score

# initialize mlflow
mlflow.set_experiment("mlflow/customer_churn_model")

# scoring function 
def score(y, preds):
    """
    Returns corresponding metric scores 
    :param y: true y values
    :param preds: predicted y values
    :return: f1_score, recall, and precision scores 
    """
    f1 = f1_score(y, preds)
    recall = recall_score(y, preds)
    precision = precision_score(y, preds)
    return [f1, recall, precision]


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# baseline model
f1_scores, recall_scores, precision_scores = [], [], []
for fold in range(5):
    # define train and validation set
    features = ["total_day_minutes", "number_customer_service_calls"]
    df_train = df[df.kfold get_ipython().getoutput("= fold].reset_index(drop=True)")
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    # target and features
    y_train = df_train['churn'].values
    y_valid = df_valid['churn'].values
    # init and fit scaler 
    scaler = StandardScaler()
    x_train = scaler.fit_transform(df_train[features])
    x_valid = scaler.transform(df_valid[features])
    # create and train model
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    preds = clf.predict(x_valid)
    # score model
    scores = score(y_valid, preds)
    f1_scores.append(scores[0])
    recall_scores.append(scores[1])
    precision_scores.append(scores[2])
# average scores over each fold
f1_avg = np.average(f1_scores)
recall_avg = np.average(recall_scores)
precision_avg = np.average(precision_scores)
print(f"Average F1 = {f1_avg}, Recall = {recall_avg}, Precision = {precision_avg}")

# log metrics on mlflow 
with mlflow.start_run(run_name="lr_baseline") as mlops_run:
        mlflow.log_metric("F1", f1_avg)
        mlflow.log_metric("Recall", recall_avg)
        mlflow.log_metric("Preision", precision_avg)


from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from mlxtend.feature_selection import ColumnSelector

def feature_pipeline(config_path="params.yaml"):
    """
    :param config_path: path to params.yaml file
    :return: preprocessing feature pipeline 
    """
    # load in config information
    config = read_params(config_path)
    num_features = config["raw_data_config"]["model_features"]["numeric"]
    cat_features = config["raw_data_config"]["model_features"]["categorical"]
    # transformers
    transforms = []
    # categorical pipeline
    transforms.append(
        (
            "catergorical",
            Pipeline(
                [
                    ("select", ColumnSelector(cols=cat_features)),
                    ("encode", OneHotEncoder()),
                ]
            ),
        )
    )
    # numeric pipeline
    transforms.append(
        (
            "numeric",
            Pipeline(
                [
                    ("select", ColumnSelector(cols=num_features)),
                    ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
                    ("scale", MinMaxScaler()),
                ]
            ),
        )
    )
    # combine features
    features = FeatureUnion(transforms)
    return features


from imblearn.over_sampling import SMOTE
def train(fold, df, model=LogisticRegression(solver='newton-cg'), smote=False):
    """
    :param fold: fold to train model on
    :param df: pandas dataframe containing our data
    :param model: model to train data on
    :param smote: boolean, perform smote over-sampling 
    :return: f1, recall, precision validation score for fold, as well as y_valid and preds
    """
    # feature pipeline
    features = feature_pipeline()

    # define train and validation set
    df_train = df[df.kfold get_ipython().getoutput("= fold].reset_index(drop=True)")
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    # target and features
    y_train = df_train['churn'].values
    y_valid = df_valid['churn'].values

    # create training and validation features
    x_train = features.fit_transform(df_train)
    x_valid = features.transform(df_valid)

    # smote
    if smote:
        smt = SMOTE(random_state=42)
        x_train, y_train = smt.fit_resample(x_train, y_train)

    # create and train model
    clf = model
    clf.fit(x_train, y_train)
    preds = clf.predict(x_valid)
    # score model
    scores = score(y_valid, preds)
    return scores, [y_valid, preds]


f1_scores, recall_scores, precision_scores = [], [], []
for fold in range(5):
    scores, _ = train(fold,df, smote=False)
    f1, recall, precision = scores
    f1_scores.append(f1)
    recall_scores.append(recall)
    precision_scores.append(precision)
# average scores over each fold
f1_avg = np.average(f1_scores)
recall_avg = np.average(recall_scores)
precision_avg = np.average(precision_scores)
print(f"Average F1 = {f1_avg}, Recall = {recall_avg}, Precision = {precision_avg}")
# log metrics on mlflow 
with mlflow.start_run(run_name="lr_all_features") as mlops_run:
        mlflow.log_metric("F1", f1_avg)
        mlflow.log_metric("Recall", recall_avg)
        mlflow.log_metric("Preision", precision_avg)


def train_and_eval(df, model=LogisticRegression(solver='newton-cg'), smote=True, model_name="", params = {}):
    '''
    train model and evaluate it on each fold
    :param df: pandas dataframe containing our data
    :param model: model to train data on
    :param model_name: string, for tracking on mlflow
    :param model_name: dict, for tracking on mlflow
    :return: average score for each metric
    '''
    f1_scores, recall_scores, precision_scores = [], [], []
    for fold in range(5):
        scores, _ = train(fold, df, model=model, smote=smote)
        f1, recall, precision = scores
        f1_scores.append(f1)
        recall_scores.append(recall)
        precision_scores.append(precision)
    # average scores over each fold
    f1_avg = np.average(f1_scores)
    recall_avg = np.average(recall_scores)
    precision_avg = np.average(precision_scores)
    print(f"Average F1 = {f1_avg}, Recall = {recall_avg}, Precision = {precision_avg}")
    # log metrics on mlflow 
    with mlflow.start_run(run_name=model_name) as mlops_run:
            mlflow.log_metric("F1", f1_avg)
            mlflow.log_metric("Recall", recall_avg)
            mlflow.log_metric("Preision", precision_avg)
            if params:
                mlflow.log_params(params)


train_and_eval(df, model_name="lr_all_features_smote")


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# get preds for non-smote and smote models
_, evals = train(0, df, smote=False)
_, evals_smote = train(0, df, smote=True)
# set axis and plot
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,10))
ax1.set_title("Model without SMOTE")
ConfusionMatrixDisplay.from_predictions(*evals, ax=ax1)
ax2.set_title("Model with SMOTE")
ConfusionMatrixDisplay.from_predictions(*evals_smote, ax=ax2)
plt.tight_layout()  
plt.show()


from sklearn.tree import DecisionTreeClassifier
train_and_eval(df, model=DecisionTreeClassifier(), model_name="dt_all_features_smote")


from scipy.stats import loguniform
# define search spaces
lr_space = {
    'solver': ['newton-cg', 'lbfgs', 'liblinear'], 
    'penalty': ['none', 'l1', 'l2', 'elasticnet'],
    'C': loguniform(1e-5, 100)
}
# limit max_depth as prone to over-fitting
dt_space = {'max_depth':[2,3,4,5], 
            'min_samples_leaf':[1,2,4,6,8,10,20,30],
            'min_samples_split':[1,2,3,4,5,6,8,10],
            'criterion': ["gini", "entropy"]
           }


# random search to find best hyperparams
from scipy.stats import loguniform
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
def hyperparameter_optim(df, model, params):
    '''
    Optimize hyperparameters for given model using random search 
    and stratified k fold cross validation
    :param df: pandas dataframe
    :param model: model to optimize
    :param params: param dictionary to search
    '''
    # feature pipeline
    features = feature_pipeline()
    # target and features
    x_train = features.fit_transform(df)
    y_train = df['churn'].values
    # smote
    smt = SMOTE(random_state=42)
    x_train, y_train = smt.fit_resample(x_train, y_train)
    # define model
    model = model
    # define cross val.
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    # define search
    search = RandomizedSearchCV(model, params, n_iter=500, scoring='f1', n_jobs=-1, cv=cv, random_state=1)
    # execute search
    result = search.fit(x_train, y_train)
    # summarize result
    print('Best Score: get_ipython().run_line_magic("s'", " % result.best_score_)")
    print('Best Hyperparameters: get_ipython().run_line_magic("s'", " % result.best_params_)")
    
# hyperparameter_optim(df, DecisionTreeClassifier(), dt_space)
# hyperparameter_optim(df, LogisticRegression(), lr_space)


lr_params = {'C': 68, 'penalty': 'l1', 'solver': 'liblinear'}
train_and_eval(df, model=LogisticRegression(**lr_params), model_name="lr_smote_tuned", params=lr_params)


dt_params = {'min_samples_split': 6, 'min_samples_leaf': 4, 'max_depth': 5, 'criterion': 'gini'}
train_and_eval(df, model=DecisionTreeClassifier(**dt_params), model_name="dt_smote_tuned", params=dt_params)


def train_full(df, model, params):
    """
    :param df: pandas dataframe containing our data
    :param model: model to train data on
    :param params: model hyperparameters
    :return: f1, recall, precision validation score for fold, as well as y_valid and preds
    """
    # feature pipeline
    features = feature_pipeline()

    # target and features
    x_train = features.fit_transform(df)
    y_train = df['churn'].values

    smt = SMOTE(random_state=42)
    x_train, y_train = smt.fit_resample(x_train, y_train)

    # create and train model
    clf = model(**params)
    clf.fit(x_train, y_train)
    return clf, features


clf, features = train_full(df=df, model=DecisionTreeClassifier, params=dt_params)


from sklearn.metrics import PrecisionRecallDisplay
# training data
x_train = features.transform(df)
y_train = df['churn'].values
# precision recall curve
display = PrecisionRecallDisplay.from_estimator(
    clf, x_train, y_train, name="Decision tree classifier"
)
_ = display.ax_.set_title("Precision-Recall curve")


from sklearn.metrics import precision_recall_curve
def optimize_threshold(clf, df, recall = 0.70):
    '''
    Uses cross-validation to optimize prob. threshold for required task
    :param clf: model to optimize
    :param df: training data with kfold col
    :param recall: desired recall 
    :return: optimal prob. threshold
    '''
    threshs = []
    # loop over each fold
    for fold in range(5):
        # define train and validation set
        df_train = df[df.kfold get_ipython().getoutput("= fold].reset_index(drop=True)")
        df_valid = df[df.kfold == fold].reset_index(drop=True)
        # target and features
        y_train = df_train['churn'].values
        y_valid = df_valid['churn'].values
        # fit model on training data 
        clf, features = train_full(df=df_train, model=DecisionTreeClassifier, params=dt_params)
        # get scores for valid data
        x_valid = features.transform(df_valid)
        y_scores = clf.predict_proba(x_valid)[:, 1]
        # locate where recall is closest to 0.70
        precisions, recalls, thresholds = precision_recall_curve(y_valid, y_scores)
        distance_to_optim = abs(recalls - recall)
        optimal_idx = np.argmin(distance_to_optim)
        threshs.append(thresholds[optimal_idx])
    # average optimal thresh across all folds
    return np.average(threshs)

thresh = optimize_threshold(clf, df)
print(thresh)


def predict(X, clf, feature_pipeline, thresh=0.71):
    '''
    Predict on new data 
    :param X: data containing features
    :param clf: trained model
    :param feature_pipeline: trained feature processing pipeline
    :param thresh: prediction threshold
    :return: predictions
    '''
    X = feature_pipeline.transform(X)
    preds = (clf.predict_proba(X)[:,1] >= thresh).astype(int)
    return preds

# train full model 
clf, features = train_full(df=df, model=DecisionTreeClassifier, params=dt_params)

# load in test data and score
df_test = pd.read_csv("data/external/test.csv")
# target mapping
target_mapping = {"no": 0, "yes": 1}
df_test.loc[:, 'churn'] = df_test['churn'].map(target_mapping)
y_test = df_test['churn'].values
# predict on unseen data
preds = predict(df_test, clf, features)
# score
f1, recall, precision = score(y_test, preds)
print(f"Average F1 = {f1}, Recall = {recall}, Precision = {precision}")


# log metrics on mlflow 
with mlflow.start_run(run_name="Final DT Model") as mlops_run:
        mlflow.log_metric("F1", f1)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("Preision", precision)
        mlflow.log_params(dt_params)
        mlflow.log_params({"prob_thresh": 0.71})
        # log model and pipeline
        mlflow.sklearn.log_model(clf, "final_model")
        mlflow.sklearn.log_model(features, "features_pipeline")
        


# import pickle

# with open('models/final_DT_model.pickle', 'wb') as f:
#     pickle.dump(clf, f)

# with open('models/feature_pipeline.pickle', 'wb') as f:
#     pickle.dump(features, f)
