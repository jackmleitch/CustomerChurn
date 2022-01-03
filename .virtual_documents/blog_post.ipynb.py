import pandas as pd
from src.data.load_data import read_params
# load in training data 
config = read_params("params.yaml")
external_data_path = config["external_data_config"]["external_data_csv"]
df = pd.read_csv(external_data_path, sep=",", encoding="utf-8")
# check we have our 20 cols
assert len(df.columns) == 20


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


def preprocess(df):
    # add new features
    df['total_minutes'] = df['total_day_minutes'] + df['total_eve_minutes'] + df['total_night_minutes']
    df['total_calls'] = df['total_day_calls'] + df['total_eve_calls'] + df['total_night_calls']
    df['total_charge'] = df['total_day_charge'] + df['total_eve_charge'] + df['total_night_charge'] 
    # target mapping
    target_mapping = {"no": 0, "yes": 1}
    df.loc[:, 'churn'] = df['churn'].map(target_mapping)
    # map state 
    state_mapping = {
        'AK': 'O', 'AL': 'S', 'AR': 'S', 'AS': 'O', 'AZ': 'W', 'CA': 'W', 'CO': 'W', 'CT': 'N', 'DC': 'N', 'DE': 'N', 'FL': 'S', 'GA': 'S',
        'GU': 'O', 'HI': 'O', 'IA': 'M', 'ID': 'W', 'IL': 'M', 'IN': 'M', 'KS': 'M', 'KY': 'S', 'LA': 'S', 'MA': 'N', 'MD': 'N', 'ME': 'N',
        'MI': 'W', 'MN': 'M', 'MO': 'M', 'MP': 'O', 'MS': 'S', 'MT': 'W', 'NA': 'O',  'NC': 'S', 'ND': 'M', 'NE': 'W', 'NH': 'N', 'NJ': 'N',
        'NM': 'W', 'NV': 'W', 'NY': 'N', 'OH': 'M', 'OK': 'S', 'OR': 'W', 'PA': 'N', 'PR': 'O', 'RI': 'N', 'SC': 'S', 'SD': 'M', 'TN': 'S', 
        'TX': 'S', 'UT': 'W', 'VA': 'S', 'VI': 'O', 'VT': 'N', 'WA': 'W', 'WI': 'M', 'WV': 'S', 'WY': 'W'
    }
    df.loc[:, 'state'] = df['state'].map(state_mapping)
    return df
# preprocess dataframe and add features
df = preprocess(df)
df.head()


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


from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from mlxtend.feature_selection import ColumnSelector
from category_encoders import HashingEncoder

def feature_pipeline(config_path="params.yaml"):
    """
    :param config_path: path to params.yaml file
    :return: preprocessing feature pipeline 
    """
    # load in config information
    config = read_params(config_path)
    num_features = config["raw_data_config"]["model_features"]["numeric"]
    cat_features = config["raw_data_config"]["model_features"]["categorical"]
    binary_features = config["raw_data_config"]["model_features"]["binary"]

    # transformers
    transforms = []
    # categorical pipeline
    transforms.append(
        (
            "categorical",
            Pipeline(
                [
                    ("select", ColumnSelector(cols=cat_features)),
                    ("encode", OneHotEncoder()),
                ]
            ),
        )
    )
    transforms.append(
        (
            "binary",
            Pipeline(
                [
                    ("select", ColumnSelector(cols=binary_features)),
                    ("encode", OrdinalEncoder()),
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
                    ("impute", SimpleImputer(missing_values=np.nan, strategy="median")),   
                ]
            ),
        )
    )
    
    # combine features
    features = FeatureUnion(transforms)
    return features


from imblearn.over_sampling import SMOTE, SMOTENC
def train(fold, df, model=LogisticRegression(solver='newton-cg'), smote=False):
    """
    :param fold: fold to train model on
    :param df: pandas dataframe containing our data
    :param model: model to train data on
    :param smote: float, if named it is the sampling strategy for SMOTE
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
        smt = SMOTE(random_state=42, sampling_strategy=smote) 
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


def train_and_eval(df, model=LogisticRegression(solver='newton-cg'), smote=0.75, model_name="", params = {}, log_mlflow=True):
    '''
    train model and evaluate it on each fold
    :param df: pandas dataframe containing our data
    :param model: model to train data on
    :param model_name: string, for tracking on mlflow
    :param params: dict, for tracking on mlflow
    :param log_mlflow: boolean, if true then log results using mlflow
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
    if log_mlflow:
        with mlflow.start_run(run_name=model_name) as mlops_run:
                mlflow.log_metric("F1", f1_avg)
                mlflow.log_metric("Recall", recall_avg)
                mlflow.log_metric("Preision", precision_avg)
                if params:
                    mlflow.log_params(params)
    return f1_avg


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


from sklearn.ensemble import RandomForestClassifier
features = feature_pipeline()
X = np.asarray(features.fit_transform(df).todense())
y = df['churn'].values

clf = RandomForestClassifier()
model = clf.fit(X,y)

import shap 
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)


# get feature names
num_features = config["raw_data_config"]["model_features"]["numeric"]
cat_features = config["raw_data_config"]["model_features"]["categorical"]
bin_features = config["raw_data_config"]["model_features"]["binary"]
cat_features = features.transformer_list[0][1][1].get_feature_names_out(cat_features)
feature_names = list(cat_features) + list(bin_features) + list(num_features)


shap.summary_plot(shap_values, features=X, feature_names=feature_names, plot_type='bar')


features_keep = [
    'total_charge', 'number_customer_service_calls', 'international_plan', 'total_day_minutes', 'total_day_charge', 
    'total_minutes', 'total_intl_calls', 'voice_mail_plan', 'total_intl_minutes', 'number_vmail_messages', 'total_intl_charge'
]


def feature_pipeline():
    """
    :return: preprocessing feature pipeline with selected features
    """
    # different features
    numeric_features = [
        'total_charge', 'number_customer_service_calls', 'total_day_minutes', 'total_day_charge', 
        'total_minutes', 'total_intl_calls', 'total_intl_minutes', 'number_vmail_messages', 'total_intl_charge'
    ]
    binary_features = ['international_plan', 'voice_mail_plan']

    # transformers
    transforms = []
    
    transforms.append(
        (
            "binary",
            Pipeline(
                [
                    ("select", ColumnSelector(cols=binary_features)),
                    ("encode", OrdinalEncoder()),
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
                    ("select", ColumnSelector(cols=numeric_features)),
                    ("impute", SimpleImputer(missing_values=np.nan, strategy="median")),   
                ]
            ),
        )
    )
    
    # combine features
    features = FeatureUnion(transforms)
    return features


train_and_eval(df, model_name="lr_selected_features_smote")


from sklearn.tree import DecisionTreeClassifier
train_and_eval(df, model=DecisionTreeClassifier(), model_name="dt_selected_features_smote")


def train_valid_f1_score(mean_train, mean_test):
    '''
    RMSE of the difference between testing and training is weighted four times less than the test accuracy 
    '''
    return np.sqrt((mean_test - mean_train)**2) + 4 * (1 - mean_test)


from optuna import Trial, create_study
from optuna.samplers import TPESampler
from scipy.stats import loguniform

def objective(trial, n_jobs=-1, random_state=42):
    '''
    Objective function to optimize our custom metric using Optuna's TPE sampler
    '''
    # smote param space 
    smote_space = {'sampling_strategy': trial.suggest_uniform('sampling_strategy', 0.5, 1)}
    # define search spaces
    if model == LogisticRegression:
        params = {
            'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']), 
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'C': trial.suggest_float("C", 1.0, 10.0, log=True),
            'tol': trial.suggest_float("tol", 0.0001, 0.01, log=True),
            'max_iter': trial.suggest_int('max_iter', 100, 1000)

        }
    else:
        params = {
            'max_depth': trial.suggest_int('max_depth',2,10), 
            'min_samples_leaf': trial.suggest_int('min_samples_leaf',1,30),
            'min_samples_split': trial.suggest_int('min_samples_split',2,10),
            'criterion': trial.suggest_categorical('criterion', ["gini", "entropy"])
        }
        
    # feature pipeline
    features = feature_pipeline()
    # create training and validation features
    X = features.fit_transform(df)
    y = df['churn'].values

    train_f1, valid_f1 = [], []
    # Create StratifiedKFold object.
    strat = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  
    for train_index, test_index in strat.split(X, y):
        # split data
        x_train, x_valid = X[train_index], X[test_index]
        y_train, y_valid = y[train_index], y[test_index]
        # feature transformations and smote
        smt = SMOTE(random_state=42, **smote_space) 
        x_train_smt, y_train_smt = smt.fit_resample(x_train, y_train)
        # train model
        clf = model(**params)
        clf.fit(x_train_smt, y_train_smt)
        # compute f1 score on valid and training data (without SMOTEget_ipython().getoutput(")")
        preds_train = clf.predict(x_train)
        preds_valid = clf.predict(x_valid)
        train_f1.append(f1_score(y_train, preds_train))
        valid_f1.append(f1_score(y_valid, preds_valid))
    # compute mean of f1 train/valid scores
    train_f1_mean, valid_f1_mean = np.array(train_f1).mean(), np.array(valid_f1).mean()
    # train/test cross score 
    return train_valid_f1_score(train_f1_mean, valid_f1_mean)


# model to optimize
model = DecisionTreeClassifier

# Bayesian sampler 
sampler = TPESampler()
study = create_study(direction="minimize", sampler=sampler)
# study.optimize(objective, n_trials=100)

# # display params
# best = study.best_params
# for key, value in best.items():
#     print(f"{key:>20s} : {value}")
# print(f"{'best objective value':>20s} : {study.best_value}")


def train(df, model, params, sampling_strategy):
    '''
    Train model and output both training and validation f1 scores
    :param df: pandas dataframe of our data
    :param model: model to train
    :param params: dict, model hyperparams
    :param sampling_strategy: float, sampling strat for SMOTE
    :return: prints training and validation F1 scores
    '''
    # feature pipeline
    features = feature_pipeline()
    # create training and validation features
    X = features.fit_transform(df)
    y = df['churn'].values

    train_f1, valid_f1 = [], []
    # Create StratifiedKFold object.
    strat = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  
    for train_index, test_index in strat.split(X, y):
        # split data
        x_train, x_valid = X[train_index], X[test_index]
        y_train, y_valid = y[train_index], y[test_index]
        # feature transformations and smote
        smt = SMOTE(random_state=42, sampling_strategy=sampling_strategy) 
        x_train_smt, y_train_smt = smt.fit_resample(x_train, y_train)
        # train model
        clf = model(**params)
        clf.fit(x_train_smt, y_train_smt)
        # compute f1 score on valid and training data (without SMOTEget_ipython().getoutput(")")
        preds_train = clf.predict(x_train)
        preds_valid = clf.predict(x_valid)
        train_f1.append(f1_score(y_train, preds_train))
        valid_f1.append(f1_score(y_valid, preds_valid))
    # compute mean of f1 train/valid scores
    train_f1_mean, valid_f1_mean = np.array(train_f1).mean(), np.array(valid_f1).mean()
    # train/test cross score 
    print(f"Average training F1 score {train_f1_mean}")
    print(f"Average validation F1 score {valid_f1_mean}")
    print(f"Overfit score {train_valid_f1_score(train_f1_mean, valid_f1_mean)}")


params = {'solver' : 'liblinear', 'penalty' : 'l2', 'C' : 1.14, 'tol' : 0.0002, 'max_iter' : 150}
train(df, LogisticRegression, params, 0.70)


params = {'max_depth': 7, 'min_samples_leaf': 4, 'min_samples_split': 10, 'criterion': 'gini'}
train(df, DecisionTreeClassifier, params, 0.70)


params = {'max_depth': 4, 'min_samples_leaf': 18, 'min_samples_split': 17, 'criterion': 'entropy'}
train(df, DecisionTreeClassifier, params, 0.78)


with mlflow.start_run(run_name="dt_tuned") as mlops_run:
        mlflow.log_metric("F1", 0.8161)


def train_full(df, model, params, smote=0.78):
    """
    train model on whole dataset
    :param df: pandas dataframe containing our data
    :param model: model to train data on
    :param params: dict, model hyperparameters
    :return: trained model and feature transformation pipeline
    """
    # feature pipeline
    features = feature_pipeline()

    # target and features
    x_train = features.fit_transform(df)
    y_train = df['churn'].values

    smt = SMOTE(random_state=42, sampling_strategy=smote)
    x_train, y_train = smt.fit_resample(x_train, y_train)

    # create and train model
    clf = model(**params)
    clf.fit(x_train, y_train)
    return clf, features


# train full model 
params = {'max_depth': 4, 'min_samples_leaf': 18, 'min_samples_split': 17, 'criterion': 'entropy'}
clf, features = train_full(df=df, model=DecisionTreeClassifier, params=params)


# get cost complexity alphas from model
ccp_alphas = clf.cost_complexity_pruning_path(x_train, y_train)['ccp_alphas']
ccp_alphas


def search_alpha_space(alphas, df=df):
    '''
    Cross validated scoring of each ccp_alpha 
    :param alphas: list, ccp_alphas from trained model 
    :param df: pandas dataframe 
    :return: dict, of alphas and avergae corresponding validation F1 score
    '''
    # optimized parameters
    params = {'max_depth': 4, 'min_samples_leaf': 18, 'min_samples_split': 17, 'criterion': 'entropy'}
    sampling_strategy = 0.78
        
    # feature pipeline
    features = feature_pipeline()
    # create training and validation features
    X = features.fit_transform(df)
    y = df['churn'].values

    f1 = {}
    # Create StratifiedKFold object.
    strat = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  
    for alpha in alphas:
        score = []
        for train_index, test_index in strat.split(X, y):
            # split data
            x_train, x_valid = X[train_index], X[test_index]
            y_train, y_valid = y[train_index], y[test_index]
            # feature transformations and smote
            smt = SMOTE(random_state=42, sampling_strategy=0.78) 
            x_train_smt, y_train_smt = smt.fit_resample(x_train, y_train)
            # train model
            clf = DecisionTreeClassifier(ccp_alpha=alpha, **params)
            clf.fit(x_train_smt, y_train_smt)
            # compute f1 score on valid and training data (without SMOTEget_ipython().getoutput(")")
            preds_valid = clf.predict(x_valid)
            score.append(f1_score(y_valid, preds_valid))
        f1[f'{alpha}'] = np.array(score).mean()
    return f1

optim_alpha = search_alpha_space(ccp_alphas)

# print scored alphas
from pprint import pprint 
pprint(optim_alpha)


params = {'max_depth': 4, 'min_samples_leaf': 18, 'min_samples_split': 17, 'criterion': 'entropy', 'ccp_alpha': 0.00811411}
clf, features = train_full(df=df, model=DecisionTreeClassifier, params=params)


def preprocess(df, target=False):
    '''
    Preprocessing unseen data by adding features and mapping target variable
    :param df: pandas dataframe 
    :param target: str, default False but if true and a string then map target to binary
    :return: processed dataframe
    '''
    # add new features
    df['total_minutes'] = df['total_day_minutes'] + df['total_eve_minutes'] + df['total_night_minutes']
    df['total_charge'] = df['total_day_charge'] + df['total_eve_charge'] + df['total_night_charge'] 
    # target mapping if req.
    if target:
        target_mapping = {"no": 0, "yes": 1}
        df.loc[:, f'{target}'] = df[f'{target}'].map(target_mapping)
    return df

def predict(X, clf, feature_pipeline, thresh=0.5):
    '''
    Predict customer churn on new data 
    :param X: data containing features
    :param clf: trained model
    :param feature_pipeline: trained feature processing pipeline
    :param thresh: prediction threshold
    :return: predictions
    '''
    X = feature_pipeline.transform(X)
    preds = (clf.predict_proba(X)[:,1] >= thresh).astype(int)
    return preds


# load in test data and score
df_test = pd.read_csv("data/external/test.csv")
# preprocess data
df_test = preprocess(df_test, target='churn')
# target mapping
y_test = df_test['churn'].values
# predict on unseen data
preds = predict(df_test, clf, features)
# score
f1, recall, precision = score(y_test, preds)
print(f"Average F1 = {f1}, Recall = {recall}, Precision = {precision}")


with mlflow.start_run(run_name="dt_final_test_data") as mlops_run:
        mlflow.log_metric("F1", 0.777)
        mlflow.log_metric("Recall", 0.659)
        mlflow.log_metric("Preision", 0.947)


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
def optimize_threshold(clf, df, recall = 0.80):
    '''
    Optimize prob. threshold on training dataset 
    :param df: pandas dataframe
    :param recall: desired recall 
    :return: optimal prob. threshold
    '''    
    # create features and target labels
    X = features.transform(df)
    y = df['churn'].values

    # get scores for valid data
    y_scores = clf.predict_proba(X)[:, 1]
    # locate where recall is closest to 0.80
    precisions, recalls, thresholds = precision_recall_curve(y, y_scores)
    distance_to_optim = abs(recalls - recall)
    optimal_idx = np.argmin(distance_to_optim)
    thresh = thresholds[optimal_idx]
    return thresh

thresh = optimize_threshold(clf, df)
print(thresh)


# predict on unseen data
preds = predict(df_test, clf, features, thresh=0.426)
# score
f1, recall, precision = score(y_test, preds)
print(f"Average F1 = {f1}, Recall = {recall}, Precision = {precision}")
with mlflow.start_run(run_name="dt_final_test_data_threshold_tuning") as mlops_run:
        mlflow.log_metric("F1", f1)
        mlflow.log_metric("Preision", precision)
        mlflow.log_metric("Recall", recall)


# log metrics on mlflow 
with mlflow.start_run(run_name="Final Decision Tree") as mlops_run:
        mlflow.log_metric("F1", f1)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("Preision", precision)
        mlflow.log_params(params)
        mlflow.log_params({"prob_thresh": 0.426})
        # log model and pipeline
        mlflow.sklearn.log_model(clf, "clf")
        mlflow.sklearn.log_model(features, "features_pipeline")


# import pickle

# with open('models/final_DT_model.pickle', 'wb') as f:
#     pickle.dump(clf, f)

# with open('models/feature_pipeline.pickle', 'wb') as f:
#     pickle.dump(features, f)


# get feature names
numeric_features = [
    'total_charge', 'number_customer_service_calls', 'total_day_minutes', 'total_day_charge', 
    'total_minutes', 'total_intl_calls', 'total_intl_minutes', 'number_vmail_messages', 'total_intl_charge'
]
binary_features = ['international_plan', 'voice_mail_plan']
feature_names = np.append(numeric_features, binary_features)

# plot tree from our model
import graphviz 
from sklearn import tree
tree_graph = tree.export_graphviz(clf, out_file=None, 
                              feature_names=feature_names,  
                              class_names=['No churn', 'Churn'],  
                              filled=True, rounded=True,  
                              special_characters=True)

graph = graphviz.Source(tree_graph) 
graph 
