from scipy.stats import loguniform
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

from src.data.load_data import read_params
from src.models.logistic_regression import feature_pipeline, score

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
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)
    

if __name__ == "__main__":
    # load config
    config = read_params("params.yaml")
    data_path = config["raw_data_config"]["raw_data_fold_csv"]
    target = config["raw_data_config"]["target"]
    # read data
    df = pd.read_csv(data_path)
    # target mapping
    target_mapping = {"no": 0, "yes": 1}
    df.loc[:, target] = df[target].map(target_mapping)
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
    hyperparameter_optim(df, DecisionTreeClassifier(), dt_space)
    # hyperparameter_optim(df, LogisticRegression(), lr_space)
