import pandas as pd
import pandas_profiling as pp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os
import plotly.graph_objects as go
import plotly.io as pio
import pickle
from sklearn.utils import resample
# Metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, auc, roc_curve
from sklearn import metrics

# Validation
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline, make_pipeline

# Tuning
from sklearn.model_selection import GridSearchCV

# Feature Extraction
from sklearn.feature_selection import RFE

# Preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer, LabelEncoder

# Models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Ensembles
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier

warnings.filterwarnings('ignore')


sns.set_style("whitegrid", {'axes.grid' : False})
pio.templates.default = "plotly_white"



################################################################################
#                                                                              #
#                            Analyze Data                                      #
#                                                                              #
################################################################################
def explore_data(df):
    print("Number of Instances and Attributes:", df.shape)
    print('\n')
    print('Dataset columns:',df.columns)
    print('\n')
    df.info()
    if df.isna().sum().any():
        print("Some data is null, have to fill the gaps")
        remove_nan(df)

################################################################################
#                                                                              #
#                      Removing the NotANumber                                 #
#                                                                              #
################################################################################
def remove_nan(df):
    null_list = df.columns[df.isnull().any()].to_list()

    for null_col in null_list:
        df[null_col] = df[null_col].fillna(df.groupby(['Potability'])[null_col].transform('mean'))
    return

################################################################################
#                                                                              #
#                      Checking for Duplicates                                 #
#                                                                              #
################################################################################
def checking_removing_duplicates(df):
    count_dups = df.duplicated().sum()
    print("Number of Duplicates: ", count_dups)
    if count_dups >= 1:
        df.drop_duplicates(inplace=True)
        print('Duplicate values removed!')
    else:
        print('No Duplicate values')
################################################################################
#                                                                              #
#                Split Data to Training and Validation set                     #
#                                                                              #
################################################################################
def read_in_and_split_data(data, target):
    explore_data(data)
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=0)
    return X_train, X_test, y_train, y_test
################################################################################
#                                                                              #
#                        Spot-Check Algorithms                                 #
#                                                                              #
################################################################################
def get_models():
    Models = []
    Models.append(('Logistic_Regression'   , LogisticRegression()))
    Models.append(('Decision_Tree'   , DecisionTreeClassifier()))
    Models.append(('Gradient_Boosting'   , GradientBoostingClassifier()))
    Models.append(('Random_Forest'  , RandomForestClassifier()))
    Models.append(('KNeighbors'  , KNeighborsClassifier()))
    Models.append(('Gaussian_NB'   , GaussianNB()))
    Models.append(('SVC'  , SVC(probability=True)))
    return Models

def ensemblemodels():
    ensembles = []
    ensembles.append(('AB'   , AdaBoostClassifier()))
    ensembles.append(('GBM'  , GradientBoostingClassifier()))
    ensembles.append(('RF'   , RandomForestClassifier()))
    ensembles.append(( 'Bagging' , BaggingClassifier()))
    ensembles.append(('ET', ExtraTreesClassifier()))
    return ensembles
################################################################################
#                                                                              #
#                 Spot-Check Normalized Models                                 #
#                                                                              #
################################################################################
def NormalizedModel(nameOfScaler):
    
    if nameOfScaler == 'standard':
        scaler = StandardScaler()
    elif nameOfScaler =='minmax':
        scaler = MinMaxScaler()
    elif nameOfScaler == 'normalizer':
        scaler = Normalizer()
    elif nameOfScaler == 'binarizer':
        scaler = Binarizer()

    pipelines = []
    pipelines.append((nameOfScaler+'LR'  , Pipeline([('Scaler', scaler),('LR'  , LogisticRegression())])))
    pipelines.append((nameOfScaler+'LDA' , Pipeline([('Scaler', scaler),('LDA' , LinearDiscriminantAnalysis())])))
    pipelines.append((nameOfScaler+'KNN' , Pipeline([('Scaler', scaler),('KNN' , KNeighborsClassifier())])))
    pipelines.append((nameOfScaler+'CART', Pipeline([('Scaler', scaler),('CART', DecisionTreeClassifier())])))
    pipelines.append((nameOfScaler+'NB'  , Pipeline([('Scaler', scaler),('NB'  , GaussianNB())])))
    pipelines.append((nameOfScaler+'SVM' , Pipeline([('Scaler', scaler),('SVM' , SVC())])))
    pipelines.append((nameOfScaler+'AB'  , Pipeline([('Scaler', scaler),('AB'  , AdaBoostClassifier())])  ))
    pipelines.append((nameOfScaler+'GBM' , Pipeline([('Scaler', scaler),('GMB' , GradientBoostingClassifier())])  ))
    pipelines.append((nameOfScaler+'RF'  , Pipeline([('Scaler', scaler),('RF'  , RandomForestClassifier())])  ))
    pipelines.append((nameOfScaler+'ET'  , Pipeline([('Scaler', scaler),('ET'  , ExtraTreesClassifier())])  ))

    return pipelines
################################################################################
#                                                                              #
#                           Train Model                                        #
#                                                                              #
################################################################################
def fit_models(X_train, y_train, models):
    # Test options and evaluation metric
    num_folds = 10
    scoring = 'accuracy'

    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=0)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
    return names, results
################################################################################
#                                                                              #
#                          Save Trained Model                                  #
#                                                                              #
################################################################################
def save_model(name, model):
    PATH="./models"
    pickle.dump(model, open(f"{PATH}/{name}.pkl", 'wb'))
################################################################################
#                                                                              #
#                          Performance Measure                                 #
#                                                                              #
################################################################################
def classification_metrics(model, conf_matrix):
    print(f"Training Accuracy Score: {model.score(X_train, y_train) * 100:.1f}%")
    print(f"Validation Accuracy Score: {model.score(X_test, y_test) * 100:.1f}%")
    fig,ax = plt.subplots(figsize=(8,6))
    sns.heatmap(pd.DataFrame(conf_matrix), annot = True, cmap = 'YlGnBu',fmt = 'g')
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.title('Confusion Matrix', fontsize=20, y=1.1)
    plt.ylabel('Actual label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.show()
    print(classification_report(y_test, y_pred))
    

################################################################################
#                                                                              #
#                              Scale Inputs                                    #
#                                                                              #
################################################################################
def scale_inputs(X_train, X_test, y_train, y_test):
    mmscaler = MinMaxScaler()
    X_train = mmscaler.fit_transform(X_train)
    X_test = mmscaler.fit(X_test)

    y_train = LabelEncoder().fit_transform(np.asarray(y_train).ravel())
    y_test = LabelEncoder().fit_transform(np.asarray(y_test).ravel())


# Load Dataset
df = pd.read_csv('water_potability.csv')

# Remove Outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df_out = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Split Data to Training and Validation set
target ='Potability'
X_train, X_test, y_train, y_test = read_in_and_split_data(df, target)

# scale inputs
scale_inputs(X_train, X_test, y_train, y_test)
models = get_models()

# Train models
names, results = fit_models(X_train, y_train, models)

# pipeline = make_pipeline(StandardScaler(), RandomForestClassifier())
# model = pipeline.fit(X_train, y_train)

# Test and save models
for name, model in models:
    pipeline = make_pipeline(MinMaxScaler(), model)
    model = pipeline.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_metrics(pipeline, conf_matrix)

    save_model(name, model)

# print(y_test)
# print(y_test.shape)
# print(type(y_test))
# np.savetxt('./misc/y_test.txt', y_test)

y_test.to_csv('./misc/y_test.csv')
