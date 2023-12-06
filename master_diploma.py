# Магістерський Диплом Студента КНСШ-23 Черещука Любомира

# load needed libs
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

import plotly.offline as pyo
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data.csv")
df.head()

# get data set info
df.info()

# get data set descriptive statistic info
df.describe()

# get number of unique values for eaach column
df.nunique()

# check if we have null values in data set
df.isnull().sum()

# plot features distribution

# plot gender, heart_disease, hypertension features
fig = make_subplots(rows=1, cols=3, subplot_titles=['gender', 'heart_disease', 'hypertension'])
fig.add_trace(go.Histogram(x=df["gender"]), row=1, col=1)
fig.add_trace(go.Histogram(x=df["heart_disease"]), row=1, col=2)
fig.add_trace(go.Histogram(x=df["hypertension"]), row=1, col=3)
fig.update_layout(showlegend=False, yaxis_title='Count', title_text="Features Distribution", bargap = 0.05)
fig.show();

# plot ever_married, work_type, Residence_type features
fig = make_subplots(rows=1, cols=3, subplot_titles=['ever_married', 'work_type', 'Residence_type'])
fig.add_trace(go.Histogram(x=df["ever_married"]), row=1, col=1)
fig.add_trace(go.Histogram(x=df["work_type"]), row=1, col=2)
fig.add_trace(go.Histogram(x=df["Residence_type"]), row=1, col=3)
fig.update_layout(showlegend=False, yaxis_title='Count', title_text="Features Distribution", bargap = 0.05)
fig.show();

# plot age, avg_glucose_level features
fig = make_subplots(rows=1, cols=2, subplot_titles=['age', 'avg_glucose_level'])
fig.add_trace(go.Histogram(x=df["age"]), row=1, col=1)
fig.add_trace(go.Histogram(x=df["avg_glucose_level"]), row=1, col=2)
fig.update_layout(showlegend=False, yaxis_title='Count', title_text="Features Distribution", bargap = 0.05)
fig.show();

# plot bmi, smoking_status features
fig = make_subplots(rows=1, cols=2, subplot_titles=['bmi', 'smoking_status'])
fig.add_trace(go.Histogram(x=df["bmi"]), row=1, col=1)
fig.add_trace(go.Histogram(x=df["smoking_status"]), row=1, col=2)
fig.update_layout(showlegend=False, yaxis_title='Count', title_text="Features Distribution", bargap = 0.05)
fig.show();

# plot stroke historgam
fig = px.histogram(df, x="stroke")
fig.update_layout(autosize=True, width=700, height=500, bargap=0.05)
fig.show();

# plot feature histrograms by stroke

for column in df.columns:
    if column == 'stroke':
        continue

    fig = px.histogram(df, x=column, color='stroke')
    fig.update_layout(title=f'Histogram for {column} by stroke', xaxis_title=column, yaxis_title='Count')
    fig.show()

# plot box plot for numerical features avg_glucose_level, bmi

fig = make_subplots(rows=2, cols=1, subplot_titles=("Avg Glucose Level", "Body Mass Index" ))
fig.add_trace(go.Box(x=df["avg_glucose_level"]), row=1, col=1)
fig.add_trace(go.Box(x=df["bmi"]), row=2, col=1)
fig.update_layout(showlegend=False, title_text="Numerical BoxPlot", height=500)

# plot scatter for bmi and avg_glucose_level to see outleirs
fig = px.scatter(df[df["stroke"]==1], x="bmi", y="avg_glucose_level")
fig.show()

# method for removing outleirs
def remove_outliers(x):
    Q1 = np.percentile(x, 25)
    Q3 = np.percentile(x, 75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return lower, upper

lower_bmi , upper_bmi = remove_outliers(df["bmi"])
lower_glu , upper_glu = remove_outliers(df["avg_glucose_level"])

# remove outliers
df["bmi"] = df["bmi"].apply(lambda x: np.mean(df["bmi"]) if x > upper_bmi or x < lower_bmi else x)
df["avg_glucose_level"] = df["avg_glucose_level"].apply(lambda x: np.mean(df["avg_glucose_level"]) if x > upper_glu or x < lower_glu else x)

df

# plot box plot for numerical features avg_glucose_level, bmi after removing outleirs

fig = make_subplots(rows=2, cols=1, subplot_titles=("Avg Glucose Level" , "Body Mass Index"))
fig.add_trace(go.Box(x=df["avg_glucose_level"]), row=1, col=1)
fig.add_trace(go.Box(x=df["bmi"]), row=2, col=1)
fig.update_layout(showlegend=False, title_text="Numerical BoxPlot", height=500)

# plot scatter for bmi and avg_glucose_level after removing outleirs
fig = px.scatter(df[df["stroke"]==1], x="bmi", y="avg_glucose_level")
fig.show()

# change gender values to male=1 and female=0
gndr_col = pd.get_dummies(df["gender"])
gndr_col = gndr_col.drop(['Female'], axis=1)

# change ever_married values to Yes=1 and No=0
mrd_col = pd.get_dummies(df["ever_married"])
mrd_col = mrd_col.drop(['No'], axis=1)

# change residence_type values to urban=1 and rural=0
urbn_col = pd.get_dummies(df["Residence_type"])
urbn_col = urbn_col.drop(['Rural'], axis=1)

# build dataset
df = pd.concat((df, gndr_col, mrd_col, urbn_col), axis=1)
df = df.drop(['gender', 'ever_married', 'Residence_type'], axis=1)
df = df.rename(columns={'Male': 'gender', 'Yes': 'ever_married'})
df.head()

# One-Hot Encoding to convert work_type and smoking_status
df = pd.get_dummies(df, drop_first=True)
df.head()

# scaling data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
columns = ['age', 'avg_glucose_level', 'bmi']
df[columns] = scaler.fit_transform(df[columns])
df.head()

df.stroke.value_counts()

# use SMOTE for fixing imbalancing of data
from imblearn.over_sampling import SMOTE

x = df.drop('stroke', axis=1)
y = df['stroke'].astype('int')

oversampling = SMOTE(sampling_strategy=0.5)
x, y = oversampling.fit_resample(x, y)

x_df = pd.DataFrame(x)
y_df = pd.DataFrame(y, columns=['stroke'])
df = pd.concat([x_df, y_df], axis=1)
df

df.stroke.value_counts()

# plot corr map
plt.figure(figsize = (10, 10))
sns.heatmap(df.corr(), annot = True)
plt.show()

# get target(y) and features(x)
x = df.drop('stroke', axis=1)
y = df.stroke

# splitting data to train and test 80% to 20%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

# DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

# AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

# StackingClassifier with RandomForestClassifier, LinearSVC and LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

# GridSearchCV
from sklearn.model_selection import GridSearchCV

# metrics for model validation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# base DecisionTreeClassifier
decision_tree_classifier = DecisionTreeClassifier()
decision_tree_classifier.fit(x_train, y_train)
decision_tree_classifier_prediction = decision_tree_classifier.predict(x_test)

# print metrics and classification report
print('Accuracy score: ', (accuracy_score(y_test, decision_tree_classifier_prediction)))
print('Precision score: ', precision_score(y_test, decision_tree_classifier_prediction))
print('Recall: ', recall_score(y_test, decision_tree_classifier_prediction))
print('F1-Score: ', f1_score(y_test, decision_tree_classifier_prediction))
print('Classification report: \n', classification_report(decision_tree_classifier_prediction, y_test))

# show confusion_matrix
sns.heatmap(confusion_matrix(decision_tree_classifier_prediction, y_test), annot=True)
plt.show()

# GridSearch for DecisionTreeClassifier
param_grid_for_decision_tree_classifier = {
    'splitter': ['best', 'random'],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [10, 30, 50],
    'min_samples_split': [2, 4, 6, 8, 10]
}

# do grid search
grid_search = GridSearchCV(decision_tree_classifier, param_grid=param_grid_for_decision_tree_classifier, cv=10, scoring='recall')
grid_search.fit(x_train, y_train)

# print best hyperparameters for DecisionTreeClassifier
print(grid_search.best_params_)
print(grid_search.best_score_)

# build DecisionTreeClassifier with grid search best params
grid_decision_tree_classifier = DecisionTreeClassifier(criterion='log_loss', max_depth=10, min_samples_split=4, splitter='best')
grid_decision_tree_classifier.fit(x_train, y_train)
grid_decision_tree_classifier_prediction = decision_tree_classifier.predict(x_test)

# print metrics and classification report
print('Accuracy score: ', (accuracy_score(y_test, grid_decision_tree_classifier_prediction)))
print('Precision score: ', precision_score(y_test, grid_decision_tree_classifier_prediction))
print('Recall: ', recall_score(y_test, grid_decision_tree_classifier_prediction))
print('F1-Score: ', f1_score(y_test, grid_decision_tree_classifier_prediction))
print('Classification report: \n', classification_report(grid_decision_tree_classifier_prediction, y_test))

# show confusion_matrix
sns.heatmap(confusion_matrix(grid_decision_tree_classifier_prediction, y_test), annot=True)
plt.show()

# base RandomForestClassifier
random_forest_classifier = RandomForestClassifier()
random_forest_classifier.fit(x_train, y_train)
random_forest_classifier_prediction = random_forest_classifier.predict(x_test)

# print metrics and classification report
print('Accuracy score: ', (accuracy_score(y_test, random_forest_classifier_prediction)))
print('Precision score: ', precision_score(y_test, random_forest_classifier_prediction))
print('Recall: ', recall_score(y_test, random_forest_classifier_prediction))
print('F1-Score: ', f1_score(y_test, random_forest_classifier_prediction))
print('Classification report: \n', classification_report(random_forest_classifier_prediction, y_test))

# show confusion_matrix
sns.heatmap(confusion_matrix(random_forest_classifier_prediction, y_test), annot=True)
plt.show()

# GridSearch for RandomForestClassifier
param_grid_for_random_forest_classifier = {
    'n_estimators': [50, 150, 300, 500],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [10, 30, 50]
}

# do grid search
grid_search = GridSearchCV(random_forest_classifier, param_grid=param_grid_for_random_forest_classifier, cv=10, scoring='recall')
grid_search.fit(x_train, y_train)

# print best hyperparameters for RandomForestClassifier
print(grid_search.best_params_)
print(grid_search.best_score_)

# build RandomForestClassifier with grid search best params
grid_random_forest_classifier = RandomForestClassifier(criterion='entropy', max_depth=30, n_estimators=500)
grid_random_forest_classifier.fit(x_train, y_train)
grid_random_forest_classifier_prediction = grid_random_forest_classifier.predict(x_test)

# print metrics and classification report
print('Accuracy score: ', (accuracy_score(y_test, grid_random_forest_classifier_prediction)))
print('Precision score: ', precision_score(y_test, grid_random_forest_classifier_prediction))
print('Recall: ', recall_score(y_test, grid_random_forest_classifier_prediction))
print('F1-Score: ', f1_score(y_test, grid_random_forest_classifier_prediction))
print('Classification report: \n', classification_report(grid_random_forest_classifier_prediction, y_test))

# show confusion_matrix
sns.heatmap(confusion_matrix(grid_random_forest_classifier_prediction, y_test), annot=True)
plt.show()

# base KNeighborsClassifier
k_neighbors_classifier = KNeighborsClassifier()
k_neighbors_classifier.fit(x_train, y_train)
k_neighbors_classifier_prediction = k_neighbors_classifier.predict(x_test)

# print metrics and classification report
print('Accuracy score: ', (accuracy_score(y_test, k_neighbors_classifier_prediction)))
print('Precision score: ', precision_score(y_test, k_neighbors_classifier_prediction))
print('Recall: ', recall_score(y_test, k_neighbors_classifier_prediction))
print('F1-Score: ', f1_score(y_test, k_neighbors_classifier_prediction))
print('Classification report: \n', classification_report(k_neighbors_classifier_prediction, y_test))

# show confusion_matrix
sns.heatmap(confusion_matrix(k_neighbors_classifier_prediction, y_test), annot=True)
plt.show()

# GridSearch for KNeighborsClassifier
param_grid_for_k_neighbors_classifier = {
    'n_neighbors': [3, 5, 8, 10],
    'weights': ['uniform','distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'metric': ['minkowski', 'manhattan', 'euclidean']
}

# do grid search
grid_search = GridSearchCV(k_neighbors_classifier, param_grid=param_grid_for_k_neighbors_classifier, cv=10, scoring='recall')
grid_search.fit(x_train, y_train)

# print best hyperparameters for KNeighborsClassifier
print(grid_search.best_params_)
print(grid_search.best_score_)

# build KNeighborsClassifier with grid search best params
grid_k_neighbors_classifier = KNeighborsClassifier(algorithm='auto', metric='manhattan', n_neighbors=8, weights='distance')
grid_k_neighbors_classifier.fit(x_train, y_train)
grid_k_neighbors_classifier_prediction = grid_k_neighbors_classifier.predict(x_test)

# print metrics and classification report
print('Accuracy score: ', (accuracy_score(y_test, grid_k_neighbors_classifier_prediction)))
print('Precision score: ', precision_score(y_test, grid_k_neighbors_classifier_prediction))
print('Recall: ', recall_score(y_test, grid_k_neighbors_classifier_prediction))
print('F1-Score: ', f1_score(y_test, grid_k_neighbors_classifier_prediction))
print('Classification report: \n', classification_report(grid_k_neighbors_classifier_prediction, y_test))

# show confusion_matrix
sns.heatmap(confusion_matrix(grid_k_neighbors_classifier_prediction, y_test), annot=True)
plt.show()

# base AdaBoostClassifier
ada_boost_classifier = AdaBoostClassifier()
ada_boost_classifier.fit(x_train, y_train)
ada_boost_classifier_prediction = ada_boost_classifier.predict(x_test)

# print metrics and classification report
print('Accuracy score: ', (accuracy_score(y_test, ada_boost_classifier_prediction)))
print('Precision score: ', precision_score(y_test, ada_boost_classifier_prediction))
print('Recall: ', recall_score(y_test, ada_boost_classifier_prediction))
print('F1-Score: ', f1_score(y_test, ada_boost_classifier_prediction))
print('Classification report: \n', classification_report(ada_boost_classifier_prediction, y_test))

# show confusion_matrix
sns.heatmap(confusion_matrix(ada_boost_classifier_prediction, y_test), annot=True)
plt.show()

# GridSearch for AdaBoostClassifier
param_grid_for_ada_boost_classifier = {
    'base_estimator': [DecisionTreeClassifier(max_depth=1), RandomForestClassifier(max_depth=1)],
    'n_estimators': [50, 150, 300, 500],
    'learning_rate': [1.0, 0.1, 0.01]
}

# do grid search
grid_search = GridSearchCV(ada_boost_classifier, param_grid=param_grid_for_ada_boost_classifier, cv=10, scoring='recall')
grid_search.fit(x_train, y_train)

# print best hyperparameters for AdaBoostClassifier
print(grid_search.best_params_)
print(grid_search.best_score_)

# build AdaBoostClassifier with grid search best params
grid_ada_boost_classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), learning_rate=0.01, n_estimators=50)
grid_ada_boost_classifier.fit(x_train, y_train)
grid_ada_boost_classifier_prediction = grid_ada_boost_classifier.predict(x_test)

# print metrics and classification report
print('Accuracy score: ', (accuracy_score(y_test, grid_ada_boost_classifier_prediction)))
print('Precision score: ', precision_score(y_test, grid_ada_boost_classifier_prediction))
print('Recall: ', recall_score(y_test, grid_ada_boost_classifier_prediction))
print('F1-Score: ', f1_score(y_test, grid_ada_boost_classifier_prediction))
print('Classification report: \n', classification_report(grid_ada_boost_classifier_prediction, y_test))

# show confusion_matrix
sns.heatmap(confusion_matrix(grid_ada_boost_classifier_prediction, y_test), annot=True)
plt.show()

# base StackingClassifier with RandomForestClassifier and LinearSVC
estimators = [
     ('rf', RandomForestClassifier(random_state=42)),
     ('svr', LinearSVC(random_state=42))
]
stacking_classifier = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacking_classifier.fit(x_train, y_train)
stacking_classifier_prediction = stacking_classifier.predict(x_test)

# print metrics and classification report
print('Accuracy score: ', (accuracy_score(y_test, stacking_classifier_prediction)))
print('Precision score: ', precision_score(y_test, stacking_classifier_prediction))
print('Recall: ', recall_score(y_test, stacking_classifier_prediction))
print('F1-Score: ', f1_score(y_test, stacking_classifier_prediction))
print('Classification report: \n', classification_report(stacking_classifier_prediction, y_test))

# show confusion_matrix
sns.heatmap(confusion_matrix(stacking_classifier_prediction, y_test), annot=True)
plt.show()

# GridSearch for StackingClassifier
param_grid_for_stacking_classifier = {
    'rf__n_estimators': [50, 150, 300, 500],
    'rf__criterion': ['gini', 'entropy', 'log_loss'],
    'rf__max_depth': [10, 30, 50]
}

# do grid search
grid_search = GridSearchCV(stacking_classifier, param_grid=param_grid_for_stacking_classifier, cv=10, scoring='recall')
grid_search.fit(x_train, y_train)

# print best hyperparameters for StackingClassifier
print(grid_search.best_params_)
print(grid_search.best_score_)

# build StackingClassifier with grid search best params
estimators = [
     ('rf', RandomForestClassifier(random_state=42, n_estimators=300, criterion='entropy', max_depth=30)),
     ('svr', LinearSVC(random_state=42))
]

grid_stacking_classifier = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
grid_stacking_classifier.fit(x_train, y_train)
grid_stacking_classifier_prediction = grid_stacking_classifier.predict(x_test)

# print metrics and classification report
print('Accuracy score: ', (accuracy_score(y_test, grid_stacking_classifier_prediction)))
print('Precision score: ', precision_score(y_test, grid_stacking_classifier_prediction))
print('Recall: ', recall_score(y_test, grid_stacking_classifier_prediction))
print('F1-Score: ', f1_score(y_test, grid_stacking_classifier_prediction))
print('Classification report: \n', classification_report(grid_stacking_classifier_prediction, y_test))

# show confusion_matrix
sns.heatmap(confusion_matrix(grid_stacking_classifier_prediction, y_test), annot=True)
plt.show()

# check if not fixing imbalancing of data

df = pd.read_csv("data.csv")

# remove outliers
df["bmi"] = df["bmi"].apply(lambda x: np.mean(df["bmi"]) if x > upper_bmi or x < lower_bmi else x)
df["avg_glucose_level"] = df["avg_glucose_level"].apply(lambda x: np.mean(df["avg_glucose_level"]) if x > upper_glu or x < lower_glu else x)


# change gender values to male=1 and female=0
gndr_col = pd.get_dummies(df["gender"])
gndr_col = gndr_col.drop(['Female'], axis=1)

# change ever_married values to Yes=1 and No=1
mrd_col = pd.get_dummies(df["ever_married"])
mrd_col = mrd_col.drop(['No'], axis=1)

# change residence_type values to urban=1 and rural=0
urbn_col = pd.get_dummies(df["Residence_type"])
urbn_col = urbn_col.drop(['Rural'], axis=1)

# build dataset
df = pd.concat((df, gndr_col, mrd_col, urbn_col), axis=1)
df = df.drop(['gender', 'ever_married', 'Residence_type'], axis=1)
df = df.rename(columns={'Male': 'gender', 'Yes': 'ever_married'})
df.head()

# One-Hot Encoding to convert work_type and smoking_status
df = pd.get_dummies(df, drop_first=True)
df.head()

# scaling data
scaler = MinMaxScaler()
columns = ['age', 'avg_glucose_level', 'bmi']
df[columns] = scaler.fit_transform(df[columns])
df.head()

# get target(y) and features(x)
x = df.drop('stroke', axis=1)
y = df.stroke

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# build RandomForestClassifier
random_forest_classifier = RandomForestClassifier()
random_forest_classifier.fit(x_train, y_train)
random_forest_classifier_prediction = random_forest_classifier.predict(x_test)

# print metrics and classification report
print('Accuracy score: ', (accuracy_score(y_test, random_forest_classifier_prediction)))
print('Precision score: ', precision_score(y_test, random_forest_classifier_prediction))
print('Recall: ', recall_score(y_test, random_forest_classifier_prediction))
print('F1-Score: ', f1_score(y_test, random_forest_classifier_prediction))
print('Classification report: \n', classification_report(random_forest_classifier_prediction, y_test))

# show confusion_matrix
sns.heatmap(confusion_matrix(random_forest_classifier_prediction, y_test), annot=True)
plt.show()