#!/usr/bin/env python
# coding: utf-8

# EE 559 Project
# 
# 
# Naman Rajendra Joshi
# 
# 
# Sanskar Tewatia

# In[138]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter
import random
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
import pickle
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA


# In[123]:


train_data = pd.read_csv("credit_card_dataset_train.csv")
test_data = pd.read_csv("credit_card_dataset_test.csv")


# In[3]:


train_data


# In[327]:


test_data


# Exploratory Data Analysis

# Objectives
# 
# 1) To understand the dataset 
# 2) To understand which variables are useful for classification 
# 3) To draw different types of plots to understand the dataset.

# In[4]:


print(train_data.isnull().sum())  # To check if there are any null values


# There are no null values

# In[6]:


print(train_data.shape)  #shape


# In[7]:


print(train_data.columns)  #columns


# In[8]:


train_data['default payment next month'].value_counts()


# Observations

# We can clearly see that the dataset is clearly imbalanced with class 0(people not defaulting) is the majority class

# In[7]:


train_data[['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
       'default payment next month']]


# In[9]:


import warnings    # to ignore the warning messages
warnings.filterwarnings('ignore')


# We are ploting a pair plot so as to see any relation between variables

# In[9]:


plt.close()
sns.set_style('whitegrid')
z = sns.pairplot(train_data[['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                             'default payment next month']], hue='default payment next month', height=6)
z.fig.suptitle('Pair plot of all features', fontsize=25)
plt.show()


# Lets check out more important plots from the above pair plots

# In[24]:


sns.set_style('darkgrid')
sns.FacetGrid(train_data, hue='default payment next month', height=20, aspect = 1) \
    .map(plt.scatter,'LIMIT_BAL','AGE')

plt.ylabel('AGE')
plt.xlabel('LIMIT_BAL')
plt.legend()
plt.title('2D-Scatter plot of AGE v/s LIMIT_BAL')
plt.show()


# Observation
# 
# From the above plot we can see that people having higher amount of credit are defaulting less.

# In[31]:


sns.displot(train_data['AGE'])
plt.title('distribution of age ')
plt.show()


# We can see that most people are in late 20's and 30's

# In[15]:


# get the number of target values where col1 has the value 'A'
num_targets = train_data.loc[train_data['SEX'] == 1, 'default payment next month'].value_counts()
num_targets


# Ratio of 1/0 = 2571/8097 = 0.3175

# In[16]:


# get the number of target values where col1 has the value 'A'
num_targets = train_data.loc[train_data['SEX'] == 2, 'default payment next month'].value_counts()
num_targets


# Ratio of 1/0 = 3401/12931 = 0.263

# Observations
# 
# 1) Firstly we can see that for the feature Education, the values (0,5 and 6) have not much relation with 1,2 and 3. Hence we need to convert these values others category(values).
# 
# 2) Then we can see from the ratios that females have a less percentage of defaults as compared to males

# Replacing 0 of marriage with 3(others)

# In[124]:


train_data['MARRIAGE']=train_data['MARRIAGE'].replace({0:3})
test_data['MARRIAGE']=test_data['MARRIAGE'].replace({0:3})


# In[27]:


train_data['MARRIAGE'].value_counts()


# In[125]:


train_data['EDUCATION']=train_data['EDUCATION'].replace({0:4,5:4,6:4})
test_data['EDUCATION']=test_data['EDUCATION'].replace({0:4,5:4,6:4})


# OUTLIERS REMOVAL IN NUMERICAL DATA

# In[57]:


sns.boxplot(x='default payment next month', y='LIMIT_BAL', data=train_data)
plt.ylabel('LIMIT_BAL')
plt.xlabel('default payment next month')
plt.title("Box plot for default payment next month v/s LIMIT_BAL")
plt.show()


# In[59]:


sns.boxplot(train_data['LIMIT_BAL'])
plt.title('Limit Balance before Outlier Treatment')
plt.show()


# In[126]:


def cap_outliers(df, column,test_data):

    q1, q3 = np.percentile(train_data[column], [25, 75])
    iqr = q3 - q1
    lower = (q1 - (1.5*iqr))
    upper = (q3 + (1.5*iqr))
    print("lower bound",lower)
    print("upper bound",upper)
    # Cap the outliers
    df[column] = np.where(df[column] < lower, lower, df[column])
    df[column] = np.where(df[column] > upper, upper, df[column])
    test_data[column] = np.where(test_data[column] < lower, lower, test_data[column])
    test_data[column] = np.where(test_data[column] > upper, upper, test_data[column])


    return df,test_data


# In[127]:


train_data,test_data=cap_outliers(train_data,'LIMIT_BAL',test_data)


# In[245]:


sns.boxplot(train_data['LIMIT_BAL'],color = 'purple')
plt.title('Limit Balance After Outlier Treatment')
plt.show()


# In[78]:


corr_matrix = train_data.corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corr_matrix, annot=True)
plt.show()


# From the above correlation matrix, we can clearly see that PAY_0 to PAY_6, the repayment status of months which are nearby have much higher correlation than others, similarly for amount of bill statement.

# In[246]:


sns.boxplot(train_data['BILL_AMT3'], color='purple')
plt.title('Box Plot of Bill Amount 3')
plt.show()


# We can see outliers here. Lets cap the outliers

# In[128]:


for i in range(1,7):
    feature='BILL_AMT'+str(i)
    print(feature)
    train_data,test_data=cap_outliers(train_data,feature,test_data)


# In[248]:


sns.boxplot(train_data['BILL_AMT3'], color='purple')
plt.title('Box Plot of Bill Amount 3 after caping')
plt.show()


# Similarly

# In[249]:


sns.boxplot(train_data['PAY_AMT6'], color='purple')
plt.title('Box Plot of Bill Amount 6 before caping')
plt.show()


# In[ ]:





# In[129]:


for i in range(1,7):
    feature='PAY_AMT'+str(i)
    print(feature)
    train_data,test_data=cap_outliers(train_data,feature,test_data)


# In[252]:


sns.boxplot(train_data['PAY_AMT6'], color='purple')
plt.title('Box Plot of Bill Amount 6 AFTER caping')
plt.show()


# Feature Standardization

# In[253]:


#Perform Data Analysis
print(train_data.hist(bins=30,figsize=(20,20), color='r'))
plt.show()


# We can clearly see that the distribution of the data follows power law, which means that general standardization techniques won't be helpful because they assume the data to be gaussian. 
# 
# We can apply Median Absolute Deviation(MAD) Scaling which is robust to outliers

# In[130]:


def mad_scale(data,feature,test_data):
    median = np.median(data[feature])
    mad = np.median(np.abs(data[feature]- median))
    scaled_test_data = (test_data[feature] - median) / mad
    return (data[feature] - median) / mad,scaled_test_data


# In[262]:


test_data['PAY_AMT1']


# In[131]:


for i in range(1,7):
    feature='PAY_AMT'+str(i)
    print(feature)
    train_data[feature],test_data[feature]=mad_scale(train_data,feature,test_data)


# In[265]:


test_data['PAY_AMT1']


# In[132]:


for i in range(1,7):
    feature='BILL_AMT'+str(i)
    print(feature)
    train_data[feature],test_data[feature]=mad_scale(train_data,feature,test_data)


# In[133]:


train_data['AGE'],test_data['AGE']=mad_scale(train_data,'AGE',test_data)


# In[134]:


train_data['LIMIT_BAL'],test_data['LIMIT_BAL']=mad_scale(train_data,'LIMIT_BAL',test_data)


# In[17]:


#Perform Data Analysis
print(train_data.hist(bins=30, figsize=(20, 20), color='r'))
plt.show()


# Correcting class imbalance by using SMOTE

# In[136]:


X_train = train_data.drop('default payment next month', axis=1)
Y_train = train_data['default payment next month']
X_test = test_data.drop('default payment next month', axis=1)
Y_test = test_data['default payment next month']


# In[139]:


print("Before applying SMOTE: ",Counter(Y_train))
SMOTE= SMOTE()


X_train_res,Y_train_res= SMOTE.fit_resample(X_train,Y_train)

# summarize class distribution
print("After applying SMOTE: ",Counter(Y_train_res))


# Creating four more different sets of data for experimentation

# 1) Keeping first three months

# In[19]:


features_to_drop = ['BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6','PAY_4','PAY_5','PAY_6']
X_train_first3 = X_train_res.drop(features_to_drop, axis=1)
X_test_first3 = X_test.drop(features_to_drop, axis=1)


# 2) Keeping last three months

# In[20]:


features_to_drop = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
                    'PAY_0', 'PAY_2', 'PAY_3']
X_train_last3 = X_train_res.drop(features_to_drop, axis=1)
X_test_last3 = X_test.drop(features_to_drop, axis=1)


# 3) Keeping first, middle and last month

# In[21]:


features_to_drop = ['BILL_AMT2', 'BILL_AMT4', 'BILL_AMT6',
                    'PAY_2', 'PAY_4', 'PAY_6']
X_train_middle = X_train_res.drop(features_to_drop, axis=1)
X_test_middle = X_test.drop(features_to_drop, axis=1)


# 4) Keeping first and last month

# In[22]:


features_to_drop = ['BILL_AMT4','BILL_AMT3','BILL_AMT2', 'BILL_AMT5','PAY_2','PAY_3','PAY_4','PAY_5']
X_train_fl = X_train_res.drop(features_to_drop, axis=1)
X_test_fl = X_test.drop(features_to_drop, axis=1)


# Keeping PAY_I and removing all bill amount except first

# In[23]:


features_to_drop = ['BILL_AMT4','BILL_AMT3','BILL_AMT2', 'BILL_AMT5','BILL_AMT6']
X_train_f = X_train_res.drop(features_to_drop, axis=1)
X_test_f = X_test.drop(features_to_drop, axis=1)


# Trivial System

# In[278]:


N0 = Y_train.value_counts()
N0


# Without SMOTE

# In[299]:


N0 = Y_train.value_counts()[0]
N1 = Y_train.value_counts()[1]
N = N0 + N1

# output random class assignments with probabilities N0/N and N1/N
class_assignments = []
for i in range(N):
    preds = random.choices([0,1], weights=[N0/N, N1/N])[0]
    class_assignments.append(preds)
class_assignments_test = []
for i in range(len(X_test)):
    preds_test = random.choices([0, 1], weights=[N0/N, N1/N])[0]
    class_assignments_test.append(preds_test)  


# In[31]:


def performance_metrics(Y_train, class_assignments):
    # compute accuracy
    accuracy = accuracy_score(Y_train, class_assignments)
    print("Accuracy:", accuracy)

    # compute f1-score
    f1 = f1_score(Y_train, class_assignments)
    print("F1-score:", f1)

    # compute confusion matrix
    cm = confusion_matrix(Y_train, class_assignments)
    cm_display = ConfusionMatrixDisplay(cm).plot()
    print("Confusion matrix:\n", cm_display)


# In[301]:


performance_metrics(Y_train, class_assignments)


# In[302]:


performance_metrics(Y_test, class_assignments_test)


# With SMOTE

# In[303]:


N0 = Y_train_res.value_counts()[0]
N1 = Y_train_res.value_counts()[1]
N = N0 + N1

# output random class assignments with probabilities N0/N and N1/N
class_assignments = []
for i in range(N):
    preds = random.choices([0,1], weights=[N0/N, N1/N])[0]
    class_assignments.append(preds)
class_assignments_test = []
for i in range(len(X_test)):
    preds_test = random.choices([0, 1], weights=[N0/N, N1/N])[0]
    class_assignments_test.append(preds_test)


# In[304]:


performance_metrics(Y_train_res, class_assignments)


# In[305]:


performance_metrics(Y_test, class_assignments_test)


# Baseline System
# 
# Nearest Means Classifier

# Without SMOTE

# In[121]:


clf = NearestCentroid()

# fit the model using the training data
clf.fit(X_train, Y_train)

# predict class labels for test data
y_pred = clf.predict(X_train)


# In[307]:


performance_metrics(Y_train, y_pred)


# In[308]:


y_pred_test = clf.predict(X_test)
performance_metrics(Y_test, y_pred_test)


# With SMOTE

# In[143]:


clf = NearestCentroid()

# fit the model using the training data
clf.fit(X_train_res, Y_train_res)

# predict class labels for test data
y_pred = clf.predict(X_train_res)


# Below are the baseline values

# In[34]:


performance_metrics(Y_train_res, y_pred)


# In[144]:


y_pred_test = clf.predict(X_test)
performance_metrics(Y_test, y_pred_test)


# Both F1 Score and Accuracy improved

# Lets check if hyperparameter tuning improves the result

# In[311]:


def tuning_NMS(X_train, Y_train):
    param_grid = {
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'shrink_threshold': [None, 0.1, 0.5, 1.0]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # create a NearestCentroid classifier object
    clf = NearestCentroid()

    # create a GridSearchCV object to perform hyperparameter tuning
    search_grid = GridSearchCV(
        clf,
        param_grid=param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1
    )


    # perform hyperparameter tuning using GridSearchCV
    search_grid.fit(X_train, Y_train)

    best_params = search_grid.best_params_
    best_score = search_grid.best_score_

    print(f"Best hyperparameters are: {best_params}")
    print(f"Best F1-score is : {best_score}")



# In[312]:


tuning_NMS(X_train, Y_train)


# In[314]:


tuning_NMS(X_train_res, Y_train_res)


# We can see no improvement here

# KNN

# Without SMOTE

# In[315]:


def tuning_knn(X_train, Y_train):
    knn = KNeighborsClassifier()
    # Define the parameter grid to search over
    list_of_paramters = {
        'n_neighbors': [3, 4,5,6, 7, 9],
        'weights': ['uniform', 'distance'],
        'p': [1, 2],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search_grid = GridSearchCV(
        knn,
        param_grid=list_of_paramters,
        cv=cv,
        scoring='f1',
        n_jobs=-1
    )


    # Perform a grid search to find the best hyperparameters
    search_grid.fit(X_train, Y_train)

    # Print the best hyperparameters and the corresponding score
    print("Best hyperparameters are: ", search_grid.best_params_)
    print("Best F1-score is: {:.2f}".format(search_grid.best_score_))


# In[316]:


tuning_knn(X_train, Y_train)


# In[317]:


def knn(n_neighbors, p, weights,X_train, Y_train,X_test,Y_test):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, p=p, weights=weights)
    clf.fit(X_train, Y_train)
    print("Train Data")
    # make predictions on the test data
    y_pred = clf.predict(X_train)
    performance_metrics(Y_train, y_pred)
    print("test data")
    y_pred_test = clf.predict(X_test)
    performance_metrics(Y_test, y_pred_test)



# Without SMOTE

# In[318]:


knn(5,2,'distance',X_train, Y_train,X_test,Y_test)


# We can clearly see overfit here

# Using SMOTE

# In[319]:


tuning_knn(X_train_res, Y_train_res)


# But after trying many compbinations of neighbors, we got 1000 neighbors as optimal

# In[321]:


knn(1000,1,'distance',X_train_res, Y_train_res,X_test,Y_test)


# We can see that this model is performing even worse when using SMOTE

# Support Vector Machines

# Without SMOTE

# In[36]:


def svm_(X_train, Y_train,X_test, Y_test):
    # initialize the SVM classifier
    clf = SVC(kernel='rbf')

    # train the model on the training data
    clf.fit(X_train, Y_train)

    print("Train Data")
    # make predictions on the test data
    y_pred = clf.predict(X_train)
    performance_metrics(Y_train, y_pred)
    print("test data")
    y_pred_test = clf.predict(X_test)
    performance_metrics(Y_test, y_pred_test)
    return clf
    


# In[323]:


svm_(X_train, Y_train,X_test, Y_test)


# With  SMOTE

# In[333]:


def svm_tune(X_train_res, Y_train_res):
    # Assuming X_train and y_train are the original train data and labels

    # Train a SVM model on the full feature set
    pca = PCA(n_components=5)
    pca.fit(X_train_res)
    top5_features = pca.transform(X_train_res)
    print("1 done")
    list_of_paramters = {
        'C': [0.01, 0.1, 1]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    svm_cv = SVC(kernel='rbf')
    search_grid = GridSearchCV(
        svm_cv,
        param_grid=list_of_paramters,
        cv=cv,
        scoring='f1',
        n_jobs=-1
    )
    print("2 done")
    search_grid.fit(top5_features, Y_train_res)

    # Print the best hyperparameters and the corresponding score
    print("Best hyperparameters are: ", search_grid.best_params_)
    print("Best F1-score is: {:.2f}".format(search_grid.best_score_))


# In[334]:


svm_tune(X_train_res, Y_train_res)


# In[37]:


clf = svm_(X_train_res, Y_train_res,X_test, Y_test)


# In[44]:


print("The number of support vectors are: ",len(clf.support_vectors_))
classes = 2
print("The degrees of freedom are: ", len(clf.support_vectors_)-classes - 1)


# After applying SMOTE, we can clearly see that overfitting has reduced by a big margin but we can see that accuracy has reduced

# Neural Networks - MLP

# In[51]:


def mlp(X_tr, y_tr,X_te,Y_te):
    # create an MLP object
    mlp = MLPClassifier(hidden_layer_sizes=(20,20,20), activation='relu', solver='adam', max_iter=300,alpha=0.001)

    # train the MLP on the training data
    mlp.fit(X_tr, y_tr)
    print("Train Data")
    # make predictions on the test data
    y_pred = mlp.predict(X_tr)
    #print(y_pred.shape)
    performance_metrics(y_tr, y_pred)
    print("test data")
    y_pred_test = mlp.predict(X_te)
    performance_metrics(Y_te, y_pred_test)
    # make predictions on the test data
    return mlp


# Without SMOTE

# In[326]:


mlp(X_train, Y_train,X_test,Y_test)


# With SMOTE

# In[335]:


def mlp_tune(X_train_res, Y_train_res):
    pca = PCA(n_components=5)
    pca.fit(X_train_res)
    top5_features = pca.transform(X_train_res)
    print("1 done")
    list_of_paramters = {
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'invscaling', 'adaptive']
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mlp_cv = MLPClassifier(hidden_layer_sizes=(
        10,), activation='relu', solver='adam', random_state=42)
    search_grid = GridSearchCV(
        mlp_cv,
        param_grid=list_of_paramters,
        cv=cv,
        scoring='f1',
        n_jobs=-1
    )
    print("2 done")
    search_grid.fit(top5_features, Y_train_res)

    # Print the best hyperparameters and the corresponding score
    print("Best hyperparameters are: ", search_grid.best_params_)
    print("Best F1-score is: {:.2f}".format(search_grid.best_score_))


# In[336]:


mlp_tune(X_train_res, Y_train_res)


# In[53]:


mlp1 = mlp(X_train_res, Y_train_res,X_test,Y_test)


# In[72]:


WEIGHT_N = sum((layer_size + 1) * next_layer_size for layer_size,
                next_layer_size in zip(mlp1.hidden_layer_sizes, mlp1.hidden_layer_sizes[1:]))
WEIGHT_N += (mlp1.n_outputs_ * (mlp1.hidden_layer_sizes[-1] + 1))
print("Total number of weights in the MLP model: ",WEIGHT_N)


# In[73]:


BIAS_T = sum([b.size for b in mlp1.intercepts_])
print("Total bias values: ", BIAS_T)


# In[74]:


total_dof = WEIGHT_N + BIAS_T
print("DOF : ",total_dof)


# In[34]:


print(X_train_res.shape)
print(Y_train_res.shape)


# Random Forest Classifier

# In[99]:


def tune_random(X_tr, y_tr):

    list_of_paramters = {
        'n_estimators': [50, 100, 200,300],
        'max_depth': [5, 10, 20, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Create a random forest classifier object
    clf = RandomForestClassifier(random_state=42, class_weight='balanced')

    search_grid_n = GridSearchCV(
        clf,
        param_grid=list_of_paramters,
        cv=cv,
        scoring='f1',
        n_jobs=-1
    )

    search_grid_n.fit(X_tr, y_tr)

    best_p = search_grid_n.best_params_
    b_score = search_grid_n.best_score_

    print(f"Best hyperparameters are: {best_p}")
    print(f"Best F1-score is: {b_score}")



# In[118]:


def random_f(X_tr, y_tr, X_te, Y_te, max_depth, max_features, min_samples_leaf, min_samples_split, n_estimators):
    rf = RandomForestClassifier(max_depth= max_depth, max_features= max_features, min_samples_leaf= min_samples_leaf, min_samples_split= min_samples_split, n_estimators= n_estimators)

    # Train the classifier on the training data
    rf.fit(X_tr, y_tr)
    print("Train Data")
    # make predictions on the test data
    with open('model.pkl', 'wb') as file:
        pickle.dump(rf, file)
    y_pred = rf.predict(X_tr)
    #print(y_pred.shape)
    performance_metrics(y_tr, y_pred)
    print("test data")
    y_pred_test = rf.predict(X_te)
    performance_metrics(Y_te, y_pred_test)
    # make predictions on the test data


# In[62]:


tune_random(X_train, Y_train)


# In[112]:


random_f(X_train, Y_train,X_test,Y_test,10,'sqrt',2,5,150)


# We can clearly see here that False positives are more than True Negative which is causing low F1 score

# In[74]:


tune_random(X_train_res, Y_train_res)


# But these parameters are causing over fitting

# In[280]:


random_f(X_train_res, Y_train_res,X_test,Y_test,10,'sqrt',1,2,200)


# This is performing the best 

# In[4]:


with open('model.pkl', 'rb') as file:
    rf = pickle.load(file)


# Lets plot feature importance using the best model

# In[28]:


imp = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
ind = np.argsort(imp)[::-1]
plt.figure()
plt.title("Feature importance")
plt.bar(range(X_train_res.shape[1]), imp[ind],
        yerr=std[ind], align="center", color='maroon')
plt.xticks(range(X_train_res.shape[1]),
           X_train_res.columns[ind], rotation='vertical')
plt.show()


# In[ ]:


with open('model.pkl', 'wb') as file:
    pickle.dump(rf, file)


# We can see that the accuracy has reduced but F1 SCORE has increased

# In[107]:


random_f(X_train_first3, Y_train_res,X_test_first3,Y_test,10,'sqrt',1,2,300)


# In[108]:


random_f(X_train_last3, Y_train_res,X_test_last3,Y_test,10,'sqrt',1,2,300)


# In[109]:


random_f(X_train_fl, Y_train_res,X_test_fl,Y_test,10,'sqrt',1,2,300)


# In[156]:


tune_random(X_train_f, Y_train_res,X_test_f,Y_test)


# In[110]:


random_f(X_train_f, Y_train_res,X_test_f,Y_test,10,'sqrt',1,2,300)


# In this case, we can see that the accuracies and f1 scores are almost the same as the dataset having all the features. It means that removing these specific multicollinear features would be beneficial in making the training time less without lossing on the accuracy. Same goes for the case when we used the first3 dataset

# We can see that removing different combinations of months from the data does not improve the performance metrics

# Logistic Regression

# In[75]:


def tune_logistic(X_tr, y_tr,X_te,Y_te):

    list_of_paramters = {
        'penalty': ['l2','l1'],
        'C': [0.01, 0.1, 1.0,10],
        'solver': ['saga', 'liblinear']
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Create a random forest classifier object
    clf = LogisticRegression()
    search_grid_n = GridSearchCV(
        clf,
        param_grid=list_of_paramters,
        cv=cv,
        scoring='f1',
        n_jobs=-1
    )

    search_grid_n.fit(X_tr, y_tr)

    best_p = search_grid_n.best_params_
    b_score = search_grid_n.best_score_
    print(f"Best hyperparameters are: {best_p}")
    print(f"Best F1-score is: {b_score}")
    clf = LogisticRegression(**best_p)
    clf.fit(X_tr, y_tr)
    print("Train Data")
    # make predictions on the test data
    y_pred = clf.predict(X_tr)
    #print(y_pred.shape)
    performance_metrics(y_tr, y_pred)
    print("test data")
    y_pred_test = clf.predict(X_te)
    performance_metrics(np.array(Y_te), y_pred_test)
    return clf


# Without SMOTE

# In[113]:


tune_logistic(X_train, Y_train,X_test,Y_test)


# With SMOTE

# In[81]:


clf_l = tune_logistic(X_train_res, Y_train_res,X_test,Y_test)


# In[79]:


n_coef = len(clf_l.coef_[0]) + 1

df = X_train_res.shape[0] - n_coef
print("DOF",df)


# In[82]:


tune_logistic(X_train_first3, Y_train_res,X_test_first3,Y_test)


# In[83]:


tune_logistic(X_train_last3, Y_train_res,X_test_last3,Y_test)


# In[84]:


tune_logistic(X_train_fl, Y_train_res,X_test_fl,Y_test)


# We can see that Accuracy has actually reduced after using SMOTE 

# Now lets try encoding some of the categorical variables using one hot encoding and see if there is any improvement in any results

# In[94]:


X_train_new = X_train_res


# In[105]:


X_test_new = X_test


# In[106]:


X_test_new


# In[107]:


X_train_new.replace({'SEX': {1: 'MALE', 2: 'FEMALE'}, 'EDUCATION': {1: 'graduate school', 2: 'university',
              3: 'high school', 4: 'others'}, 'MARRIAGE': {1: 'married', 2: 'single', 3: 'others'}}, inplace=True)
X_test_new.replace({'SEX': {1: 'MALE', 2: 'FEMALE'}, 'EDUCATION': {1: 'graduate school', 2: 'university',
              3: 'high school', 4: 'others'}, 'MARRIAGE': {1: 'married', 2: 'single', 3: 'others'}}, inplace=True)


# In[108]:


X_test_new


# In[97]:


X_train_new.head()


# In[109]:


encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(X_train_new[['PAY_0', 'PAY_2', 'PAY_3',
                     'PAY_4', 'PAY_5', 'PAY_6', 'EDUCATION', 'MARRIAGE']])

X_train_encoded = pd.concat([X_train_new.drop(['PAY_0', 'PAY_2', 'PAY_3',
                                           'PAY_4', 'PAY_5', 'PAY_6', 'EDUCATION', 'MARRIAGE'], axis=1),
                             pd.DataFrame(encoder.transform(X_train_new[['PAY_0', 'PAY_2', 'PAY_3',
                                                                     'PAY_4', 'PAY_5', 'PAY_6', 'EDUCATION', 'MARRIAGE']]).toarray(),
                                          columns=encoder.get_feature_names_out(['PAY_0', 'PAY_2', 'PAY_3',
                                                                             'PAY_4', 'PAY_5', 'PAY_6', 'EDUCATION', 'MARRIAGE']))],
                            axis=1)

X_test_encoded = pd.concat([X_test_new.drop(['PAY_0', 'PAY_2', 'PAY_3',
                                         'PAY_4', 'PAY_5', 'PAY_6', 'EDUCATION', 'MARRIAGE'], axis=1),
                            pd.DataFrame(encoder.transform(X_test_new[['PAY_0', 'PAY_2', 'PAY_3',
                                                                   'PAY_4', 'PAY_5', 'PAY_6', 'EDUCATION', 'MARRIAGE']]).toarray(),
                                         columns=encoder.get_feature_names_out(['PAY_0', 'PAY_2', 'PAY_3',
                                                                            'PAY_4', 'PAY_5', 'PAY_6', 'EDUCATION', 'MARRIAGE']))],
                           axis=1)


# In[110]:


X_test_encoded


# In[206]:


X_train_new.head()


# In[111]:


# LABEL ENCODING FOR SEX
encoders_nums = {
    "SEX": {"FEMALE": 0, "MALE": 1}
}
X_train_encoded= X_train_encoded.replace(encoders_nums)
X_test_encoded = X_test_encoded.replace(encoders_nums)


# In[112]:


scaler = StandardScaler()
X_train_new1 = scaler.fit_transform(X_train_encoded)


# In[113]:


X_test_new1 = scaler.transform(X_test_encoded)


# In[114]:


tune_logistic(X_train_new1, Y_train_res,X_test_new1,Y_test)


# In[116]:


X_train_new1


# In[119]:


random_f(X_train_new1, Y_train_res,X_test_new1,Y_test,10,'sqrt',1,2,300)


# We can clearly see that there is no improvement when using one hot encoding

# In[120]:


clf = svm_(X_train_new1, Y_train_res,X_test_new1,Y_test)


# In[145]:


mlp1 = mlp(X_train_new1, Y_train_res,X_test_new1,Y_test)


# References
# 
# [1]https://www.kaggle.com/code/bansodesandeep/credit-card-default-prediction#One-Hot-Encoding.
# 
# 
# [2] https://github.com/SanikaDharwadker/CreditCardDefault.
# 
