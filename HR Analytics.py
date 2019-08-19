
# coding: utf-8

# ### Problem Statement
# Your client is a large MNC and they have 9 broad verticals across the organisation. One of the problem your client is facing is around identifying the right people for promotion (only for manager position and below) and prepare them in time. Currently the process, they are following is:
# 1.	They first identify a set of employees based on recommendations/ past performance
# 2.	Selected employees go through the separate training and evaluation program for each vertical. These programs are based on the required skill of each vertical
# 3.	At the end of the program, based on various factors such as training performance, KPI completion (only employees with KPIs completed greater than 60% are considered) etc., employee gets promotion
# 
# For above mentioned process, the final promotions are only announced after the evaluation and this leads to delay in transition to their new roles. Hence, company needs your help in identifying the eligible candidates at a particular checkpoint so that they can expedite the entire promotion cycle. 
# 
# They have provided multiple attributes around Employee's past and current performance along with demographics. Now, The task is to predict whether a potential promotee at checkpoint in the test set will be promoted or not after the evaluation process.
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve


# In[2]:


df_train = pd.read_csv('train_LZdllcl.csv', index_col = 0)
df_train.head()


# In[3]:


# looking at info
df_train.info()


# In[4]:


df_train.describe()


# In[5]:


df_train.columns


# In[6]:


#cbar removes the legend colormap (default : True)
sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='viridis') 


# In[7]:


df_train.groupby('department')['education'].value_counts()


# In[8]:


df_train['education'].value_counts()


# In[9]:


df_train['previous_year_rating'].fillna(value = 0, inplace = True)


# In[10]:


df_train['education'].fillna(value = "New", inplace = True)


# In[11]:


df_train.groupby('education').size()


# In[12]:


df_train[df_train['education'] == "Master's & above"]['age'].mean()


# In[13]:


df_train[df_train['education'] == "Bachelor's"]['age'].mean()


# In[14]:


df_train[df_train['education'] == "Below Secondary"]['age'].mean()


# In[15]:


df_train[df_train['education'] == "New"]['age'].mean()


# In[16]:


df_train[df_train['education'] == "New"]['department'].value_counts()


# New is changes to Bachelor's as it has the maximum occurence

# In[17]:


df_train['education'] = df_train['education'].map({"New" : "Bachelor's", 
                                       "Bachelor's" : "Bachelor's", 
                                       "Master's & above" : "Master's & above",
                                       "Below Secondary" : "Below Secondary"
                                      })


# In[18]:


df_train.groupby('department')['education'].value_counts()


# In[19]:


#cbar removes the legend colormap (default : True)
sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='viridis') 

# now no null value so whole plot will be in same color


# In[20]:


df_train["is_promoted"].value_counts()


# In[21]:


# Out of 54808, only 4668 are recommended for promotion, ie. only 8.5%
print(4668/54808 * 100)


# ### Visualization

# In[22]:


#sns.pairplot(df_train)


# In[23]:


sns.set_style('whitegrid')
sns.countplot(x='department',data=df_train,palette='RdBu_r')


# In[24]:


sns.set_style('whitegrid')
sns.countplot(x='education',data=df_train,palette='RdBu_r', hue = "recruitment_channel")


# In[25]:


sns.jointplot(x = "length_of_service", y = "avg_training_score", data = df_train)


# With increase in service areas, people tend to loose the motivation to get promotion and hence achive less score. But the youngsters thrive to get promotion and hence performs well.

# In[26]:


sns.jointplot(x = "no_of_trainings", y = "avg_training_score", data = df_train)


# In[27]:


sns.jointplot(x = "no_of_trainings", y = "length_of_service", data = df_train)


# Using jointplots to check the average_score attained by employees in their service period and age

# In[28]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='department',y='avg_training_score',data=df_train,palette='winter', hue="is_promoted")


# Maximum training score is achived by employees in departments ['Analytics', 'R&D', 'Technology'] 

# In[29]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='recruitment_channel',y='avg_training_score',data=df_train,palette='winter')


# ## Converting Categorical Features
# 
# We'll need to convert categorical features to dummy variables using pandas! Otherwise our machine learning algorithm won't be able to directly take in those features as inputs.
# 

# In[30]:


dept = pd.get_dummies(df_train['department'],drop_first=True, prefix="department")
edu = pd.get_dummies(df_train['education'],drop_first=True, prefix="education")
male = pd.get_dummies(df_train['gender'],drop_first=True, prefix="gender")
recruit = pd.get_dummies(df_train['recruitment_channel'],drop_first=True, prefix="recruitment_channel")
yr_rating = pd.get_dummies(df_train['previous_year_rating'],drop_first=True, prefix="previous_year_rating")


# In[31]:


dept.head()


# In[32]:


edu.head()


# In[33]:


male.head()


# In[34]:


recruit.head()


# In[35]:


yr_rating.head()


# In[36]:


df_train.drop(["region", "department", "education", "gender", "recruitment_channel", "previous_year_rating"],axis=1,inplace=True)
df_train.head()


# In[37]:


df_train = pd.concat([df_train, dept, edu, male, recruit, yr_rating],axis=1)
df_train.head()


# In[38]:


df_train.rename(columns = {"department_Sales & Marketing" : "department_Sales&Marketing",
                          "education_Below Secondary" : "education_BelowSecondary",
                          "education_Master's & above" : "education_Master's&above"}, inplace = True)


# In[39]:


df_train.head()


# # Building a Logistic Regression model

# In[40]:


#X_train = df_train.drop("is_promoted", axis = 1)
#y_train = df_train["is_promoted"]


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(df_train.drop('is_promoted',axis=1), df_train['is_promoted'], test_size=0.30,random_state=101)


# In[42]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[43]:


predictions = logmodel.predict(X_test)


# ## Evaluation

# In[44]:


print(classification_report(y_test,predictions))


# In[45]:


cm = confusion_matrix(y_test,predictions)
print(confusion_matrix(y_test,predictions))


# Here 0 means 'no promotion' and 1 means 'promotion'.
# 
# Therefore, here we are talking about no promotion as output.
# 
# for instance, with 93% accuracy I can say that the person will not get promoted

# In[46]:


TP = cm[0][0]
FN = cm[0][1]
FP = cm[1][0]
TN = cm[1][1]


# In[47]:


accuracy = (TP + TN) / (TP + TN + FP + FN)
accuracy


# In[48]:


error_rate = (FP + FN) / (TP + TN + FP + FN)
error_rate


# In[49]:


recall = (TP) / (TP + FN)
recall


# In[50]:


precision = (TP) / (TP + FP)
precision


# In[51]:


f_score = (2 * recall * precision) / (recall + precision)
f_score


# In[52]:


from sklearn.metrics import roc_auc_score, roc_curve

# predict probabilities
probs = logmodel.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]


# In[53]:


# calculate AUC
auc = roc_auc_score(y_test, probs)
auc


# In[54]:


# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)

plt.plot([0, 1], [0, 1], linestyle='--')   #plt.plot(x, y)
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.show()


# ## Evaluationg with different models

# In[55]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import time


# In[56]:


from matplotlib import cm as cm

fig = plt.figure()
ax1 = fig.add_subplot(111)
cmap = cm.get_cmap('jet', 10)      #10 here is number of different colors
cax = ax1.imshow(df_train.corr(), interpolation="none", cmap=cmap)
ax1.grid(True)
plt.title('Weather Attributes Correlation')
# Add colorbar, make sure to specify tick locations to match desired ticklabels
#fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
fig.colorbar(cax)
plt.show()


# ### Baseline algorithm checking
# 
# From the dataset, we will analysis and build a model to predict if precipitation is rain/snow/sunny. This is a binary classification problem, and a few algorithms are appropriate for use. Since we do not know which one will perform the best at the point, we will do a quick test on the few appropriate algorithms with default setting to get an early indication of how each of them perform. We will use 10 fold cross validation for each testing.
# 
# The following non-linear algorithms will be used, namely: Classification and Regression Trees (CART), Linear Support Vector Machines (SVM), Gaussian Naive Bayes (NB) and k-Nearest Neighbors (KNN).
# 

# In[57]:


models_list = []
models_list.append(('CART', DecisionTreeClassifier()))
#models_list.append(('SVM', SVC())) # It is taking 16.5 minutes here, so avoid for now
models_list.append(('NB', GaussianNB()))
models_list.append(('KNN', KNeighborsClassifier()))


# In[58]:


num_folds = 10
results = []
names = []

for name, model in models_list:
    kfold = KFold(n_splits=num_folds, random_state=123)
    start = time.time()
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    end = time.time()
    results.append(cv_results)
    names.append(name)
    print( "%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), end-start))


# In[59]:


fig = plt.figure()
fig.suptitle('Performance Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# From the initial run, it looks like CART, KNN and SVM performed the best given the dataset (all above 98% mean accuracy). GuassianNB has also given good erformance here. However, if we standardise the input dataset, it's performance should improve.

# ## Evaluation of algorithm on Standardised Data
# 
# The performance of the few machine learning algorithm could be improved if a standardised dataset is being used. The improvement is likely for all the models. I will use pipelines that standardize the data and build the model for each fold in the cross-validation test harness. That way we can get a fair estimation of how each model with standardized data might perform on unseen data

# In[60]:


import warnings

# Standardize the dataset
pipelines = []

pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
#pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC())]))) # It is taking 5 minutes here, very less as compared to 76 minutes before
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))  # it takes 2 minutes
results = []
names = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    kfold = KFold(n_splits=num_folds, random_state=123)
    for name, model in pipelines:
        start = time.time()
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
        end = time.time()
        results.append(cv_results)
        names.append(name)
        print( "%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), end-start))


# In[61]:


fig = plt.figure()
fig.suptitle('Performance Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# ### Algorithm Tuning - Tuning SVM
# 
# We will focus on SVM for the algorithm tuning. We can tune two key parameter of the SVM algorithm - the value of C and the type of kernel. The default C for SVM is 1.0 and the kernel is Radial Basis Function (RBF). We will use the grid search method using 10-fold cross-validation with a standardized copy of the sample training dataset. We will try over a combination of C values and the following kernel types 'linear', 'poly', 'rbf' and 'sigmoid

# In[62]:


#it's going to take 6 hours so be careful.

'''
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
#gamma can also be taken
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(C=c_values, kernel=kernel_values)
model = SVC()
kfold = KFold(n_splits=num_folds, random_state=21)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)
grid_result = grid.fit(rescaledX, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
'''


# ### Application of SVC on dataset
# 
# Let's fit the SVM to the dataset and see how it performs given the test data.
# 

# In[63]:


# prepare the model
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
model = SVC(C=2.0, kernel='rbf')
start = time.time()
model.fit(X_train_scaled, y_train)
end = time.time()
print( "Run Time: %f" % (end-start))


# In[64]:


# estimate accuracy on test dataset
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    X_test_scaled = scaler.transform(X_test)
predictions = model.predict(X_test_scaled)


# In[65]:


print("Accuracy score %f" % accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))


# In[66]:


cm = confusion_matrix(y_test, predictions)
print(cm)


# ##  Working on testing dataset_now

# In[68]:


df_test = pd.read_csv("test_2umaH9m.csv", index_col = 0)
df_test.head()


# In[69]:


df_test['education'].fillna(value = "Bachelor's", inplace = True)
df_test['previous_year_rating'].fillna(value = 0, inplace = True)


# In[70]:


dept = pd.get_dummies(df_test['department'],drop_first=True, prefix="department")
edu = pd.get_dummies(df_test['education'],drop_first=True, prefix="education")
male = pd.get_dummies(df_test['gender'],drop_first=True, prefix="gender")
recruit = pd.get_dummies(df_test['recruitment_channel'],drop_first=True, prefix="recruitment_channel")
yr_rating = pd.get_dummies(df_test['previous_year_rating'],drop_first=True, prefix="previous_year_rating")


# In[71]:


df_test.drop(["region", "department", "education", "gender", "recruitment_channel", "previous_year_rating"],axis=1,inplace=True)
df_test.head()


# In[72]:


df_test = pd.concat([df_test, dept, edu, male, recruit, yr_rating],axis=1)
df_test.head()


# In[73]:


df_test.shape


# In[74]:


df_test.rename(columns = {"department_Sales & Marketing" : "department_Sales&Marketing",
                          "education_Below Secondary" : "education_BelowSecondary",
                          "education_Master's & above" : "education_Master's&above"}, inplace = True)


# In[75]:


df_train.shape


# In[76]:


df_train.columns


# In[77]:


df_test.columns


# In[78]:


# estimate accuracy on test dataset
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    df_test_scaled = scaler.transform(df_test)
predictions = model.predict(df_test_scaled)


# In[79]:


predictions


# In[80]:


len(predictions)


# In[85]:


predict = pd.DataFrame(predictions, columns=['predictions'])


# In[86]:


predict.head()


# In[87]:


predict.to_csv('y_predicted.csv')

