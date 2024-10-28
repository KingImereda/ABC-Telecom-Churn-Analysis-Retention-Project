#!/usr/bin/env python
# coding: utf-8

# ## Build diff ML models
# 
# New notebook

# In[1]:


#Load cleaned data from lakehouse
df = spark.sql("SELECT * FROM ABC_Telecom.clean_data")
display(df)


# In[2]:


# Import pandas library and convert spark dataframe to pandas dataframe
import pandas as pd
df = df.toPandas()
df.head()


# In[3]:


# Display all the columns in the pandas dataframe.
df.columns.values


#  ##### 1. Linear Regression Model

# In[4]:


#Split dataset into Train and Test data
from sklearn.model_selection import train_test_split


X, y = df[['SeniorCitizen', 'MonthlyCharges', 'TotalCharges',
       'gender_Female', 'gender_Male', 'Partner_No', 'Partner_Yes',
       'Dependents_No', 'Dependents_Yes', 'PhoneService_No',
       'PhoneService_Yes', 'MultipleLines_No',
       'MultipleLines_No_phone_service', 'MultipleLines_Yes',
       'InternetService_DSL', 'InternetService_Fiber_optic',
       'InternetService_No', 'OnlineSecurity_No',
       'OnlineSecurity_No_internet_service', 'OnlineSecurity_Yes',
       'OnlineBackup_No', 'OnlineBackup_No_internet_service',
       'OnlineBackup_Yes', 'DeviceProtection_No',
       'DeviceProtection_No_internet_service', 'DeviceProtection_Yes',
       'TechSupport_No', 'TechSupport_No_internet_service',
       'TechSupport_Yes', 'StreamingTV_No',
       'StreamingTV_No_internet_service', 'StreamingTV_Yes',
       'StreamingMovies_No', 'StreamingMovies_No_internet_service',
       'StreamingMovies_Yes', 'Contract_Month-to-month',
       'Contract_One_year', 'Contract_Two_year', 'PaperlessBilling_No',
       'PaperlessBilling_Yes', 'PaymentMethod_Bank_transfer__automatic_',
       'PaymentMethod_Credit_card__automatic_',
       'PaymentMethod_Electronic_check', 'PaymentMethod_Mailed_check',
       'tenure_group_1_-_12', 'tenure_group_13_-_24',
       'tenure_group_25_-_36', 'tenure_group_37_-_48',
       'tenure_group_49_-_60', 'tenure_group_61_-_72']].values, df['Churn'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)



# In[5]:


# Name Experiment:
import mlflow
experiment_name = "linear_regression"
mlflow.set_experiment(experiment_name)


# In[6]:


#Linear Regression:
from sklearn.linear_model import LinearRegression
    
with mlflow.start_run():
   mlflow.autolog()
    
   model = LinearRegression()
   model.fit(X_train, y_train)
    
   mlflow.log_param("estimator", "LinearRegression")


# ##### 2. Logistic Regression or Classification Model

# In[7]:


#Split dataset into Train and Test data
from sklearn.model_selection import train_test_split


X, y = df[['SeniorCitizen', 'MonthlyCharges', 'TotalCharges',
       'gender_Female', 'gender_Male', 'Partner_No', 'Partner_Yes',
       'Dependents_No', 'Dependents_Yes', 'PhoneService_No',
       'PhoneService_Yes', 'MultipleLines_No',
       'MultipleLines_No_phone_service', 'MultipleLines_Yes',
       'InternetService_DSL', 'InternetService_Fiber_optic',
       'InternetService_No', 'OnlineSecurity_No',
       'OnlineSecurity_No_internet_service', 'OnlineSecurity_Yes',
       'OnlineBackup_No', 'OnlineBackup_No_internet_service',
       'OnlineBackup_Yes', 'DeviceProtection_No',
       'DeviceProtection_No_internet_service', 'DeviceProtection_Yes',
       'TechSupport_No', 'TechSupport_No_internet_service',
       'TechSupport_Yes', 'StreamingTV_No',
       'StreamingTV_No_internet_service', 'StreamingTV_Yes',
       'StreamingMovies_No', 'StreamingMovies_No_internet_service',
       'StreamingMovies_Yes', 'Contract_Month-to-month',
       'Contract_One_year', 'Contract_Two_year', 'PaperlessBilling_No',
       'PaperlessBilling_Yes', 'PaymentMethod_Bank_transfer__automatic_',
       'PaymentMethod_Credit_card__automatic_',
       'PaymentMethod_Electronic_check', 'PaymentMethod_Mailed_check',
       'tenure_group_1_-_12', 'tenure_group_13_-_24',
       'tenure_group_25_-_36', 'tenure_group_37_-_48',
       'tenure_group_49_-_60', 'tenure_group_61_-_72']].values, df['Churn'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[8]:


# Name Experiment:
import mlflow
experiment_name = "classification_regression"
mlflow.set_experiment(experiment_name)


# In[9]:


# Classification or Logistic Regression:
from sklearn.linear_model import LogisticRegression
    
with mlflow.start_run():
    mlflow.sklearn.autolog()

    model = LogisticRegression(C=1/0.1, solver="liblinear").fit(X_train, y_train)


# ##### 3. Decision Tree Model

# In[10]:


#Split dataset into Train and Test data
from sklearn.model_selection import train_test_split


X, y = df[['SeniorCitizen', 'MonthlyCharges', 'TotalCharges',
       'gender_Female', 'gender_Male', 'Partner_No', 'Partner_Yes',
       'Dependents_No', 'Dependents_Yes', 'PhoneService_No',
       'PhoneService_Yes', 'MultipleLines_No',
       'MultipleLines_No_phone_service', 'MultipleLines_Yes',
       'InternetService_DSL', 'InternetService_Fiber_optic',
       'InternetService_No', 'OnlineSecurity_No',
       'OnlineSecurity_No_internet_service', 'OnlineSecurity_Yes',
       'OnlineBackup_No', 'OnlineBackup_No_internet_service',
       'OnlineBackup_Yes', 'DeviceProtection_No',
       'DeviceProtection_No_internet_service', 'DeviceProtection_Yes',
       'TechSupport_No', 'TechSupport_No_internet_service',
       'TechSupport_Yes', 'StreamingTV_No',
       'StreamingTV_No_internet_service', 'StreamingTV_Yes',
       'StreamingMovies_No', 'StreamingMovies_No_internet_service',
       'StreamingMovies_Yes', 'Contract_Month-to-month',
       'Contract_One_year', 'Contract_Two_year', 'PaperlessBilling_No',
       'PaperlessBilling_Yes', 'PaymentMethod_Bank_transfer__automatic_',
       'PaymentMethod_Credit_card__automatic_',
       'PaymentMethod_Electronic_check', 'PaymentMethod_Mailed_check',
       'tenure_group_1_-_12', 'tenure_group_13_-_24',
       'tenure_group_25_-_36', 'tenure_group_37_-_48',
       'tenure_group_49_-_60', 'tenure_group_61_-_72']].values, df['Churn'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[11]:


# Name Experiment:
import mlflow
experiment_name = "decision_tree"
mlflow.set_experiment(experiment_name)


# In[12]:


# Decision Tree Classification:
from sklearn.tree import DecisionTreeRegressor
    
with mlflow.start_run():
   mlflow.autolog()
    
   model = DecisionTreeRegressor(max_depth=5) 
   model.fit(X_train, y_train)
    
   mlflow.log_param("estimator", "DecisionTreeRegressor")


# ##### 4. Random Forest Model

# In[13]:


#Split dataset into Train and Test data
from sklearn.model_selection import train_test_split


X, y = df[['SeniorCitizen', 'MonthlyCharges', 'TotalCharges',
       'gender_Female', 'gender_Male', 'Partner_No', 'Partner_Yes',
       'Dependents_No', 'Dependents_Yes', 'PhoneService_No',
       'PhoneService_Yes', 'MultipleLines_No',
       'MultipleLines_No_phone_service', 'MultipleLines_Yes',
       'InternetService_DSL', 'InternetService_Fiber_optic',
       'InternetService_No', 'OnlineSecurity_No',
       'OnlineSecurity_No_internet_service', 'OnlineSecurity_Yes',
       'OnlineBackup_No', 'OnlineBackup_No_internet_service',
       'OnlineBackup_Yes', 'DeviceProtection_No',
       'DeviceProtection_No_internet_service', 'DeviceProtection_Yes',
       'TechSupport_No', 'TechSupport_No_internet_service',
       'TechSupport_Yes', 'StreamingTV_No',
       'StreamingTV_No_internet_service', 'StreamingTV_Yes',
       'StreamingMovies_No', 'StreamingMovies_No_internet_service',
       'StreamingMovies_Yes', 'Contract_Month-to-month',
       'Contract_One_year', 'Contract_Two_year', 'PaperlessBilling_No',
       'PaperlessBilling_Yes', 'PaymentMethod_Bank_transfer__automatic_',
       'PaymentMethod_Credit_card__automatic_',
       'PaymentMethod_Electronic_check', 'PaymentMethod_Mailed_check',
       'tenure_group_1_-_12', 'tenure_group_13_-_24',
       'tenure_group_25_-_36', 'tenure_group_37_-_48',
       'tenure_group_49_-_60', 'tenure_group_61_-_72']].values, df['Churn'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[14]:


# Name Experiment:
import mlflow
experiment_name = "random_forest"
mlflow.set_experiment(experiment_name)


# In[15]:


# Random Forest Regressor:
from sklearn.ensemble import RandomForestRegressor
from mlflow import autolog

with mlflow.start_run():
  mlflow.autolog()

  # Set hyperparameters for Random Forest Regressor (adjust as needed)
  model = RandomForestRegressor(n_estimators=100, max_depth=5) 

  model.fit(X_train, y_train)

  mlflow.log_param("estimator", "RandomForestRegressor")  


# ##### To list all experiments, use the following code
# 

# In[16]:


import mlflow
experiments = mlflow.search_experiments()
for exp in experiments:
    print(exp.name)


# **Result:** 
# 
# ##### Below are the key metric of all the different ML models.The classification model has the highest accuracy among all the 4 models. So,I ampicking the Classification model for the Batch Prediction.

# 
# **Classification_regression Run metrics (7)**
#  ```
# training_accuracy_score
# 0.8017777777777778
# training_f1_score
# 0.7926115006868967
# training_log_loss
# 0.4246052050225107
# training_precision_score
# 0.791361113567693
# training_recall_score
# 0.8017777777777778
# training_roc_auc
# 0.8404142492737763
# training_score
# 0.8017777777777778
# 
# ```
# 
# **Decision Tree Run metrics (5)**
# ```
# training_mean_absolute_error
# 0.26874471565719127
# training_mean_squared_error
# 0.13437235782859563
# training_r2_score
# 0.31576040098486224
# training_root_mean_squared_error
# 0.36656835355578044
# training_score
# 0.31576040098486224
# ```
# 
# **Linear Regression Run metrics (5)**
# ```
# training_mean_absolute_error
# 0.2907650372194981
# training_mean_squared_error
# 0.1393653745240242
# training_r2_score
# 0.2903353820541541
# training_root_mean_squared_error
# 0.37331672146318895
# training_score
# 0.2903353820541541
# 
# ```
# 
# 
# **Random Forest Run metrics (5)**
# ```
# training_mean_absolute_error
# 0.27013785344586644
# training_mean_squared_error
# 0.1298880487590777
# training_r2_score
# 0.3385950218039808
# training_root_mean_squared_error
# 0.36039984567016353
# training_score
# 0.3385950218039808
# 
# ```
