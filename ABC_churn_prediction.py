#!/usr/bin/env python
# coding: utf-8

# ## ABC_churn_prediction
# 
# New notebook

# In[1]:


# Load data of new customers/subscribers

df = spark.sql("SELECT * FROM ABC_Telecom.new_customers LIMIT 1000")
display(df)


# In[2]:


import mlflow
from synapse.ml.predict import MLFlowTransformer
    
# read new customers/subscribers data through the table path

df = spark.read.format("delta").load("abfss://374380e1-d804-491d-861f-cc99bfbb443d@onelake.dfs.fabric.microsoft.com/26716f58-3490-425b-b25d-56272dfa71d5/Tables/new_customers")  # Your input table filepath here

# MLFlow Transformer    
model = MLFlowTransformer(
        inputCols=["SeniorCitizen","MonthlyCharges","TotalCharges","gender_Female","gender_Male","Partner_No","Partner_Yes","Dependents_No","Dependents_Yes","PhoneService_No","PhoneService_Yes","MultipleLines_No","MultipleLines_No_phone_service","MultipleLines_Yes","InternetService_DSL","InternetService_Fiber_optic","InternetService_No","OnlineSecurity_No","OnlineSecurity_No_internet_service","OnlineSecurity_Yes","OnlineBackup_No","OnlineBackup_No_internet_service","OnlineBackup_Yes","DeviceProtection_No","DeviceProtection_No_internet_service","DeviceProtection_Yes","TechSupport_No","TechSupport_No_internet_service","TechSupport_Yes","StreamingTV_No","StreamingTV_No_internet_service","StreamingTV_Yes","StreamingMovies_No","StreamingMovies_No_internet_service","StreamingMovies_Yes","Contract_Month-to-month","Contract_One_year","Contract_Two_year","PaperlessBilling_No","PaperlessBilling_Yes","PaymentMethod_Bank_transfer__automatic_","PaymentMethod_Credit_card__automatic_","PaymentMethod_Electronic_check","PaymentMethod_Mailed_check","tenure_group_1_-_12","tenure_group_13_-_24","tenure_group_25_-_36","tenure_group_37_-_48","tenure_group_49_-_60","tenure_group_61_-_72"], # Your input columns here
        outputCol="predictions", # Your new column name here
        modelName="ABC_Telecom", # Your model name here
        modelVersion=2 # Your model version here
    )
df = model.transform(df)


# In[3]:


# Generate predictions by applying our pre-trained ML model to dataset df

predictions = model.transform(df)
display(df)


# In[4]:


# Overwrite result aspredictions

predictions.write.format("delta").mode("overwrite").save(f"Tables/predictions")

