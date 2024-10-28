#!/usr/bin/env python
# coding: utf-8

# ## Register & define Model Signature/Create New Customer Data
# 
# New notebook

# In[1]:


# Load cleaned data

df = spark.sql("SELECT * FROM ABC_Telecom.clean_data")
display(df)


# In[2]:


import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

# Get the data
df = spark.sql("SELECT * FROM ABC_Telecom.clean_data")
# Convert to pandas
df = df.toPandas()


# Split the features and label for training
X, y = df[["SeniorCitizen", "MonthlyCharges", "TotalCharges",
       "gender_Female", "gender_Male", "Partner_No", "Partner_Yes",
       "Dependents_No", "Dependents_Yes", "PhoneService_No",
       "PhoneService_Yes", "MultipleLines_No",
       "MultipleLines_No_phone_service", "MultipleLines_Yes",
       "InternetService_DSL", "InternetService_Fiber_optic",
       "InternetService_No", "OnlineSecurity_No",
       "OnlineSecurity_No_internet_service", "OnlineSecurity_Yes",
       "OnlineBackup_No", "OnlineBackup_No_internet_service",
       "OnlineBackup_Yes", "DeviceProtection_No",
       "DeviceProtection_No_internet_service", "DeviceProtection_Yes",
       "TechSupport_No", "TechSupport_No_internet_service",
       "TechSupport_Yes", "StreamingTV_No",
       "StreamingTV_No_internet_service", "StreamingTV_Yes",
       "StreamingMovies_No", "StreamingMovies_No_internet_service",
       "StreamingMovies_Yes", "Contract_Month-to-month",
       "Contract_One_year", "Contract_Two_year", "PaperlessBilling_No",
       "PaperlessBilling_Yes", "PaymentMethod_Bank_transfer__automatic_",
       "PaymentMethod_Credit_card__automatic_",
       "PaymentMethod_Electronic_check", "PaymentMethod_Mailed_check",
       "tenure_group_1_-_12", "tenure_group_13_-_24",
       "tenure_group_25_-_36", "tenure_group_37_-_48",
       "tenure_group_49_-_60", "tenure_group_61_-_72" ]].values,  df["Churn"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

#Train the model in an MLflow experiment
experiment_name = "ABC_Telecom"
mlflow.set_experiment(experiment_name)
with mlflow.start_run():
    mlflow.sklearn.autolog()

    model = LogisticRegression(C=1/0.1, solver="liblinear").fit(X_train, y_train)

    


    # Define the model signature, 
    # ***specify the structure and data types of the input and output data expected by the model
    input_schema = Schema([
        ColSpec("integer","SeniorCitizen"),
        ColSpec("double","MonthlyCharges"), 
        ColSpec("double", "TotalCharges"),
        ColSpec("integer", "gender_Female"), 
        ColSpec("integer", "gender_Male"), 
        ColSpec("integer","Partner_No"), 
        ColSpec("integer","Partner_Yes"),
        ColSpec("integer", "Dependents_No"), 
        ColSpec("integer","Dependents_Yes"), 
        ColSpec("integer", "PhoneService_No"),
        ColSpec("integer", "PhoneService_Yes"), 
        ColSpec("integer","MultipleLines_No"),
        ColSpec("integer", "MultipleLines_No_phone_service"), 
        ColSpec("integer", "MultipleLines_Yes"),
        ColSpec("integer", "InternetService_DSL"), 
        ColSpec("integer", "InternetService_Fiber_optic"),
        ColSpec("integer", "InternetService_No"), 
        ColSpec("integer", "OnlineSecurity_No"),
        ColSpec("integer", "OnlineSecurity_No_internet_service"),
        ColSpec("integer", "OnlineSecurity_Yes"),
        ColSpec("integer", "OnlineBackup_No"),
        ColSpec("integer", "OnlineBackup_No_internet_service"),
        ColSpec("integer", "OnlineBackup_Yes"),
        ColSpec("integer", "DeviceProtection_No"),
        ColSpec("integer", "DeviceProtection_No_internet_service"),
        ColSpec("integer", "DeviceProtection_Yes"),
        ColSpec("integer", "TechSupport_No"),
        ColSpec("integer", "TechSupport_No_internet_service"),
        ColSpec("integer", "TechSupport_Yes"),
        ColSpec("integer", "StreamingTV_No"),
        ColSpec("integer", "StreamingTV_No_internet_service"),
        ColSpec("integer", "StreamingTV_Yes"),
        ColSpec("integer", "StreamingMovies_No"),
        ColSpec("integer", "StreamingMovies_No_internet_service"),
        ColSpec("integer", "StreamingMovies_Yes"),
        ColSpec("integer", "Contract_Month-to-month"),
        ColSpec("integer", "Contract_One_year"),
        ColSpec("integer", "Contract_Two_year"),
        ColSpec("integer", "PaperlessBilling_No"),
        ColSpec("double", "PaperlessBilling_Yes"),
        ColSpec("integer", "PaymentMethod_Bank_transfer__automatic_"),
        ColSpec("integer", "PaymentMethod_Credit_card__automatic_"),
        ColSpec("integer", "PaymentMethod_Electronic_check"),
        ColSpec("integer", "PaymentMethod_Mailed_check"),
        ColSpec("integer", "tenure_group_1_-_12"),
        ColSpec("integer", "tenure_group_13_-_24"),
        ColSpec("integer", "tenure_group_25_-_36"),
        ColSpec("integer", "tenure_group_37_-_48"),
        ColSpec("integer", "tenure_group_49_-_60"),
        ColSpec("integer", "tenure_group_61_-_72"),
        ])
    output_schema = Schema([ColSpec("integer")])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
   
    # Log the model
    mlflow.sklearn.log_model(model, "model", signature=signature)



# In[3]:


# Get the most recent experiement run
exp = mlflow.get_experiment_by_name(experiment_name)
last_run = mlflow.search_runs(exp.experiment_id, order_by=["start_time DESC"], max_results=1)
last_run_id = last_run.iloc[0]["run_id"]

# Register the model that was trained in that run
print("Registering the model from run :", last_run_id)
model_uri = "runs:/{}/model".format(last_run_id)
mv = mlflow.register_model(model_uri, "ABC_Telecom")
print("Name: {}".format(mv.name))
print("Version: {}".format(mv.version))



# ***  **Records of new customers to predict if they will churn or not.**

# In[4]:


from pyspark.sql.types import IntegerType, DoubleType

# Create a new DataFrame capturing data of new subscribers.
data = [
    (0, 70.35, 1407.00, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,),
    (1, 90.65, 5439.00, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0,),
    (1, 58.43, 1040.51, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,),
    (1, 55.89, 870.30, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1,),
    (0, 75.44, 2000.15, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0,),
    (0, 60.72, 950.43, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1,),
    (1, 85.90, 1550.66, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1,),
    (0, 68.34, 1300.85, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,),
    (1, 50.82, 750.33, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0,),
    (0, 78.71, 1800.08, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0,)
]

columns = ["SeniorCitizen", "MonthlyCharges", "TotalCharges", "gender_Female", "gender_Male", "Partner_No", "Partner_Yes",
           "Dependents_No", "Dependents_Yes", "PhoneService_No", "PhoneService_Yes", "MultipleLines_No",
           "MultipleLines_No_phone_service", "MultipleLines_Yes", "InternetService_DSL", "InternetService_Fiber_optic",
           "InternetService_No", "OnlineSecurity_No", "OnlineSecurity_No_internet_service", "OnlineSecurity_Yes",
           "OnlineBackup_No", "OnlineBackup_No_internet_service", "OnlineBackup_Yes", "DeviceProtection_No",
           "DeviceProtection_No_internet_service", "DeviceProtection_Yes", "TechSupport_No",
           "TechSupport_No_internet_service", "TechSupport_Yes", "StreamingTV_No", "StreamingTV_No_internet_service",
           "StreamingTV_Yes", "StreamingMovies_No", "StreamingMovies_No_internet_service", "StreamingMovies_Yes",
           "Contract_Month-to-month", "Contract_One_year", "Contract_Two_year", "PaperlessBilling_No",
           "PaperlessBilling_Yes", "PaymentMethod_Bank_transfer__automatic_", "PaymentMethod_Credit_card__automatic_",
           "PaymentMethod_Electronic_check", "PaymentMethod_Mailed_check", "tenure_group_1_-_12", "tenure_group_13_-_24",
           "tenure_group_25_-_36", "tenure_group_37_-_48", "tenure_group_49_-_60", "tenure_group_61_-_72"]

df = spark.createDataFrame(data, schema=columns)

# Convert data types to match the model input schema
df = df.withColumn("SeniorCitizen", df["SeniorCitizen"].cast(IntegerType()))
df = df.withColumn("MonthlyCharges", df["MonthlyCharges"].cast(DoubleType()))  # Changed to DoubleType
df = df.withColumn("TotalCharges", df["TotalCharges"].cast(DoubleType()))
df = df.withColumn("gender_Female", df["gender_Female"].cast(IntegerType()))
df = df.withColumn("gender_Male", df["gender_Male"].cast(IntegerType()))
df = df.withColumn("Partner_No", df["Partner_No"].cast(IntegerType()))
df = df.withColumn("Partner_Yes", df["Partner_Yes"].cast(IntegerType()))
df = df.withColumn("Dependents_No", df["Dependents_No"].cast(IntegerType()))
df = df.withColumn("Dependents_Yes", df["Dependents_Yes"].cast(IntegerType()))
df = df.withColumn("PhoneService_No", df["PhoneService_No"].cast(IntegerType()))
df = df.withColumn("PhoneService_Yes", df["PhoneService_Yes"].cast(IntegerType()))
df = df.withColumn("MultipleLines_No", df["MultipleLines_No"].cast(IntegerType()))
df = df.withColumn("MultipleLines_No_phone_service", df["MultipleLines_No_phone_service"].cast(IntegerType()))
df = df.withColumn("MultipleLines_Yes", df["MultipleLines_Yes"].cast(IntegerType()))
df = df.withColumn("InternetService_DSL", df["InternetService_DSL"].cast(IntegerType()))
df = df.withColumn("InternetService_Fiber_optic", df["InternetService_Fiber_optic"].cast(IntegerType()))
df = df.withColumn("InternetService_No", df["InternetService_No"].cast(IntegerType()))
df = df.withColumn("OnlineSecurity_No", df["OnlineSecurity_No"].cast(IntegerType()))
df = df.withColumn("OnlineSecurity_No_internet_service", df["OnlineSecurity_No_internet_service"].cast(IntegerType()))
df = df.withColumn("OnlineSecurity_Yes", df["OnlineSecurity_Yes"].cast(IntegerType()))
df = df.withColumn("OnlineBackup_No", df["OnlineBackup_No"].cast(IntegerType()))
df = df.withColumn("OnlineBackup_No_internet_service", df["OnlineBackup_No_internet_service"].cast(IntegerType()))
df = df.withColumn("OnlineBackup_Yes", df["OnlineBackup_Yes"].cast(IntegerType()))
df = df.withColumn("DeviceProtection_No", df["DeviceProtection_No"].cast(IntegerType()))
df = df.withColumn("DeviceProtection_No_internet_service", df["DeviceProtection_No_internet_service"].cast(IntegerType()))
df = df.withColumn("DeviceProtection_Yes", df["DeviceProtection_Yes"].cast(IntegerType()))
df = df.withColumn("TechSupport_No", df["TechSupport_No"].cast(IntegerType()))
df = df.withColumn("TechSupport_No_internet_service", df["TechSupport_No_internet_service"].cast(IntegerType()))
df = df.withColumn("TechSupport_Yes", df["TechSupport_Yes"].cast(IntegerType()))
df = df.withColumn("StreamingTV_No", df["StreamingTV_No"].cast(IntegerType()))
df = df.withColumn("StreamingTV_No_internet_service", df["StreamingTV_No_internet_service"].cast(IntegerType()))
df = df.withColumn("StreamingTV_Yes", df["StreamingTV_Yes"].cast(IntegerType()))
df = df.withColumn("StreamingMovies_No", df["StreamingMovies_No"].cast(IntegerType()))
df = df.withColumn("StreamingMovies_No_internet_service", df["StreamingMovies_No_internet_service"].cast(IntegerType()))
df = df.withColumn("StreamingMovies_Yes", df["StreamingMovies_Yes"].cast(IntegerType()))
df = df.withColumn("Contract_Month-to-month", df["Contract_Month-to-month"].cast(IntegerType()))
df = df.withColumn("Contract_One_year", df["Contract_One_year"].cast(IntegerType()))
df = df.withColumn("Contract_Two_year", df["Contract_Two_year"].cast(IntegerType()))
df = df.withColumn("PaperlessBilling_No", df["PaperlessBilling_No"].cast(IntegerType()))
df = df.withColumn("PaperlessBilling_Yes", df["PaperlessBilling_Yes"].cast(IntegerType()))
df = df.withColumn("PaymentMethod_Bank_transfer__automatic_", df["PaymentMethod_Bank_transfer__automatic_"].cast(IntegerType()))
df = df.withColumn("PaymentMethod_Credit_card__automatic_", df["PaymentMethod_Credit_card__automatic_"].cast(IntegerType()))
df = df.withColumn("PaymentMethod_Electronic_check", df["PaymentMethod_Electronic_check"].cast(IntegerType()))
df = df.withColumn("PaymentMethod_Mailed_check", df["PaymentMethod_Mailed_check"].cast(IntegerType()))
df = df.withColumn("tenure_group_1_-_12", df["tenure_group_1_-_12"].cast(IntegerType()))
df = df.withColumn("tenure_group_13_-_24", df["tenure_group_13_-_24"].cast(IntegerType()))
df = df.withColumn("tenure_group_25_-_36", df["tenure_group_25_-_36"].cast(IntegerType()))
df = df.withColumn("tenure_group_37_-_48", df["tenure_group_37_-_48"].cast(IntegerType()))
df = df.withColumn("tenure_group_49_-_60", df["tenure_group_49_-_60"].cast(IntegerType()))
df = df.withColumn("tenure_group_61_-_72", df["tenure_group_61_-_72"].cast(IntegerType()))


# Save the data in a delta table
table_name = "new_customers"
df.write.format("delta").mode("overwrite").saveAsTable(table_name)
print(f"Spark dataframe saved to delta table: {table_name}")

