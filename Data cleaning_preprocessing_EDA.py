#!/usr/bin/env python
# coding: utf-8

# ## Data cleaning/preprocessing/EDA
# 
# New notebook

# - **This work is done in python environment and not pyspark.**
# - **Load data into pandas dataframe**
# 

# In[1]:


import pandas as pd
# Load data into pandas DataFrame from "/lakehouse/default/Files/WA_Fn-UseC_-Telco-Customer-Churn.csv"
telco_base_data = pd.read_csv("/lakehouse/default/Files/WA_Fn-UseC_-Telco-Customer-Churn.csv")
display(telco_base_data)


# In[2]:


#Show the first 5 rows

telco_base_data.head(5)



# In[3]:


#Import necessary python Libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')



# In[4]:


#Check number of rows and column in the dataframe

telco_base_data.shape



# **Result:** The telco_base_data has 7043 rows and 21 columns as shown by the reuslt above.

# In[5]:


#Show all columns in the pandas dataframe

telco_base_data.columns.values 



# **Result :** Array showing list of columns in telco_base_data as shown by the result above.

# In[6]:


#Determing data type for each column

telco_base_data.dtypes


# On investigating the data type of each column shown above, it was discovered that the [TotalCharges] column is presented as object instead of float.
# 

# In[7]:


# convert [TotalCharges] from string to double.

telco_base_data['TotalCharges'] = pd.to_numeric(telco_base_data['TotalCharges'], errors = 'coerce')


# ##### ***The dataset is grouped into Features(independent variables) and Label(dependent variables). The Features can further be sub-divided as 'Category' values and 'Numerical' values.
# 

# In[8]:


# Check descriptive statistics of numerical variables

telco_base_data.describe()



# In[9]:


# Count the number of churners as against number of active accounts.

telco_base_data['Churn'].value_counts()


# 
# **Insights from descriptive statistics:**
# ```
# 1. That the total number of subscribers of ABC_Telecom is 7043.
# 
# 2. The maximum tenure a customer has stayed with the ABC_Telecom is 72 months and the average tenure of customers is 32.4 months.
# 
# 3. 75% of the subscribers have spent less than 55 month with the company, 50% of the subscribers have spent less than 29 months with the company.
# 
# 4. That the average monthly charges per customer is 64.67 USD and only 25% of the customers spent more than 89.85 USD per month.
# 
# 5. That the average total charges in period under review 2283.30 USD and 75% of customers spent less than 3794.74 USD.
# ```
# 
# 

# In[10]:


telco_base_data['Churn'].value_counts()


# ###### **Result:** 
# - That from a total number of 7043 subscribers, 5174 are active subscribers and 1869 are churners(inactive account).
# 
# - In % terms, 73.46% of customers have active account with ABC_Telecom while 26.53% of customers have churn(left)
# 
# 
# 

# In[11]:


# Plot the number of active account as against inactive(churn) account and plot result in a bar chart.
telco_base_data['Churn'].value_counts().plot(kind='bar', figsize=(8, 6))
plt.xlabel("Count", labelpad=14)
plt.ylabel("Target Variable", labelpad=14)
plt.title("Count of TARGET Variable per category", y=1.02);



# ### **Data Cleaning & Preprocessing.**

# In[12]:


# Create a copy of the base data for manipulation and processing
telco_data = telco_base_data.copy()


# ###### Finding the total of null values in each column

# In[13]:


#Find the total of all missing values in each column
telco_data.isnull().sum()


# **Result:** Each column has zero missing values except [TotalCharge] that has 11 missing values.

# In[14]:


# To find specific rows that have null values in ['TotalCharge'] column
telco_data.loc[telco_data ['TotalCharges'].isnull() == True]


# In[15]:


# Plot the % of missing values
missing = pd.DataFrame((telco_base_data.isnull().sum()) * 100 / telco_base_data.shape[0]).reset_index()
missing.columns = ['Feature', 'Percentage']

plt.figure(figsize=(16, 5))
ax = sns.pointplot(x='Feature', y='Percentage', data=missing)
plt.xticks(rotation=90, fontsize=7)
plt.title("Percentage of Missing Values")
plt.ylabel("PERCENTAGE")
plt.show()


# **P.S.**
# ```
# Handling Missing Values:
# - For features with less or few missing values- we can use regression to predict the missing values or fill  them with the mean of the values present, or use a moving average depending on the feature.
# 
# - For features with very high number of missing values- it is better to drop those columns as they provide little insights on analysis.
# 
# - As there 're no rule of thumb on what criteria do we delete the columns with high number of missing values, but generally you can delete the columns, if you have more than 30-40% of missing values. But again there's a catch here, for example, Is_Car & Car_Type, People having no cars, will obviously have Car_Type as NaN (null), but that doesn't make this column useless, so decisions has to be taken wisely.
# ```

# **Result:**
# 
# Since the total number of missing value is 11 i.e., 0.15% in % terms compared to the  total records in the  dataset, it is safe to ignore them, by deleting them from further processing.
# 
# However, missing values can also be handle by inputting average values, Zero, or moving average etc.
# 

# In[16]:


# This code remove any rows with missing value
telco_data.dropna(how = 'any', inplace = True)

# Alternatively, we can fill the  empty field in ['TotalCharge'] column with zero value using this code "telco_data.fillna(0)""


# In[17]:


# Calculate the max 'Tenure'
print(telco_data['tenure'].max())


# **Result:** The maximum tenure of any of the subscriber is 72 months.

# In[18]:


#Divide customers into bins or groups based on tenure e.g. for tenure < 12 months: assign a tenure group if 1-12, for tenure between 1 to 2 Yrs, tenure group of 13-24; so on...
# Group the tenure column in bins of 12 months for ease of analyzing the tenure column

labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]

telco_data['tenure_group'] = pd.cut(telco_data.tenure, range(1, 80, 12), right=False, labels=labels)


# In[19]:


# Show result
telco_data['tenure_group'].value_counts()


# In[20]:


#Drop Columns that are insignificant to our analysis.
#drop column customerID and tenure, since we have replace Tenure with Tenure-group

telco_data.drop(columns= ['customerID','tenure'], axis=1, inplace=True)
telco_data.head()



# #### **Explorative Data Analysis**

# **Univariate analysis:**
# 
# **1.** This focuses on single 'feature' in relation to a Label(churn) per time. Let's Plot distribution for individual predictor by Churn.
# 
# 

# In[21]:


for i, predictor in enumerate(telco_data.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges'])):
    plt.figure(i)
    sns.countplot(data=telco_data, x=predictor, hue='Churn')


# **2.** Convert the target variable 'Churn' in a binary numeric variable i.e. Yes=1 ; No = 0
# 

# In[22]:


telco_data['Churn'] = np.where(telco_data.Churn == 'Yes',1,0)

telco_data.head()


# **3.** Convert all non-numerical variables into dummy variables
# ```
# Label Encoding vs One-Hot Encoding(dummy variables):
# 
# Label encoding and one-hot encoding are two common techniques used in machine learning to convert categorical data into numerical data that can be processed by algorithms.
# 
# Label Encoding
# Label encoding is a simple technique that assigns a unique integer to each category in a categorical variable. It is often used when the categorical variable has an 'ordinal' relationship between its categories (e.g., "low", "medium", "high").
# For example:
# Gender                 Label Encoding
# Male				            0
# Female                          1
# Trans-gender                    2
# 
# By default, the computer will assign more weight to Trans-gender(2) over Female(1), and Female(1) over Male(0). Which is not ideal if vatiables are not ordinal or ranked. The computer assumes that trans-gender is greater than female, and that female is greater than male.-- increasing in value.
# Note:Label encoding should only be used on target variable, i.e. Y or dependent variable
# 
# One-Hot Encoding is a technique that uses binary vector for each category variable.The vector has a length equal to the number of categories, and only one element in the vector is set to 1, while the rest are set to 0.
# For example:
# Color          Color_Red  Color_Green   Color_Blue
# Red                1		0	           0
# Green		       0		1	           0
# Blue		       0		0	           1
# ```

# In[23]:


telco_data_dummies = pd.get_dummies(telco_data, dtype=int)
telco_data_dummies.head()


# **4.** Using scatter plot,create a relationship between Monthly Charges and Total Charges

# In[24]:


sns.lmplot(data=telco_data_dummies, x='MonthlyCharges', y='TotalCharges', fit_reg=False)


# **Result:** Total Charges increase as Monthly Charges increase - as expected.

# **5.** Churn by Monthly Charges and Total Charges

# In[25]:


# Distribution of MonthlyCharges for customers who churned and those who didn't.

Mth = sns.kdeplot(telco_data_dummies.MonthlyCharges[(telco_data_dummies["Churn"] == 0) ],
                color="Red", shade = True)
Mth = sns.kdeplot(telco_data_dummies.MonthlyCharges[(telco_data_dummies["Churn"] == 1) ],
                ax =Mth, color="Blue", shade= True)
Mth.legend(["No Churn","Churn"],loc='upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('Monthly Charges')
Mth.set_title('Monthly charges by churn')


# **Result:** The above graph shows that, Churn is high when Monthly Charges are high

# **6.** Using  KDE (Kernel Density Estimate) plot for TotalCharges, comparing customers who have churned versus those who have not, using Seaborn.

# In[26]:


Tot = sns.kdeplot(telco_data_dummies.TotalCharges[(telco_data_dummies["Churn"] == 0) ],
                color="Red", shade = True)
Tot = sns.kdeplot(telco_data_dummies.TotalCharges[(telco_data_dummies["Churn"] == 1) ],
                ax =Tot, color="Blue", shade= True)
Tot.legend(["No Churn","Churn"],loc='upper right')
Tot.set_ylabel('Density')
Tot.set_xlabel('Total Charges')
Tot.set_title('Total charges by churn')


# **Result**: Surprising insight, there is higher Churn at lower Total Charges
# 
# However if we combine the insights of 3 parameters i.e. **Tenure**, **Monthly Charges** and **Total Charges** then the picture is bit clear :- Higher Monthly Charge at lower tenure results into lower Total Charge. Hence, all these 3 factors viz Higher Monthly Charge, Lower tenure and Lower Total Charge are linkd to High Churn.

# **7.** Build a correlation of all predictors(features) with 'Churn'

# In[27]:


plt.figure(figsize=(20,8))
telco_data_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')


# **Result:**
# - HIGH Churn seen in case of **Month to month contracts**, **No online security**, **No Tech support**, **First year of subscription** and **Fibre Optics Internet**
# - LOW Churn is seen in case of **Long term contracts**, **Subscriptions without internet** **service** and The **customers engaged for 5+ years**
# - Factors like **Gender**, **Availability of PhoneService** and **# of multiple lines** have alomost **NO** impact on Churn.
# 
# This is also evident from the Heatmap below

# In[28]:


plt.figure(figsize=(12,12))
sns.heatmap(telco_data_dummies.corr(), cmap="Paired")


# **Bivariate Analysis**: 
# 
# This focus on the relationship between two 'features' in regards to a Label(churn) per time. Let's combine the relationship between two features and plot their distribution  predictor by Churn

# *****Spliting telco_data dataframe into churners and non churners, focus will be on churners(customers that have left, reasons why they left)**

# In[29]:


# Filtering the python dataframe into customers who have not churn(active customers)

new_df1_target0=telco_data.loc[telco_data["Churn"]==0]

#Filtering the python dataframe into customers who have churn(inactive customers)
new_df1_target1=telco_data.loc[telco_data["Churn"]==1]


# In[30]:


def uniplot(df,col,title,hue =None):
    
    sns.set_style('whitegrid')
    sns.set_context('talk')
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.titlepad'] = 30
    
    
    temp = pd.Series(data = hue)
    fig, ax = plt.subplots()
    width = len(df[col].unique()) + 7 + 4*len(temp.unique())
    fig.set_size_inches(width , 8)
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.title(title)
    ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index,hue = hue,palette='bright') 
        
    plt.show()


# In[31]:


# Gender distribution of churned subscribers in regards to gender
uniplot(new_df1_target1,col='Partner',title='Distribution of Gender for Churned Customers',hue='gender')


# In[32]:


#Gender Distribution of non-churners(active subscribers) in regards to gender
uniplot(new_df1_target0,col='Partner',title='Distribution of Gender for Non Churned Customers',hue='gender')


# In[33]:


# Distribution of differrent payment-method for inactive subscribers(churners) in regard to gender.

uniplot(new_df1_target1,col='PaymentMethod',title='Distribution of PaymentMethod for Churned Customers',hue='gender')


# In[34]:


# Distribution of different subscription contracts of churned customers in regards to gender

uniplot(new_df1_target1,col='Contract',title='Distribution of Contract for Churned Customers',hue='gender')


# In[35]:


#Distribution of subscribers who had access to tech suppot that have left(churned) per gender 
uniplot(new_df1_target1,col='TechSupport',title='Distribution of TechSupport for Churned Customers',hue='gender')


# In[36]:


#Distribution of SeniorCitizens who have left(churn) per gender

uniplot(new_df1_target1,col='SeniorCitizen',title='Distribution of SeniorCitizen for Churned Customers',hue='gender')


# **CONCLUSION**
# ```
# These are some of the quick insights from this exercise:
# 
# - Among churners, customers that used Electronic-Check payment method are the highest churners for both male and female. And we have more female churners under the category of Credit_card payment method.
# - Contract Type - Customers on monthly subscription have very high frequency of churn while customers on two years contract subscription have very low frequency of churn for both male and female.
# - We have high churners among customers that had no access to Tech_Support and few churners among customers that had No Internet Service for both male and female.
# - Customers that are Non senior Citizens are high churners  while customers that are senior citizens are likely not to churn for both male and female.
# ```

# ##### Save telco_data_dummies as a delta table

# In[37]:


# Replace invalid characters in column names
telco_data_dummies.columns = telco_data_dummies.columns.str.replace('[ ,;{}()\n\t=]', '_', regex=True)

# Convert pandas DataFrame to PySpark DataFrame
telco_data_spark = spark.createDataFrame(telco_data_dummies)

# Write the DataFrame as a Delta table
telco_data_spark.write.format("delta").mode("overwrite").saveAsTable("ABC_Telecom.clean_data")

