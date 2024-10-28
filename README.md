# ABC_Telecom Churn Analysis & Retention Report- A Microsoft Fabric Project using Synapse Data Engineer, Synapse Data Science and Power BI.

## Table of Content.
- [Bussiness Case](#business-case)
- [Problem Statement](#problem-statement)
- [Proposed Solution](#proposed-solution)
- [Solution Architecture](#solution-architecture)
- [Environment Setup](#environment-setup)
- [Tools Used](#tools-used)
- [Data Cleaning Preprocessing And Explorative Data Analysis](#data-cleaning-preprocessing-and-explorative-data-analysis)
- [Machine Learning models](#machine-learning-models)
- [Register Define ML model Structure And Create New Customers Data](#register-define-ml-model-structure-and-create-new-customers-data)
- [Model Deployment And Predictions](#model-deployment-and-predictions)
- [Project Benefits And Costs](#project-benefits-and-costs)
- [Risk Assessment and Mitigation](#risk-assessment-and-mitigation)
- [Insights And Analysis](#insights-and-analysis)
- [Recommendations](#recommendations)
- [Conclusion](#conclusion)

## Business Case: Telecom Churn Analysis & Retention
ABC Telecoms is facing a significant decline in revenue and market share due to increasing customer churn. To address this challenge, I propose a comprehensive churn analysis and retention strategy. By leveraging data-driven insights, we will accurately predict customer churn, identify at-risk customers, and implement targeted retention efforts to minimize customer attrition and maximize profitability.

## Problem Statement
ABC Telecom is experiencing substantial financial losses and operational challenges as a result of high customer churn. The inability to effectively predict and address customer churn is exacerbating this problem. This business case aims to develop a robust framework to mitigate the negative impacts of customer attrition.

## Proposed Solution
1. Collect relevant customer data from various sources, including billing records, usage patterns, demographic information, and customer satisfaction surveys.
2. Clean and preprocess the collected data to handle missing values, outliers, and inconsistencies, ensuring data quality and accuracy.
3. Conduct Exploratory Data Analysis (EDA): Analyze the data to identify trends, patterns, and correlations, gaining valuable insights into customer behavior and churn drivers.
4. Build a Predictive Churn Model: Develop a machine learning model using appropriate algorithms (e.g.,linear regression, logistic regression, random forest, decision tree) to predict customer churn probability. Select the model with the highest accuracy, such as 80%.
5. Perform Batch Prediction: Apply the trained model to new customers to assess their churn risk and proactively implement retention strategies.
6. Create a Power BI Dashboard: Visualize key performance metrics to track churn rates, customer lifetime value, and the effectiveness of retention efforts.
7. Recommendations.

## Solution Architecture.

![BBB](https://github.com/user-attachments/assets/92363eca-3df3-4029-b3cc-438d750e6b02)

## The following steps were taken in carrying out the following Churn and Retention Analysis

## Environment Setup
### Create and configure Microsoft Fabric Workspace for this project
Prerequisite: Enable Microsoft Fabric in Power BI Account as an Admin or Tenant.
- Go to (app.powerbi.com)
- Navigate to "workspaces" tab on the left
- At the bottom, click( + New Workspace )
  - A drop down at the top right; Enter name of workspace " Churn Analysis & Customer Retention "
  - Optional: In the description box, give detail description of project.
  - Scroll downward to "Advance" assign licensing to the workspace by clicking on "Trial", if you using trial version. Or " Premium Capacity", if you are using premium license.
  - Click Apply button

### Create and configure Storage in Fabric environment, i.e. Lakehouse Database.
Switch from Power BI environment to Synapse Data Engineering environment
- Click on the Power BI icon on the bottom left.
- Then, click on the "Data Engineering " Component
- Then, click on Lakehouse icon
- From the dropdown, "Name Lakehouse"- <ABC_Telecom>
- Click "create"

## Tools Used:
- Synapse Data Engineer
  - Data Cleaning and Preprocessing.
  - Explorative Data Analysis

- Synapse Data Science 
  - Building Machine learning model
  - Batch Prediction

- Power BI
  - Dashboard and Report Visualization.
 
# The following  task  were carried out in performing a Churn Analysis and Retension Strategy for ABC Telecom
 
## Create Lakehouse and Load raw csv file from local machine into the Lakehouse - file sub-section
- Click on created Lakehouse
- Under 'Explorer', click on 'File', then the three '...'
- From the dropdown, click 'upload', then choose 'upload file'
- On top right, click on file icon
- Select file from your local machine
- Click upload

## Add created lakehouse to each of the spark Notebook to be used on this project.
- In Synapse Data Engineer persona, to your  top-right click 'Notebook'
- In Notebook environment,On the left, click "Add Lakehouse" button.- This help in accessing the different tables and files that reside in the Lakehouse  directly from the Notebook.
- Choose "Existing Lakehouse".
- Click "Add".
- Check or choose the Lakehouse where the raw json data resides.
- Click "Add".
- From the imported Lakehouse Database to the left, click on "File " (-This shows all files that reside in the Lakehouse Database),then "..." , then "Load Data"
- There are two options (Spark or Pandas), Choose "Spark". A code is automatically generated to read the raw json file as a Pyspark DataFrame.


## Data Cleaning Preprocessing And Explorative Data Analysis
### Data Cleaning.
- Import necessary python libraries
- Read the csv file
- Fetch the shape of the data (numbers of rows and columns) and check if each column has the right data type and convert columns with wrong data type tothe right data type
- Fetch the summary statistics of each numerical column as well as number of churners against total number of customers.
- Fetch number of missing values from the dataset. Delete records with missing values because they are so small in size compare to the number ofrecords in the dataset, thus there impact is expected to be infinitesimal.
- Group the <tenure column> in to bins

### Explorative Data Analysis
- Perform Univariates Analysis of each of the 'Features' in respect to the 'Label'[churn]
- Convert [churn] into binary values
- Convert all non-numerical variables into dummy variables by applying One-Hot Encoding
- Perform Bivariates Analysis of the 'Features' in respect to the 'Label' [churn]
-  Analyse and draw business insights.
-  Write/Save 'One-Hot Encoded' data into created Lakehouse


## Machine Learning models
Using the saved 'One_Hot Encoding' dataset, the below ML model were developed with their performance metrics as follows 

![Screenshot 2024-10-28 150703](https://github.com/user-attachments/assets/07f97ccd-1227-4ce5-b661-471dadcb199f)

![Screenshot 2024-10-28 150548](https://github.com/user-attachments/assets/87d27c32-47e6-4e48-90c7-c1302cc9facb)

![Screenshot 2024-10-28 150638](https://github.com/user-attachments/assets/a2278bde-741f-4fd8-bde2-2f563ba08089)

![Screenshot 2024-10-28 150717](https://github.com/user-attachments/assets/0c9c49cb-8981-48cf-9da2-e53368bb454d)

#### From the above results, it can deduce that the classification model has the highest performance accuracy for our churn prediction among the four models as shown below

- Accuracy (0.80): Your model correctly predicts the outcome about 80% of the time.

- F1 Score (0.79): This balance between precision and recall is quite good, especially for imbalanced datasets.

- Log Loss (0.42): Lower log loss indicates better model performance. Your log loss shows that the model is fairly confident in its predictions.

- Precision (0.79): About 79% of the positive predictions made by your model are true positives.

- Recall (0.80): The model correctly identifies 80% of all actual positives.

- ROC AUC (0.85): A high ROC AUC indicates your model has a good ability to distinguish between classes.

#### Overall, these metrics suggest a well-performing model, though there’s always room for improvement. This model is selected for the 'Churn' prediction


## Register Define ML model Structure And Create New Customers Data
- Register the  ML classification model.
- Define model signature, specify the structure and data types of the input and output data of the model
- Creat new customers data to be later use for churn prediction
- Apply selected Classification model to New customer data is created


## Model Deployment And Predictions
- Registered model is apply to new customer for prediction.

## Project Benefits And Costs

### Benefits:
1. Reduced Churn: Significantly lower churn rates through accurate prediction and targeted retention.
2. Increased Customer Lifetime Value: Enhanced revenue generation from retaining high-value customers.
3. Improved Customer Satisfaction: Personalized retention strategies boost customer satisfaction and loyalty.
4.Optimized Operations: Data-driven insights enable efficient resource allocation and decision-making.

### Costs:
1. Microsoft Fabric Capacity
2. Data Acquisition: Costs associated with collecting customer data.
3. Retention Campaign Costs: Expenses related to implementing targeted retention strategies, such as promotional offers and customer support initiatives.

## Risk Assessment and Mitigation
### Potential Risks:
1. Data Quality Issues: Inaccurate or incomplete data can compromise model accuracy.
2. Model Accuracy: Model performance may vary over time due to changing customer behavior.
3. Resistance to Change: Employees may resist new processes or systems.

### Mitigation Strategies:
1. Data Quality Assurance: Implement robust data validation and cleaning procedures.
2. Model Refinement: Continuously monitor and update the model to maintain accuracy.
3. Change Management: Provide training and support to facilitate the adoption of new processes.
Return on Investment (ROI)
4. The expected ROI depends on factors such as reduced churn rates, increased customer lifetime value, and implementation costs. A detailed ROI analysis will quantify the financial benefits and justify the investment.

## Insights And Analysis:
1. #### Customer Demographics & Billing:
Non-senior citizens, customers without dependents, and those using electronic check or paperless billing show higher churn rates.
  
2. #### Contract Type:
Monthly contracts correlate with high churn, while long-term contracts (two years or more) see significantly lower churn.
  
3. #### Tenure Influence:
Churn is highest within the first 12 months and decreases with longer customer tenure.
  
4. #### Service & Support:
Lack of tech support, online security, and device protection are major churn drivers.
  
5. #### Internet Type:
Fibre optic internet users experience the highest churn compared to DSL or no internet service.
  
6. #### Charges:
Higher monthly charges are associated with higher churn, while total charges increase as monthly charges rise, as expected.

## Recommendations:
Here are some recommendations based on the insights gathered:

1. #### Promote Long-Term Contracts:
Encourage customers to choose yearly or two-year plans by offering discounts or additional benefits, as longer contracts are linked to lower churn rates.

2. #### Enhance Onboarding Support:
Provide targeted support, particularly during the first 12 months, to reduce early-stage churn. Implement welcome calls, personalized check-ins, and proactive engagement to help customers fully utilize their services.

3. #### Expand Customer Service Options:
Increase access to tech support, online security, and device protection services. Consider bundling these with internet packages to improve customer retention.

4. #### Offer Personalized Billing and Payment Options:
Given higher churn rates among electronic check and paperless billing users, introduce flexible billing cycles, personalized payment methods, and incentives for timely payments.

5. #### Introduce Discounts on High Monthly Charges:
For customers with higher monthly charges, offer loyalty discounts or tiered pricing options to increase affordability and satisfaction.

6. #### Invest in Retention for Fibre Optic Users:
Since fibre optic users have higher churn, consider exclusive promotions, loyalty rewards, and service quality enhancements for this segment to boost satisfaction and retention.

## Conclusion
By implementing this comprehensive churn analysis and retention strategy, ABC Telecom can significantly improve customer retention, increase profitability, and gain a competitive advantage in the telecom market. The proposed solution provides a data-driven approach to address the challenges of customer churn and maximize long-term success.







