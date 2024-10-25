# Telco Churn Analysis & Retention Report- A Microsoft Fabric Project using Synapse Data Engineer, Synapse Data Science and Power BI.

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
7. Recommendations

## Benefits and Costs
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


## Insights
1. Descriptive Statistics Insights
- Total Subscribers: ABC_Telecom has 7,043 subscribers.
- Customer Tenure: Maximum tenure is 72 months, with an average of 32.4 months.
- 75% of subscribers have stayed less than 55 months; 50% less than 29 months.
- Monthly Charges: Average monthly charge per customer is $64.67, with only 25% paying more than $89.85.
- Total Charges: Average total charges are $2,283.30, with 75% of customers spending under $3,794.74.

2. Customer Status
- Active vs. Churned: Of 7,043 subscribers, 5,174 are active and 1,869 have churned (inactive), meaning 73.46% remain active, while 26.53% have left

3. Tenure Group Breakdown
- Tenure distribution shows higher churn in the initial months:
  - 1-12 months: 2,175
  - 61-72 months: 1,407
  - Other groups range from 762 to 1,024.
 
4. Partner Status and Churn
- Marital Status: More married customers are retained compared to those without a partner, with higher churn among customers with no partner.

5. Dependent Status
- Dependents: Customers without dependents are more likely to churn compared to those with dependents.
6. Internet Service Type
- Internet Service: Fibre optic users have the highest churn rates compared to DSL users.

7. Security & Device Protection
- Additional Services: High churn rates among customers without online security, backup, or device protection.

8. Billing Preferences
- Paperless Billing: Churn is higher among those who use paperless billing.

9. Tenure and Churn Rates
- Early Tenure Churn: Highest churn in the 1-12 month group, lowest churn in the 60-72 month group.

10. Monthly vs. Total Charges
- Charges Relationship: Total charges increase with monthly charges, as expected.

11. Charges and Churn
- High Charges and Churn: Churn rates increase with higher monthly charges.

12. High vs. Low Churn Characteristics
- High Churn: Observed with month-to-month contracts, no online security, no tech support, initial subscription year, and fibre optic internet.
- Low Churn: Noted with long-term contracts, no internet service, and customers with 5+ years tenure.
- Neutral Factors: Gender, phone service availability, and multiple lines have minimal impact on churn.
13. Demographic and Payment Insights
- Payment Method: Highest churn with electronic check payment; more female churners in the credit card category.
- Contract Type: Monthly contracts see high churn, while two-year contracts have low churn.
- Tech Support & Internet Service: High churn among customers without tech support; low churn among those without internet.
- Age Group: Non-senior citizens are more likely to churn, while seniors show lower churn rates across genders.





## Recommendations

## Conclusion
By implementing this comprehensive churn analysis and retention strategy, ABC Telecom can significantly improve customer retention, increase profitability, and gain a competitive advantage in the telecom market. The proposed solution provides a data-driven approach to address the challenges of customer churn and maximize long-term success.

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
 
## Load raw csv file from local machine into Fabric environment, into Lakehouse - file subsection
- Click on created Lakehouse
- Under 'Explorer', click on 'File', then the three '...'
- From the dropdown, click 'upload', then choose 'upload file'
- On top right, click on file icon
- Select file from your local machine
- Click upload

## Add  lakehouse to your  Synapse Data Engineering Notebook
- In Synapse Data Engineer persona, to your  top-right click 'Notebook'
- In Notebook environment,On the left, click "Add Lakehouse" button.- This help in accessing the different tables and files that reside in the Lakehouse  directly from the Notebook.
- Choose "Existing Lakehouse".
- Click "Add".
- Check or choose the Lakehouse where the raw json data resides.
- Click "Add".
- From the imported Lakehouse Database to the left, click on "File " (-This shows all files that reside in the Lakehouse Database),then "..." , then "Load Data"
- There are two options (Spark or Pandas), Choose "Spark". A code is automatically generated to read the raw json file as a Pyspark DataFrame.










