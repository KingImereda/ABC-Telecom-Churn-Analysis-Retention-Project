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








