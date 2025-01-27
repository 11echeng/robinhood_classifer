# Data Science & Engineering Portfolio
This code repository is for the Data Science and Engineering OA's. This repository stores Robinhood, Klaviyo, and MLB projects.

### RH Problem:

In order to improve user retention and lower churn, the growth team at Robinhood is interested in understanding why and which users withdraw money from their Robinhood account.  A user is considered churn when their equity value (amount of money in Robinhood account) falls below $10 for a period of 28 consecutive calendar days or longer.

### RH Data:

features_data.csv - contains user level data such as:
* user_id - unique id for every user
* risk_tolerance - self-reported risk tolerance of the user
* investment_experience - self-reported investment experience of the user
* liquidity_needs - self-reported liquidity needs of the user
* time_horizon - self-reported investment time horizon of the user
* platform - which platform (iOS or Android) the user is on
* time_spent - amount of time spent on the app
* first_deposit_amount - $ value of the amount first deposited
* instrument_type_first_traded - type of instrument first traded


equity_value_data.csv - contains user_id and equity_value for user along with timestamps for days when the user's equity value is greater than or equal to $10.

### RH Questions:

* Q1. What percentage of users have churned in the data provided?
* Q2. Build a classifier that given a user with their features assigns a churn probability for each user and predicts which users will churn. Based on the classifier output classify each user in the dataset as churned or not churned. What's the AUC score of your test set?
* Q3. What's the distribution of the given feature that has the highest correlation with churn (Right/left skewed, normal, multinomial distributions)
* Q4. What are the top 3 features? List the most important features that correlate to user churn

### Klaviyo Questions:

* Q1. Assemble a dataframe with one row per customer and the following columns: customer_id, gender, most_recent_order_date, order_count (number of orders placed by this customer). Sort the dataframe by customer_id ascending and display the first 10 rows.
* Q2. Plot the count of orders per week for the store.
* Q3. Compute the mean order value for gender 0 and for gender 1. Do you think the difference is significant? Justify your choice of method.
* Q4. Generate a confusion matrix for the gender predictions of customers in this dataset. You should assume that there is only one gender prediction for each customer. What does the confusion matrix tell you about the quality of the predictions?
* Q5. Describe one of your favorite tools or techniques and give a small example of how it's helped you solve a problem. Limit your answer to one paragraph, and please be specific.
* Note: For each question, state any considerations or assumptions you made.

### MLB Questions:

Consider data set 1 (ds1.csv). The data set comprises features (the Five xs) along with three sequences
that may or may not be generated from the features (3 ys).
Note: Consider data set 1 (ds1.csv). The data set comprises features (the Five xs) along with three sequences
that may or may not be generated from the features (3 ys).
* Q1. Describe the data set in a few sentences. E.g. What are the distributions of each feature?
Summary statistics?
* Q2. Try to come up with a predictive model, e.g. y = f(x_1 , … , x_n) for each y sequence. Describe
your models and how you came up with them. What (if any) are the predictive variables?
How good would you say each of your models is?
* Q3. Consider data set 2 (ds2.csv). The dataset comprises a set of observations that correspond to multiple
groups. Describe the data in a few sentences. How would you visualize this data set? Can you identify the number of groups in the data and assign each row to its group? Can you create a good visualization of your groupings?
<br></br>
NOTE: sql ide is in [Stack Overflow](https://data.stackexchange.com/stackoverflow/query/new):
* Q4. How many posts were created in 2017?
* Q5. What post/question received the most answers?
* Q6. For posts created in 2020, what were the top 10 tags?
* Q7. For the questions created in 2017, what was the average time (in seconds) between

### Amazon Questions:
* Q1. Write a query to identify Merchant that got churned out for Feb 2017?
Also, how would you write in dynamic way to identify churn for any given month and year?
Definition of Churn – Merchant having at least 1 order for 2016 and 0 order for 2017

* Q2. Write a query to report WOW (Week over Week) sales for each product category for trailing 6 weeks?

### Dicks Questions:
* Q1. See imgs directory

* Q2. See imgs directory

### Getting Started:
Check out this project repo, then add an .env credentials, case-sensitive, to the root if not there already:
```
HOST="robinhood.cjomywuwikjl.us-east-2.rds.amazonaws.com"
DATABASE="postgres"
USER="postgres"
PASSWORD="password"
PORT="5432"
```
In VScode terminal, run this command:
```
pc: python -m venv .venv
macOS: python3 -m venv .venv
```
In VScode terminal, run this command:
```
pc: .venv\Scripts\activate
macOS: source .venv/bin/activate
```
While the virutal environment is running, run the following command:
```
pip install -r requirements.txt
```
 Packages include:
```
polars - parallel distribution dataframe library
pandas - single distribution dataframe library
numpy - matrix library
pytorch - neural network library
scikit-learn - standard ml library
matplotlib - visualization library
seaborn - visualization library
imbalanced-learn - handling imabalnce dataset library
xgboost - gradient boosting library
lightgbm - gradient boosting library
mlflow - manage ml lifecycle platform
pyspark - distributed computing framework
shap - model interpretability library
optuna - hyperparameter optimization library
featuretools - automated feature engineering library
sqlalchemy - ORM to connect aws postgresql db library
```
After the installs complete, run any notebook

### AWS RDS PostgreSQL Access Setup:
Navigate to "EC2" in the [AWS](https://us-east-2.console.aws.amazon.com/console/home?region=us-east-2):
```
aws account: eric.cheng003c@gmail.com
aws pw: secret secret
```
Under "Network & Security", click "Security Groups". Select it and click "Edit inbound rules" on the bottom to add new rule:
```
Type: PostgreSQL
Protocol: TCP
Port Range: 5432
Source: Choose "My IP"
```
Click "Save Changes"