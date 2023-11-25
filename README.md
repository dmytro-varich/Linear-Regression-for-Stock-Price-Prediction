# Linear Regression for Stock Price Prediction

This documentation is dedicated to the exploration and application of supervised machine learning methods, with a focus on linear regression, in the context of forecasting stock prices.

> With our goal set on developing and evaluating a model for predicting stock prices one month ahead, we emphasize the utilization of linear regression within the realm of machine learning. Our aim is to uncover patterns and forecast future values based on historical data.

## Introduction 📃

Explore the development of a stock price prediction model using [Python](https://www.python.org/) in this documentation. Delve into the breakdown of essential sections within this document, mirroring my work in **Jupyter Notebook**. Covering topics from the **Linear Regression 📈** to **Dataset Exploration 📊**, **Correlation Analysis 🧪**, and **Implementation Details⚙️**, the document is a reflection of my three days of effort.

> I chose linear regression for its simplicity and interpretability, making it a suitable model to understand and predict stock prices. Its straightforward nature aligns well with the objective of uncovering patterns in historical data and forecasting future values within the given context.

### Table of contents

1. [Introduction](#Introduction)

2. [Linear Regression](##Linear-Regression)

3. [Dataset Exploration](#Dataset-Exploration)

4. [Correlation](#Correlation)

5. [Implementation Details](#Implementation-Details)

6. [Conclusion](#Conclusion)

7. [Author](#Author)

## Linear Regression 📈

**Linear regression** is a fundamental supervised learning algorithm used for predicting a continuous target variable (or dependent variable) based on one or more input features (independent variables). It models the relationship between the input features and the target variable as a linear equation. Linear regression is widely used in various fields, such as economics, finance, biology, and engineering.

In linear regression, the goal is to find the best-fitting linear equation that minimizes the error between the predicted values and the actual target values. The basic components of linear regression include:

- **Input Features (X)**: These are the independent variables, also known as predictors or features. They represent the data used to make predictions.

- **Target Variable (Y)**: This is the dependent variable that we want to predict based on the input features.

- **Linear Equation**: The linear equation that describes the relationship between the input features and the target variable is represented as follows:

  - Y = β₀ + β₁ _ X₁ + β₂ _ X₂ + ... + βₖ \* Xₖ + ε

  - Y is the predicted target variable.

  - β₀, β₁, β₂, ..., βₖ are the model coefficients that need to be estimated during training.

  - X₁, X₂, ..., Xₖ are the input features.

  - ε represents the error term, which accounts for the difference between the predicted and actual values.

- **Objective**: The objective in linear regression is to find the values of β₀, β₁, β₂, ..., βₖ that minimize the sum of squared errors (SSE) or mean squared error (MSE) between the predicted values and the actual target values.

**Implementing linear regression**

- Data Preparation: Prepare the dataset, ensuring it includes input features (X) and the target variable (Y). Split the data into training and testing sets.

- Model Creation: Create a linear regression model using the appropriate library (e.g., scikit-learn).

- Model Training: Train the model using the training data to estimate the coefficients (β₀, β₁, β₂, ..., βₖ).

- Model Evaluation: Evaluate the model's performance using metrics like mean squared error (MSE), root mean squared error (RMSE), and coefficient of determination (R²).

- Prediction: Use the trained model to make predictions on new or unseen data.

## Dataset Exploration 📊

To address the task of data analysis and linear regression model creation, I conducted a search for a suitable dataset using parameters and filters in the data search engine. One of the primary criteria was the tag "linear regression", which was applied in the data search engine to refine the results.

Upon further inquiry using the keyword "Stock Price", I discovered and selected a dataset on the [Kaggle](https://www.kaggle.com/datasets/varpit94/apple-stock-data-updated-till-22jun2021) containing historical data on the stock prices of Apple Inc. (AAPL) from 1980 to 2020. This dataset provides daily-level information and measures the data in US dollars.

---

**Dataset Characteristics**

The chosen dataset consists of over _10,000 rows_ and includes the following _7 columns_:

1. **Date:** Date
2. **Open:** Opening stock price
3. **High:** Maximum stock price for the day
4. **Low:** Minimum stock price for the day
5. **Close:** Closing stock price
6. **Adj Close:** Adjusted closing stock price
7. **Volume:** Trading volume

The data offers the opportunity for analysis and modeling, allowing for a detailed exploration of the stock price dynamics of Apple Inc. over the specified time period. Below is the **tabular** structure of the dataset:

```python
df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1980-12-12</td>
      <td>0.128348</td>
      <td>0.128906</td>
      <td>0.128348</td>
      <td>0.128348</td>
      <td>0.100323</td>
      <td>469033600</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1980-12-15</td>
      <td>0.122210</td>
      <td>0.122210</td>
      <td>0.121652</td>
      <td>0.121652</td>
      <td>0.095089</td>
      <td>175884800</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1980-12-16</td>
      <td>0.113281</td>
      <td>0.113281</td>
      <td>0.112723</td>
      <td>0.112723</td>
      <td>0.088110</td>
      <td>105728000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1980-12-17</td>
      <td>0.115513</td>
      <td>0.116071</td>
      <td>0.115513</td>
      <td>0.115513</td>
      <td>0.090291</td>
      <td>86441600</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1980-12-18</td>
      <td>0.118862</td>
      <td>0.119420</td>
      <td>0.118862</td>
      <td>0.118862</td>
      <td>0.092908</td>
      <td>73449600</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10404</th>
      <td>2022-03-18</td>
      <td>160.509995</td>
      <td>164.479996</td>
      <td>159.759995</td>
      <td>163.979996</td>
      <td>163.979996</td>
      <td>123351200</td>
    </tr>
    <tr>
      <th>10405</th>
      <td>2022-03-21</td>
      <td>163.509995</td>
      <td>166.350006</td>
      <td>163.009995</td>
      <td>165.380005</td>
      <td>165.380005</td>
      <td>95811400</td>
    </tr>
    <tr>
      <th>10406</th>
      <td>2022-03-22</td>
      <td>165.509995</td>
      <td>169.419998</td>
      <td>164.910004</td>
      <td>168.820007</td>
      <td>168.820007</td>
      <td>81532000</td>
    </tr>
    <tr>
      <th>10407</th>
      <td>2022-03-23</td>
      <td>167.990005</td>
      <td>172.639999</td>
      <td>167.649994</td>
      <td>170.210007</td>
      <td>170.210007</td>
      <td>98062700</td>
    </tr>
    <tr>
      <th>10408</th>
      <td>2022-03-24</td>
      <td>171.059998</td>
      <td>174.139999</td>
      <td>170.210007</td>
      <td>174.070007</td>
      <td>174.070007</td>
      <td>90018700</td>
    </tr>
  </tbody>
</table>
<p>10409 rows × 7 columns</p>
</div>

# Correlation 🧪

To uncover patterns and determine suitable values for use as input features (X), I conducted a correlation analysis using the [Seaborn](https://seaborn.pydata.org/) library (Below is the concise code). It generates a _heatmap_ illustrating the correlation coefficients between respective pairs of variables. Darker shades of red indicate stronger correlation, while lighter shades of blue may suggest weak or no correlation. In my case, correlations exist among all parameters except for "value", which seems to correlate only with itself.

**Correlation Scale:**

- **1.0:** Positive Correlation: Positive relationship between variables. An increase in one variable is accompanied by an increase in the other.
- **0.0:** Neutral Correlation: Absence of a linear relationship between variables.
- **-1.0:** Negative Correlation: Inverse relationship between variables. An increase in one variable is accompanied by a decrease in the other.

```python
import seaborn as sns

# Create a correlation matrix
correlation_matrix = df[['Open', 'High', 'Low', 'Adj Close', 'Volume']].corr()

# Building a heat map
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
```

![png](output_3_0.png)

# Implementation Details ⚙️

After loading the dataset and conducting a correlation analysis among various parameters, I proceeded to implement the task of stock price forecasting. In this context, I utilized necessary libraries such as `pandas`, `matplotlib`, and `sklearn`.

Firstly, the Apple stocks dataset was loaded using the `pandas` library. Subsequently, to create predictions, I introduced a new column named "Predict", which contains anticipated closing prices for the next 30 days. This was achieved by shifting the values of the "Adj Close" column 30 rows forward.

Next, to prepare the data for model training, I extracted _features_ and the _target variable_. Features included the parameters "Open", "High", "Low", and "Adj Close", while "Predict" became the target variable.

For model training, I opted to use linear regression from the `scikit-learn` library. For enhanced model performance, the data was standardized using `StandardScaler`.

Subsequently, the dataset was split into training and testing sets using `train_test_split`. The model was trained on the training data, and predictions were generated on the test set (20% of the data will be allocated for testing the model, and the remaining 80% will be used for training).

Performance evaluation of the model included calculating the _Mean Squared Error_ and the _R-squared coefficient_. These metrics provide insights into the accuracy and explanatory power of the model.

Following model training and evaluation, _regression coefficients_ and _performance metrics_ were displayed. For illustrative purposes, a graph was also created, showcasing actual data points alongside the regression line.

```python
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Load the Apple stocks dataset
path = r"C:\Users\admin\Desktop\MachineLearning\AAPL.csv"
df = pd.read_csv(path)


# Adding the 'Predict' column with predicted closing prices for the next 30 days
projection = 30
df['Predict'] = df["Adj Close"].shift(-projection)


# Extract features and target variable
X = df[['Open', 'High', 'Low', 'Adj Close']]
Y = df['Predict']
X, Y = X[:-projection], Y[:-projection]


# Split the dataset into a training set and a test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# Create a linear regression model and use standardization
model = make_pipeline(StandardScaler(), LinearRegression())


# Train the model on the training data
model.fit(X_train, Y_train)


# Make predictions on the test data
Y_pred = model.predict(X_test)


# Evaluate the model's performance
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)


# Print the model's coefficients and performance metrics
print('Intercept:', model.named_steps['linearregression'].intercept_)
print('Coefficients:', model.named_steps['linearregression'].coef_)
print("Mean Squared Error:", mse)
print("R-squared:", r2)


# Plot the regression line
plt.scatter(X_test['Adj Close'], Y_test, color='orange', label='Actual Data')
plt.plot(X_test['Adj Close'], Y_pred, color='c', linewidth=3, label='Regression Line')
plt.xlabel("Input Feature")
plt.ylabel("Target Variable")
plt.legend()
plt.show()
```

    Intercept: 13.49192846224256
    Coefficients: [-13.33603907 -10.69745852  20.81849554  33.0263593 ]
    Mean Squared Error: 10.117849269866701
    R-squared: 0.9886313790563631

![png](output_5_1.png)

# Conclusion 📓

In conclusion, it can be confidently stated that linear regression is suitable for price forecasting, as demonstrated by our results boasting a **98%** accuracy. However, my initial objective was to predict stock returns, for which I created a column named "Return". It was calculated using the formula:

$$
\text{Return} = \frac{{\text{price}_{\text{(t+30)}} - \text{current price}}}{{\text{current price}}} \times 100
$$

Despite employing various regression models, such as Support Vector Regression (SVR) and Lasso, I encountered significant challenges. The mean squared error soared to nearly 252, with a negative R-squared score.

Upon experimenting with alternative approaches, I shifted my focus to predicting prices first and subsequently calculating returns. This revealed a deviation issue, even with a commendable 98% accuracy. Notably, the deviation in returns was substantial, with an 80% difference between predicted and actual returns on a decimal basis.

Subsequently, I sought a less precision-dependent parameter and decided to predict whether to buy, sell, or hold. Initially, I attempted to integrate this decision-making process with regression, incorporating additional if statements. However, precision requirements surpassed 98%. After delving into literature, I opted for Logistic Regression. The model predicted overall accuracy of 68%. Closer inspection revealed challenges in predicting sales (0%), while holding and buying were around 60%.

Ultimately, my efforts to diversify and enhance the utility of the program proved unsuccessful. I chose to maintain the program's primary success in forecasting stock prices through the application of logistic regression.

# Author 🧑🏻

My name is Dmytro Varich, and I am a student at [TUKE](https://www.tuke.sk/wps/portal) University, majoring in Intelligent Systems. This article is intended for the completion of Assignment 1 in the subject of Artificial Intelligence. Similar content is also shared on my [Telegram](https://t.me/varich_channel) channel.

Email: [dmytro.varich@student.tuke.sk](mailto:dmytro.varich@student.tuke.sk)

This documentation was also written with the intention of delving deeper into the field of Machine Learning.
