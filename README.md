# Forecasting Time Series Data with SARIMAX + SARIMA Hyperparameter Tuning

Time series forecasting plays a critical role in various fields such as finance, economics, weather forecasting, and more. One powerful tool in the time series forecasting toolbox is SARIMAX, which stands for Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors. In this article, we will guide you through the process of applying SARIMAX to forecast time series data.

**Step 1: Understand the Data**

Before diving into modeling, it's crucial to familiarize yourself with the time series data you're working with. This involves understanding any existing patterns, trends, and potential seasonality in the data.

**Step 2: Install Necessary Libraries**

Ensure you have the required Python libraries installed. These typically include `pandas` for data manipulation, `statsmodels` for time series analysis, and `matplotlib` for plotting.


```python
# Import necessary libraries for data manipulation and analysis
import numpy as np # Numerical operations
import pandas as pd # Data manipulation
from datetime import datetime, timedelta # Date operations

# Import libraries for plotting and visualization
import matplotlib.pyplot as plt # Matplotlib for basic plotting
import seaborn as sns # Seaborn for enhanced visualization
import plotly.express as px # Plotly for interactive plots

# Import libraries for time series analysis and modeling
import yfinance as yf # Yahoo Finance for retrieving financial data
from statsmodels.tsa.stattools import adfuller # Augmented Dickey-Fuller Test for stationarity check
from statsmodels.tsa.statespace.sarimax import SARIMAX # SARIMAX model for time series forecasting
from statsmodels.tsa.seasonal import seasonal_decompose # Seasonal decomposition for trend, seasonal, and residual components
```

**Step 3: Load and Preprocess Data**

Load the time series data into a suitable data structure, such as a pandas DataFrame or a numpy array. Check for missing values and outliers, and handle them appropriately, either by imputing missing values or removing outliers.


```python
# Define the stock ticker symbol
ticker = 'AMD'
# Download the data
df = yf.download(ticker, start='2022-06-24', end='2023-06-24') # Download data from Yahoo Finance
# Print the first few rows of the data
print(df.head()) # Display the first few rows of the downloaded data
```

    [*********************100%%**********************]  1 of 1 completed
                     Open       High        Low      Close  Adj Close     Volume
    Date                                                                        
    2022-06-24  83.559998  87.529999  83.080002  87.080002  87.080002   88553900
    2022-06-27  87.360001  88.220001  85.250000  86.160004  86.160004   74663500
    2022-06-28  85.709999  86.730003  80.430000  80.779999  80.779999   95618600
    2022-06-29  79.550003  79.750000  76.510002  77.989998  77.989998  104140900
    2022-06-30  77.730003  78.910004  75.480003  76.470001  76.470001  105368600


**Data Preprocessing and Column Selection**


```python
# Add a 'Date' column based on the index
df["Date"] = df.index
# Reset the index and drop the old index column
df.reset_index(drop=True, inplace=True)
# Keep only the 'Date' and 'Close' columns
df = df[['Date', 'Close']]
```

**Step 4: Visualize the Data**

Plotting the time series data provides a visual understanding of its characteristics. This can help in identifying trends, seasonality, and any unusual patterns.


```python
# Create the line plot
fig1 = px.line(df, x='Date', y='Close')
# Set the title using the ticker value
fig1.update_layout(title=f"Stock Price for {ticker}")
# Display the plot
fig1.show()
```

![Alt Text](https://i.ibb.co/XsSJ576/newplot.png)


**Step 5: Stationarity**

Stationarity is a crucial concept in time series forecasting. It refers to a property of a time series where the statistical properties (such as mean, variance, and autocovariance) remain constant over time. In other words, a stationary time series does not exhibit trends or seasonality in its behavior.
Here are several reasons why stationarity is important in time series forecasting:
1. Statistical Validity:
 - Many time series models, including ARIMA and SARIMAX, assume stationarity. If the data is not stationary, the results from these models may not be reliable or interpretable.
2. Model Assumptions:
 - Stationarity is a key assumption for various time series models. When the data is non-stationary, it may violate these assumptions and lead to inaccurate forecasts.
3. Mean and Variance Stability:
 - In a stationary time series, the mean and variance are constant over time. This makes it easier to understand and interpret the data, as there are no major shifts or fluctuations in these properties.
4. Forecasting Accuracy:
 - Stationary data often leads to more accurate and reliable forecasts. Models trained on stationary data can better capture the underlying patterns and relationships, leading to more accurate predictions.
5. Removal of Trends and Seasonality:
 - Non-stationary data often contains trends or seasonal patterns. By making the data stationary through techniques like differencing or detrending, we can isolate the underlying patterns from the trend or seasonality.
6. Easier Interpretation:
 - Stationary time series are easier to interpret and analyze. Patterns, relationships, and anomalies are more apparent when the data exhibits stationarity.
7. Time-Invariance:
 - Stationarity implies that the time series is time-invariant. This means that the behavior of the series is consistent regardless of when the observations were made.
8. Model Stability:
 - Stationary time series are typically more stable over time. This means that the behavior of the time series is less likely to change dramatically in the future.
9. Mean Reversion:
 - In finance and economics, stationarity is often associated with mean reversion, where a series tends to revert back to its mean over time. This can have important implications for investment strategies.
In summary, ensuring stationarity in a time series is a critical step in preparing the data for accurate forecasting. It helps to meet the assumptions of many forecasting models, leads to more reliable predictions, and facilitates a better understanding of the underlying patterns in the data.

**Decomposing time series data**

Decomposing time series data into its constituent components (trend, seasonality, and residuals) is a crucial step in understanding and modeling time-dependent patterns. This decomposition helps identify underlying patterns and enables more accurate forecasting. Here's a detailed explanation of each component:
**Trend Component**

The trend component represents the long-term progression or direction of the data. It captures the overall pattern that is not attributable to seasonal or cyclical fluctuations. Trends can be increasing (upward), decreasing (downward), or stable (horizontal).

Methods for Identifying Trends:

- Moving Averages: Calculating the average of a fixed-size window of data points to smooth out short-term fluctuations and highlight the overall trend.
- Exponential Smoothing: Assigning exponentially decreasing weights to past observations, giving more importance to recent data.
- Regression Analysis: Fitting a regression model to the time series data to estimate the trend component.
Seasonality Component
The seasonality component captures patterns that repeat at regular intervals, typically within a year. For example, retail sales often exhibit higher numbers during holiday seasons. Seasonality can be additive (constant amplitude) or multiplicative (varying amplitude).
Methods for Identifying Seasonality:
- Seasonal Decomposition of Time Series (STL): This method decomposes time series data into trend, seasonal, and residual components using a seasonal-trend decomposition procedure based on loess.
- Periodogram Analysis: Examining the frequency domain of the data to identify dominant seasonal frequencies.
Residual Component
The residual component, also known as the error or noise, represents the random, unexplained variation in the data after accounting for the trend and seasonality. It contains any irregularities, noise, or unexpected events that cannot be attributed to the trend or seasonal patterns.
Methods for Identifying Residuals:
- Residuals are typically obtained by subtracting the estimated trend and seasonality components from the original time series data.
Decomposition Process
The process of decomposing a time series involves separating the observed data (Xt) into its constituent components:
The resulting trend, seasonality, and residuals are then used to model and forecast future values.
 Importance of Decomposition
- Improved Understanding: Decomposition provides insight into the underlying patterns and dynamics of the time series data.
- Enhanced Forecasting: Analyzing and modeling individual components allows for more accurate and interpretable forecasts.
- Anomaly Detection: Residuals can help identify unusual or unexpected events in the data.
By decomposing time series data, analysts can gain a deeper understanding of the patterns within the data, allowing for more informed decision-making and better forecasting capabilities. This process is a fundamental step in time series analysis.

# Additive and Multiplicative Seasonality
Additive and multiplicative seasonality are two different ways in which seasonal patterns can be expressed in a time series. Understanding the nature of seasonality (whether additive or multiplicative) is crucial for accurate modeling and forecasting.

**Additive Seasonality**

In an additive seasonal pattern, the seasonal component is added to the trend and error terms. Mathematically, it can be represented as:
Characteristics:
 - The seasonal fluctuations have a constant amplitude (the same amount of fluctuation occurs regardless of the level of the time series).
 - The impact of seasonality is consistent over time.

 - For example, if you're looking at monthly ice cream sales, an additive seasonal pattern would imply a consistent increase in sales during summer months, regardless of the overall level of sales.
 
![Alt Text](https://i.ibb.co/3kpL6rM/1.png)



# Multiplicative Seasonality

In a multiplicative seasonal pattern, the seasonal component is multiplied with the trend and error terms. Mathematically, it can be represented as:
- Characteristics:
 - The seasonal fluctuations are proportional to the level of the time series. As the level of the time series increases, so does the seasonal effect.
 - The impact of seasonality grows with the level of the time series.

 - Using the ice cream sales example, if there's multiplicative seasonality, it would mean that during high sales months, the increase in sales is proportionally larger compared to lower sales months.

Example

![Alt Text](https://i.ibb.co/g7gZzgH/2.png)

**Determining Seasonality Type**
Deciding whether seasonality is additive or multiplicative depends on analyzing the data and visual inspection. It can also be determined through statistical methods and model diagnostics.

It's worth noting that incorrectly identifying the type of seasonality can lead to inaccurate forecasts. Therefore, a careful examination of the data and consideration of the underlying mechanisms driving the seasonality are crucial steps in the time series analysis process.

Understanding whether seasonality is additive or multiplicative helps in selecting appropriate models, as different models are better suited for handling different types of seasonality.

**Transformation from Multiplicative to Additive**

Multiplicative time series data can often be transformed into additive time series through methods like taking logarithms. This transformation can be advantageous in specific situations. When a seasonal time series exhibits multiplicative seasonality, it can be advantageous to convert it into an additive series by applying a logarithmic transformation. This transformation changes the multiplicative elements into additive components, resulting in stabilized variance. This logarithmic transformation is particularly valuable when the seasonal component's amplitude is proportionate to the level of the series, resulting in a varying seasonal amplitude. The transformation effectively stabilizes the fluctuation in seasonal amplitude, facilitating more straightforward forecasting.

# Augmented Dickey-Fuller (ADF)

The Augmented Dickey-Fuller (ADF) test is a statistical hypothesis test used to determine whether a unit root is present in a univariate time series dataset. In simpler terms, it helps us assess whether a time series is stationary or non-stationary.

**Purpose of the ADF Test**

The presence of a unit root in a time series indicates non-stationarity. Non-stationary data can exhibit trends, which can lead to inaccurate forecasts when using models that assume stationarity. The ADF test is employed to assess whether differencing the data (to achieve stationarity) is necessary before applying certain time series models.

**How the ADF Test Works**

1. Null Hypothesis : The null hypothesis of the ADF test is that the time series has a unit root, indicating it is non-stationary.
2. Alternative Hypothesis: The alternative hypothesis is that the time series is stationary (i.e., it does not have a unit root).
3. Test Statistic: The ADF test statistic is computed. This statistic is used to compare against critical values to determine the likelihood of rejecting the null hypothesis.
4. Critical Values: The ADF test provides critical values at various confidence levels. These critical values depend on the sample size and the chosen significance level.
5. Decision: Based on the test statistic and critical values, you can decide whether to reject or fail to reject the null hypothesis.

**Interpretation**

Rejecting the Null Hypothesis: If the test statistic is less than the critical value, you reject the null hypothesis. This suggests that the time series is stationary, and differencing may not be necessary.

Failing to Reject the Null Hypothesis: If the test statistic is greater than the critical value, you fail to reject the null hypothesis. This implies that the time series may be non-stationary and differencing could be beneficial.


```python

def perform_adf_test(data):
    # Perform ADF test
    result = adfuller(data)

    # Print the results
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])

    # Check if the data is stationary based on the p-value
    if result[1] <= 0.05:
        print("The data is stationary")
    else:
        print("The data is not stationary, Data can be processed further")

# Example Usage
perform_adf_test(df['Close'])

```

    ADF Statistic: -2.013213174219679
    p-value: 0.28080980278926104
    Critical Values: {'1%': -3.4577787098622674, '5%': -2.873608704758507, '10%': -2.573201765981991}
    The data is not stationary


# Step 6: Differencing (if needed)
If the data is not stationary, apply differencing to make it stationary. This process involves subtracting the previous value from the current value.


```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def plot_differencing_acf_pacf(data):
    fig, axes = plt.subplots(3, 2, sharex=True)

    # Original Series
    axes[0, 0].plot(data)
    axes[0, 0].set_title('Original Series')
    plot_acf(data, ax=axes[0, 1])

    # 1st Differencing
    diff_1 = data.diff().dropna()
    axes[1, 0].plot(diff_1)
    axes[1, 0].set_title('1st Order Differencing')
    plot_acf(diff_1, ax=axes[1, 1])

    # 2nd Differencing
    diff_2 = data.diff().diff().dropna()
    axes[2, 0].plot(diff_2)
    axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(diff_2, ax=axes[2, 1])

    plt.tight_layout()
    plt.show()

# Example Usage
plot_differencing_acf_pacf(df['Close'])
```


![Alt Text](https://i.ibb.co/LRKtbzm/3.png)

    


# Step 7: Identify Seasonality and Autocorrelations
Utilize autocorrelation and partial autocorrelation plots to identify the order of autoregressive (AR) and moving average (MA) components, as well as the seasonal order.


```python
# Import necessary libraries
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # For autocorrelation and partial autocorrelation plots
from statsmodels.tsa.stattools import acf, pacf  # For computing autocorrelation and partial autocorrelation

# Visualize autocorrelation using pandas
pd.plotting.autocorrelation_plot(df['Close'])  # Plot autocorrelation using pandas

# Visualize autocorrelation using plot_acf
plot_acf(df['Close'], alpha=0.05)  # Plot autocorrelation using plot_acf with confidence interval

# Compute autocorrelation function (ACF) values
x_acf = pd.DataFrame(acf(df['Close']))  # Calculate autocorrelation function (ACF) values

# Print ACF values
print(x_acf)  # Print the ACF values

# Generate a partial autocorrelation plot
plot_pacf(df['Close'], lags=20, alpha=0.05)
```

               0
    0   1.000000
    1   0.980285
    2   0.957557
    3   0.935598
    4   0.914009
    5   0.893179
    6   0.868831
    7   0.839151
    8   0.808492
    9   0.774980
    10  0.740682
    11  0.707750
    12  0.676324
    13  0.641283
    14  0.607955
    15  0.574249
    16  0.539390
    17  0.506053
    18  0.474119
    19  0.440244
    20  0.409584
    21  0.383322
    22  0.358802
    23  0.337378


![Alt Text](https://i.ibb.co/dL7s67V/4.png)
![Alt Text](https://i.ibb.co/TPtNXbH/5.png)
![Alt Text](https://i.ibb.co/0C642SR/6.png)
![Alt Text](https://i.ibb.co/dL7s67V/4.png)





# Step 8: Fit the SARIMAX Model
Based on the insights from step 7, select the appropriate orders for the AR, I (differencing), MA, seasonal AR, seasonal I, and seasonal MA components.
for example our values are
q = 2
d = 2
p = 1


```python
q = 2
d = 2
p = 1
```

# **SARIMAX**


```python
pip install pmdarima
```


```python
from pmdarima.arima import auto_arima
```


```python
model_sarimax =auto_arima(df['Close'],start_p=0,d=1,start_q=0,
          max_p=1,max_d=2,max_q=2, start_P=0,
          D=1, start_Q=2, max_P=2,max_D=1,
          max_Q=5, m=12, seasonal=True,
          error_action='warn',trace=True,
          supress_warnings=True,stepwise=True,
          random_state=20,n_fits=50)
```

    Performing stepwise search to minimize aic
     ARIMA(0,1,0)(0,1,2)[12]             : AIC=inf, Time=0.89 sec
     ARIMA(0,1,0)(0,1,0)[12]             : AIC=1328.314, Time=0.04 sec
     ARIMA(1,1,0)(1,1,0)[12]             : AIC=1280.778, Time=0.18 sec
     ARIMA(0,1,1)(0,1,1)[12]             : AIC=inf, Time=0.45 sec
     ARIMA(1,1,0)(0,1,0)[12]             : AIC=1327.712, Time=0.05 sec
     ARIMA(1,1,0)(2,1,0)[12]             : AIC=1240.871, Time=0.58 sec
     ARIMA(1,1,0)(2,1,1)[12]             : AIC=1215.670, Time=2.83 sec
     ARIMA(1,1,0)(1,1,1)[12]             : AIC=inf, Time=0.55 sec
     ARIMA(1,1,0)(2,1,2)[12]             : AIC=1214.822, Time=2.38 sec
     ARIMA(1,1,0)(1,1,2)[12]             : AIC=inf, Time=2.44 sec
     ARIMA(1,1,0)(2,1,3)[12]             : AIC=inf, Time=12.11 sec
     ARIMA(1,1,0)(1,1,3)[12]             : AIC=1214.779, Time=3.38 sec
     ARIMA(1,1,0)(0,1,3)[12]             : AIC=1214.716, Time=2.04 sec
     ARIMA(1,1,0)(0,1,2)[12]             : AIC=inf, Time=2.02 sec
     ARIMA(1,1,0)(0,1,4)[12]             : AIC=1216.231, Time=4.88 sec
     ARIMA(1,1,0)(1,1,4)[12]             : AIC=1216.752, Time=14.12 sec
     ARIMA(0,1,0)(0,1,3)[12]             : AIC=1214.650, Time=1.57 sec
     ARIMA(0,1,0)(1,1,3)[12]             : AIC=1215.229, Time=2.66 sec
     ARIMA(0,1,0)(0,1,4)[12]             : AIC=1216.323, Time=4.97 sec
     ARIMA(0,1,0)(1,1,2)[12]             : AIC=inf, Time=1.82 sec
     ARIMA(0,1,0)(1,1,4)[12]             : AIC=1217.110, Time=13.65 sec
     ARIMA(0,1,1)(0,1,3)[12]             : AIC=1214.490, Time=2.40 sec
     ARIMA(0,1,1)(0,1,2)[12]             : AIC=inf, Time=1.35 sec
     ARIMA(0,1,1)(1,1,3)[12]             : AIC=1214.453, Time=4.89 sec
     ARIMA(0,1,1)(1,1,2)[12]             : AIC=inf, Time=3.98 sec
     ARIMA(0,1,1)(2,1,3)[12]             : AIC=1216.073, Time=13.09 sec
     ARIMA(0,1,1)(1,1,4)[12]             : AIC=1216.430, Time=15.94 sec
     ARIMA(0,1,1)(0,1,4)[12]             : AIC=1215.970, Time=4.96 sec
     ARIMA(0,1,1)(2,1,2)[12]             : AIC=1214.496, Time=4.23 sec
     ARIMA(0,1,1)(2,1,4)[12]             : AIC=inf, Time=21.78 sec
     ARIMA(1,1,1)(1,1,3)[12]             : AIC=1216.257, Time=8.60 sec
     ARIMA(0,1,2)(1,1,3)[12]             : AIC=1215.926, Time=4.79 sec
     ARIMA(1,1,2)(1,1,3)[12]             : AIC=1217.335, Time=8.03 sec
     ARIMA(0,1,1)(1,1,3)[12] intercept   : AIC=1215.707, Time=5.05 sec
    
    Best model:  ARIMA(0,1,1)(1,1,3)[12]          
    Total fit time: 172.810 seconds



```python
print(model_sarimax.summary())
# predict next 30 days
forecast = model_sarimax.predict(len(df["Close"]), len(df['Close'])+30)
print(forecast)

#plot forecast
plt.figure(figsize=(10, 5))
plt.plot(df['Close'], label='Actual')
plt.plot(forecast, label='Forecast')
```

                                             SARIMAX Results                                          
    ==================================================================================================
    Dep. Variable:                                          y   No. Observations:                  251
    Model:             SARIMAX(0, 1, 1)x(1, 1, [1, 2, 3], 12)   Log Likelihood                -601.227
    Date:                                    Tue, 07 Nov 2023   AIC                           1214.453
    Time:                                            13:41:31   BIC                           1235.287
    Sample:                                                 0   HQIC                          1222.850
                                                        - 251                                         
    Covariance Type:                                      opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    ma.L1          0.1142      0.061      1.878      0.060      -0.005       0.233
    ar.S.L12      -0.6155      0.249     -2.474      0.013      -1.103      -0.128
    ma.S.L12      -0.2396      0.252     -0.951      0.342      -0.734       0.254
    ma.S.L24      -0.6872      0.205     -3.350      0.001      -1.089      -0.285
    ma.S.L36       0.1214      0.108      1.126      0.260      -0.090       0.333
    sigma2         8.4233      0.631     13.359      0.000       7.187       9.659
    ===================================================================================
    Ljung-Box (L1) (Q):                   0.01   Jarque-Bera (JB):                31.07
    Prob(Q):                              0.92   Prob(JB):                         0.00
    Heteroskedasticity (H):               1.37   Skew:                             0.21
    Prob(H) (two-sided):                  0.16   Kurtosis:                         4.72
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    251    107.574462
    252    109.041287
    253    110.842939
    254    112.268545
    255    110.882080
              ...    
    497    166.860710
    498    165.528492
    499    165.689534
    500    165.560889
    501    165.569422
    Length: 251, dtype: float64





    [<matplotlib.lines.Line2D at 0x7b157c5803d0>]




    
![Alt Text](https://i.ibb.co/ZVsFqMv/8.png)

    


SARIMAX using Statsmodel


```python
import statsmodels.api as sm
import warnings

p, d, q = 2, 1, 2

model = sm.tsa.statespace.SARIMAX(df['Close'],
                                  order=(p, d, q),
                                  seasonal_order=(p, d, q, 12))
model = model.fit()
print(model.summary())
```

    /usr/local/lib/python3.10/dist-packages/statsmodels/tsa/statespace/sarimax.py:966: UserWarning:
    
    Non-stationary starting autoregressive parameters found. Using zeros as starting parameters.
    


                                         SARIMAX Results                                      
    ==========================================================================================
    Dep. Variable:                              Close   No. Observations:                  251
    Model:             SARIMAX(2, 1, 2)x(2, 1, 2, 12)   Log Likelihood                -600.589
    Date:                            Tue, 07 Nov 2023   AIC                           1219.177
    Time:                                    13:47:52   BIC                           1250.428
    Sample:                                         0   HQIC                          1231.772
                                                - 251                                         
    Covariance Type:                              opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    ar.L1          1.4826      0.103     14.451      0.000       1.282       1.684
    ar.L2         -0.9195      0.089    -10.315      0.000      -1.094      -0.745
    ma.L1         -1.5071      0.114    -13.235      0.000      -1.730      -1.284
    ma.L2          0.9043      0.108      8.398      0.000       0.693       1.115
    ar.S.L12      -0.6702      0.206     -3.251      0.001      -1.074      -0.266
    ar.S.L24      -0.0905      0.121     -0.751      0.453      -0.327       0.146
    ma.S.L12      -0.1486      0.201     -0.740      0.460      -0.543       0.245
    ma.S.L24      -0.6625      0.221     -2.994      0.003      -1.096      -0.229
    sigma2         8.3340      0.642     12.983      0.000       7.076       9.592
    ===================================================================================
    Ljung-Box (L1) (Q):                   2.80   Jarque-Bera (JB):                29.13
    Prob(Q):                              0.09   Prob(JB):                         0.00
    Heteroskedasticity (H):               1.45   Skew:                             0.25
    Prob(H) (two-sided):                  0.10   Kurtosis:                         4.64
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).



```python
# predict next 30 days BN
forecast = model.predict(len(df["Close"]), len(df['Close'])+30)
print(forecast)

#plot forecast
plt.figure(figsize=(10, 5))
plt.plot(df['Close'], label='Actual')
plt.plot(forecast, label='Forecast')
```

    251    106.342095
    252    108.990280
    253    111.292078
    254    113.456288
    255    114.363421
    256    115.542002
    257    114.213088
    258    112.174290
    259    113.304572
    260    112.585059
    261    112.572702
    262    113.051447
    263    109.365605
    264    112.261605
    265    114.586721
    266    116.807268
    267    118.165806
    268    119.315343
    269    117.912382
    270    115.851244
    271    117.207688
    272    116.693073
    273    116.711063
    274    117.323382
    275    113.276801
    276    116.474547
    277    118.917424
    278    121.324298
    279    122.814593
    280    124.029864
    281    122.435882
    Name: predicted_mean, dtype: float64





    [<matplotlib.lines.Line2D at 0x7b156b7e0310>]




    
![Alt Text](https://i.ibb.co/pXn605g/9.png)

    


# SARIMA Hyperparameter Tuning
SARIMA (Seasonal AutoRegressive Integrated Moving Average) hyperparameter tuning involves the process of selecting the optimal values for the parameters of the SARIMA model in order to improve its forecasting accuracy. The hyperparameters in a SARIMA model include:


p (AR Order): The number of lag observations included in the model (AutoRegressive term).

d (Differencing Order): The number of times the data needs to be differenced to achieve stationarity.

q (MA Order): The size of the moving average window (Moving Average term).

P (Seasonal AR Order): The seasonal autoregressive order.

D (Seasonal Differencing Order): The number of seasonal differences.

Q (Seasonal MA Order): The seasonal moving average order.

s (Seasonal Period): The number of time units in a seasonal cycle.

Hyperparameter tuning involves finding the combination of these parameters that minimizes a chosen evaluation metric (such as Mean Absolute Error, Mean Squared Error, etc.) on a validation dataset. This process is crucial for improving the accuracy of the SARIMA model.


```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import itertools
import statsmodels.api as sm
import warnings

# Define the p, d, q parameters to take any value between
p = d = q = range(0, 3)

# Generate all different combinations of p, d and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, d, q and quadruplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

best_aic = np.inf
best_pdq = None
best_seasonal_pdq = None
temp_model = None

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            temp_model = sm.tsa.statespace.SARIMAX(df['Close'],
                                                  order=param,
                                                  seasonal_order=param_seasonal,
                                                  enforce_stationarity=False,
                                                  enforce_invertibility=False)
            results = temp_model.fit()
            print(f"SARIMA{param}{param_seasonal} - AIC: {results.aic}")
            if results.aic < best_aic:
                best_aic = results.aic
                best_pdq = param
                best_seasonal_pdq = param_seasonal
        except:
            continue

print(f"Best SARIMA model - AIC: {best_pdq}, {best_seasonal_pdq}, {best_aic}")


```

    SARIMA(0, 0, 0)(0, 0, 0, 12) - AIC: 2933.606783482486
    SARIMA(0, 0, 0)(0, 0, 1, 12) - AIC: 2521.51081015389
    SARIMA(0, 0, 0)(0, 0, 2, 12) - AIC: 2192.7449171077005
    SARIMA(0, 0, 0)(0, 1, 0, 12) - AIC: 1814.2676132829092
    SARIMA(0, 0, 0)(0, 1, 1, 12) - AIC: 1701.1750645358497
    SARIMA(0, 0, 0)(0, 1, 2, 12) - AIC: 1600.0280415071468
    SARIMA(0, 0, 0)(0, 2, 0, 12) - AIC: 1830.3018096808125
    SARIMA(0, 0, 0)(0, 2, 1, 12) - AIC: 1666.3681520080168
    SARIMA(0, 0, 0)(0, 2, 2, 12) - AIC: 1532.5873805062627
    SARIMA(0, 0, 0)(1, 0, 0, 12) - AIC: 1817.1876688058792
    SARIMA(0, 0, 0)(1, 0, 1, 12) - AIC: 1794.5007992416
    SARIMA(0, 0, 0)(1, 0, 2, 12) - AIC: 1694.7482483569452
    SARIMA(0, 0, 0)(1, 1, 0, 12) - AIC: 1724.2921969159556
    SARIMA(0, 0, 0)(1, 1, 1, 12) - AIC: 1707.2610540400096
    SARIMA(0, 0, 0)(1, 1, 2, 12) - AIC: 1601.4973489340196
    SARIMA(0, 0, 0)(1, 2, 0, 12) - AIC: 1735.8045239527
    SARIMA(0, 0, 0)(1, 2, 1, 12) - AIC: 1664.1931059076865
    SARIMA(0, 0, 0)(1, 2, 2, 12) - AIC: 1534.569160596503
    SARIMA(0, 0, 0)(2, 0, 0, 12) - AIC: 1724.2624510985056
    SARIMA(0, 0, 0)(2, 0, 1, 12) - AIC: 1715.6675891135976
    SARIMA(0, 0, 0)(2, 0, 2, 12) - AIC: 1694.3111780689833
    SARIMA(0, 0, 0)(2, 1, 0, 12) - AIC: 1591.3916116603941
    SARIMA(0, 0, 0)(2, 1, 1, 12) - AIC: 1588.5867175069225
    SARIMA(0, 0, 0)(2, 1, 2, 12) - AIC: 1576.4153733129483
    SARIMA(0, 0, 0)(2, 2, 0, 12) - AIC: 1557.5495455736523
    SARIMA(0, 0, 0)(2, 2, 1, 12) - AIC: 1513.8591472528378
    SARIMA(0, 0, 0)(2, 2, 2, 12) - AIC: 1485.6770796112
    SARIMA(0, 0, 1)(0, 0, 0, 12) - AIC: 2593.563507205501
    SARIMA(0, 0, 1)(0, 0, 1, 12) - AIC: 2201.5979265719643
    SARIMA(0, 0, 1)(0, 0, 2, 12) - AIC: 1922.578320396512
    SARIMA(0, 0, 1)(0, 1, 0, 12) - AIC: 1574.298265635709
    SARIMA(0, 0, 1)(0, 1, 1, 12) - AIC: 1493.3206782590114
    SARIMA(0, 0, 1)(0, 1, 2, 12) - AIC: 1402.6257177964408
    SARIMA(0, 0, 1)(0, 2, 0, 12) - AIC: 1634.0844505911532
    SARIMA(0, 0, 1)(0, 2, 1, 12) - AIC: 1453.606698026995
    SARIMA(0, 0, 1)(0, 2, 2, 12) - AIC: 1350.302338739888
    SARIMA(0, 0, 1)(1, 0, 0, 12) - AIC: 1586.204172323477
    SARIMA(0, 0, 1)(1, 0, 1, 12) - AIC: 1574.7992635472951


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(0, 0, 1)(1, 0, 2, 12) - AIC: 1504.7698795401118
    SARIMA(0, 0, 1)(1, 1, 0, 12) - AIC: 1510.1339768896778
    SARIMA(0, 0, 1)(1, 1, 1, 12) - AIC: 1495.291202929871
    SARIMA(0, 0, 1)(1, 1, 2, 12) - AIC: 1403.546710867597
    SARIMA(0, 0, 1)(1, 2, 0, 12) - AIC: 1545.0455002725203
    SARIMA(0, 0, 1)(1, 2, 1, 12) - AIC: 1455.5997777824196
    SARIMA(0, 0, 1)(1, 2, 2, 12) - AIC: 1349.2954178480554
    SARIMA(0, 0, 1)(2, 0, 0, 12) - AIC: 1511.7841767689597
    SARIMA(0, 0, 1)(2, 0, 1, 12) - AIC: 1513.8948453782655


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(0, 0, 1)(2, 0, 2, 12) - AIC: 1487.2005482701595
    SARIMA(0, 0, 1)(2, 1, 0, 12) - AIC: 1406.4454792281413
    SARIMA(0, 0, 1)(2, 1, 1, 12) - AIC: 1406.9212528918436
    SARIMA(0, 0, 1)(2, 1, 2, 12) - AIC: 1395.940627766261
    SARIMA(0, 0, 1)(2, 2, 0, 12) - AIC: 1388.406264485038
    SARIMA(0, 0, 1)(2, 2, 1, 12) - AIC: 1346.5048225950534
    SARIMA(0, 0, 1)(2, 2, 2, 12) - AIC: 1354.2987292199916
    SARIMA(0, 0, 2)(0, 0, 0, 12) - AIC: 2356.368129141507
    SARIMA(0, 0, 2)(0, 0, 1, 12) - AIC: 1943.4376066755099
    SARIMA(0, 0, 2)(0, 0, 2, 12) - AIC: 1755.2964124579848
    SARIMA(0, 0, 2)(0, 1, 0, 12) - AIC: 1458.1886793037372
    SARIMA(0, 0, 2)(0, 1, 1, 12) - AIC: 1381.8692466831221
    SARIMA(0, 0, 2)(0, 1, 2, 12) - AIC: 1298.106271357512
    SARIMA(0, 0, 2)(0, 2, 0, 12) - AIC: 1531.2373858869787
    SARIMA(0, 0, 2)(0, 2, 1, 12) - AIC: 1347.976164420937
    SARIMA(0, 0, 2)(0, 2, 2, 12) - AIC: 1253.8869004140624
    SARIMA(0, 0, 2)(1, 0, 0, 12) - AIC: 1477.060835980769
    SARIMA(0, 0, 2)(1, 0, 1, 12) - AIC: 1459.804732673664
    SARIMA(0, 0, 2)(1, 0, 2, 12) - AIC: 1369.7078069080771
    SARIMA(0, 0, 2)(1, 1, 0, 12) - AIC: 1402.7980688906387
    SARIMA(0, 0, 2)(1, 1, 1, 12) - AIC: 1382.017785903155
    SARIMA(0, 0, 2)(1, 1, 2, 12) - AIC: 1298.5679923435098
    SARIMA(0, 0, 2)(1, 2, 0, 12) - AIC: 1450.0045883541902
    SARIMA(0, 0, 2)(1, 2, 1, 12) - AIC: 1349.951890076526
    SARIMA(0, 0, 2)(1, 2, 2, 12) - AIC: 1250.843467972781
    SARIMA(0, 0, 2)(2, 0, 0, 12) - AIC: 1406.3075453208571


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(0, 0, 2)(2, 0, 1, 12) - AIC: 1409.057525860952


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(0, 0, 2)(2, 0, 2, 12) - AIC: 1401.3975095551875
    SARIMA(0, 0, 2)(2, 1, 0, 12) - AIC: 1307.6617890391149
    SARIMA(0, 0, 2)(2, 1, 1, 12) - AIC: 1308.7708149523505
    SARIMA(0, 0, 2)(2, 1, 2, 12) - AIC: 1293.002863978336
    SARIMA(0, 0, 2)(2, 2, 0, 12) - AIC: 1306.4174086338487
    SARIMA(0, 0, 2)(2, 2, 1, 12) - AIC: 1261.6640654231835
    SARIMA(0, 0, 2)(2, 2, 2, 12) - AIC: 1232.979680456461
    SARIMA(0, 1, 0)(0, 0, 0, 12) - AIC: 1232.8402196234074
    SARIMA(0, 1, 0)(0, 0, 1, 12) - AIC: 1177.3521677023632
    SARIMA(0, 1, 0)(0, 0, 2, 12) - AIC: 1119.56998719648
    SARIMA(0, 1, 0)(0, 1, 0, 12) - AIC: 1323.4803493156328
    SARIMA(0, 1, 0)(0, 1, 1, 12) - AIC: 1139.9864080516286
    SARIMA(0, 1, 0)(0, 1, 2, 12) - AIC: 1091.5665594673019
    SARIMA(0, 1, 0)(0, 2, 0, 12) - AIC: 1483.0328578444883
    SARIMA(0, 1, 0)(0, 2, 1, 12) - AIC: 1225.3007862174807
    SARIMA(0, 1, 0)(0, 2, 2, 12) - AIC: 1065.0316397130546
    SARIMA(0, 1, 0)(1, 0, 0, 12) - AIC: 1182.2393944955986
    SARIMA(0, 1, 0)(1, 0, 1, 12) - AIC: 1177.2831746978873
    SARIMA(0, 1, 0)(1, 0, 2, 12) - AIC: 1120.4147434988622
    SARIMA(0, 1, 0)(1, 1, 0, 12) - AIC: 1213.0260455767414
    SARIMA(0, 1, 0)(1, 1, 1, 12) - AIC: 1141.9856451684714
    SARIMA(0, 1, 0)(1, 1, 2, 12) - AIC: 1086.1462465592163
    SARIMA(0, 1, 0)(1, 2, 0, 12) - AIC: 1327.277502181399
    SARIMA(0, 1, 0)(1, 2, 1, 12) - AIC: 1189.9679830433784
    SARIMA(0, 1, 0)(1, 2, 2, 12) - AIC: 1067.0306508756455
    SARIMA(0, 1, 0)(2, 0, 0, 12) - AIC: 1123.0464120484326
    SARIMA(0, 1, 0)(2, 0, 1, 12) - AIC: 1124.3133732820907
    SARIMA(0, 1, 0)(2, 0, 2, 12) - AIC: 1121.875128999218
    SARIMA(0, 1, 0)(2, 1, 0, 12) - AIC: 1113.8664119671212
    SARIMA(0, 1, 0)(2, 1, 1, 12) - AIC: 1095.1552395299486
    SARIMA(0, 1, 0)(2, 1, 2, 12) - AIC: 1085.8569603224876
    SARIMA(0, 1, 0)(2, 2, 0, 12) - AIC: 1166.378473530242
    SARIMA(0, 1, 0)(2, 2, 1, 12) - AIC: 1097.2931693347773
    SARIMA(0, 1, 0)(2, 2, 2, 12) - AIC: 1093.4992403342876
    SARIMA(0, 1, 1)(0, 0, 0, 12) - AIC: 1225.0029595564984
    SARIMA(0, 1, 1)(0, 0, 1, 12) - AIC: 1172.9831477199925
    SARIMA(0, 1, 1)(0, 0, 2, 12) - AIC: 1116.1070791021007
    SARIMA(0, 1, 1)(0, 1, 0, 12) - AIC: 1314.1452781074313
    SARIMA(0, 1, 1)(0, 1, 1, 12) - AIC: 1135.3861561080619
    SARIMA(0, 1, 1)(0, 1, 2, 12) - AIC: 1087.9047161781
    SARIMA(0, 1, 1)(0, 2, 0, 12) - AIC: 1476.8669722409315
    SARIMA(0, 1, 1)(0, 2, 1, 12) - AIC: 1220.3147704699036
    SARIMA(0, 1, 1)(0, 2, 2, 12) - AIC: 1059.28369904086
    SARIMA(0, 1, 1)(1, 0, 0, 12) - AIC: 1182.3521869837734
    SARIMA(0, 1, 1)(1, 0, 1, 12) - AIC: 1173.829257000661
    SARIMA(0, 1, 1)(1, 0, 2, 12) - AIC: 1117.238009969493
    SARIMA(0, 1, 1)(1, 1, 0, 12) - AIC: 1211.7907758230526
    SARIMA(0, 1, 1)(1, 1, 1, 12) - AIC: 1137.3858865504208
    SARIMA(0, 1, 1)(1, 1, 2, 12) - AIC: 1082.1761313609172
    SARIMA(0, 1, 1)(1, 2, 0, 12) - AIC: 1327.2369651408112
    SARIMA(0, 1, 1)(1, 2, 1, 12) - AIC: 1183.3084200127623
    SARIMA(0, 1, 1)(1, 2, 2, 12) - AIC: 1079.6023769378305
    SARIMA(0, 1, 1)(2, 0, 0, 12) - AIC: 1124.2876112450526
    SARIMA(0, 1, 1)(2, 0, 1, 12) - AIC: 1125.3098568343016
    SARIMA(0, 1, 1)(2, 0, 2, 12) - AIC: 1118.9833117071103
    SARIMA(0, 1, 1)(2, 1, 0, 12) - AIC: 1113.2997848049426
    SARIMA(0, 1, 1)(2, 1, 1, 12) - AIC: 1096.400204884703
    SARIMA(0, 1, 1)(2, 1, 2, 12) - AIC: 1081.533305820415
    SARIMA(0, 1, 1)(2, 2, 0, 12) - AIC: 1164.7096309550684
    SARIMA(0, 1, 1)(2, 2, 1, 12) - AIC: 1096.5142551860536
    SARIMA(0, 1, 1)(2, 2, 2, 12) - AIC: 1083.646618919271
    SARIMA(0, 1, 2)(0, 0, 0, 12) - AIC: 1222.276339403938
    SARIMA(0, 1, 2)(0, 0, 1, 12) - AIC: 1170.8465338125657
    SARIMA(0, 1, 2)(0, 0, 2, 12) - AIC: 1113.0380053956
    SARIMA(0, 1, 2)(0, 1, 0, 12) - AIC: 1310.8705608611692
    SARIMA(0, 1, 2)(0, 1, 1, 12) - AIC: 1134.2908165469034
    SARIMA(0, 1, 2)(0, 1, 2, 12) - AIC: 1083.842750097093
    SARIMA(0, 1, 2)(0, 2, 0, 12) - AIC: 1471.083718998325
    SARIMA(0, 1, 2)(0, 2, 1, 12) - AIC: 1215.8346218262504
    SARIMA(0, 1, 2)(0, 2, 2, 12) - AIC: 1057.6923940592394
    SARIMA(0, 1, 2)(1, 0, 0, 12) - AIC: 1184.165730308599
    SARIMA(0, 1, 2)(1, 0, 1, 12) - AIC: 1171.7016456476927
    SARIMA(0, 1, 2)(1, 0, 2, 12) - AIC: 1114.1602327638745
    SARIMA(0, 1, 2)(1, 1, 0, 12) - AIC: 1213.7744778607716
    SARIMA(0, 1, 2)(1, 1, 1, 12) - AIC: 1136.2905232084463
    SARIMA(0, 1, 2)(1, 1, 2, 12) - AIC: 1078.8039320891185
    SARIMA(0, 1, 2)(1, 2, 0, 12) - AIC: 1329.2341431082177
    SARIMA(0, 1, 2)(1, 2, 1, 12) - AIC: 1179.7034405617933
    SARIMA(0, 1, 2)(1, 2, 2, 12) - AIC: 1059.692194921274
    SARIMA(0, 1, 2)(2, 0, 0, 12) - AIC: 1125.2219057199727
    SARIMA(0, 1, 2)(2, 0, 1, 12) - AIC: 1126.30878001045
    SARIMA(0, 1, 2)(2, 0, 2, 12) - AIC: 1115.413702148719
    SARIMA(0, 1, 2)(2, 1, 0, 12) - AIC: 1114.1781943980936
    SARIMA(0, 1, 2)(2, 1, 1, 12) - AIC: 1096.7064200727632
    SARIMA(0, 1, 2)(2, 1, 2, 12) - AIC: 1077.3842210759367
    SARIMA(0, 1, 2)(2, 2, 0, 12) - AIC: 1165.7271040214778
    SARIMA(0, 1, 2)(2, 2, 1, 12) - AIC: 1097.4811105123517
    SARIMA(0, 1, 2)(2, 2, 2, 12) - AIC: 1078.6304031534705
    SARIMA(0, 2, 0)(0, 0, 0, 12) - AIC: 1374.4891198663067
    SARIMA(0, 2, 0)(0, 0, 1, 12) - AIC: 1316.4205442579932
    SARIMA(0, 2, 0)(0, 0, 2, 12) - AIC: 1256.5440106077917
    SARIMA(0, 2, 0)(0, 1, 0, 12) - AIC: 1453.7331680681923
    SARIMA(0, 2, 0)(0, 1, 1, 12) - AIC: 1272.0835091995423
    SARIMA(0, 2, 0)(0, 1, 2, 12) - AIC: 1214.6530770646846
    SARIMA(0, 2, 0)(0, 2, 0, 12) - AIC: 1616.253728341802
    SARIMA(0, 2, 0)(0, 2, 1, 12) - AIC: 1350.2975368599416
    SARIMA(0, 2, 0)(0, 2, 2, 12) - AIC: 1181.8854745078024
    SARIMA(0, 2, 0)(1, 0, 0, 12) - AIC: 1321.2240043870056
    SARIMA(0, 2, 0)(1, 0, 1, 12) - AIC: 1317.566256073661
    SARIMA(0, 2, 0)(1, 0, 2, 12) - AIC: 1254.6600454159416
    SARIMA(0, 2, 0)(1, 1, 0, 12) - AIC: 1334.508295785496
    SARIMA(0, 2, 0)(1, 1, 1, 12) - AIC: 1274.0832622168905
    SARIMA(0, 2, 0)(1, 1, 2, 12) - AIC: 1207.0573471700895
    SARIMA(0, 2, 0)(1, 2, 0, 12) - AIC: 1445.282980077532
    SARIMA(0, 2, 0)(1, 2, 1, 12) - AIC: 1304.840342325498
    SARIMA(0, 2, 0)(1, 2, 2, 12) - AIC: 1183.884682549463
    SARIMA(0, 2, 0)(2, 0, 0, 12) - AIC: 1260.8150507450673
    SARIMA(0, 2, 0)(2, 0, 1, 12) - AIC: 1259.2830322709947
    SARIMA(0, 2, 0)(2, 0, 2, 12) - AIC: 1256.6597862426395
    SARIMA(0, 2, 0)(2, 1, 0, 12) - AIC: 1232.9107098485956
    SARIMA(0, 2, 0)(2, 1, 1, 12) - AIC: 1224.9608696670948
    SARIMA(0, 2, 0)(2, 1, 2, 12) - AIC: 1208.3365406437392
    SARIMA(0, 2, 0)(2, 2, 0, 12) - AIC: 1271.6545952653635
    SARIMA(0, 2, 0)(2, 2, 1, 12) - AIC: 1207.5912316958188
    SARIMA(0, 2, 0)(2, 2, 2, 12) - AIC: 1197.9736310209314
    SARIMA(0, 2, 1)(0, 0, 0, 12) - AIC: 1229.9271452309558
    SARIMA(0, 2, 1)(0, 0, 1, 12) - AIC: 1175.0478037140892
    SARIMA(0, 2, 1)(0, 0, 2, 12) - AIC: 1117.016387328761
    SARIMA(0, 2, 1)(0, 1, 0, 12) - AIC: 1319.6847708176176
    SARIMA(0, 2, 1)(0, 1, 1, 12) - AIC: 1142.106016572819
    SARIMA(0, 2, 1)(0, 1, 2, 12) - AIC: 1091.6476719776
    SARIMA(0, 2, 1)(0, 2, 0, 12) - AIC: 1477.259392230324
    SARIMA(0, 2, 1)(0, 2, 1, 12) - AIC: 1228.272057945561
    SARIMA(0, 2, 1)(0, 2, 2, 12) - AIC: 1066.4189794448907
    SARIMA(0, 2, 1)(1, 0, 0, 12) - AIC: 1185.2534947365903
    SARIMA(0, 2, 1)(1, 0, 1, 12) - AIC: 1174.481491981182
    SARIMA(0, 2, 1)(1, 0, 2, 12) - AIC: 1118.042451272997
    SARIMA(0, 2, 1)(1, 1, 0, 12) - AIC: 1215.9548901462672
    SARIMA(0, 2, 1)(1, 1, 1, 12) - AIC: 1143.8806328272667
    SARIMA(0, 2, 1)(1, 1, 2, 12) - AIC: 1082.6246227761176
    SARIMA(0, 2, 1)(1, 2, 0, 12) - AIC: 1329.2396846847969
    SARIMA(0, 2, 1)(1, 2, 1, 12) - AIC: 1190.7802073698379
    SARIMA(0, 2, 1)(1, 2, 2, 12) - AIC: 1067.808104786408
    SARIMA(0, 2, 1)(2, 0, 0, 12) - AIC: 1125.8678266244015
    SARIMA(0, 2, 1)(2, 0, 1, 12) - AIC: 1127.1249164953779
    SARIMA(0, 2, 1)(2, 0, 2, 12) - AIC: 1119.1677799028091
    SARIMA(0, 2, 1)(2, 1, 0, 12) - AIC: 1117.02484114143
    SARIMA(0, 2, 1)(2, 1, 1, 12) - AIC: 1099.8340730840819
    SARIMA(0, 2, 1)(2, 1, 2, 12) - AIC: 1084.738931073058
    SARIMA(0, 2, 1)(2, 2, 0, 12) - AIC: 1168.9383543409456
    SARIMA(0, 2, 1)(2, 2, 1, 12) - AIC: 1102.3798656883405
    SARIMA(0, 2, 1)(2, 2, 2, 12) - AIC: 1087.4299912485621
    SARIMA(0, 2, 2)(0, 0, 0, 12) - AIC: 1225.854171855526
    SARIMA(0, 2, 2)(0, 0, 1, 12) - AIC: 1170.5120234330327
    SARIMA(0, 2, 2)(0, 0, 2, 12) - AIC: 1111.6427964569498
    SARIMA(0, 2, 2)(0, 1, 0, 12) - AIC: 1313.478800995227
    SARIMA(0, 2, 2)(0, 1, 1, 12) - AIC: 1137.7145543014176
    SARIMA(0, 2, 2)(0, 1, 2, 12) - AIC: 1085.2121767675626
    SARIMA(0, 2, 2)(0, 2, 0, 12) - AIC: 1472.2213206364918
    SARIMA(0, 2, 2)(0, 2, 1, 12) - AIC: 1220.8588191755823
    SARIMA(0, 2, 2)(0, 2, 2, 12) - AIC: 1061.0745617537489
    SARIMA(0, 2, 2)(1, 0, 0, 12) - AIC: 1185.276112392676
    SARIMA(0, 2, 2)(1, 0, 1, 12) - AIC: 1169.9772965381464
    SARIMA(0, 2, 2)(1, 0, 2, 12) - AIC: 1113.3371042750948
    SARIMA(0, 2, 2)(1, 1, 0, 12) - AIC: 1214.5263094643587
    SARIMA(0, 2, 2)(1, 1, 1, 12) - AIC: 1139.568671088502
    SARIMA(0, 2, 2)(1, 1, 2, 12) - AIC: 1075.256487137216
    SARIMA(0, 2, 2)(1, 2, 0, 12) - AIC: 1329.03489515432
    SARIMA(0, 2, 2)(1, 2, 1, 12) - AIC: 1182.9687209521737
    SARIMA(0, 2, 2)(1, 2, 2, 12) - AIC: 1062.6020477611025
    SARIMA(0, 2, 2)(2, 0, 0, 12) - AIC: 1127.220638014182
    SARIMA(0, 2, 2)(2, 0, 1, 12) - AIC: 1128.374772738424
    SARIMA(0, 2, 2)(2, 0, 2, 12) - AIC: 1114.3858841088982
    SARIMA(0, 2, 2)(2, 1, 0, 12) - AIC: 1116.24154937127
    SARIMA(0, 2, 2)(2, 1, 1, 12) - AIC: 1101.141769127622
    SARIMA(0, 2, 2)(2, 1, 2, 12) - AIC: 1078.8614382571282
    SARIMA(0, 2, 2)(2, 2, 0, 12) - AIC: 1167.0090254348424
    SARIMA(0, 2, 2)(2, 2, 1, 12) - AIC: 1102.3436653782337
    SARIMA(0, 2, 2)(2, 2, 2, 12) - AIC: 1081.0089282756576
    SARIMA(1, 0, 0)(0, 0, 0, 12) - AIC: 1238.7813049498518
    SARIMA(1, 0, 0)(0, 0, 1, 12) - AIC: 1183.2422925608466


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(1, 0, 0)(0, 0, 2, 12) - AIC: 1129.2626356215628
    SARIMA(1, 0, 0)(0, 1, 0, 12) - AIC: 1322.908300404827
    SARIMA(1, 0, 0)(0, 1, 1, 12) - AIC: 1145.5615891441084
    SARIMA(1, 0, 0)(0, 1, 2, 12) - AIC: 1098.1825666919076
    SARIMA(1, 0, 0)(0, 2, 0, 12) - AIC: 1478.3705714094522


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(1, 0, 0)(0, 2, 1, 12) - AIC: 1228.2414127490756
    SARIMA(1, 0, 0)(0, 2, 2, 12) - AIC: 1070.042170152292
    SARIMA(1, 0, 0)(1, 0, 0, 12) - AIC: 1183.99865297107
    SARIMA(1, 0, 0)(1, 0, 1, 12) - AIC: 1183.2384828243028
    SARIMA(1, 0, 0)(1, 0, 2, 12) - AIC: 1126.8687709025885
    SARIMA(1, 0, 0)(1, 1, 0, 12) - AIC: 1210.3124591831918
    SARIMA(1, 0, 0)(1, 1, 1, 12) - AIC: 1147.5585493902577
    SARIMA(1, 0, 0)(1, 1, 2, 12) - AIC: 1092.3080510357588
    SARIMA(1, 0, 0)(1, 2, 0, 12) - AIC: 1323.984756681787
    SARIMA(1, 0, 0)(1, 2, 1, 12) - AIC: 1194.022932396092
    SARIMA(1, 0, 0)(1, 2, 2, 12) - AIC: 1072.038213680441
    SARIMA(1, 0, 0)(2, 0, 0, 12) - AIC: 1125.0196322027641
    SARIMA(1, 0, 0)(2, 0, 1, 12) - AIC: 1126.305109128632
    SARIMA(1, 0, 0)(2, 0, 2, 12) - AIC: 1128.2803240086873
    SARIMA(1, 0, 0)(2, 1, 0, 12) - AIC: 1112.4400146740395
    SARIMA(1, 0, 0)(2, 1, 1, 12) - AIC: 1096.026630556145
    SARIMA(1, 0, 0)(2, 1, 2, 12) - AIC: 1092.033624364726
    SARIMA(1, 0, 0)(2, 2, 0, 12) - AIC: 1163.7435098346516
    SARIMA(1, 0, 0)(2, 2, 1, 12) - AIC: 1096.056275936577
    SARIMA(1, 0, 0)(2, 2, 2, 12) - AIC: 1096.4661610795547
    SARIMA(1, 0, 1)(0, 0, 0, 12) - AIC: 1234.420035560365
    SARIMA(1, 0, 1)(0, 0, 1, 12) - AIC: 1179.5640890396603


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(1, 0, 1)(0, 0, 2, 12) - AIC: 1125.8802914274456
    SARIMA(1, 0, 1)(0, 1, 0, 12) - AIC: 1315.491042374868
    SARIMA(1, 0, 1)(0, 1, 1, 12) - AIC: 1141.3816685571664
    SARIMA(1, 0, 1)(0, 1, 2, 12) - AIC: 1093.1759709147475
    SARIMA(1, 0, 1)(0, 2, 0, 12) - AIC: 1471.135771298094
    SARIMA(1, 0, 1)(0, 2, 1, 12) - AIC: 1221.2058988334907
    SARIMA(1, 0, 1)(0, 2, 2, 12) - AIC: 1066.6407231700505
    SARIMA(1, 0, 1)(1, 0, 0, 12) - AIC: 1184.1877273387533
    SARIMA(1, 0, 1)(1, 0, 1, 12) - AIC: 1179.8722584002924
    SARIMA(1, 0, 1)(1, 0, 2, 12) - AIC: 1123.6537151502962
    SARIMA(1, 0, 1)(1, 1, 0, 12) - AIC: 1208.094309685178
    SARIMA(1, 0, 1)(1, 1, 1, 12) - AIC: 1143.379245902223
    SARIMA(1, 0, 1)(1, 1, 2, 12) - AIC: 1087.268577855265
    SARIMA(1, 0, 1)(1, 2, 0, 12) - AIC: 1322.5024149135484
    SARIMA(1, 0, 1)(1, 2, 1, 12) - AIC: 1185.4773888225677
    SARIMA(1, 0, 1)(1, 2, 2, 12) - AIC: 1068.6380135110874
    SARIMA(1, 0, 1)(2, 0, 0, 12) - AIC: 1126.2728509132507
    SARIMA(1, 0, 1)(2, 0, 1, 12) - AIC: 1127.309406180747


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(1, 0, 1)(2, 0, 2, 12) - AIC: 1125.1746893782722
    SARIMA(1, 0, 1)(2, 1, 0, 12) - AIC: 1111.1201858804334
    SARIMA(1, 0, 1)(2, 1, 1, 12) - AIC: 1097.058271897616
    SARIMA(1, 0, 1)(2, 1, 2, 12) - AIC: 1086.9458052058028
    SARIMA(1, 0, 1)(2, 2, 0, 12) - AIC: 1160.8245679596291
    SARIMA(1, 0, 1)(2, 2, 1, 12) - AIC: 1094.4884372158863
    SARIMA(1, 0, 1)(2, 2, 2, 12) - AIC: 1088.8372651906607
    SARIMA(1, 0, 2)(0, 0, 0, 12) - AIC: 1228.996604644098


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(1, 0, 2)(0, 0, 1, 12) - AIC: 1176.7021694083987


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(1, 0, 2)(0, 0, 2, 12) - AIC: 1122.8529132736276
    SARIMA(1, 0, 2)(0, 1, 0, 12) - AIC: 1309.436631532708
    SARIMA(1, 0, 2)(0, 1, 1, 12) - AIC: 1139.2344447260266
    SARIMA(1, 0, 2)(0, 1, 2, 12) - AIC: 1090.6488170333712
    SARIMA(1, 0, 2)(0, 2, 0, 12) - AIC: 1466.9930312494325
    SARIMA(1, 0, 2)(0, 2, 1, 12) - AIC: 1217.554260003428
    SARIMA(1, 0, 2)(0, 2, 2, 12) - AIC: 1062.8663201990967
    SARIMA(1, 0, 2)(1, 0, 0, 12) - AIC: 1185.970508879208


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(1, 0, 2)(1, 0, 1, 12) - AIC: 1177.6455789722108
    SARIMA(1, 0, 2)(1, 0, 2, 12) - AIC: 1120.3210091167143
    SARIMA(1, 0, 2)(1, 1, 0, 12) - AIC: 1210.0237970818873
    SARIMA(1, 0, 2)(1, 1, 1, 12) - AIC: 1141.2331874794304
    SARIMA(1, 0, 2)(1, 1, 2, 12) - AIC: 1085.6957867139563
    SARIMA(1, 0, 2)(1, 2, 0, 12) - AIC: 1323.941762389324
    SARIMA(1, 0, 2)(1, 2, 1, 12) - AIC: 1181.9052018459072
    SARIMA(1, 0, 2)(1, 2, 2, 12) - AIC: 1064.8655118723914
    SARIMA(1, 0, 2)(2, 0, 0, 12) - AIC: 1127.1859603380499


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(1, 0, 2)(2, 0, 1, 12) - AIC: 1128.298112460137


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(1, 0, 2)(2, 0, 2, 12) - AIC: 1121.7182799641866
    SARIMA(1, 0, 2)(2, 1, 0, 12) - AIC: 1112.54069974487
    SARIMA(1, 0, 2)(2, 1, 1, 12) - AIC: 1097.6925870010818
    SARIMA(1, 0, 2)(2, 1, 2, 12) - AIC: 1084.0987979230263
    SARIMA(1, 0, 2)(2, 2, 0, 12) - AIC: 1162.466190906236
    SARIMA(1, 0, 2)(2, 2, 1, 12) - AIC: 1095.9878461513727
    SARIMA(1, 0, 2)(2, 2, 2, 12) - AIC: 1081.5460178439985
    SARIMA(1, 1, 0)(0, 0, 0, 12) - AIC: 1232.5416167015276
    SARIMA(1, 1, 0)(0, 0, 1, 12) - AIC: 1177.7473819747806
    SARIMA(1, 1, 0)(0, 0, 2, 12) - AIC: 1120.756309838843
    SARIMA(1, 1, 0)(0, 1, 0, 12) - AIC: 1322.878798472721
    SARIMA(1, 1, 0)(0, 1, 1, 12) - AIC: 1141.146110267157
    SARIMA(1, 1, 0)(0, 1, 2, 12) - AIC: 1092.032894717092
    SARIMA(1, 1, 0)(0, 2, 0, 12) - AIC: 1484.1056199617842
    SARIMA(1, 1, 0)(0, 2, 1, 12) - AIC: 1226.1095164037488
    SARIMA(1, 1, 0)(0, 2, 2, 12) - AIC: 1067.4720486223648
    SARIMA(1, 1, 0)(1, 0, 0, 12) - AIC: 1178.392974673179
    SARIMA(1, 1, 0)(1, 0, 1, 12) - AIC: 1178.1300749068464
    SARIMA(1, 1, 0)(1, 0, 2, 12) - AIC: 1121.4502548291343
    SARIMA(1, 1, 0)(1, 1, 0, 12) - AIC: 1207.2253247026317
    SARIMA(1, 1, 0)(1, 1, 1, 12) - AIC: 1143.1460517191144
    SARIMA(1, 1, 0)(1, 1, 2, 12) - AIC: 1086.9080582810514
    SARIMA(1, 1, 0)(1, 2, 0, 12) - AIC: 1320.8934932647362
    SARIMA(1, 1, 0)(1, 2, 1, 12) - AIC: 1188.961024743413
    SARIMA(1, 1, 0)(1, 2, 2, 12) - AIC: 1085.5698768130242
    SARIMA(1, 1, 0)(2, 0, 0, 12) - AIC: 1119.8965277678105
    SARIMA(1, 1, 0)(2, 0, 1, 12) - AIC: 1121.2240076966834
    SARIMA(1, 1, 0)(2, 0, 2, 12) - AIC: 1123.21384046758
    SARIMA(1, 1, 0)(2, 1, 0, 12) - AIC: 1108.9169098020889
    SARIMA(1, 1, 0)(2, 1, 1, 12) - AIC: 1091.073747995279
    SARIMA(1, 1, 0)(2, 1, 2, 12) - AIC: 1086.255931770739
    SARIMA(1, 1, 0)(2, 2, 0, 12) - AIC: 1160.4090735963714
    SARIMA(1, 1, 0)(2, 2, 1, 12) - AIC: 1092.0655961232787
    SARIMA(1, 1, 0)(2, 2, 2, 12) - AIC: 1092.3362124681753
    SARIMA(1, 1, 1)(0, 0, 0, 12) - AIC: 1226.973379103324
    SARIMA(1, 1, 1)(0, 0, 1, 12) - AIC: 1174.9134357284593
    SARIMA(1, 1, 1)(0, 0, 2, 12) - AIC: 1117.9477118058692
    SARIMA(1, 1, 1)(0, 1, 0, 12) - AIC: 1316.0763392625572
    SARIMA(1, 1, 1)(0, 1, 1, 12) - AIC: 1137.0631376376766
    SARIMA(1, 1, 1)(0, 1, 2, 12) - AIC: 1089.8639339841957
    SARIMA(1, 1, 1)(0, 2, 0, 12) - AIC: 1472.959900280736
    SARIMA(1, 1, 1)(0, 2, 1, 12) - AIC: 1222.1976066273314
    SARIMA(1, 1, 1)(0, 2, 2, 12) - AIC: 1060.138374025335
    SARIMA(1, 1, 1)(1, 0, 0, 12) - AIC: 1180.2290817018957
    SARIMA(1, 1, 1)(1, 0, 1, 12) - AIC: 1175.7426235523055
    SARIMA(1, 1, 1)(1, 0, 2, 12) - AIC: 1117.1382569862647
    SARIMA(1, 1, 1)(1, 1, 0, 12) - AIC: 1209.061918618745
    SARIMA(1, 1, 1)(1, 1, 1, 12) - AIC: 1139.0629146338365
    SARIMA(1, 1, 1)(1, 1, 2, 12) - AIC: 1083.950479702542
    SARIMA(1, 1, 1)(1, 2, 0, 12) - AIC: 1325.954360310076
    SARIMA(1, 1, 1)(1, 2, 1, 12) - AIC: 1185.0161430543046
    SARIMA(1, 1, 1)(1, 2, 2, 12) - AIC: 1062.1381448471334
    SARIMA(1, 1, 1)(2, 0, 0, 12) - AIC: 1121.7630902156934
    SARIMA(1, 1, 1)(2, 0, 1, 12) - AIC: 1122.8779278547458
    SARIMA(1, 1, 1)(2, 0, 2, 12) - AIC: 1120.7719149335076
    SARIMA(1, 1, 1)(2, 1, 0, 12) - AIC: 1109.7034764157825
    SARIMA(1, 1, 1)(2, 1, 1, 12) - AIC: 1092.552270386118
    SARIMA(1, 1, 1)(2, 1, 2, 12) - AIC: 1083.3561426087292
    SARIMA(1, 1, 1)(2, 2, 0, 12) - AIC: 1154.7506110263885
    SARIMA(1, 1, 1)(2, 2, 1, 12) - AIC: 1092.842468827464
    SARIMA(1, 1, 1)(2, 2, 2, 12) - AIC: 1085.2500552892948
    SARIMA(1, 1, 2)(0, 0, 0, 12) - AIC: 1223.3988851647377
    SARIMA(1, 1, 2)(0, 0, 1, 12) - AIC: 1171.473350743042
    SARIMA(1, 1, 2)(0, 0, 2, 12) - AIC: 1113.8764681182197
    SARIMA(1, 1, 2)(0, 1, 0, 12) - AIC: 1311.4689712526929
    SARIMA(1, 1, 2)(0, 1, 1, 12) - AIC: 1135.991913974463
    SARIMA(1, 1, 2)(0, 1, 2, 12) - AIC: 1084.9776173916896
    SARIMA(1, 1, 2)(0, 2, 0, 12) - AIC: 1470.1746796464631
    SARIMA(1, 1, 2)(0, 2, 1, 12) - AIC: 1216.388459621785
    SARIMA(1, 1, 2)(0, 2, 2, 12) - AIC: 1059.63808405008
    SARIMA(1, 1, 2)(1, 0, 0, 12) - AIC: 1180.944395768498
    SARIMA(1, 1, 2)(1, 0, 1, 12) - AIC: 1173.0358335666674
    SARIMA(1, 1, 2)(1, 0, 2, 12) - AIC: 1115.4370860412637
    SARIMA(1, 1, 2)(1, 1, 0, 12) - AIC: 1210.8129176903367
    SARIMA(1, 1, 2)(1, 1, 1, 12) - AIC: 1137.9915964098811
    SARIMA(1, 1, 2)(1, 1, 2, 12) - AIC: 1078.8878181122593
    SARIMA(1, 1, 2)(1, 2, 0, 12) - AIC: 1324.1935158484944
    SARIMA(1, 1, 2)(1, 2, 1, 12) - AIC: 1181.4349429126928
    SARIMA(1, 1, 2)(1, 2, 2, 12) - AIC: 1076.1189428614498
    SARIMA(1, 1, 2)(2, 0, 0, 12) - AIC: 1121.764821224034
    SARIMA(1, 1, 2)(2, 0, 1, 12) - AIC: 1123.3694375952498
    SARIMA(1, 1, 2)(2, 0, 2, 12) - AIC: 1116.7972313569892
    SARIMA(1, 1, 2)(2, 1, 0, 12) - AIC: 1111.1158306501698
    SARIMA(1, 1, 2)(2, 1, 1, 12) - AIC: 1092.3660862654624
    SARIMA(1, 1, 2)(2, 1, 2, 12) - AIC: 1079.0130249350134
    SARIMA(1, 1, 2)(2, 2, 0, 12) - AIC: 1162.6774659761627
    SARIMA(1, 1, 2)(2, 2, 1, 12) - AIC: 1094.317376279384
    SARIMA(1, 1, 2)(2, 2, 2, 12) - AIC: 1080.5158193373125
    SARIMA(1, 2, 0)(0, 0, 0, 12) - AIC: 1325.7493708745649
    SARIMA(1, 2, 0)(0, 0, 1, 12) - AIC: 1269.8542150759474
    SARIMA(1, 2, 0)(0, 0, 2, 12) - AIC: 1212.6685429922004
    SARIMA(1, 2, 0)(0, 1, 0, 12) - AIC: 1412.7919429250555
    SARIMA(1, 2, 0)(0, 1, 1, 12) - AIC: 1228.689590603405
    SARIMA(1, 2, 0)(0, 1, 2, 12) - AIC: 1173.6661983250654
    SARIMA(1, 2, 0)(0, 2, 0, 12) - AIC: 1577.0677346121922
    SARIMA(1, 2, 0)(0, 2, 1, 12) - AIC: 1311.5678829622698
    SARIMA(1, 2, 0)(0, 2, 2, 12) - AIC: 1145.4337229134599
    SARIMA(1, 2, 0)(1, 0, 0, 12) - AIC: 1270.1063714392492
    SARIMA(1, 2, 0)(1, 0, 1, 12) - AIC: 1271.1994090409003
    SARIMA(1, 2, 0)(1, 0, 2, 12) - AIC: 1212.6980641180426
    SARIMA(1, 2, 0)(1, 1, 0, 12) - AIC: 1288.1968461202648
    SARIMA(1, 2, 0)(1, 1, 1, 12) - AIC: 1230.689541598925
    SARIMA(1, 2, 0)(1, 1, 2, 12) - AIC: 1167.6627914605792
    SARIMA(1, 2, 0)(1, 2, 0, 12) - AIC: 1397.9081407603167
    SARIMA(1, 2, 0)(1, 2, 1, 12) - AIC: 1266.196321746363
    SARIMA(1, 2, 0)(1, 2, 2, 12) - AIC: 1147.4341181051254
    SARIMA(1, 2, 0)(2, 0, 0, 12) - AIC: 1212.4436653725024
    SARIMA(1, 2, 0)(2, 0, 1, 12) - AIC: 1212.953426897719
    SARIMA(1, 2, 0)(2, 0, 2, 12) - AIC: 1214.6979260538953
    SARIMA(1, 2, 0)(2, 1, 0, 12) - AIC: 1192.0941081533642
    SARIMA(1, 2, 0)(2, 1, 1, 12) - AIC: 1179.7515955975232
    SARIMA(1, 2, 0)(2, 1, 2, 12) - AIC: 1169.311399066732
    SARIMA(1, 2, 0)(2, 2, 0, 12) - AIC: 1234.5374576508407
    SARIMA(1, 2, 0)(2, 2, 1, 12) - AIC: 1169.516629290918
    SARIMA(1, 2, 0)(2, 2, 2, 12) - AIC: 1166.6056344345773
    SARIMA(1, 2, 1)(0, 0, 0, 12) - AIC: 1230.7069089189795
    SARIMA(1, 2, 1)(0, 0, 1, 12) - AIC: 1175.4821876441201
    SARIMA(1, 2, 1)(0, 0, 2, 12) - AIC: 1118.3225862270142
    SARIMA(1, 2, 1)(0, 1, 0, 12) - AIC: 1318.8299691007978
    SARIMA(1, 2, 1)(0, 1, 1, 12) - AIC: 1142.4571776969935
    SARIMA(1, 2, 1)(0, 1, 2, 12) - AIC: 1091.9262979598843
    SARIMA(1, 2, 1)(0, 2, 0, 12) - AIC: 1478.2898575591194
    SARIMA(1, 2, 1)(0, 2, 1, 12) - AIC: 1228.493803573788
    SARIMA(1, 2, 1)(0, 2, 2, 12) - AIC: 1067.5789453391549
    SARIMA(1, 2, 1)(1, 0, 0, 12) - AIC: 1180.7105053328264
    SARIMA(1, 2, 1)(1, 0, 1, 12) - AIC: 1175.7438243985405
    SARIMA(1, 2, 1)(1, 0, 2, 12) - AIC: 1119.3499514338105
    SARIMA(1, 2, 1)(1, 1, 0, 12) - AIC: 1209.6465199748409
    SARIMA(1, 2, 1)(1, 1, 1, 12) - AIC: 1145.8539126672968
    SARIMA(1, 2, 1)(1, 1, 2, 12) - AIC: 1083.7393585926163
    SARIMA(1, 2, 1)(1, 2, 0, 12) - AIC: 1323.3141503541124
    SARIMA(1, 2, 1)(1, 2, 1, 12) - AIC: 1190.3113487800656
    SARIMA(1, 2, 1)(1, 2, 2, 12) - AIC: 1069.106113485948
    SARIMA(1, 2, 1)(2, 0, 0, 12) - AIC: 1122.7306101796435
    SARIMA(1, 2, 1)(2, 0, 1, 12) - AIC: 1123.8622743951305
    SARIMA(1, 2, 1)(2, 0, 2, 12) - AIC: 1120.8364391494822
    SARIMA(1, 2, 1)(2, 1, 0, 12) - AIC: 1110.1421564663472
    SARIMA(1, 2, 1)(2, 1, 1, 12) - AIC: 1095.1718447233357
    SARIMA(1, 2, 1)(2, 1, 2, 12) - AIC: 1084.9599671028993
    SARIMA(1, 2, 1)(2, 2, 0, 12) - AIC: 1160.8729919666405
    SARIMA(1, 2, 1)(2, 2, 1, 12) - AIC: 1095.9273334534926
    SARIMA(1, 2, 1)(2, 2, 2, 12) - AIC: 1086.9409862029274
    SARIMA(1, 2, 2)(0, 0, 0, 12) - AIC: 1227.8340048264704
    SARIMA(1, 2, 2)(0, 0, 1, 12) - AIC: 1172.4420973159768
    SARIMA(1, 2, 2)(0, 0, 2, 12) - AIC: 1113.5899443653384
    SARIMA(1, 2, 2)(0, 1, 0, 12) - AIC: 1315.4196943831123
    SARIMA(1, 2, 2)(0, 1, 1, 12) - AIC: 1139.7130683019263
    SARIMA(1, 2, 2)(0, 1, 2, 12) - AIC: 1087.081159616581
    SARIMA(1, 2, 2)(0, 2, 0, 12) - AIC: 1474.0329351200155
    SARIMA(1, 2, 2)(0, 2, 1, 12) - AIC: 1222.8148275231733
    SARIMA(1, 2, 2)(0, 2, 2, 12) - AIC: 1062.8280374328024
    SARIMA(1, 2, 2)(1, 0, 0, 12) - AIC: 1182.5184820415539
    SARIMA(1, 2, 2)(1, 0, 1, 12) - AIC: 1171.8484857121114
    SARIMA(1, 2, 2)(1, 0, 2, 12) - AIC: 1115.2272553566204
    SARIMA(1, 2, 2)(1, 1, 0, 12) - AIC: 1211.5092718330209
    SARIMA(1, 2, 2)(1, 1, 1, 12) - AIC: 1141.5608979859658
    SARIMA(1, 2, 2)(1, 1, 2, 12) - AIC: 1076.7061249785338
    SARIMA(1, 2, 2)(1, 2, 0, 12) - AIC: 1325.2463927749186
    SARIMA(1, 2, 2)(1, 2, 1, 12) - AIC: 1184.9634753604282
    SARIMA(1, 2, 2)(1, 2, 2, 12) - AIC: 1064.0806091623817
    SARIMA(1, 2, 2)(2, 0, 0, 12) - AIC: 1124.4242016167684
    SARIMA(1, 2, 2)(2, 0, 1, 12) - AIC: 1125.5740636388748
    SARIMA(1, 2, 2)(2, 0, 2, 12) - AIC: 1116.2692478684787
    SARIMA(1, 2, 2)(2, 1, 0, 12) - AIC: 1111.1494400383935
    SARIMA(1, 2, 2)(2, 1, 1, 12) - AIC: 1096.6445255241945
    SARIMA(1, 2, 2)(2, 1, 2, 12) - AIC: 1080.7195947237478
    SARIMA(1, 2, 2)(2, 2, 0, 12) - AIC: 1155.0056115851626
    SARIMA(1, 2, 2)(2, 2, 1, 12) - AIC: 1096.412148848525
    SARIMA(1, 2, 2)(2, 2, 2, 12) - AIC: 1082.5726482227942
    SARIMA(2, 0, 0)(0, 0, 0, 12) - AIC: 1234.4737227808619


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 0, 0)(0, 0, 1, 12) - AIC: 1183.6764525315316


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 0, 0)(0, 0, 2, 12) - AIC: 1132.073527790566
    SARIMA(2, 0, 0)(0, 1, 0, 12) - AIC: 1315.56901185883
    SARIMA(2, 0, 0)(0, 1, 1, 12) - AIC: 1146.3003941427974
    SARIMA(2, 0, 0)(0, 1, 2, 12) - AIC: 1098.1674920647708
    SARIMA(2, 0, 0)(0, 2, 0, 12) - AIC: 1471.2577684462226
    SARIMA(2, 0, 0)(0, 2, 1, 12) - AIC: 1227.4696767345854
    SARIMA(2, 0, 0)(0, 2, 2, 12) - AIC: 1072.1223309430657
    SARIMA(2, 0, 0)(1, 0, 0, 12) - AIC: 1180.2498202932572
    SARIMA(2, 0, 0)(1, 0, 1, 12) - AIC: 1179.9938265688925
    SARIMA(2, 0, 0)(1, 0, 2, 12) - AIC: 1127.8329406425482
    SARIMA(2, 0, 0)(1, 1, 0, 12) - AIC: 1203.1354072496913
    SARIMA(2, 0, 0)(1, 1, 1, 12) - AIC: 1143.6215429447948
    SARIMA(2, 0, 0)(1, 1, 2, 12) - AIC: 1092.6221564865612
    SARIMA(2, 0, 0)(1, 2, 0, 12) - AIC: 1316.0506182669012
    SARIMA(2, 0, 0)(1, 2, 1, 12) - AIC: 1185.4833169083936
    SARIMA(2, 0, 0)(1, 2, 2, 12) - AIC: 1074.1222845156453
    SARIMA(2, 0, 0)(2, 0, 0, 12) - AIC: 1121.8918857481635
    SARIMA(2, 0, 0)(2, 0, 1, 12) - AIC: 1123.284299477989


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 0, 0)(2, 0, 2, 12) - AIC: 1125.2545367058472
    SARIMA(2, 0, 0)(2, 1, 0, 12) - AIC: 1106.9609339913445
    SARIMA(2, 0, 0)(2, 1, 1, 12) - AIC: 1091.7790200491845
    SARIMA(2, 0, 0)(2, 1, 2, 12) - AIC: 1087.1526087538375
    SARIMA(2, 0, 0)(2, 2, 0, 12) - AIC: 1156.2644191645309
    SARIMA(2, 0, 0)(2, 2, 1, 12) - AIC: 1090.0569786510987
    SARIMA(2, 0, 0)(2, 2, 2, 12) - AIC: 1089.2769502228941
    SARIMA(2, 0, 1)(0, 0, 0, 12) - AIC: 1236.4184803882538


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 0, 1)(0, 0, 1, 12) - AIC: 1181.5534298648479


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 0, 1)(0, 0, 2, 12) - AIC: 1128.1285293980322
    SARIMA(2, 0, 1)(0, 1, 0, 12) - AIC: 1317.4543991806254
    SARIMA(2, 0, 1)(0, 1, 1, 12) - AIC: 1143.1445007016184
    SARIMA(2, 0, 1)(0, 1, 2, 12) - AIC: 1095.1645235665142
    SARIMA(2, 0, 1)(0, 2, 0, 12) - AIC: 1473.1193681440404
    SARIMA(2, 0, 1)(0, 2, 1, 12) - AIC: 1223.2058985500053
    SARIMA(2, 0, 1)(0, 2, 2, 12) - AIC: 1067.5584382176062
    SARIMA(2, 0, 1)(1, 0, 0, 12) - AIC: 1182.154704707992


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 0, 1)(1, 0, 1, 12) - AIC: 1181.8246535597557


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 0, 1)(1, 0, 2, 12) - AIC: 1125.773709753221
    SARIMA(2, 0, 1)(1, 1, 0, 12) - AIC: 1205.0927852995296
    SARIMA(2, 0, 1)(1, 1, 1, 12) - AIC: 1145.1427546227637
    SARIMA(2, 0, 1)(1, 1, 2, 12) - AIC: 1089.0780088309346
    SARIMA(2, 0, 1)(1, 2, 0, 12) - AIC: 1318.0506173737276
    SARIMA(2, 0, 1)(1, 2, 1, 12) - AIC: 1187.4388819396472


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 0, 1)(1, 2, 2, 12) - AIC: 1069.5842776351064


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 0, 1)(2, 0, 0, 12) - AIC: 1123.824684756145
    SARIMA(2, 0, 1)(2, 0, 1, 12) - AIC: 1125.3611248504046


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 0, 1)(2, 0, 2, 12) - AIC: 1127.3185527428739
    SARIMA(2, 0, 1)(2, 1, 0, 12) - AIC: 1108.166933167227
    SARIMA(2, 0, 1)(2, 1, 1, 12) - AIC: 1093.277871804367
    SARIMA(2, 0, 1)(2, 1, 2, 12) - AIC: 1088.8189090830706
    SARIMA(2, 0, 1)(2, 2, 0, 12) - AIC: 1157.4105437579346
    SARIMA(2, 0, 1)(2, 2, 1, 12) - AIC: 1091.2820669723576
    SARIMA(2, 0, 1)(2, 2, 2, 12) - AIC: 1090.7116582223994
    SARIMA(2, 0, 2)(0, 0, 0, 12) - AIC: 1229.1133724928159


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 0, 2)(0, 0, 1, 12) - AIC: 1178.0868543984877


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 0, 2)(0, 0, 2, 12) - AIC: 1134.9045079599427
    SARIMA(2, 0, 2)(0, 1, 0, 12) - AIC: 1311.3636071083588
    SARIMA(2, 0, 2)(0, 1, 1, 12) - AIC: 1141.1349275595355


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 0, 2)(0, 1, 2, 12) - AIC: 1090.2270108714952
    SARIMA(2, 0, 2)(0, 2, 0, 12) - AIC: 1468.613132499469
    SARIMA(2, 0, 2)(0, 2, 1, 12) - AIC: 1219.1340999408199
    SARIMA(2, 0, 2)(0, 2, 2, 12) - AIC: 1064.6180103162555
    SARIMA(2, 0, 2)(1, 0, 0, 12) - AIC: 1182.8264899704004


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 0, 2)(1, 0, 1, 12) - AIC: 1180.5877768893554


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 0, 2)(1, 0, 2, 12) - AIC: 1122.3114746897104
    SARIMA(2, 0, 2)(1, 1, 0, 12) - AIC: 1206.9510357069962
    SARIMA(2, 0, 2)(1, 1, 1, 12) - AIC: 1143.1700746926876


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 0, 2)(1, 1, 2, 12) - AIC: 1086.125918309167
    SARIMA(2, 0, 2)(1, 2, 0, 12) - AIC: 1319.3007287220341
    SARIMA(2, 0, 2)(1, 2, 1, 12) - AIC: 1183.809783625461


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 0, 2)(1, 2, 2, 12) - AIC: 1066.6190737881673


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 0, 2)(2, 0, 0, 12) - AIC: 1124.528963257328


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 0, 2)(2, 0, 1, 12) - AIC: 1125.6698086900142


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 0, 2)(2, 0, 2, 12) - AIC: 1124.1261079952676
    SARIMA(2, 0, 2)(2, 1, 0, 12) - AIC: 1109.784081250805
    SARIMA(2, 0, 2)(2, 1, 1, 12) - AIC: 1093.4973476194903


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 0, 2)(2, 1, 2, 12) - AIC: 1085.8666426946988
    SARIMA(2, 0, 2)(2, 2, 0, 12) - AIC: 1159.3053444352508
    SARIMA(2, 0, 2)(2, 2, 1, 12) - AIC: 1092.9663724171105
    SARIMA(2, 0, 2)(2, 2, 2, 12) - AIC: 1083.5318341321686
    SARIMA(2, 1, 0)(0, 0, 0, 12) - AIC: 1226.652153135218
    SARIMA(2, 1, 0)(0, 0, 1, 12) - AIC: 1179.1290766204374
    SARIMA(2, 1, 0)(0, 0, 2, 12) - AIC: 1121.5870345182193
    SARIMA(2, 1, 0)(0, 1, 0, 12) - AIC: 1315.0748695917077
    SARIMA(2, 1, 0)(0, 1, 1, 12) - AIC: 1142.180273415232
    SARIMA(2, 1, 0)(0, 1, 2, 12) - AIC: 1093.490250931649
    SARIMA(2, 1, 0)(0, 2, 0, 12) - AIC: 1476.4228963292512
    SARIMA(2, 1, 0)(0, 2, 1, 12) - AIC: 1226.7602172707534
    SARIMA(2, 1, 0)(0, 2, 2, 12) - AIC: 1068.2586184962834
    SARIMA(2, 1, 0)(1, 0, 0, 12) - AIC: 1174.961361643169
    SARIMA(2, 1, 0)(1, 0, 1, 12) - AIC: 1175.4649302831672
    SARIMA(2, 1, 0)(1, 0, 2, 12) - AIC: 1122.1897319113118
    SARIMA(2, 1, 0)(1, 1, 0, 12) - AIC: 1203.5773109380345
    SARIMA(2, 1, 0)(1, 1, 1, 12) - AIC: 1140.216409309934
    SARIMA(2, 1, 0)(1, 1, 2, 12) - AIC: 1088.2095841986775
    SARIMA(2, 1, 0)(1, 2, 0, 12) - AIC: 1316.6553052603958
    SARIMA(2, 1, 0)(1, 2, 1, 12) - AIC: 1185.0854936668115


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 1, 0)(1, 2, 2, 12) - AIC: 1070.2577134283777
    SARIMA(2, 1, 0)(2, 0, 0, 12) - AIC: 1116.17435270023
    SARIMA(2, 1, 0)(2, 0, 1, 12) - AIC: 1117.6086983929195
    SARIMA(2, 1, 0)(2, 0, 2, 12) - AIC: 1119.552583132843
    SARIMA(2, 1, 0)(2, 1, 0, 12) - AIC: 1103.576039136915
    SARIMA(2, 1, 0)(2, 1, 1, 12) - AIC: 1086.2578622937235
    SARIMA(2, 1, 0)(2, 1, 2, 12) - AIC: 1082.799761604168
    SARIMA(2, 1, 0)(2, 2, 0, 12) - AIC: 1151.935587540022
    SARIMA(2, 1, 0)(2, 2, 1, 12) - AIC: 1084.7340730693015
    SARIMA(2, 1, 0)(2, 2, 2, 12) - AIC: 1084.7981403903998
    SARIMA(2, 1, 1)(0, 0, 0, 12) - AIC: 1226.985042993016
    SARIMA(2, 1, 1)(0, 0, 1, 12) - AIC: 1176.1366803550386
    SARIMA(2, 1, 1)(0, 0, 2, 12) - AIC: 1117.8016495428715
    SARIMA(2, 1, 1)(0, 1, 0, 12) - AIC: 1315.5196241022359
    SARIMA(2, 1, 1)(0, 1, 1, 12) - AIC: 1139.7965463011192
    SARIMA(2, 1, 1)(0, 1, 2, 12) - AIC: 1090.8002115624267
    SARIMA(2, 1, 1)(0, 2, 0, 12) - AIC: 1471.318344544196
    SARIMA(2, 1, 1)(0, 2, 1, 12) - AIC: 1221.0539162694295
    SARIMA(2, 1, 1)(0, 2, 2, 12) - AIC: 1062.1383599710043
    SARIMA(2, 1, 1)(1, 0, 0, 12) - AIC: 1175.7155894656398


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 1, 1)(1, 0, 1, 12) - AIC: 1176.7056030519157
    SARIMA(2, 1, 1)(1, 0, 2, 12) - AIC: 1119.1407553188437
    SARIMA(2, 1, 1)(1, 1, 0, 12) - AIC: 1205.8692063931876
    SARIMA(2, 1, 1)(1, 1, 1, 12) - AIC: 1141.7949552967493
    SARIMA(2, 1, 1)(1, 1, 2, 12) - AIC: 1084.3688089108164
    SARIMA(2, 1, 1)(1, 2, 0, 12) - AIC: 1318.304769001128
    SARIMA(2, 1, 1)(1, 2, 1, 12) - AIC: 1186.4919041408853


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 1, 1)(1, 2, 2, 12) - AIC: 1064.7066990493372
    SARIMA(2, 1, 1)(2, 0, 0, 12) - AIC: 1117.0318902388801
    SARIMA(2, 1, 1)(2, 0, 1, 12) - AIC: 1118.6536734665808
    SARIMA(2, 1, 1)(2, 0, 2, 12) - AIC: 1120.6044167535777
    SARIMA(2, 1, 1)(2, 1, 0, 12) - AIC: 1105.7004145099177


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 1, 1)(2, 1, 1, 12) - AIC: 1086.8089066297653
    SARIMA(2, 1, 1)(2, 1, 2, 12) - AIC: 1083.805542911227
    SARIMA(2, 1, 1)(2, 2, 0, 12) - AIC: 1148.5359148216837
    SARIMA(2, 1, 1)(2, 2, 1, 12) - AIC: 1086.472184264438
    SARIMA(2, 1, 1)(2, 2, 2, 12) - AIC: 1088.134307811798
    SARIMA(2, 1, 2)(0, 0, 0, 12) - AIC: 1225.6592861330573
    SARIMA(2, 1, 2)(0, 0, 1, 12) - AIC: 1171.6114256652722
    SARIMA(2, 1, 2)(0, 0, 2, 12) - AIC: 1110.226043238397
    SARIMA(2, 1, 2)(0, 1, 0, 12) - AIC: 1298.943303904589
    SARIMA(2, 1, 2)(0, 1, 1, 12) - AIC: 1131.5111285543576
    SARIMA(2, 1, 2)(0, 1, 2, 12) - AIC: 1083.2054306704345
    SARIMA(2, 1, 2)(0, 2, 0, 12) - AIC: 1460.7749946611225
    SARIMA(2, 1, 2)(0, 2, 1, 12) - AIC: 1206.3083397680296


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 1, 2)(0, 2, 2, 12) - AIC: 1055.2972871173502
    SARIMA(2, 1, 2)(1, 0, 0, 12) - AIC: 1177.0050739268477
    SARIMA(2, 1, 2)(1, 0, 1, 12) - AIC: 1170.514701084226
    SARIMA(2, 1, 2)(1, 0, 2, 12) - AIC: 1111.3371178085433
    SARIMA(2, 1, 2)(1, 1, 0, 12) - AIC: 1205.1129920404896
    SARIMA(2, 1, 2)(1, 1, 1, 12) - AIC: 1133.5154690106974
    SARIMA(2, 1, 2)(1, 1, 2, 12) - AIC: 1076.7412573222732
    SARIMA(2, 1, 2)(1, 2, 0, 12) - AIC: 1307.4872279987694
    SARIMA(2, 1, 2)(1, 2, 1, 12) - AIC: 1180.6231990942083


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 1, 2)(1, 2, 2, 12) - AIC: 1062.3118070138007
    SARIMA(2, 1, 2)(2, 0, 0, 12) - AIC: 1114.3291463097487
    SARIMA(2, 1, 2)(2, 0, 1, 12) - AIC: 1116.3076187550137


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 1, 2)(2, 0, 2, 12) - AIC: 1113.2509236154876
    SARIMA(2, 1, 2)(2, 1, 0, 12) - AIC: 1106.303379061537
    SARIMA(2, 1, 2)(2, 1, 1, 12) - AIC: 1086.1172459241416


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 1, 2)(2, 1, 2, 12) - AIC: 1072.3731261132823
    SARIMA(2, 1, 2)(2, 2, 0, 12) - AIC: 1158.0411819678154
    SARIMA(2, 1, 2)(2, 2, 1, 12) - AIC: 1090.3492691608772
    SARIMA(2, 1, 2)(2, 2, 2, 12) - AIC: 1082.0624600759452
    SARIMA(2, 2, 0)(0, 0, 0, 12) - AIC: 1303.7871443590668
    SARIMA(2, 2, 0)(0, 0, 1, 12) - AIC: 1252.9166721476558
    SARIMA(2, 2, 0)(0, 0, 2, 12) - AIC: 1194.984868376605
    SARIMA(2, 2, 0)(0, 1, 0, 12) - AIC: 1394.6700635978577
    SARIMA(2, 2, 0)(0, 1, 1, 12) - AIC: 1212.306751813165
    SARIMA(2, 2, 0)(0, 1, 2, 12) - AIC: 1157.5599909594816
    SARIMA(2, 2, 0)(0, 2, 0, 12) - AIC: 1560.2270439190875
    SARIMA(2, 2, 0)(0, 2, 1, 12) - AIC: 1299.8792257591012
    SARIMA(2, 2, 0)(0, 2, 2, 12) - AIC: 1129.2285087320465
    SARIMA(2, 2, 0)(1, 0, 0, 12) - AIC: 1248.6176088775087
    SARIMA(2, 2, 0)(1, 0, 1, 12) - AIC: 1250.0352296547742
    SARIMA(2, 2, 0)(1, 0, 2, 12) - AIC: 1194.3897764966998
    SARIMA(2, 2, 0)(1, 1, 0, 12) - AIC: 1267.4168336106686
    SARIMA(2, 2, 0)(1, 1, 1, 12) - AIC: 1209.1480415398169
    SARIMA(2, 2, 0)(1, 1, 2, 12) - AIC: 1152.7195515533922
    SARIMA(2, 2, 0)(1, 2, 0, 12) - AIC: 1382.1959839343892
    SARIMA(2, 2, 0)(1, 2, 1, 12) - AIC: 1246.2476779863619
    SARIMA(2, 2, 0)(1, 2, 2, 12) - AIC: 1131.228345981539
    SARIMA(2, 2, 0)(2, 0, 0, 12) - AIC: 1189.247814669478
    SARIMA(2, 2, 0)(2, 0, 1, 12) - AIC: 1189.1011018039435
    SARIMA(2, 2, 0)(2, 0, 2, 12) - AIC: 1190.9882000370958
    SARIMA(2, 2, 0)(2, 1, 0, 12) - AIC: 1164.1596210108619
    SARIMA(2, 2, 0)(2, 1, 1, 12) - AIC: 1156.344760371727
    SARIMA(2, 2, 0)(2, 1, 2, 12) - AIC: 1144.603928050347
    SARIMA(2, 2, 0)(2, 2, 0, 12) - AIC: 1204.7719500132466
    SARIMA(2, 2, 0)(2, 2, 1, 12) - AIC: 1140.3556020030596
    SARIMA(2, 2, 0)(2, 2, 2, 12) - AIC: 1136.901142531931
    SARIMA(2, 2, 1)(0, 0, 0, 12) - AIC: 1229.8751511515952
    SARIMA(2, 2, 1)(0, 0, 1, 12) - AIC: 1176.8031634502445
    SARIMA(2, 2, 1)(0, 0, 2, 12) - AIC: 1118.9428544540708
    SARIMA(2, 2, 1)(0, 1, 0, 12) - AIC: 1319.539027595026
    SARIMA(2, 2, 1)(0, 1, 1, 12) - AIC: 1143.3002623952054
    SARIMA(2, 2, 1)(0, 1, 2, 12) - AIC: 1093.3476888821942
    SARIMA(2, 2, 1)(0, 2, 0, 12) - AIC: 1477.7450658857758
    SARIMA(2, 2, 1)(0, 2, 1, 12) - AIC: 1229.5302711239397
    SARIMA(2, 2, 1)(0, 2, 2, 12) - AIC: 1068.272685884125
    SARIMA(2, 2, 1)(1, 0, 0, 12) - AIC: 1177.6263784487192
    SARIMA(2, 2, 1)(1, 0, 1, 12) - AIC: 1177.1730230146827
    SARIMA(2, 2, 1)(1, 0, 2, 12) - AIC: 1119.7127318389892
    SARIMA(2, 2, 1)(1, 1, 0, 12) - AIC: 1206.0747737164563
    SARIMA(2, 2, 1)(1, 1, 1, 12) - AIC: 1145.296804778968
    SARIMA(2, 2, 1)(1, 1, 2, 12) - AIC: 1085.0387299846002
    SARIMA(2, 2, 1)(1, 2, 0, 12) - AIC: 1319.4685216549947
    SARIMA(2, 2, 1)(1, 2, 1, 12) - AIC: 1191.9578006118022


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 2, 1)(1, 2, 2, 12) - AIC: 1070.2237885843485
    SARIMA(2, 2, 1)(2, 0, 0, 12) - AIC: 1118.039761416192
    SARIMA(2, 2, 1)(2, 0, 1, 12) - AIC: 1119.3268083319424
    SARIMA(2, 2, 1)(2, 0, 2, 12) - AIC: 1121.323235617162
    SARIMA(2, 2, 1)(2, 1, 0, 12) - AIC: 1104.8008215179802
    SARIMA(2, 2, 1)(2, 1, 1, 12) - AIC: 1089.7554681606362
    SARIMA(2, 2, 1)(2, 1, 2, 12) - AIC: 1086.1884795010092
    SARIMA(2, 2, 1)(2, 2, 0, 12) - AIC: 1152.2774885500357
    SARIMA(2, 2, 1)(2, 2, 1, 12) - AIC: 1086.5119744092508
    SARIMA(2, 2, 1)(2, 2, 2, 12) - AIC: 1086.449655839379
    SARIMA(2, 2, 2)(0, 0, 0, 12) - AIC: 1228.1481818009315
    SARIMA(2, 2, 2)(0, 0, 1, 12) - AIC: 1173.318748356809
    SARIMA(2, 2, 2)(0, 0, 2, 12) - AIC: 1112.0945246508286
    SARIMA(2, 2, 2)(0, 1, 0, 12) - AIC: 1314.5602084902791
    SARIMA(2, 2, 2)(0, 1, 1, 12) - AIC: 1141.014053431779
    SARIMA(2, 2, 2)(0, 1, 2, 12) - AIC: 1087.447647446746
    SARIMA(2, 2, 2)(0, 2, 0, 12) - AIC: 1469.55702610512
    SARIMA(2, 2, 2)(0, 2, 1, 12) - AIC: 1225.0395260079865
    SARIMA(2, 2, 2)(0, 2, 2, 12) - AIC: 1064.6614027495798
    SARIMA(2, 2, 2)(1, 0, 0, 12) - AIC: 1179.1794687141619
    SARIMA(2, 2, 2)(1, 0, 1, 12) - AIC: 1173.843410145089
    SARIMA(2, 2, 2)(1, 0, 2, 12) - AIC: 1113.732761686236
    SARIMA(2, 2, 2)(1, 1, 0, 12) - AIC: 1206.2683168899355


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 2, 2)(1, 1, 1, 12) - AIC: 1144.1510209267649
    SARIMA(2, 2, 2)(1, 1, 2, 12) - AIC: 1078.7058496716413
    SARIMA(2, 2, 2)(1, 2, 0, 12) - AIC: 1321.736186992774
    SARIMA(2, 2, 2)(1, 2, 1, 12) - AIC: 1185.9298055267136


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    


    SARIMA(2, 2, 2)(1, 2, 2, 12) - AIC: 1066.3971899598587
    SARIMA(2, 2, 2)(2, 0, 0, 12) - AIC: 1118.3849469381053
    SARIMA(2, 2, 2)(2, 0, 1, 12) - AIC: 1119.7640781447465
    SARIMA(2, 2, 2)(2, 0, 2, 12) - AIC: 1115.1871622240242
    SARIMA(2, 2, 2)(2, 1, 0, 12) - AIC: 1107.6700023955286
    SARIMA(2, 2, 2)(2, 1, 1, 12) - AIC: 1089.9734207392398
    SARIMA(2, 2, 2)(2, 1, 2, 12) - AIC: 1081.8522223744822
    SARIMA(2, 2, 2)(2, 2, 0, 12) - AIC: 1148.4782608982675
    SARIMA(2, 2, 2)(2, 2, 1, 12) - AIC: 1083.5731869065223
    SARIMA(2, 2, 2)(2, 2, 2, 12) - AIC: 1083.8394403537395
    Best SARIMA model - AIC: (2, 1, 2), (0, 2, 2, 12), 1055.2972871173502


    /usr/local/lib/python3.10/dist-packages/statsmodels/base/model.py:607: ConvergenceWarning:
    
    Maximum Likelihood optimization failed to converge. Check mle_retvals
    



    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-16-0ee908b777d3> in <cell line: 49>()
         47 predictions = model_fit.predict(len(df['Close']), len(df['Close'])+29)
         48 
    ---> 49 predictions.summary()
         50 
         51 


    /usr/local/lib/python3.10/dist-packages/pandas/core/generic.py in __getattr__(self, name)
       5900         ):
       5901             return self[name]
    -> 5902         return object.__getattribute__(self, name)
       5903 
       5904     def __setattr__(self, name: str, value) -> None:


    AttributeError: 'Series' object has no attribute 'summary'



```python


# plot the predictions
plt.figure(figsize=(10, 5))

plt.plot(df["Close"], label='Actual')
plt.plot(predictions, color="red", label='Predicted')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title(f'{ticker} Stock Price')
plt.legend(loc='upper left')

plt.show()
```


    
![Alt Text](https://i.ibb.co/hsn2VmK/10.png)

    


