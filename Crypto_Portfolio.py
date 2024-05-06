#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import statsmodels.api as sm
import numpy as np

import sqlite3
import statsmodels.formula.api as smf


# In[3]:


import pandas as pd

url = 'https://raw.githubusercontent.com/Ajim63/FinTech-Teaching-Materials/main/End%20of%20Year%20Project%20Data_Cryto.csv'
df = pd.read_csv(url, index_col=0, parse_dates=['Date'])


# In[4]:


# Calculate logarithmic returns
df_ret = np.log(df) - np.log(df.shift(1))
df_ret = df_ret.replace([np.inf, -np.inf], np.nan)

# Remove the first observation as it is missing for all
df_ret = df_ret.iloc[1:]

# Extract lists of cryptocurrencies and dates
crypto_list = list(df_ret.columns)
date_list = list(df_ret.index)

# Calculate statistics for each cryptocurrency
res = pd.DataFrame()
for crypto in crypto_list:
    X = df_ret[crypto]

    for i in range(42, len(date_list) + 1):
        df_42 = X.iloc[i - 42:i]

        w = [df_42.mean(), df_42.median(), df_42.max(), df_42.min(), df_42.quantile(0.05), df_42.quantile(0.95), df_42.std(), df_42.skew(), df_42.kurt()]
        w = pd.DataFrame(w).T
        w['Date'] = date_list[i - 1]
        w['crypto'] = crypto
        res = pd.concat([res, w], axis=0)

# Rename columns for clarity
res.rename(columns={0: 'mean', 1: 'median', 2: 'max', 3: 'min', 4: 'q_05', 5: 'q_95', 6: 'std', 7: 'skew', 8: 'kurt'}, inplace=True)

# Transform logarithmic returns DataFrame
crypto_ret = df_ret.unstack().reset_index().rename(columns={"level_0": "crypto", 0: "Ret"})

# Merge statistics with logarithmic returns
crypto_ret = res.merge(crypto_ret, on=['crypto', 'Date'])

# Display the merged DataFrame
print(crypto_ret)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[13]:


# Create a subset of your data with relevant columns do for all pther variables and for bivariated you do "Mean and anyother"
# Create a subset of your data with relevant columns
X = crypto_ret[['crypto', 'Date','mean', 'median', 'max', 'min', 'q_05', 'q_95', 'std', 'skew', 'kurt', 'Ret']]
 # Assuming 'mean' corresponds to 'm_1' and 'Ret' corresponds to 'ex_ret'

# Drop any remaining rows with missing values
X.dropna(inplace=True)

# Sort data and calculate breakpoints
grp_x = X.groupby(['Date'])['mean'].describe(percentiles=[0.2, 0.4, 0.6, 0.8]).reset_index()
grp_x = grp_x[['Date', '20%', '40%', '60%', '80%']]

# Merge breakpoints with the main dataframe
X = X.merge(grp_x, how='left', on=['Date'])

# Use the breakpoints to form the portfolios
def grp_bucket(row):
    if row['mean'] <= row['20%']:
        value = 'H20'
    elif row['mean'] <= row['40%']:
        value = 'H40'
    elif row['mean'] <= row['60%']:
        value = 'H60'
    elif row['mean'] <= row['80%']:
        value = 'H80'
    elif row['mean'] > row['80%']:
        value = 'H100'
    else:
        value = ''
    return value

X['port'] = X.apply(grp_bucket, axis=1)
import pandas as pd


# Create a subset of your data with relevant columns
X = crypto_ret[['crypto', 'Date', 'mean', 'Ret']]  # Assuming 'mean' corresponds to 'm_1' and 'Ret' corresponds to 'ex_ret'

# Drop any remaining rows with missing values
X.dropna(inplace=True)

# Sort data and calculate breakpoints
grp_x = X.groupby(['Date'])['mean'].describe(percentiles=[0.2, 0.4, 0.6, 0.8]).reset_index()
grp_x = grp_x[['Date', '20%', '40%', '60%', '80%']]

# Merge breakpoints with the main dataframe
X = X.merge(grp_x, how='left', on=['Date'])

# Use the breakpoints to form the portfolios
def grp_bucket(row):
    if row['mean'] <= row['20%']:
        value = 'H20'
    elif row['mean'] <= row['40%']:
        value = 'H40'
    elif row['mean'] <= row['60%']:
        value = 'H60'
    elif row['mean'] <= row['80%']:
        value = 'H80'
    elif row['mean'] > row['80%']:
        value = 'H100'
    else:
        value = ''
    return value

X['port'] = X.apply(grp_bucket, axis=1)

import pandas as pd



# Create a subset of your data with relevant columns
X = crypto_ret[['crypto', 'Date', 'mean', 'Ret']]  # Assuming 'mean' corresponds to 'm_1' and 'Ret' corresponds to 'ex_ret'

# Drop any remaining rows with missing values
X.dropna(inplace=True)

# Sort data and calculate breakpoints
grp_x = X.groupby(['Date'])['mean'].describe(percentiles=[0.2, 0.4, 0.6, 0.8]).reset_index()
grp_x = grp_x[['Date', '20%', '40%', '60%', '80%']]

# Merge breakpoints with the main dataframe
X = X.merge(grp_x, how='left', on=['Date'])

# Use the breakpoints to form the portfolios
def grp_bucket(row):
    if row['mean'] <= row['20%']:
        value = 'H20'
    elif row['mean'] <= row['40%']:
        value = 'H40'
    elif row['mean'] <= row['60%']:
        value = 'H60'
    elif row['mean'] <= row['80%']:
        value = 'H80'
    elif row['mean'] > row['80%']:
        value = 'H100'
    else:
        value = ''
    return value

X['port'] = X.apply(grp_bucket, axis=1)
Portfolio = X.groupby(['Date', 'port'])['Ret'].mean().reset_index()
print(X)


# In[14]:


# Calculate average value of the outcome variable 'Ret' (assuming 'Ret' corresponds to 'ex_ret' in your DataFrame)

Portfolio = X.groupby(['Date', 'port'])['Ret'].mean().reset_index()


# In[15]:


# Calculate Difference Portfolio for your DataFrame
Portfolio_1 = pd.pivot_table(Portfolio, values='Ret', index=['Date'], columns=['port'])
Portfolio_1['HML'] = Portfolio_1['H100'] - Portfolio_1['H20']
print(Portfolio_1)


# In[16]:


# Calculate average returns for each portfolio
average_port = (Portfolio_1.mean() * 100).reset_index(name='return (%)')

# Calculate t-statistics for each portfolio #Also Run newey west Tstat
tstat = (Portfolio_1.mean() / Portfolio_1.sem()).reset_index(name='tstat')

# Merge average returns and t-statistics into a single DataFrame
average_port = average_port.merge(tstat, on='port')

print(Portfolio)
average_port.transpose()


# In[17]:


ff3 = pd.read_csv("C:\\Users\\ulric\\OneDrive\\Bureau\\FF3.csv")
ff3.rename(columns={"Unnamed: 0": "Date"}, inplace = True)
ff3.head()


# In[ ]:





# In[156]:


#Fama Macbeth Regression: First merge data set 


# In[10]:


import pandas as pd

# Assuming you have loaded ff3 and Portfolio_1 DataFrames
ff3 = pd.read_csv("C:\\Users\\ulric\\OneDrive\\Bureau\\FF3.csv")

# Check data types of the 'Date' column
print(ff3['Date'].dtype)  # Verify the data type of 'Date' column

# Convert 'Date' column to datetime if needed
ff3['Date'] = pd.to_datetime(ff3['Date'])

# Now check the data type again
print(ff3['Date'].dtype)  # Should show datetime64[ns]

# Now try to merge on 'Date'
merged_data = pd.merge(ff3, Portfolio_1, on='Date')
print(merged_data.head())





# In[12]:


# Select relevant columns for regression
X = merged_data[['Mkt-RF', 'SMB', 'HML_y', 'RF']]  # Fama-French factors
y = merged_data['HML_x']  # Dependent variable (HML_y)

# Add constant to independent variables (X) for regression intercept
X = sm.add_constant(X)

# Fit Fama-French regression model
model = sm.OLS(y, X)
results = model.fit()

# Print regression summary
print(results.summary())


# In[ ]:





# In[ ]:





# In[9]:


#Calculate the time-seris beta (from Market), Size (from SMB), and BM (from HML)


# In[8]:


import pandas as pd

# Assuming you already have the 'crypto_ret' DataFrame and 'Portfolio_1' DataFrame
# Assuming you have loaded ff3 and Portfolio_1 DataFrames
ff3 = pd.read_csv("C:\\Users\\ulric\\OneDrive\\Bureau\\FF3.csv")

# Check data types of the 'Date' column
print(ff3['Date'].dtype)  # Verify the data type of 'Date' column

# Convert 'Date' column to datetime if needed
ff3['Date'] = pd.to_datetime(ff3['Date'])
# Merge 'crypto_ret' DataFrame with 'Portfolio_1' DataFrame on 'Date'
merged_data = pd.merge(crypto_ret, ff3, on='Date')

# Display the merged DataFrame
print(merged_data)

# Save the merged DataFrame to a CSV file
merged_data.to_csv("merged_data.csv")



# In[133]:


print(crypto_ret.columns)


# In[139]:


# Initialize empty DataFrame to store regression results, WITHOUT RET COLUMN
regression_results = pd.DataFrame()#SIMPLE REGRESSION

# Loop over each cryptocurrency
for crypto in crypto_ret['crypto'].unique():
    # Filter data for the current cryptocurrency
    crypto_data = merged_data[merged_data['crypto'] == crypto]
    
    # Extract independent variables (Fama-French factors) and dependent variable (crypto returns)
    X = crypto_data[['Mkt-RF', 'SMB', 'HML']]
    y = crypto_data['Ret']
    
    # Add constant to independent variables for regression intercept
    X = sm.add_constant(X)
    
    # Fit OLS regression model
    model = sm.OLS(y, X)
    results = model.fit()
    
    # Extract coefficients and other relevant statistics
    coefficients = results.params
    std_errors = results.bse
    t_values = results.tvalues
    
    # Create a DataFrame to store results for the current cryptocurrency
    crypto_results = pd.DataFrame({
        'crypto': crypto,  
        'Beta_Mkt-RF': coefficients['Mkt-RF'],
        'Size_SMB': coefficients['SMB'],
        'BM_HML': coefficients['HML'],
        'Std_Err_Mkt-RF': std_errors['Mkt-RF'],
        'Std_Err_SMB': std_errors['SMB'],
        'Std_Err_HML': std_errors['HML'],
        't_Mkt-RF': t_values['Mkt-RF'],
        't_SMB': t_values['SMB'],
        't_HML': t_values['HML']
    }, index=[0])
    
    # Append results to the overall regression results DataFrame
    regression_results = pd.concat([regression_results, crypto_results], ignore_index=True)

# Display the final regression results
print(regression_results)


# In[149]:


import pandas as pd
import numpy as np
import statsmodels.api as sm

#RET COLUMN INCLUDED 
# Initialize empty DataFrame to store regression results
regression_results = pd.DataFrame() #SIMPLE REGRESSION

# Loop over each cryptocurrency
for crypto in crypto_ret['crypto'].unique():
    # Filter data for the current cryptocurrency
    crypto_data = merged_data[merged_data['crypto'] == crypto]
    
    # Extract independent variables (Fama-French factors) and dependent variable (crypto returns)
    X = crypto_data[['Mkt-RF', 'SMB', 'HML']]
    y = crypto_data['Ret']  # Extract 'Ret' column as the dependent variable
    
    # Add constant to independent variables for regression intercept
    X = sm.add_constant(X)
    
    # Fit OLS regression model
    model = sm.OLS(y, X)
    results = model.fit()
    
    # Extract coefficients and other relevant statistics
    coefficients = results.params
    std_errors = results.bse
    t_values = results.tvalues
    
    # Create a DataFrame to store results for the current cryptocurrency
    crypto_results = pd.DataFrame({
        'crypto': crypto,  
        'Beta_Mkt-RF': coefficients['Mkt-RF'],
        'Size_SMB': coefficients['SMB'],
        'BM_HML': coefficients['HML'],
        'Std_Err_Mkt-RF': std_errors['Mkt-RF'],
        'Std_Err_SMB': std_errors['SMB'],
        'Std_Err_HML': std_errors['HML'],
        't_Mkt-RF': t_values['Mkt-RF'],
        't_SMB': t_values['SMB'],
        't_HML': t_values['HML']
    }, index=[0])
    
    # Append 'Ret' column to crypto_results
    crypto_results['Ret'] = y.mean()  # Add mean of 'Ret' as a summary statistic
    
    # Append results to the overall regression results DataFrame
    regression_results = pd.concat([regression_results, crypto_results], ignore_index=True)

# Display the final regression results
print(regression_results)


# In[35]:


import pandas as pd
import numpy as np
import statsmodels.api as sm

# Assuming you have loaded your data and performed the necessary merging
# merged_data = ...

# Get unique cryptocurrencies and date list from merged_data
portfolios = merged_data['crypto'].unique()
date_list = merged_data['Date'].unique().tolist()

# Initialize empty DataFrame to store regression results
res = pd.DataFrame()

# Specify the rolling window period (12 days)
rolling_window = 12

for port in portfolios:
    df_port = merged_data[['Date', 'Mkt-RF', 'SMB', 'HML', 'Ret']][merged_data['crypto'] == port].sort_values(by='Date')
    
    # Iterate over the data using the rolling window
    for i in range(rolling_window, len(date_list) + 1):
        df_rolling = df_port.iloc[i - rolling_window:i]  # Select data for the rolling window
        X = sm.add_constant(df_rolling[['Mkt-RF', 'SMB', 'HML']])  # Add constant to independent variables
        y = df_rolling['Ret']  # Dependent variable (returns) for the rolling window
        
        # Perform linear regression and get coefficients
        model = sm.OLS(y, X)
        results = model.fit()
        
        # Extract coefficients and other relevant statistics
        coefficients = results.params
        std_errors = results.bse
        
        # Create a DataFrame to store the regression results for the current window
        window_results = pd.DataFrame({
            'crypto': port,
            'Date': date_list[i - 1],  # Use the end date of the rolling window
            'alpha': coefficients['const'],  # Intercept (alpha)
            'bMkt': coefficients['Mkt-RF'],  # Coefficient for Market-RF
            'bsmb': coefficients['SMB'],  # Coefficient for SMB
            'bhml': coefficients['HML'],  # Coefficient for HML
            'std_error_alpha': std_errors['const'],  # Standard error of intercept
            'std_error_bMkt': std_errors['Mkt-RF'],  # Standard error of Market-RF coefficient
            'std_error_bsmb': std_errors['SMB'],  # Standard error of SMB coefficient
            'std_error_bhml': std_errors['HML']  # Standard error of HML coefficient
        }, index=[0])
        
        window_results['Ret'] = y.mean()
        
        # Append the window results to the overall regression results DataFrame
        res = pd.concat([res, window_results], ignore_index=True)

# Display the regression results
print(res.head())

# Optional: Save the results to a CSV file
res.to_csv("rolling_regression_results.csv", index=False)


# In[43]:


res


# In[36]:


final_port = pd.DataFrame()

factors = ['bMkt', 'bsmb', 'bhml']

for col in factors:

    X = res[['Date','crypto','bMkt','bsmb','bhml', 'Ret']]
    X.dropna(inplace = True)

    grp_x = X.groupby(['Date'])[col].describe(percentiles=[0.2,0.4,0.6,0.8]).reset_index()


    grp_x = grp_x[["Date", '20%',  '40%', '60%', '80%' ]]

    X = X.merge(grp_x, how='left', on=['Date'])



    # Use the breakpoints to form the portfolios
    def grp_bucket(row):
        if row[col]<=row['20%']:
            value='H20'
        elif row[col]<=row['40%']:
            value='H40'
        elif row[col]<=row['60%']:
            value='H60'
        elif row[col]<=row['80%']:
            value='H80'
        
          
        else:
            value='H100'

        return value


    X['port'] = X.apply(grp_bucket, axis=1)


    Portfolio = X.groupby(['Date','port'])['Ret'].mean().reset_index()
    Portfolio




# In[39]:


import pandas as pd
import numpy as np

# Calculate Difference Portfolio
Portfolio_1 = pd.pivot_table(Portfolio, values='Ret', index=['Date'], columns=['port'])
Portfolio_1['HML'] = Portfolio_1['H100'] - Portfolio_1['H20']

# Calculate mean and t-statistics
average_port = (Portfolio_1.reset_index()
                .melt(id_vars="Date", var_name="portfolio", value_name="estimate")
                .groupby("portfolio")["estimate"]
                .apply(lambda x: pd.Series({
                    "risk_premium": 100 * x.mean(),
                    "t_statistic": x.mean() / x.std() * np.sqrt(len(x))
                }))
                .reset_index()
                .pivot(index="portfolio", columns="level_1", values="estimate")
                .reset_index()
                )

# Display the results
print(average_port.head())



# In[40]:


# Calculate Newey-West t-statistics
nw_t_stat = (Portfolio_1.reset_index()
             .melt(id_vars="Date", var_name="portfolio", value_name="estimate")
             .groupby("portfolio")
             .apply(lambda x: (
                 x["estimate"].mean() /
                 smf.ols("estimate ~ 1", x)
                 .fit(cov_type="HAC", cov_kwds={"maxlags": 6}).bse
             ))
             .reset_index()
             .rename(columns={"Intercept": "t_statistic_newey_west"})
             )

# Merge Newey-West t-statistics with average_port DataFrame
average_port = average_port.merge(nw_t_stat, on="portfolio").round(3)
average_port['factors'] = col  # Assuming 'col' is defined elsewhere

# Concatenate or merge with final_port DataFrame
final_port = pd.concat([final_port, average_port])  # Assuming 'final_port' is defined elsewhere

# Display or return final_port DataFrame
print(final_port.head())  # Display the head of the updated final_port DataFrame


# In[42]:


final_port


# In[45]:


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Assuming 'final_port' and other necessary variables are defined elsewhere

# Loop over each factor (e.g., 'bhml', 'size', etc.)
for col in ['bMkt', 'bsmb', 'bhml']:  # Update with actual factors

    # Calculate Difference Portfolio
    Portfolio_1 = pd.pivot_table(Portfolio, values='Ret', index=['Date'], columns=['port'])
    Portfolio_1['HML'] = Portfolio_1['H100'] - Portfolio_1['H20']

    # Calculate mean and t-statistics
    average_port = (Portfolio_1.reset_index()
                    .melt(id_vars="Date", var_name="portfolio", value_name="estimate")
                    .groupby("portfolio")["estimate"]
                    .apply(lambda x: pd.Series({
                        "risk_premium": 100 * x.mean(),
                        "t_statistic": x.mean() / x.std() * np.sqrt(len(x))
                    }))
                    .reset_index()
                    .pivot(index="portfolio", columns="level_1", values="estimate")
                    .reset_index()
                    )

    # Calculate Newey-West t-statistics
    nw_t_stat = (Portfolio_1.reset_index()
                 .melt(id_vars="Date", var_name="portfolio", value_name="estimate")
                 .groupby("portfolio")
                 .apply(lambda x: (
                     x["estimate"].mean() /
                     smf.ols("estimate ~ 1", x)
                     .fit(cov_type="HAC", cov_kwds={"maxlags": 6}).bse
                 ))
                 .reset_index()
                 .rename(columns={"Intercept": "t_statistic_newey_west"})
                 )

    # Merge Newey-West t-statistics with average_port DataFrame
    average_port = average_port.merge(nw_t_stat, on="portfolio").round(3)
    average_port['factors'] = col  # Assign the current factor name to 'factors'

    # Concatenate or merge with final_port DataFrame
    final_port = pd.concat([final_port, average_port])  # Append to final_port DataFrame

# Display or return final_port DataFrame
print(final_port.head())  # Display the head of the updated final_port DataFrame


# In[46]:


final_port


# In[47]:


final_port[final_port["factors"]=="bmkt"]


# In[48]:


#7. Perform a Fama-MacBeth regression to test the significance of each of these three factors.

risk_premiums = (res.groupby("Date").apply(lambda x: smf.ols(formula="Ret ~ + bMkt + bsmb + bhml",
                                                                 data=x).fit().params).reset_index())



price_of_risk = (risk_premiums.melt(id_vars="Date", var_name="factor", value_name="estimate")
  .groupby("factor")["estimate"]
  .apply(lambda x: pd.Series({
      "risk_premium": 100*x.mean(),
      "t_statistic": x.mean()/x.std()*np.sqrt(len(x))
    })
  )
  .reset_index()
  .pivot(index="factor", columns="level_1", values="estimate")
  .reset_index()
)


price_of_risk_newey_west = (risk_premiums
  .melt(id_vars="Date", var_name="factor", value_name="estimate")
  .groupby("factor")
  .apply(lambda x: (
      x["estimate"].mean()/
        smf.ols("estimate ~ 1", x)
        .fit(cov_type="HAC", cov_kwds={"maxlags": 6}).bse
    )
  )
  .reset_index()
  .rename(columns={"Intercept": "t_statistic_newey_west"})
)

price_of_risk = (price_of_risk
  .merge(price_of_risk_newey_west, on="factor")
  .round(3))

price_of_risk


# In[ ]:


Create a 3*5 bivariate Independent-Sort portfolio based on Market and Size coefficients (3 market and 5 size portfolios). Calculate the portfolio return for each of the 15 portfolios, the difference in portfolio returns (high-low), and test the significance of the difference in portfolio returns.


# In[65]:


res


# In[66]:


res.groupby(['Date']).size()


# In[67]:


grp_sz


# In[68]:


X = res[[ 'Date','crypto','bMkt','bsmb','bhml', 'Ret']]
X.dropna(inplace = True)
X.groupby(['Date'])['bMkt'].describe(percentiles=[0.33, 0.66]).reset_index()

grp_mkt = X.groupby(['Date'])['bMkt'].describe(percentiles=[0.33, 0.66]).reset_index()

grp_mkt  = grp_mkt[["Date", '33%', '66%']]

grp_mkt  = grp_mkt.rename(columns={'33%':'mkt33','66%':'mkt66'})


# size breakdown
grp_sz = X.groupby(['Date'])['bsmb'].describe(percentiles=[0.2,0.4,0.6,0.8]).reset_index()


grp_sz = grp_sz[["Date",'20%', '40%', '60%',  '80%']]

grp_sz = grp_sz.rename(columns={'20%':'bsmb20','40%':'bsmb40','60%':'bsmb60', '80%':'bsmb80'})

### Asign Portfolios into Brackets

def mkt_bucket(row):
    if row['bMkt']<=row['mkt33']:
        value = 'MKT33'
    elif row['bMkt']<=row['mkt66']:
        value='MKT66'
    elif row['bMkt']>row['mkt66']:
        value='MKT99'
    else:
        value=''
    return value


def bsmb_bucket(row):
    if row['bsmb']<=row['bsmb20']:
        value = 'BSMB20'
    elif row['bsmb']<=row['bsmb40']:
        value='BSMB40'
    elif row['bsmb']<=row['bsmb60']:
        value='BSMB60'
    elif row['bsmb']<=row['bsmb80']:
        value='BSMB80'
    elif row['bsmb']>row['bsmb80']:
        value='BSMB100'

    else:
        value=''
    return value

mkt_grp = X.merge(grp_mkt, on=['Date'])
mkt_grp['tr_port'] = mkt_grp.apply(mkt_bucket, axis=1)


bsmb_grp = X.merge(grp_sz, on=['Date'])
bsmb_grp['size_port'] = bsmb_grp.apply(bsmb_bucket, axis=1)


final_grp = pd.merge(mkt_grp[[ 'Date', 'Ret', 'bMkt', 'bsmb','tr_port','crypto']],
                     bsmb_grp[[ 'Date', 'size_port', 'crypto']], on =['Date',  'crypto'])


# In[69]:


final_grp


# In[72]:


final_grp.groupby("crypto")["Ret"].apply(lambda x: pd.Series({
          "risk_premium": 100*x.mean(),
          "t_statistic": x.mean()/x.std()*np.sqrt(len(x))})).reset_index().pivot(index="crypto", columns="level_1", values="Ret")


# In[75]:


average_port = final_grp.groupby("crypto")["Ret"].apply(lambda x: pd.Series({
          "risk_premium": 100*x.mean(),
          "t_statistic": x.mean()/x.std()*np.sqrt(len(x))})).reset_index().pivot(index="crypto", columns="level_1", values="Ret").reset_index()


#calculate Newey_West t-stat and add it to mean and t-stat.
nw_t_stat = final_grp.groupby("crypto").apply(lambda x:
            (x["Ret"].mean()/smf.ols("Ret ~ 1", x).fit(cov_type="HAC", cov_kwds={"maxlags": 6}).bse)).reset_index().rename(columns={"Intercept": "t_statistic_newey_west"})


average_port = (average_port.merge(nw_t_stat, on="crypto").round(3))

average_port


# In[81]:


X = res[[ 'Date','crypto','bMkt','bsmb','bhml', 'Ret']]
X.dropna(inplace = True)
X.groupby(['Date'])['bMkt'].describe(percentiles=[0.33, 0.66]).reset_index()

grp_mkt = X.groupby(['Date'])['bMkt'].describe(percentiles=[0.33, 0.66]).reset_index()

grp_mkt  = grp_mkt[["Date", '33%', '66%']]

grp_mkt  = grp_mkt.rename(columns={'33%':'mkt33','66%':'mkt66'})


# size breakdown
grp_sz = X.groupby(['Date'])['bsmb'].describe(percentiles=[0.2,0.4,0.6,0.8]).reset_index()


grp_sz = grp_sz[["Date",'20%', '40%', '60%',  '80%']]

grp_sz = grp_sz.rename(columns={'20%':'bsmb20','40%':'bsmb40','60%':'bsmb60', '80%':'bsmb80'})

### Asign Portfolios into Brackets

def mkt_bucket(row):
    if row['bMkt']<=row['mkt33']:
        value = 'MKT33'
    elif row['bMkt']<=row['mkt66']:
        value='MKT66'
    elif row['bMkt']>row['mkt66']:
        value='MKT99'
    else:
        value=''
    return value


def bsmb_bucket(row):
    if row['bsmb']<=row['bsmb20']:
        value = 'BSMB20'
    elif row['bsmb']<=row['bsmb40']:
        value='BSMB40'
    elif row['bsmb']<=row['bsmb60']:
        value='BSMB60'
    elif row['bsmb']<=row['bsmb80']:
        value='BSMB80'
    elif row['bsmb']>row['bsmb80']:
        value='BSMB100'

    else:
        value=''
    return value

mkt_grp = X.merge(grp_mkt, on=['Date'])
mkt_grp['tr_port'] = mkt_grp.apply(mkt_bucket, axis=1)


bsmb_grp = X.merge(grp_sz, on=['Date'])
bsmb_grp['size_port'] = bsmb_grp.apply(bsmb_bucket, axis=1)


final_grp = pd.merge(mkt_grp[[ 'Date', 'Ret', 'bMkt', 'bhml','tr_port','crypto']],
                     bsmb_grp[[ 'Date', 'size_port', 'crypto']], on =['Date',  'crypto'])


# In[84]:


final_grp


# In[85]:


final_grp.groupby("crypto")["Ret"].apply(lambda x: pd.Series({
          "risk_premium": 100*x.mean(),
          "t_statistic": x.mean()/x.std()*np.sqrt(len(x))})).reset_index().pivot(index="crypto", columns="level_1", values="Ret")


# In[86]:


average_port = final_grp.groupby("crypto")["Ret"].apply(lambda x: pd.Series({
          "risk_premium": 100*x.mean(),
          "t_statistic": x.mean()/x.std()*np.sqrt(len(x))})).reset_index().pivot(index="crypto", columns="level_1", values="Ret").reset_index()


#calculate Newey_West t-stat and add it to mean and t-stat.
nw_t_stat = final_grp.groupby("crypto").apply(lambda x:
            (x["Ret"].mean()/smf.ols("Ret ~ 1", x).fit(cov_type="HAC", cov_kwds={"maxlags": 6}).bse)).reset_index().rename(columns={"Intercept": "t_statistic_newey_west"})


average_port = (average_port.merge(nw_t_stat, on="crypto").round(3))

average_port


# In[99]:


X = crypto_ret[['crypto', 'Date','mean', 'median', 'max', 'min', 'q_05', 'q_95', 'std', 'skew', 'kurt', 'Ret']]
 # Assuming 'mean' corresponds to 'm_1' and 'Ret' corresponds to 'ex_ret'
#Sort data and calculate breakpoints# Assuming 'mean' corresponds to 'm_1' and 'Ret' corresponds to 'ex_ret'
# Sort data and calculate breakpoints
grp_x = X.groupby(['Date'])['median'].describe(percentiles=[0.2, 0.4, 0.6, 0.8]).reset_index()
grp_x = grp_x[['Date', '20%', '40%', '60%', '80%']]

# Merge breakpoints with the main dataframe
X = X.merge(grp_x, how='left', on=['Date'])

# Use the breakpoints to form the portfolios
def grp_bucket(row):
    if row['median'] <= row['20%']:
        value = 'H20'
    elif row['median'] <= row['40%']:
        value = 'H40'
    elif row['median'] <= row['60%']:
        value = 'H60'
    elif row['median'] <= row['80%']:
        value = 'H80'
    elif row['median'] > row['80%']:
        value = 'H100'
    else:
        value = ''
    return value

X['port'] = X.apply(grp_bucket, axis=1)

import pandas as pd

# Create a subset of your data with relevant columns
X = crypto_ret[['crypto', 'Date', 'median', 'Ret']]  # Assuming 'mean' corresponds to 'm_1' and 'Ret' corresponds to 'ex_ret'

# Drop any remaining rows with missing values
X.dropna(inplace=True)

# Sort data and calculate breakpoints
grp_x = X.groupby(['Date'])['median'].describe(percentiles=[0.2, 0.4, 0.6, 0.8]).reset_index()
grp_x = grp_x[['Date', '20%', '40%', '60%', '80%']]

# Merge breakpoints with the main dataframe
X = X.merge(grp_x, how='left', on=['Date'])

# Use the breakpoints to form the portfolios
def grp_bucket(row):
    if row['median'] <= row['20%']:
        value = 'H20'
    elif row['median'] <= row['40%']:
        value = 'H40'
    elif row['median'] <= row['60%']:
        value = 'H60'
    elif row['median'] <= row['80%']:
        value = 'H80'
    elif row['median'] > row['80%']:
        value = 'H100'
    else:
        value = ''
    return value

X['port'] = X.apply(grp_bucket, axis=1)
Portfolio = X.groupby(['Date', 'port'])['Ret'].mean().reset_index()
print(X)


# In[ ]:





# In[100]:


Portfolio = X.groupby(['Date', 'port'])['Ret'].mean().reset_index()
Portfolio


# In[101]:


# Calculate Difference Portfolio for your DataFrame
Portfolio_1 = pd.pivot_table(Portfolio, values='Ret', index=['Date'], columns=['port'])
Portfolio_1['HML'] = Portfolio_1['H100'] - Portfolio_1['H20']
print(Portfolio_1)


# In[102]:


# Calculate average returns for each portfolio
average_port = (Portfolio_1.mean() * 100).reset_index(name='return (%)')

# Calculate t-statistics for each portfolio #Also Run newey west Tstat
tstat = (Portfolio_1.mean() / Portfolio_1.sem()).reset_index(name='tstat')

# Merge average returns and t-statistics into a single DataFrame
average_port = average_port.merge(tstat, on='port')

print(Portfolio)
average_port.transpose()


# In[103]:


import pandas as pd

# Assuming you have loaded ff3 and Portfolio_1 DataFrames
ff3 = pd.read_csv("C:\\Users\\ulric\\OneDrive\\Bureau\\FF3.csv")

# Check data types of the 'Date' column
print(ff3['Date'].dtype)  # Verify the data type of 'Date' column

# Convert 'Date' column to datetime if needed
ff3['Date'] = pd.to_datetime(ff3['Date'])

# Now check the data type again
print(ff3['Date'].dtype)  # Should show datetime64[ns]

# Now try to merge on 'Date'
merged_data = pd.merge(ff3, Portfolio_1, on='Date')
print(merged_data.head())


# In[104]:


# Select relevant columns for regression
X = merged_data[['Mkt-RF', 'SMB', 'HML_x', 'RF']]  # Fama-French factors
y = merged_data['HML_y']  # Dependent variable (HML_y)

# Add constant to independent variables (X) for regression intercept
X = sm.add_constant(X)

# Fit Fama-French regression model
model = sm.OLS(y, X)
results = model.fit()

# Print regression summary
print(results.summary())


# In[105]:


import pandas as pd
import statsmodels.formula.api as smf

# Assuming 'Portfolio' is your DataFrame containing portfolio returns by date and portfolio
# 'port' column contains the portfolio names ('H20', 'H40', 'H60', 'H80', 'H100')

# Iterate over each portfolio
for portfolio in ['H20', 'H40', 'H60', 'H80', 'H100']:
    # Filter data for the current portfolio
    portfolio_data = Portfolio[Portfolio['port'] == portfolio]
    
    # Define the formula for the OLS model
    formula = 'Ret ~ 1'
    
    # Fit OLS model with Newey-West standard errors
    model = smf.ols(formula, data=portfolio_data).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
    
    # Extract the t-statistic for the intercept
    t_statistic_newey_west = model.tvalues['Intercept']
    
    # Print the Newey-West t-statistic for the current portfolio
    print(f"Portfolio: {portfolio}, Newey-West t-statistic: {t_statistic_newey_west}")


# In[ ]:





# In[27]:


X = crypto_ret[['crypto', 'Date','mean', 'median', 'max', 'min', 'q_05', 'q_95', 'std', 'skew', 'kurt', 'Ret']]
 # Assuming 'mean' corresponds to 'm_1' and 'Ret' corresponds to 'ex_ret'
#Sort data and calculate breakpoints# Assuming 'mean' corresponds to 'm_1' and 'Ret' corresponds to 'ex_ret'
# Sort data and calculate breakpoints
grp_x = X.groupby(['Date'])['max'].describe(percentiles=[0.2, 0.4, 0.6, 0.8]).reset_index()
grp_x = grp_x[['Date', '20%', '40%', '60%', '80%']]

# Merge breakpoints with the main dataframe
X = X.merge(grp_x, how='left', on=['Date'])

# Use the breakpoints to form the portfolios
def grp_bucket(row):
    if row['max'] <= row['20%']:
        value = 'H20'
    elif row['max'] <= row['40%']:
        value = 'H40'
    elif row['max'] <= row['60%']:
        value = 'H60'
    elif row['max'] <= row['80%']:
        value = 'H80'
    elif row['max'] > row['80%']:
        value = 'H100'
    else:
        value = ''
    return value

X['port'] = X.apply(grp_bucket, axis=1)

import pandas as pd

# Create a subset of your data with relevant columns
X = crypto_ret[['crypto', 'Date', 'max', 'Ret']]  # Assuming 'mean' corresponds to 'm_1' and 'Ret' corresponds to 'ex_ret'

# Drop any remaining rows with missing values
X.dropna(inplace=True)

# Sort data and calculate breakpoints
grp_x = X.groupby(['Date'])['max'].describe(percentiles=[0.2, 0.4, 0.6, 0.8]).reset_index()
grp_x = grp_x[['Date', '20%', '40%', '60%', '80%']]

# Merge breakpoints with the main dataframe
X = X.merge(grp_x, how='left', on=['Date'])

# Use the breakpoints to form the portfolios
def grp_bucket(row):
    if row['max'] <= row['20%']:
        value = 'H20'
    elif row['max'] <= row['40%']:
        value = 'H40'
    elif row['max'] <= row['60%']:
        value = 'H60'
    elif row['max'] <= row['80%']:
        value = 'H80'
    elif row['max'] > row['80%']:
        value = 'H100'
    else:
        value = ''
    return value

X['port'] = X.apply(grp_bucket, axis=1)
Portfolio = X.groupby(['Date', 'port'])['Ret'].mean().reset_index()
print(X)


# In[28]:


Portfolio = X.groupby(['Date', 'port'])['Ret'].mean().reset_index()
Portfolio


# In[29]:


# Calculate Difference Portfolio for your DataFrame
Portfolio_1 = pd.pivot_table(Portfolio, values='Ret', index=['Date'], columns=['port'])
Portfolio_1['HML'] = Portfolio_1['H100'] - Portfolio_1['H20']
print(Portfolio_1)


# In[30]:


# Calculate average returns for each portfolio
average_port = (Portfolio_1.mean() * 100).reset_index(name='return (%)')

# Calculate t-statistics for each portfolio #Also Run newey west Tstat
tstat = (Portfolio_1.mean() / Portfolio_1.sem()).reset_index(name='tstat')

# Merge average returns and t-statistics into a single DataFrame
average_port = average_port.merge(tstat, on='port')

print(Portfolio)
average_port.transpose()


# In[31]:


import pandas as pd
import statsmodels.formula.api as smf

# Assuming 'Portfolio' is your DataFrame containing portfolio returns by date and portfolio
# 'port' column contains the portfolio names ('H20', 'H40', 'H60', 'H80', 'H100')

# Iterate over each portfolio
for portfolio in ['H20', 'H40', 'H60', 'H80', 'H100']:
    # Filter data for the current portfolio
    portfolio_data = Portfolio[Portfolio['port'] == portfolio]
    
    # Define the formula for the OLS model
    formula = 'Ret ~ 1'
    
    # Fit OLS model with Newey-West standard errors
    model = smf.ols(formula, data=portfolio_data).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
    
    # Extract the t-statistic for the intercept
    t_statistic_newey_west = model.tvalues['Intercept']
    
    # Print the Newey-West t-statistic for the current portfolio
    print(f"Portfolio: {portfolio}, Newey-West t-statistic: {t_statistic_newey_west}")


# In[32]:


import pandas as pd

# Assuming you have loaded ff3 and Portfolio_1 DataFrames
ff3 = pd.read_csv("C:\\Users\\ulric\\OneDrive\\Bureau\\FF3.csv")

# Check data types of the 'Date' column
print(ff3['Date'].dtype)  # Verify the data type of 'Date' column

# Convert 'Date' column to datetime if needed
ff3['Date'] = pd.to_datetime(ff3['Date'])

# Now check the data type again
print(ff3['Date'].dtype)  # Should show datetime64[ns]

# Now try to merge on 'Date'
merged_data = pd.merge(ff3, Portfolio_1, on='Date')
print(merged_data.head())


# In[33]:


# Select relevant columns for regression
X = merged_data[['Mkt-RF', 'SMB', 'HML_x', 'RF']]  # Fama-French factors
y = merged_data['HML_y']  # Dependent variable (HML_y)

# Add constant to independent variables (X) for regression intercept
X = sm.add_constant(X)

# Fit Fama-French regression model
model = sm.OLS(y, X)
results = model.fit()

# Print regression summary
print(results.summary())


# In[ ]:





# In[34]:


X = crypto_ret[['crypto', 'Date','mean', 'median', 'max', 'min', 'q_05', 'q_95', 'std', 'skew', 'kurt', 'Ret']]
 # Assuming 'mean' corresponds to 'm_1' and 'Ret' corresponds to 'ex_ret'
#Sort data and calculate breakpoints# Assuming 'mean' corresponds to 'm_1' and 'Ret' corresponds to 'ex_ret'
# Sort data and calculate breakpoints# Assuming 'mean' corresponds to 'm_1' and 'Ret' corresponds to 'ex_ret'
# Sort data and calculate breakpoints
grp_x = X.groupby(['Date'])['min'].describe(percentiles=[0.2, 0.4, 0.6, 0.8]).reset_index()
grp_x = grp_x[['Date', '20%', '40%', '60%', '80%']]

# Merge breakpoints with the main dataframe
X = X.merge(grp_x, how='left', on=['Date'])

# Use the breakpoints to form the portfolios
def grp_bucket(row):
    if row['min'] <= row['20%']:
        value = 'H20'
    elif row['min'] <= row['40%']:
        value = 'H40'
    elif row['min'] <= row['60%']:
        value = 'H60'
    elif row['min'] <= row['80%']:
        value = 'H80'
    elif row['min'] > row['80%']:
        value = 'H100'
    else:
        value = ''
    return value

X['port'] = X.apply(grp_bucket, axis=1)

import pandas as pd

# Create a subset of your data with relevant columns
X = crypto_ret[['crypto', 'Date', 'min', 'Ret']]  # Assuming 'mean' corresponds to 'm_1' and 'Ret' corresponds to 'ex_ret'

# Drop any remaining rows with missing values
X.dropna(inplace=True)

# Sort data and calculate breakpoints
grp_x = X.groupby(['Date'])['min'].describe(percentiles=[0.2, 0.4, 0.6, 0.8]).reset_index()
grp_x = grp_x[['Date', '20%', '40%', '60%', '80%']]

# Merge breakpoints with the main dataframe
X = X.merge(grp_x, how='left', on=['Date'])

# Use the breakpoints to form the portfolios
def grp_bucket(row):
    if row['min'] <= row['20%']:
        value = 'H20'
    elif row['min'] <= row['40%']:
        value = 'H40'
    elif row['min'] <= row['60%']:
        value = 'H60'
    elif row['min'] <= row['80%']:
        value = 'H80'
    elif row['min'] > row['80%']:
        value = 'H100'
    else:
        value = ''
    return value

X['port'] = X.apply(grp_bucket, axis=1)
Portfolio = X.groupby(['Date', 'port'])['Ret'].mean().reset_index()
print(X)


# In[35]:


Portfolio = X.groupby(['Date', 'port'])['Ret'].mean().reset_index()
Portfolio   


# In[36]:


# Calculate Difference Portfolio for your DataFrame
Portfolio_1 = pd.pivot_table(Portfolio, values='Ret', index=['Date'], columns=['port'])
Portfolio_1['HML'] = Portfolio_1['H100'] - Portfolio_1['H20']
print(Portfolio_1)


# In[37]:


# Calculate average returns for each portfolio
average_port = (Portfolio_1.mean() * 100).reset_index(name='return (%)')

# Calculate t-statistics for each portfolio #Also Run newey west Tstat
tstat = (Portfolio_1.mean() / Portfolio_1.sem()).reset_index(name='tstat')

# Merge average returns and t-statistics into a single DataFrame
average_port = average_port.merge(tstat, on='port')

print(Portfolio)
average_port.transpose()


# In[38]:


import pandas as pd
import statsmodels.formula.api as smf

# Assuming 'Portfolio' is your DataFrame containing portfolio returns by date and portfolio
# 'port' column contains the portfolio names ('H20', 'H40', 'H60', 'H80', 'H100')

# Iterate over each portfolio
for portfolio in ['H20', 'H40', 'H60', 'H80', 'H100']:
    # Filter data for the current portfolio
    portfolio_data = Portfolio[Portfolio['port'] == portfolio]
    
    # Define the formula for the OLS model
    formula = 'Ret ~ 1'
    
    # Fit OLS model with Newey-West standard errors
    model = smf.ols(formula, data=portfolio_data).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
    
    # Extract the t-statistic for the intercept
    t_statistic_newey_west = model.tvalues['Intercept']
    
    # Print the Newey-West t-statistic for the current portfolio
    print(f"Portfolio: {portfolio}, Newey-West t-statistic: {t_statistic_newey_west}")


# In[39]:


import pandas as pd

# Assuming you have loaded ff3 and Portfolio_1 DataFrames
ff3 = pd.read_csv("C:\\Users\\ulric\\OneDrive\\Bureau\\FF3.csv")

# Check data types of the 'Date' column
print(ff3['Date'].dtype)  # Verify the data type of 'Date' column

# Convert 'Date' column to datetime if needed
ff3['Date'] = pd.to_datetime(ff3['Date'])

# Now check the data type again
print(ff3['Date'].dtype)  # Should show datetime64[ns]

# Now try to merge on 'Date'
merged_data = pd.merge(ff3, Portfolio_1, on='Date')
print(merged_data.head())


# In[40]:


# Select relevant columns for regression
X = merged_data[['Mkt-RF', 'SMB', 'HML_x', 'RF']]  # Fama-French factors
y = merged_data['HML_y']  # Dependent variable (HML_y)

# Add constant to independent variables (X) for regression intercept
X = sm.add_constant(X)

# Fit Fama-French regression model
model = sm.OLS(y, X)
results = model.fit()

# Print regression summary
print(results.summary())


# In[41]:


X = crypto_ret[['crypto', 'Date','mean', 'median', 'max', 'min', 'q_05', 'q_95', 'std', 'skew', 'kurt', 'Ret']]
 # Assuming 'mean' corresponds to 'm_1' and 'Ret' corresponds to 'ex_ret'
#Sort data and calculate breakpoints# Assuming 'mean' corresponds to 'm_1' and 'Ret' corresponds to 'ex_ret'
# Assuming 'mean' corresponds to 'm_1' and 'Ret' corresponds to 'ex_ret'
# Sort data and calculate breakpoints
grp_x = X.groupby(['Date'])['q_05'].describe(percentiles=[0.2, 0.4, 0.6, 0.8]).reset_index()
grp_x = grp_x[['Date', '20%', '40%', '60%', '80%']]

# Merge breakpoints with the main dataframe
X = X.merge(grp_x, how='left', on=['Date'])

# Use the breakpoints to form the portfolios
def grp_bucket(row):
    if row['q_05'] <= row['20%']:
        value = 'H20'
    elif row['q_05'] <= row['40%']:
        value = 'H40'
    elif row['q_05'] <= row['60%']:
        value = 'H60'
    elif row['q_05'] <= row['80%']:
        value = 'H80'
    elif row['q_05'] > row['80%']:
        value = 'H100'
    else:
        value = ''
    return value

X['port'] = X.apply(grp_bucket, axis=1)

import pandas as pd

# Create a subset of your data with relevant columns
X = crypto_ret[['crypto', 'Date', 'q_05', 'Ret']]  # Assuming 'mean' corresponds to 'm_1' and 'Ret' corresponds to 'ex_ret'

# Drop any remaining rows with missing values
X.dropna(inplace=True)

# Sort data and calculate breakpoints
grp_x = X.groupby(['Date'])['q_05'].describe(percentiles=[0.2, 0.4, 0.6, 0.8]).reset_index()
grp_x = grp_x[['Date', '20%', '40%', '60%', '80%']]

# Merge breakpoints with the main dataframe
X = X.merge(grp_x, how='left', on=['Date'])

# Use the breakpoints to form the portfolios
def grp_bucket(row):
    if row['q_05'] <= row['20%']:
        value = 'H20'
    elif row['q_05'] <= row['40%']:
        value = 'H40'
    elif row['q_05'] <= row['60%']:
        value = 'H60'
    elif row['q_05'] <= row['80%']:
        value = 'H80'
    elif row['q_05'] > row['80%']:
        value = 'H100'
    else:
        value = ''
    return value

X['port'] = X.apply(grp_bucket, axis=1)
Portfolio = X.groupby(['Date', 'port'])['Ret'].mean().reset_index()
print(X)


# In[42]:


Portfolio = X.groupby(['Date', 'port'])['Ret'].mean().reset_index()
Portfolio   


# In[43]:


# Calculate Difference Portfolio for your DataFrame
Portfolio_1 = pd.pivot_table(Portfolio, values='Ret', index=['Date'], columns=['port'])
Portfolio_1['HML'] = Portfolio_1['H100'] - Portfolio_1['H20']
print(Portfolio_1)


# In[44]:


# Calculate average returns for each portfolio
average_port = (Portfolio_1.mean() * 100).reset_index(name='return (%)')

# Calculate t-statistics for each portfolio #Also Run newey west Tstat
tstat = (Portfolio_1.mean() / Portfolio_1.sem()).reset_index(name='tstat')

# Merge average returns and t-statistics into a single DataFrame
average_port = average_port.merge(tstat, on='port')

print(Portfolio)
average_port.transpose()


# In[45]:


import pandas as pd
import statsmodels.formula.api as smf

# Assuming 'Portfolio' is your DataFrame containing portfolio returns by date and portfolio
# 'port' column contains the portfolio names ('H20', 'H40', 'H60', 'H80', 'H100')

# Iterate over each portfolio
for portfolio in ['H20', 'H40', 'H60', 'H80', 'H100']:
    # Filter data for the current portfolio
    portfolio_data = Portfolio[Portfolio['port'] == portfolio]
    
    # Define the formula for the OLS model
    formula = 'Ret ~ 1'
    
    # Fit OLS model with Newey-West standard errors
    model = smf.ols(formula, data=portfolio_data).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
    
    # Extract the t-statistic for the intercept
    t_statistic_newey_west = model.tvalues['Intercept']
    
    # Print the Newey-West t-statistic for the current portfolio
    print(f"Portfolio: {portfolio}, Newey-West t-statistic: {t_statistic_newey_west}")


# In[46]:


import pandas as pd

# Assuming you have loaded ff3 and Portfolio_1 DataFrames
ff3 = pd.read_csv("C:\\Users\\ulric\\OneDrive\\Bureau\\FF3.csv")

# Check data types of the 'Date' column
print(ff3['Date'].dtype)  # Verify the data type of 'Date' column

# Convert 'Date' column to datetime if needed
ff3['Date'] = pd.to_datetime(ff3['Date'])

# Now check the data type again
print(ff3['Date'].dtype)  # Should show datetime64[ns]

# Now try to merge on 'Date'
merged_data = pd.merge(ff3, Portfolio_1, on='Date')
print(merged_data.head())


# In[47]:


# Select relevant columns for regression
X = merged_data[['Mkt-RF', 'SMB', 'HML_x', 'RF']]  # Fama-French factors
y = merged_data['HML_y']  # Dependent variable (HML_y)

# Add constant to independent variables (X) for regression intercept
X = sm.add_constant(X)

# Fit Fama-French regression model
model = sm.OLS(y, X)
results = model.fit()

# Print regression summary
print(results.summary())


# In[48]:


X = crypto_ret[['crypto', 'Date','mean', 'median', 'max', 'min', 'q_05', 'q_95', 'std', 'skew', 'kurt', 'Ret']]
# Assuming 'mean' corresponds to 'm_1' and 'Ret' corresponds to 'ex_ret'
# Sort data and calculate breakpoints
grp_x = X.groupby(['Date'])['q_95'].describe(percentiles=[0.2, 0.4, 0.6, 0.8]).reset_index()
grp_x = grp_x[['Date', '20%', '40%', '60%', '80%']]

# Merge breakpoints with the main dataframe
X = X.merge(grp_x, how='left', on=['Date'])

# Use the breakpoints to form the portfolios
def grp_bucket(row):
    if row['q_95'] <= row['20%']:
        value = 'H20'
    elif row['q_95'] <= row['40%']:
        value = 'H40'
    elif row['q_95'] <= row['60%']:
        value = 'H60'
    elif row['q_95'] <= row['80%']:
        value = 'H80'
    elif row['q_95'] > row['80%']:
        value = 'H100'
    else:
        value = ''
    return value

X['port'] = X.apply(grp_bucket, axis=1)

import pandas as pd

# Create a subset of your data with relevant columns
X = crypto_ret[['crypto', 'Date', 'q_95', 'Ret']]  # Assuming 'mean' corresponds to 'm_1' and 'Ret' corresponds to 'ex_ret'

# Drop any remaining rows with missing values
X.dropna(inplace=True)

# Sort data and calculate breakpoints
grp_x = X.groupby(['Date'])['q_95'].describe(percentiles=[0.2, 0.4, 0.6, 0.8]).reset_index()
grp_x = grp_x[['Date', '20%', '40%', '60%', '80%']]

# Merge breakpoints with the main dataframe
X = X.merge(grp_x, how='left', on=['Date'])

# Use the breakpoints to form the portfolios
def grp_bucket(row):
    if row['q_95'] <= row['20%']:
        value = 'H20'
    elif row['q_95'] <= row['40%']:
        value = 'H40'
    elif row['q_95'] <= row['60%']:
        value = 'H60'
    elif row['q_95'] <= row['80%']:
        value = 'H80'
    elif row['q_95'] > row['80%']:
        value = 'H100'
    else:
        value = ''
    return value

X['port'] = X.apply(grp_bucket, axis=1)
Portfolio = X.groupby(['Date', 'port'])['Ret'].mean().reset_index()

print(X)


# In[49]:


Portfolio = X.groupby(['Date', 'port'])['Ret'].mean().reset_index()
Portfolio


# In[50]:


# Calculate Difference Portfolio for your DataFrame
Portfolio_1 = pd.pivot_table(Portfolio, values='Ret', index=['Date'], columns=['port'])
Portfolio_1['HML'] = Portfolio_1['H100'] - Portfolio_1['H20']
print(Portfolio_1)


# In[51]:


# Calculate average returns for each portfolio
average_port = (Portfolio_1.mean() * 100).reset_index(name='return (%)')

# Calculate t-statistics for each portfolio #Also Run newey west Tstat
tstat = (Portfolio_1.mean() / Portfolio_1.sem()).reset_index(name='tstat')

# Merge average returns and t-statistics into a single DataFrame
average_port = average_port.merge(tstat, on='port')

print(Portfolio)
average_port.transpose()


# In[52]:


import pandas as pd
import statsmodels.formula.api as smf

# Assuming 'Portfolio' is your DataFrame containing portfolio returns by date and portfolio
# 'port' column contains the portfolio names ('H20', 'H40', 'H60', 'H80', 'H100')

# Iterate over each portfolio
for portfolio in ['H20', 'H40', 'H60', 'H80', 'H100']:
    # Filter data for the current portfolio
    portfolio_data = Portfolio[Portfolio['port'] == portfolio]
    
    # Define the formula for the OLS model
    formula = 'Ret ~ 1'
    
    # Fit OLS model with Newey-West standard errors
    model = smf.ols(formula, data=portfolio_data).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
    
    # Extract the t-statistic for the intercept
    t_statistic_newey_west = model.tvalues['Intercept']
    
    # Print the Newey-West t-statistic for the current portfolio
    print(f"Portfolio: {portfolio}, Newey-West t-statistic: {t_statistic_newey_west}")


# In[53]:


import pandas as pd

# Assuming you have loaded ff3 and Portfolio_1 DataFrames
ff3 = pd.read_csv("C:\\Users\\ulric\\OneDrive\\Bureau\\FF3.csv")

# Check data types of the 'Date' column
print(ff3['Date'].dtype)  # Verify the data type of 'Date' column

# Convert 'Date' column to datetime if needed
ff3['Date'] = pd.to_datetime(ff3['Date'])

# Now check the data type again
print(ff3['Date'].dtype)  # Should show datetime64[ns]

# Now try to merge on 'Date'
merged_data = pd.merge(ff3, Portfolio_1, on='Date')
print(merged_data.head())


# In[54]:


# Select relevant columns for regression
X = merged_data[['Mkt-RF', 'SMB', 'HML_x', 'RF']]  # Fama-French factors
y = merged_data['HML_y']  # Dependent variable (HML_y)

# Add constant to independent variables (X) for regression intercept
X = sm.add_constant(X)

# Fit Fama-French regression model
model = sm.OLS(y, X)
results = model.fit()

# Print regression summary
print(results.summary())


# In[55]:


X = crypto_ret[['crypto', 'Date','mean', 'median', 'max', 'min', 'q_05', 'q_95', 'std', 'skew', 'kurt', 'Ret']]
# Assuming 'mean' corresponds to 'm_1' and 'Ret' corresponds to 'ex_ret'
# Sort data and calculate breakpoints
grp_x = X.groupby(['Date'])['std'].describe(percentiles=[0.2, 0.4, 0.6, 0.8]).reset_index()
grp_x = grp_x[['Date', '20%', '40%', '60%', '80%']]

# Merge breakpoints with the main dataframe
X = X.merge(grp_x, how='left', on=['Date'])

# Use the breakpoints to form the portfolios
def grp_bucket(row):
    if row['std'] <= row['20%']:
        value = 'H20'
    elif row['std'] <= row['40%']:
        value = 'H40'
    elif row['std'] <= row['60%']:
        value = 'H60'
    elif row['std'] <= row['80%']:
        value = 'H80'
    elif row['std'] > row['80%']:
        value = 'H100'
    else:
        value = ''
    return value

X['port'] = X.apply(grp_bucket, axis=1)

import pandas as pd

# Create a subset of your data with relevant columns
X = crypto_ret[['crypto', 'Date', 'std', 'Ret']]  # Assuming 'mean' corresponds to 'm_1' and 'Ret' corresponds to 'ex_ret'

# Drop any remaining rows with missing values
X.dropna(inplace=True)

# Sort data and calculate breakpoints
grp_x = X.groupby(['Date'])['std'].describe(percentiles=[0.2, 0.4, 0.6, 0.8]).reset_index()
grp_x = grp_x[['Date', '20%', '40%', '60%', '80%']]

# Merge breakpoints with the main dataframe
X = X.merge(grp_x, how='left', on=['Date'])

# Use the breakpoints to form the portfolios
def grp_bucket(row):
    if row['std'] <= row['20%']:
        value = 'H20'
    elif row['std'] <= row['40%']:
        value = 'H40'
    elif row['std'] <= row['60%']:
        value = 'H60'
    elif row['std'] <= row['80%']:
        value = 'H80'
    elif row['std'] > row['80%']:
        value = 'H100'
    else:
        value = ''
    return value

X['port'] = X.apply(grp_bucket, axis=1)
Portfolio = X.groupby(['Date', 'port'])['Ret'].mean().reset_index()

print(X)


# In[56]:


Portfolio = X.groupby(['Date', 'port'])['Ret'].mean().reset_index()
Portfolio 


# In[57]:


# Calculate Difference Portfolio for your DataFrame
Portfolio_1 = pd.pivot_table(Portfolio, values='Ret', index=['Date'], columns=['port'])
Portfolio_1['HML'] = Portfolio_1['H100'] - Portfolio_1['H20']
print(Portfolio_1)


# In[58]:


# Calculate average returns for each portfolio
average_port = (Portfolio_1.mean() * 100).reset_index(name='return (%)')

# Calculate t-statistics for each portfolio #Also Run newey west Tstat
tstat = (Portfolio_1.mean() / Portfolio_1.sem()).reset_index(name='tstat')

# Merge average returns and t-statistics into a single DataFrame
average_port = average_port.merge(tstat, on='port')

print(Portfolio)
average_port.transpose()


# In[59]:


import pandas as pd
import statsmodels.formula.api as smf

# Assuming 'Portfolio' is your DataFrame containing portfolio returns by date and portfolio
# 'port' column contains the portfolio names ('H20', 'H40', 'H60', 'H80', 'H100')

# Iterate over each portfolio
for portfolio in ['H20', 'H40', 'H60', 'H80', 'H100']:
    # Filter data for the current portfolio
    portfolio_data = Portfolio[Portfolio['port'] == portfolio]
    
    # Define the formula for the OLS model
    formula = 'Ret ~ 1'
    
    # Fit OLS model with Newey-West standard errors
    model = smf.ols(formula, data=portfolio_data).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
    
    # Extract the t-statistic for the intercept
    t_statistic_newey_west = model.tvalues['Intercept']
    
    # Print the Newey-West t-statistic for the current portfolio
    print(f"Portfolio: {portfolio}, Newey-West t-statistic: {t_statistic_newey_west}")


# In[60]:


import pandas as pd

# Assuming you have loaded ff3 and Portfolio_1 DataFrames
ff3 = pd.read_csv("C:\\Users\\ulric\\OneDrive\\Bureau\\FF3.csv")

# Check data types of the 'Date' column
print(ff3['Date'].dtype)  # Verify the data type of 'Date' column

# Convert 'Date' column to datetime if needed
ff3['Date'] = pd.to_datetime(ff3['Date'])

# Now check the data type again
print(ff3['Date'].dtype)  # Should show datetime64[ns]

# Now try to merge on 'Date'
merged_data = pd.merge(ff3, Portfolio_1, on='Date')
print(merged_data.head())


# In[61]:


# Select relevant columns for regression
X = merged_data[['Mkt-RF', 'SMB', 'HML_x', 'RF']]  # Fama-French factors
y = merged_data['HML_y']  # Dependent variable (HML_y)

# Add constant to independent variables (X) for regression intercept
X = sm.add_constant(X)

# Fit Fama-French regression model
model = sm.OLS(y, X)
results = model.fit()

# Print regression summary
print(results.summary())


# In[62]:


X = crypto_ret[['crypto', 'Date','mean', 'median', 'max', 'min', 'q_05', 'q_95', 'std', 'skew', 'kurt', 'Ret']]
# Assuming 'mean' corresponds to 'm_1' and 'Ret' corresponds to 'ex_ret'
# Sort data and calculate breakpoints
grp_x = X.groupby(['Date'])['std'].describe(percentiles=[0.2, 0.4, 0.6, 0.8]).reset_index()
grp_x = grp_x[['Date', '20%', '40%', '60%', '80%']]

# Merge breakpoints with the main dataframe
X = X.merge(grp_x, how='left', on=['Date'])

# Use the breakpoints to form the portfolios
def grp_bucket(row):
    if row['std'] <= row['20%']:
        value = 'H20'
    elif row['std'] <= row['40%']:
        value = 'H40'
    elif row['std'] <= row['60%']:
        value = 'H60'
    elif row['std'] <= row['80%']:
        value = 'H80'
    elif row['std'] > row['80%']:
        value = 'H100'
    else:
        value = ''
    return value

X['port'] = X.apply(grp_bucket, axis=1)

import pandas as pd

# Create a subset of your data with relevant columns
X = crypto_ret[['crypto', 'Date', 'std', 'Ret']]  # Assuming 'mean' corresponds to 'm_1' and 'Ret' corresponds to 'ex_ret'

# Drop any remaining rows with missing values
X.dropna(inplace=True)

# Sort data and calculate breakpoints
grp_x = X.groupby(['Date'])['std'].describe(percentiles=[0.2, 0.4, 0.6, 0.8]).reset_index()
grp_x = grp_x[['Date', '20%', '40%', '60%', '80%']]

# Merge breakpoints with the main dataframe
X = X.merge(grp_x, how='left', on=['Date'])

# Use the breakpoints to form the portfolios
def grp_bucket(row):
    if row['std'] <= row['20%']:
        value = 'H20'
    elif row['std'] <= row['40%']:
        value = 'H40'
    elif row['std'] <= row['60%']:
        value = 'H60'
    elif row['std'] <= row['80%']:
        value = 'H80'
    elif row['std'] > row['80%']:
        value = 'H100'
    else:
        value = ''
    return value

X['port'] = X.apply(grp_bucket, axis=1)
Portfolio = X.groupby(['Date', 'port'])['Ret'].mean().reset_index()

print(X)


# In[63]:


Portfolio = X.groupby(['Date', 'port'])['Ret'].mean().reset_index()
Portfolio 


# In[64]:


# Calculate Difference Portfolio for your DataFrame
Portfolio_1 = pd.pivot_table(Portfolio, values='Ret', index=['Date'], columns=['port'])
Portfolio_1['HML'] = Portfolio_1['H100'] - Portfolio_1['H20']
print(Portfolio_1)


# In[65]:


# Calculate average returns for each portfolio
average_port = (Portfolio_1.mean() * 100).reset_index(name='return (%)')

# Calculate t-statistics for each portfolio #Also Run newey west Tstat
tstat = (Portfolio_1.mean() / Portfolio_1.sem()).reset_index(name='tstat')

# Merge average returns and t-statistics into a single DataFrame
average_port = average_port.merge(tstat, on='port')

print(Portfolio)
average_port.transpose()


# In[66]:


X = crypto_ret[['crypto', 'Date','mean', 'median', 'max', 'min', 'q_05', 'q_95', 'std', 'skew', 'kurt', 'Ret']]
# Assuming 'mean' corresponds to 'm_1' and 'Ret' corresponds to 'ex_ret'
# Sort data and calculate breakpoints
grp_x = X.groupby(['Date'])['skew'].describe(percentiles=[0.2, 0.4, 0.6, 0.8]).reset_index()
grp_x = grp_x[['Date', '20%', '40%', '60%', '80%']]

# Merge breakpoints with the main dataframe
X = X.merge(grp_x, how='left', on=['Date'])

# Use the breakpoints to form the portfolios
def grp_bucket(row):
    if row['skew'] <= row['20%']:
        value = 'H20'
    elif row['skew'] <= row['40%']:
        value = 'H40'
    elif row['skew'] <= row['60%']:
        value = 'H60'
    elif row['skew'] <= row['80%']:
        value = 'H80'
    elif row['skew'] > row['80%']:
        value = 'H100'
    else:
        value = ''
    return value

X['port'] = X.apply(grp_bucket, axis=1)

import pandas as pd

# Create a subset of your data with relevant columns
X = crypto_ret[['crypto', 'Date', 'skew', 'Ret']]  # Assuming 'mean' corresponds to 'm_1' and 'Ret' corresponds to 'ex_ret'

# Drop any remaining rows with missing values
X.dropna(inplace=True)

# Sort data and calculate breakpoints
grp_x = X.groupby(['Date'])['skew'].describe(percentiles=[0.2, 0.4, 0.6, 0.8]).reset_index()
grp_x = grp_x[['Date', '20%', '40%', '60%', '80%']]

# Merge breakpoints with the main dataframe
X = X.merge(grp_x, how='left', on=['Date'])

# Use the breakpoints to form the portfolios
def grp_bucket(row):
    if row['skew'] <= row['20%']:
        value = 'H20'
    elif row['skew'] <= row['40%']:
        value = 'H40'
    elif row['skew'] <= row['60%']:
        value = 'H60'
    elif row['skew'] <= row['80%']:
        value = 'H80'
    elif row['skew'] > row['80%']:
        value = 'H100'
    else:
        value = ''
    return value

X['port'] = X.apply(grp_bucket, axis=1)
Portfolio = X.groupby(['Date', 'port'])['Ret'].mean().reset_index()

print(X)


# In[67]:


Portfolio = X.groupby(['Date', 'port'])['Ret'].mean().reset_index()
Portfolio 


# In[68]:


# Calculate Difference Portfolio for your DataFrame
Portfolio_1 = pd.pivot_table(Portfolio, values='Ret', index=['Date'], columns=['port'])
Portfolio_1['HML'] = Portfolio_1['H100'] - Portfolio_1['H20']
print(Portfolio_1)


# In[69]:


# Calculate average returns for each portfolio
average_port = (Portfolio_1.mean() * 100).reset_index(name='return (%)')

# Calculate t-statistics for each portfolio #Also Run newey west Tstat
tstat = (Portfolio_1.mean() / Portfolio_1.sem()).reset_index(name='tstat')

# Merge average returns and t-statistics into a single DataFrame
average_port = average_port.merge(tstat, on='port')

print(Portfolio)
average_port.transpose()


# In[70]:


import pandas as pd
import statsmodels.formula.api as smf

# Assuming 'Portfolio' is your DataFrame containing portfolio returns by date and portfolio
# 'port' column contains the portfolio names ('H20', 'H40', 'H60', 'H80', 'H100')

# Iterate over each portfolio
for portfolio in ['H20', 'H40', 'H60', 'H80', 'H100']:
    # Filter data for the current portfolio
    portfolio_data = Portfolio[Portfolio['port'] == portfolio]
    
    # Define the formula for the OLS model
    formula = 'Ret ~ 1'
    
    # Fit OLS model with Newey-West standard errors
    model = smf.ols(formula, data=portfolio_data).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
    
    # Extract the t-statistic for the intercept
    t_statistic_newey_west = model.tvalues['Intercept']
    
    # Print the Newey-West t-statistic for the current portfolio
    print(f"Portfolio: {portfolio}, Newey-West t-statistic: {t_statistic_newey_west}")


# In[71]:


import pandas as pd

# Assuming you have loaded ff3 and Portfolio_1 DataFrames
ff3 = pd.read_csv("C:\\Users\\ulric\\OneDrive\\Bureau\\FF3.csv")

# Check data types of the 'Date' column
print(ff3['Date'].dtype)  # Verify the data type of 'Date' column

# Convert 'Date' column to datetime if needed
ff3['Date'] = pd.to_datetime(ff3['Date'])

# Now check the data type again
print(ff3['Date'].dtype)  # Should show datetime64[ns]

# Now try to merge on 'Date'
merged_data = pd.merge(ff3, Portfolio_1, on='Date')
print(merged_data.head())


# In[72]:


# Select relevant columns for regression
X = merged_data[['Mkt-RF', 'SMB', 'HML_x', 'RF']]  # Fama-French factors
y = merged_data['HML_y']  # Dependent variable (HML_y)

# Add constant to independent variables (X) for regression intercept
X = sm.add_constant(X)

# Fit Fama-French regression model
model = sm.OLS(y, X)
results = model.fit()

# Print regression summary
print(results.summary())


# In[73]:


X = crypto_ret[['crypto', 'Date','mean', 'median', 'max', 'min', 'q_05', 'q_95', 'std', 'skew', 'kurt', 'Ret']]
# Assuming 'mean' corresponds to 'm_1' and 'Ret' corresponds to 'ex_ret'
# Sort data and calculate breakpoints
grp_x = X.groupby(['Date'])['kurt'].describe(percentiles=[0.2, 0.4, 0.6, 0.8]).reset_index()
grp_x = grp_x[['Date', '20%', '40%', '60%', '80%']]

# Merge breakpoints with the main dataframe
X = X.merge(grp_x, how='left', on=['Date'])

# Use the breakpoints to form the portfolios
def grp_bucket(row):
    if row['kurt'] <= row['20%']:
        value = 'H20'
    elif row['kurt'] <= row['40%']:
        value = 'H40'
    elif row['kurt'] <= row['60%']:
        value = 'H60'
    elif row['kurt'] <= row['80%']:
        value = 'H80'
    elif row['kurt'] > row['80%']:
        value = 'H100'
    else:
        value = ''
    return value

X['port'] = X.apply(grp_bucket, axis=1)

import pandas as pd

# Create a subset of your data with relevant columns
X = crypto_ret[['crypto', 'Date', 'kurt', 'Ret']]  # Assuming 'mean' corresponds to 'm_1' and 'Ret' corresponds to 'ex_ret'

# Drop any remaining rows with missing values
X.dropna(inplace=True)

# Sort data and calculate breakpoints
grp_x = X.groupby(['Date'])['kurt'].describe(percentiles=[0.2, 0.4, 0.6, 0.8]).reset_index()
grp_x = grp_x[['Date', '20%', '40%', '60%', '80%']]

# Merge breakpoints with the main dataframe
X = X.merge(grp_x, how='left', on=['Date'])

# Use the breakpoints to form the portfolios
def grp_bucket(row):
    if row['kurt'] <= row['20%']:
        value = 'H20'
    elif row['kurt'] <= row['40%']:
        value = 'H40'
    elif row['kurt'] <= row['60%']:
        value = 'H60'
    elif row['kurt'] <= row['80%']:
        value = 'H80'
    elif row['kurt'] > row['80%']:
        value = 'H100'
    else:
        value = ''
    return value

X['port'] = X.apply(grp_bucket, axis=1)
Portfolio = X.groupby(['Date', 'port'])['Ret'].mean().reset_index()

print(X)


# In[74]:


Portfolio = X.groupby(['Date', 'port'])['Ret'].mean().reset_index()
Portfolio 


# In[75]:


# Calculate Difference Portfolio for your DataFrame
Portfolio_1 = pd.pivot_table(Portfolio, values='Ret', index=['Date'], columns=['port'])
Portfolio_1['HML'] = Portfolio_1['H100'] - Portfolio_1['H20']
print(Portfolio_1)


# In[76]:


# Calculate average returns for each portfolio
average_port = (Portfolio_1.mean() * 100).reset_index(name='return (%)')

# Calculate t-statistics for each portfolio #Also Run newey west Tstat
tstat = (Portfolio_1.mean() / Portfolio_1.sem()).reset_index(name='tstat')

# Merge average returns and t-statistics into a single DataFrame
average_port = average_port.merge(tstat, on='port')

print(Portfolio)
average_port.transpose()


# In[77]:


import pandas as pd
import statsmodels.formula.api as smf

# Assuming 'Portfolio' is your DataFrame containing portfolio returns by date and portfolio
# 'port' column contains the portfolio names ('H20', 'H40', 'H60', 'H80', 'H100')

# Iterate over each portfolio
for portfolio in ['H20', 'H40', 'H60', 'H80', 'H100']:
    # Filter data for the current portfolio
    portfolio_data = Portfolio[Portfolio['port'] == portfolio]
    
    # Define the formula for the OLS model
    formula = 'Ret ~ 1'
    
    # Fit OLS model with Newey-West standard errors
    model = smf.ols(formula, data=portfolio_data).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
    
    # Extract the t-statistic for the intercept
    t_statistic_newey_west = model.tvalues['Intercept']
    
    # Print the Newey-West t-statistic for the current portfolio
    print(f"Portfolio: {portfolio}, Newey-West t-statistic: {t_statistic_newey_west}")


# In[78]:


import pandas as pd

# Assuming you have loaded ff3 and Portfolio_1 DataFrames
ff3 = pd.read_csv("C:\\Users\\ulric\\OneDrive\\Bureau\\FF3.csv")

# Check data types of the 'Date' column
print(ff3['Date'].dtype)  # Verify the data type of 'Date' column

# Convert 'Date' column to datetime if needed
ff3['Date'] = pd.to_datetime(ff3['Date'])

# Now check the data type again
print(ff3['Date'].dtype)  # Should show datetime64[ns]

# Now try to merge on 'Date'
merged_data = pd.merge(ff3, Portfolio_1, on='Date')
print(merged_data.head())


# In[79]:


# Select relevant columns for regression
X = merged_data[['Mkt-RF', 'SMB', 'HML_x', 'RF']]  # Fama-French factors
y = merged_data['HML_y']  # Dependent variable (HML_y)

# Add constant to independent variables (X) for regression intercept
X = sm.add_constant(X)

# Fit Fama-French regression model
model = sm.OLS(y, X)
results = model.fit()

# Print regression summary
print(results.summary())


# In[ ]:





# In[21]:


# Load the saved CSV file for further calculations
loaded_crypto_ret = pd.read_csv('crypto_ret.csv')



# In[22]:


print(loaded_crypto_ret.head())


# In[23]:


ff3 = pd.read_csv("C:\\Users\\ulric\\OneDrive\\Bureau\\FF3.csv")
ff3.rename(columns={"Unnamed: 0": "Date"}, inplace = True)
ff3.head()


# In[ ]:





# In[ ]:





# In[ ]:


#Bivariated Portfolio based on Stats


# In[ ]:


#Mean and q_05


# In[87]:


import pandas as pd

# Read the data from CSV files
ff3 = pd.read_csv("C:\\Users\\ulric\\OneDrive\\Bureau\\FF3.csv")
ff3.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
loaded_crypto_ret = pd.read_csv('crypto_ret.csv')

# Select only the desired columns from ff3
ff3_subset = ff3[['Date', 'Mkt-RF', 'SMB', 'HML', 'RF']]

# Merge the DataFrames on the "Date" column
merged_df = pd.merge(loaded_crypto_ret, ff3_subset, on='Date', how='inner')

# Display the merged DataFrame
print(merged_df.head())


# In[96]:


X = merged_df[['crypto', 'Date', 'Ret', 'mean', 'q_05']]

grp_mean = X.groupby(['Date'])['mean'].describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).reset_index()

grp_mean = grp_mean[["Date", '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']]

grp_mean = grp_mean.rename(columns={'10%': 'mn10', '20%': 'mn20', '30%': 'mn30', '40%': 'mn40',
                                     '50%': 'mn50', '60%': 'mn60', '70%': 'mn70', '80%': 'mn80',
                                     '90%': 'mn90'})


# Size breakdown
grp_q_05 = X.groupby(['Date'])['q_05'].describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).reset_index()

grp_q_05 = grp_q_05[["Date", '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']]

grp_q_05 = grp_q_05.rename(columns={'10%': 'q10', '20%': 'q20', '30%': 'q30', '40%': 'q40',
                                     '50%': 'q50', '60%': 'q60', '70%': 'q70', '80%': 'q80',
                                     '90%': 'q90'})
#Mean and q_05


# In[97]:


def mn_bucket(row):
    if 0 <= row['mean'] <= row['mn10']:
        value = 'MN10'
    elif row['mean'] <= row['mn20']:
        value = 'MN20'
    elif row['mean'] <= row['mn30']:
        value = 'MN30'
    elif row['mean'] <= row['mn40']:
        value = 'MN40'
    elif row['mean'] <= row['mn50']:
        value = 'MN50'
    elif row['mean'] <= row['mn60']:
        value = 'MN60'
    elif row['mean'] <= row['mn70']:
        value = 'MN70'
    elif row['mean'] <= row['mn80']:
        value = 'MN80'
    elif row['mean'] <= row['mn90']:
        value = 'MN90'
    elif row['mean'] > row['mn90']:
        value = 'MN100'
    else:
        value = ''
    return value


def q_bucket(row):
    if 0 <= row['q_05'] <= row['q10']:
        value = 'Q10'
    elif row['q_05'] <= row['q20']:
        value = 'Q20'
    elif row['q_05'] <= row['q30']:
        value = 'Q30'
    elif row['q_05'] <= row['q40']:
        value = 'Q40'
    elif row['q_05'] <= row['q50']:
        value = 'Q50'
    elif row['q_05'] <= row['q60']:
        value = 'Q60'
    elif row['q_05'] <= row['q70']:
        value = 'Q70'
    elif row['q_05'] <= row['q80']:
        value = 'Q80'
    elif row['q_05'] <= row['q90']:
        value = 'Q90'
    elif row['q_05'] > row['q90']:
        value = 'Q100'
    else:
        value = ''
    return value


# In[100]:


mean_grp = X.merge(grp_mean, on=['Date'])
mean_grp['mn_port'] = mean_grp.apply(mn_bucket, axis=1)


q_05_grp = X.merge(grp_q_05, on=['Date'])
q_05_grp['q_port'] = q_05_grp.apply(q_bucket, axis=1)


# In[101]:


q_05_grp


# In[103]:


final_grp = pd.merge(mean_grp[['crypto', 'Date', 'Ret', 'mean', 'q_05','mn_port' ]], 
                     q_05_grp[['crypto', 'Date', 'q_port']], on =['Date', 'crypto'])


# In[104]:


final_grp 


# In[105]:


#Portfolio_ret_2['sbport'] = Portfolio_ret_2['cn_port_1']+Portfolio_ret_2['sz_port_1']
FBD = final_grp.groupby(['Date', 'mn_port', 'q_port'])['Ret'].mean().reset_index()
FBDS = FBD.groupby(['mn_port', 'q_port'])['Ret'].mean()


# In[106]:


FBDS.unstack()


# In[107]:


(FBD.groupby(['mn_port', 'q_port'])['Ret'].mean()/FBD.groupby(['mn_port', 'q_port'])['Ret'].sem()).unstack()


# In[ ]:


#q_05 and standard.Dev


# In[125]:


X = merged_df[['crypto', 'Date', 'Ret', 'std', 'q_95']]

grp_q_95 = X.groupby(['Date'])['q_95'].describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).reset_index()

grp_q_95 = grp_q_95[["Date", '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']]

ggrp_q_95 = grp_q_95.rename(columns={'10%':'v10','20%': 'v20', '30%': 'v30', '40%': 'v40',
                                     '50%': 'v50', '60%': 'v60', '70%': 'v70', '80%': 'v80',
                                     '90%': 'v90'})

# Size breakdown
grp_std = X.groupby(['Date'])['std'].describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).reset_index()

grp_std = grp_std[["Date", '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']]

grp_std = grp_std.rename(columns={'10%': 'std10', '20%': 'std20', '30%': 'std30', '40%': 'std40',
                                     '50%': 'std50', '60%': 'std60', '70%': 'std70', '80%': 'std80',
                                     '90%': 'std90'})


# In[129]:


X = merged_df[['crypto', 'Date', 'Ret', 'std', 'q_95']]

# Calculate percentiles for q_95
grp_q_95 = X.groupby(['Date'])['q_95'].describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).reset_index()

# Select percentile columns and rename them
grp_q_95 = grp_q_95[['Date', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']]
grp_q_95 = grp_q_95.rename(columns={'10%': 'v10', '20%': 'v20', '30%': 'v30', '40%': 'v40',
                                     '50%': 'v50', '60%': 'v60', '70%': 'v70', '80%': 'v80',
                                     '90%': 'v90'})

# Merge q_95 percentiles back to the original DataFrame X
q_95_grp = X.merge(grp_q_95, on=['Date'])

# Apply the v_bucket function to create 'v_port' column
q_95_grp['v_port'] = q_95_grp.apply(v_bucket, axis=1)

# Similarly, calculate percentiles for 'std'
grp_std = X.groupby(['Date'])['std'].describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).reset_index()

# Select percentile columns and rename them for 'std'
grp_std = grp_std[['Date', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']]
grp_std = grp_std.rename(columns={'10%': 'std10', '20%': 'std20', '30%': 'std30', '40%': 'std40',
                                   '50%': 'std50', '60%': 'std60', '70%': 'std70', '80%': 'std80',
                                   '90%': 'std90'})

# Merge std percentiles back to the original DataFrame X
std_grp = X.merge(grp_std, on=['Date'])

# Apply the std_bucket function to create 'std_port' column
std_grp['std_port'] = std_grp.apply(std_bucket, axis=1)


# In[128]:


print(q_95_grp.columns)


# In[130]:


def v_bucket(row):
    if 0 <= row['q_95'] <= row['v10']:
        value = 'V10'
    elif row['q_95'] <= row['v20']:
        value = 'V20'
    elif row['q_95'] <= row['v30']:
        value = 'V30'
    elif row['q_95'] <= row['v40']:
        value = 'V40'
    elif row['q_95'] <= row['v50']:
        value = 'V50'
    elif row['q_95'] <= row['v60']:
        value = 'V60'
    elif row['q_95'] <= row['v70']:
        value = 'V70'
    elif row['q_95'] <= row['v80']:
        value = 'V80'
    elif row['q_95'] <= row['v90']:
        value = 'V90'
    elif row['q_95'] > row['v90']:
        value = 'V100'
    else:
        value = ''
    return value

def std_bucket(row):
    if 0 <= row['std'] <= row['std10']:
        value = 'STD10'
    elif row['std'] <= row['std20']:
        value = 'STD20'
    elif row['std'] <= row['std30']:
        value = 'STD30'
    elif row['std'] <= row['std40']:
        value = 'STD40'
    elif row['std'] <= row['std50']:
        value = 'STD50'
    elif row['std'] <= row['std60']:
        value = 'STD60'
    elif row['std'] <= row['std70']:
        value = 'STD70'
    elif row['std'] <= row['std80']:
        value = 'STD80'
    elif row['std'] <= row['std90']:
        value = 'STD90'
    elif row['std'] > row['std90']:
        value = 'STD100'
    else:
        value = ''
    return value



# In[131]:


q_95_grp = X.merge(grp_q_95, on=['Date'])
q_95_grp['v_port'] = q_95_grp.apply(v_bucket, axis=1)


std_grp = X.merge(grp_std, on=['Date'])
std_grp['std_port'] = std_grp.apply(std_bucket, axis=1)


# In[133]:


std_grp


# In[134]:


final_grp = pd.merge(q_95_grp[['crypto', 'Date', 'Ret', 'std', 'q_95','v_port' ]], 
                     std_grp[['crypto', 'Date', 'std_port']], on =['Date', 'crypto'])


# In[135]:


final_grp


# In[137]:


#Portfolio_ret_2['sbport'] = Portfolio_ret_2['cn_port_1']+Portfolio_ret_2['sz_port_1']
FBD = final_grp.groupby(['Date', 'v_port', 'std_port'])['Ret'].mean().reset_index()
FBDS = FBD.groupby(['v_port', 'std_port'])['Ret'].mean()
FBDS.unstack()


# In[138]:


(FBD.groupby(['v_port', 'std_port'])['Ret'].mean()/FBD.groupby(['v_port', 'std_port'])['Ret'].sem()).unstack()


# In[154]:


X = merged_df[['crypto', 'Date', 'Ret', 'max', 'min']]

grp_max = X.groupby(['Date'])['max'].describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).reset_index()

grp_max = grp_max[["Date", '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']]

grp_max = grp_max.rename(columns={'10%':'ma10','20%': 'ma20', '30%': 'ma30', '40%': 'ma40',
                                     '50%': 'ma50', '60%': 'ma60', '70%': 'ma70', '80%': 'ma80',
                                     '90%': 'ma90'})

# Size breakdown
grp_min = X.groupby(['Date'])['min'].describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).reset_index()

grp_min = grp_min [["Date", '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']]

grp_min = grp_min .rename(columns={'10%': 'min10', '20%': 'min20', '30%': 'min30', '40%': 'min40',
                                     '50%': 'min50', '60%': 'min60', '70%': 'min70', '80%': 'min80',
                                     '90%': 'min90'})


# In[157]:


def ma_bucket(row):
    if 0 <= row['max'] <= row['ma10']:
        value = 'MA10'
    elif row['max'] <= row['ma20']:
        value = 'MA20'
    elif row['max'] <= row['ma30']:
        value = 'MA30'
    elif row['max'] <= row['ma40']:
        value = 'MA40'
    elif row['max'] <= row['ma50']:
        value = 'MA50'
    elif row['max'] <= row['ma60']:
        value = 'MA60'
    elif row['max'] <= row['ma70']:
        value = 'MA70'
    elif row['max'] <= row['ma80']:
        value = 'MA80'
    elif row['max'] <= row['ma90']:
        value = 'MA90'
    elif row['max'] > row['ma90']:
        value = 'MA100'
    else:
        value = ''
    return value

def min_bucket(row):
    if 0 <= row['min'] <= row['min10']:
        value = 'MIN10'
    elif row['min'] <= row['min20']:
        value = 'MIN20'
    elif row['min'] <= row['min30']:
        value = 'MIN30'
    elif row['min'] <= row['min40']:
        value = 'MIN40'
    elif row['min'] <= row['min50']:
        value = 'MIN50'
    elif row['min'] <= row['min60']:
        value = 'MIN60'
    elif row['min'] <= row['min70']:
        value = 'MIN70'
    elif row['min'] <= row['min80']:
        value = 'MIN80'
    elif row['min'] <= row['min90']:
        value = 'MIN90'
    elif row['min'] > row['min90']:
        value = 'MIN100'
    else:
        value = ''
    return value


# In[158]:


grp_max = X.merge(grp_max, on=['Date'])
grp_max['ma_port'] = grp_max.apply(ma_bucket, axis=1)


min_grp = X.merge(min_grp , on=['Date'])
min_grp ['min_port'] = min_grp.apply(min_bucket, axis=1)


# In[ ]:





# In[ ]:


#merged_data  = np.log(df) - np.log(df.shift(1))
#merged_data  = merged_data.replace([np.inf, -np.inf], np.nan)

