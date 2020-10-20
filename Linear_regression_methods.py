import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()

# import data and look at top values to check what it looks like
raw_data = pd.read_csv('1.04. Real-life example.csv')
raw_data.head()

# Get an overview of the data
print(raw_data.describe(include='all'))

# Remove unwanted columns (redundant bits of info)
data = raw_data.drop(['Model'], axis=1)

# Find how many values are non existent
print(data.isnull().sum())
# If they account for fewer than 5% of thje data set, they may be removed and this is how
data_no_mv = data.dropna(axis=0)

# Exploring the PDFs

sns.displot(data_no_mv['Price'])
# plt.show()

# Dealing with outliers
q = data_no_mv['Price'].quantile(0.99)
data_1 = data_no_mv[data_no_mv['Price']<q]
print(data_1.describe(include='all'))

# Repeat for other categories checking for mis-input values
# Removing top 1%
q = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage']<q]
# Top value can only be 6.5 so any others are incorrect entries
data_3 = data_2[data_2['EngineV']<6.5]
# Removing the bottom 1%
q = data_3['Year'].quantile(0.01)
data_4 = data_3[data_3['Year']>q]

# Group all cleaned data and reset indexes
data_cleaned = data_4.reset_index(drop=True)

# Checking the OLS assumptions
# Relaxing the assumptions by converting exponential price into linear log_price
log_price = np.log(data_cleaned['Price'])
data_cleaned['log_price'] = log_price
# First with scatter plots
f, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True, figsize=(15,3))
ax1.scatter(data_cleaned['Year'], data_cleaned['log_price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'], data_cleaned['log_price'])
ax2.set_title('EngineV and Price')
ax3.scatter(data_cleaned['Mileage'], data_cleaned['log_price'])
ax3.set_title('Price and Mileage')

# plt.show()

# Drop original price variable
data_cleaned = data_cleaned.drop(['Price'],axis=1)

# Checking multicollinearity with variance inflation factor (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_cleaned[['Mileage', 'Year', 'EngineV']]
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['Features'] = variables.columns
print(vif)
# VIF = 1: No multicollinearity; VIF = 1-5: All's good; VIF above 5 or 6 may be unacceptable but can be accepted up to 10
data_no_multicollinearity = data_cleaned.drop(['Year'], axis=1)

# Create dummy variables
data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)

# Rearrange data
print(data_with_dummies.columns.values)
cols = ['log_price', 'Mileage', 'EngineV',  'Brand_BMW', 'Brand_Mercedes-Benz',
 'Brand_Mitsubishi', 'Brand_Renault', 'Brand_Toyota', 'Brand_Volkswagen',
 'Body_hatch', 'Body_other', 'Body_sedan', 'Body_vagon', 'Body_van',
 'Engine Type_Gas', 'Engine Type_Other', 'Engine Type_Petrol',
 'Registration_yes']
data_preprocessed = data_with_dummies[cols]

# Declare the inputs and the targets
targets = data_preprocessed['log_price']
inputs = data_preprocessed.drop(['log_price'],axis=1)

# Scale the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(inputs)

inputs_scaled = scaler.transform(inputs)

# Train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365)

# Create the regression
reg = LinearRegression()
reg.fit(x_train, y_train)
# Compare predicted values to target values
y_hat = reg.predict(x_train)
plt.scatter(y_train, y_hat)
plt.xlabel('Targets (y_train)', size=18)
plt.ylabel('Predictions (y_hat)', size=18)
plt.xlim(6,13)
plt.ylim(6,13)
# plt.show()

# Residual plot
sns.displot(y_train - y_hat)
plt.title('Residuals PDF', size=18)
# plt.show()

# Find R squared
reg.score(x_train, y_train)

# Finding the weights and biases
reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
print(reg_summary)

# A positive weight shows a feature has a positive correlation with price (log price)
# A negative weight shows a feature has a negative correlation with price (log price)
# Looking at variables with dummies
data_cleaned['Brand'].unique()
# As audi is the benchmark, all the dummy variables are based around Audi at 0
# e.g. Positive values mean a brand is more expensive than audi

# Testing

y_hat_test = reg.predict(x_test)
plt.scatter(y_train, y_hat_test
            # Can have an alpha value here to have a 'heatmap' of concentration of values
            )
plt.xlabel('Targets (y_train)', size=18)
plt.ylabel('Predictions (y_hat_test)', size=18)
plt.xlim(6,13)
plt.ylim(6,13)
# plt.show()

# Show predictions compared to targets
df_pf = pd.DataFrame(y_hat_test, columns=['Prediction'])
# This resets indexes so numpy doesn't break it
y_test = y_test.reset_index(drop=True)
df_pf['Target'] = np.exp(y_test)

# Calculate the residuals to see how good the model is with test data
df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']
df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf.describe()

# Investigate the dataframe
# Set max rows
pd.options.display.max_rows = 999
# Set floats to 2 decimal places
pd.set_option('display.float_format', lambda x: '%.2f' % x)
df_pf.sort_values(by=['Difference%'])

