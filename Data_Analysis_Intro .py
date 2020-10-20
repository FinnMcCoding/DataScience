import pandas as pd
import seaborn as sns

#listings_df = pd.read_csv('Airbnb_data.csv')

for value in range(1,2):
    print(value)
# Show columns and rows:
# listings_df.shape

# Show column types:
# listings_df.dtypes
# Object=String, Int=Integer

# If these do not work may need to pass delimiter parameter in .read command

# .head and .tail show first and last 5 data points respectively

# **Cleaning the data
# Shows how many fields are empty
# print(listings_df.isnull().sum())

# Dropping some columns (data types)
# columns_to_drop = ['id', 'host_name', 'last_review']
# listings_df.drop(columns_to_drop, axis="columns", inplace=True)

# Replacing the empty cells in the dataframe with a default value (0 here)
# listings_df.fillna({'reviews_per_month': 0}, inplace=True)

# Filtering by column
# listings_df['name'] Makes a Series
# listings_df[['name', 'neighbourhood_group', 'price']] Makes a Dataframe

# Filtering by row
# listings_df[5:100]

# **Boolean Indexing**
# a = listings_df['price'] < 100 This will produce a list of true/false values attached to an id
# listings_df[a] produces all of the data values for which a is True

# Question 1: What are the ten most reviewed listings:
# listings_df.nlargest(10, 'number_of_reviews')

# Question 2: What are the NY neighbourhood groups with listings?
# listings_df['neighbourhood_group'].unique()
# Sub-Question 2a: How many listings per neighbourhood group?
# listings_df['neighbourhood_group'].value_counts()

# Question 3: What are the top 10 neighbourhoods with airbnb listings?
# listings_df['neighbourhood'].value_counts().head(10)

# Plotting a graph:
# listings_df['neighbourhood'].value_counts().head(10).plot(kind='bar)
# or
# sns.countplot(data=listings_df, x='neigbourhood_group')


