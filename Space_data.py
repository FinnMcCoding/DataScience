import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read file
listings_df = pd.read_csv('Space_Corrected.csv')

# Rename unnanmed column for readability
listings_df.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)

# Reduce to a dataframe of only needed columns
graph_df = listings_df[['Company Name', 'Status Mission']]

# Removing all comapnies with fewer than 100 rockets fired
counts = graph_df['Company Name'].value_counts()
res = graph_df[~graph_df['Company Name'].isin(counts[counts < 100].index)]

# Replacing specific versions of launch failure with a blanket failure term
res.loc[res['Status Mission'] != 'Success', 'Status Mission'] = 'Failure'

# Calculating and assigning success rates to each company in a new dataframe
success_df = pd.DataFrame(columns=['Company Name', 'Success Rate'])
for company in res['Company Name'].unique():
    specific_company = res[res['Company Name']==company]
    success_value = specific_company.value_counts().values
    success_rate = (success_value[0]/(success_value[1] + success_value[0]))*100
    success_df = success_df.append({
        'CompanyName' : f'{company}',
        'Success Rate' : success_rate},
        ignore_index=True)

# Plotting the graph
sns.barplot(data=success_df, x='CompanyName', y='Success Rate', order=success_df.sort_values('Success Rate', ascending=False).CompanyName)
plt.xlabel("Company Name", size=15)
plt.ylabel("Success rate", size=15)
plt.title("Rocket launch Success Rates", size=18)
plt.tight_layout()
plt.ylim(80, 100)
plt.show()
