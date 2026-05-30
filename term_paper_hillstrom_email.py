import pandas as pd
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Task 1: Load Hillstrom.csv into Python. 
df = pd.read_csv('Hillstrom.csv')
print(df.head())
print(df.columns)

# Task 1: Create a new column got_email: set it to 1 for rows where segment is “Mens E-Mail” or “Womens E-Mail”, and 0 for rows where segment is “No E-Mail. 
df['got_email'] = (df['segment'].isin(['Mens E-Mail', 'Womens E-Mail'])).astype(int)
print(df['got_email'].value_counts())

# Task 1: Sanity check: compute the mean of conversion in the got_email = 1 and got_email = 0 groups; the difference between the two means is βˆ.
email_mean = df[df['got_email'] == 1]['conversion'].mean()
no_email_mean = df[df['got_email'] == 0]['conversion'].mean()
beta_hat = email_mean - no_email_mean
print(email_mean)
print(no_email_mean)
print(beta_hat)

# Task 1: Estimate the regression and read off βˆ with its standard error. Refer to the in-class.ipynb notebooks from the RCT lectures for the regression code template.
X = sm.add_constant(df['got_email'])
y = df['conversion']
model = sm.OLS(y, X).fit()
print(model.summary())

# Task 2: Pick a customer characteristic with 2–3 distinct levels.
# Zip code 

# Task 2: Split the dataset by that characteristic.
urban = df[df['zip_code'] == 'Urban']
suburban = df[df['zip_code'] == 'Surburban']
rural = df[df['zip_code'] == 'Rural']

# Task 2:  Re-run the Task 1 regression within each subgroup, using the same code template as before

# Urban 
X = sm.add_constant(urban['got_email'])
y = urban['conversion']
urban_model = sm.OLS(y, X).fit()
print('Urban')
print(urban_model.params['got_email'])
print(urban_model.bse['got_email'])

# Suburban 
X = sm.add_constant(suburban['got_email'])
y = suburban['conversion']
suburban_model = sm.OLS(y, X).fit()
print('Suburban')
print(suburban_model.params['got_email'])
print(suburban_model.bse['got_email'])

# Rural 
X = sm.add_constant(rural['got_email'])
y = rural['conversion']
rural_model = sm.OLS(y, X).fit()
print('Rural')
print(rural_model.params['got_email'])
print(rural_model.bse['got_email'])

# Task 2:  Compare βˆg across subgroups; discuss what would change your interpretation.

# To examine whether the email effect differed across customer groups, we split the sample by zip code and estimated the treatment effect separately for Urban, Suburban, and Rural customers. 
# The estimated effects were β̂Urban = 0.00513, β̂Suburban = 0.00499, and β̂Rural = 0.00433.
# Since all three estimates are positive and very similar to the overall treatment effect (β̂ = 0.00495), we find little evidence that the effectiveness of the email campaign varies by zip code. 
# If the subgroup estimates had differed substantially, we would conclude that the campaign's impact depends on customer location and that a single overall treatment effect may mask important differences across groups.

# Task 3: Pick a small set of covariates to match on 
covariates = [
    'recency',
    'history',
    'newbie']

# Task 3: For each customer with got_email = 1, find one customer with got_email = 0whose covariate values are closest. 
covariates = [
    'recency',
    'history',
    'newbie']
treated = df[df['got_email'] == 1].copy()
control = df[df['got_email'] == 0].copy()
scaler = StandardScaler()
treated_scaled = scaler.fit_transform(treated[covariates])
control_scaled = scaler.transform(control[covariates])
nn = NearestNeighbors(n_neighbors=1)
nn.fit(control_scaled)
distances, indices = nn.kneighbors(treated_scaled)
matched_controls = control.iloc[indices.flatten()].copy()
print(len(treated))
print(len(matched_controls))

# Task 3: Compute the average difference in conversion across matched pairs — call this βˆmatch.
treated_conversion = treated['conversion'].reset_index(drop=True)
matched_conversion = matched_controls['conversion'].reset_index(drop=True)
beta_match = (
    treated_conversion -
    matched_conversion).mean()
print(beta_match)

# Task 3: Compare βˆmatch to βˆ from Task 1. Did matching recover the experimental answer? If not, what assumption is failing?
# The experimental estimate from Task 1 was β̂ = 0.00495, while the matching estimate was β̂match = -0.00403. 
# Since the estimates differ substantially, matching did not recover the experimental result. 
# This suggests that matching on recency, history, and newbie status may not fully account for all factors related to treatment assignment and conversion.



