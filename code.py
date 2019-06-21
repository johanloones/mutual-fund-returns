# --------------
# import libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# Code Starts here
#The path for the dataset had been stored in variable path. Load the dataframe from the 'path' using pd.read_csv() and 
#store the dataframe in a variable called 'data'.
data=pd.read_csv(path)

#Display the shape of the dataframe using shape method
data.shape

#Use describe() method to display the summary statistics of the dataframe
data.describe()

#Feature of Serial Number is just a representative number and hence is not useful in model building. 
#Drop the column of Serial Number from the dataframe.
data.drop(columns='Serial Number',inplace=True)
data.head()

# Code ends here




# --------------
#Importing header files
from scipy.stats import chi2_contingency
import scipy.stats as stats

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 11)   # Df = number of variable categories(in purpose) - 1

# Code Starts here
#Create a variable 'return_rating' which is the value counts of morningstar_return_rating
return_rating=data['morningstar_return_rating'].value_counts()

#Create a variable 'risk_raing' which is the value counts of morningstar_risk_rating
risk_rating=data['morningstar_risk_rating'].value_counts()

#Concat 'return_rating.transpose()' and 'risk_rating.transpose()' along axis=1 with keys= ['return','risk'] 
#and store it in a variable called 'observed'
observed=pd.concat([return_rating.transpose(),risk_rating.transpose()],axis=1,keys= ['return','risk'])
#print(return_rating,risk_raing,observed)

#Apply "chi2_contingency()" on 'observed' and store the result in variables named chi2, p, dof, ex respectively.
chi2, p, dof, ex=chi2_contingency(observed)
print(chi2, p, dof, ex)
#Compare chi2 with critical_value(given)
#If chi-squared statistic exceeds the critical_value, reject the null hypothesis that the both features are independent 
#from each other, else null hypothesis cannot be rejected.
if chi2 > critical_value:
  print('Reject Null Hypothesis that both Features are Independent')

# Code ends here


# --------------
# Code Starts here
#Use corr() method to calculate the correlation between all the features. Use abs() method to get only the absolute values 
#of the correlation. Store the values in dataframe correlation
correlation=abs(data.corr())

#Print correlation.
print(correlation)

#As you can observe we get a dataframe with correlation calculated for each pair, this dataframe needs some transformation 
#to extract the features with correlation greater than 0.75
#Use unstack() method on correlation and store the same in us_correlationUse sort_values() method to sort the correlation with 
#ascending=False parameter. Store the sorted series in us_correlation
us_correlation=correlation.unstack().sort_values(ascending=False)

#Apply a filter to us_correlation to extract the pairs with correlatio higher that 0.75 but less than 1. 
#Store the filtered values to max_correlated
max_correlated=us_correlation[us_correlation>0.75]

#You can observe that we have 4 pairs of features which have correlation higher than 0.75. Based on this information 
#drop the features of morningstar_rating, portfolio_stocks, category_12 and sharpe_ratio_3y from the data
data.drop(columns=['morningstar_rating','portfolio_stocks','category_12','sharpe_ratio_3y'],inplace=True)
print(data.head())
# Code ends here


# --------------
# Code Starts here
#Create two subplots with axes as 'ax_1', 'ax_2'.
fig, (ax_1, ax_2) =plt.subplots(nrows=1, ncols=2)

#Plot a boxplot of price_earning column using ax_1. Set axis title as price_earning.
ax_1.boxplot(data['price_earning'])
ax_1.set_title('price_earning')

#Plot a boxplot of net_annual_expenses_ratio column using ax_2. Set axis title as net_annual_expenses_ratio
ax_2.boxplot(data['net_annual_expenses_ratio'])
ax_2.set_title('net_annual_expenses_ratio')

# Code ends here


# --------------
# import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

# Code Starts here
#Store all the features(independent values) in a variable called X
X=data.drop(columns='bonds_aaa')

#Store the target variable bonds_aaa(dependent value) in a variable called y
y=data['bonds_aaa']

#Split the dataframe data into X_train,X_test,y_train,y_test using train_test_split() function. 
#Use test_size = 0.3 and random_state = 3
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size = 0.3,random_state = 3)

#Instantiate a linear regression model with LinearRegression() and save it to a variable called lr.
lr=LinearRegression()

#Fit the model on the training data X_train and y_train.
lr.fit(X_train, y_train)

#Make predictions on the X_test features and save the results in a variable called 'y_pred'.
y_pred=lr.predict(X_test)

#Calculate the root mean squared error and store the result in a variable called rmse.
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)
# Code ends here


# --------------
# import libraries
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge,Lasso

# regularization parameters for grid search
ridge_lambdas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60]
lasso_lambdas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1]

# Code Starts here
#Instantiate a model with Ridge() with and save it to a variable called ridge_model.
ridge_model=Ridge()

#Apply GridSearchCV as GridSearchCV(estimator=ridge_model, param_grid=dict(alpha=ridge_lambdas)) and store it in variable ridge_grid.
ridge_grid=GridSearchCV(estimator=ridge_model, param_grid=dict(alpha=ridge_lambdas))

#Fit the ridge_grid on the training data X_train and y_train.
ridge_grid.fit(X_train, y_train)

#Calculate the rmse score for the above model and store the value in ridge_rmse
ridge_rmse=np.sqrt(mean_squared_error(y_test,ridge_grid.predict(X_test)))

#Instantiate a model with lasso() with and save it to a variable called lasso_model.
lasso_model=Lasso()

#Apply GridSearchCV as GridSearchCV(estimator=lasso_model, param_grid=dict(alpha=lasso_lambdas)) and store it in variable lasso_grid.
lasso_grid=GridSearchCV(estimator=lasso_model, param_grid=dict(alpha=lasso_lambdas))

#Fit the lasso_grid on the training data X_train and y_train.
lasso_grid.fit(X_train, y_train)

#Calculate the rmse score for above model and store the value in lasso_rmse
lasso_rmse=np.sqrt(mean_squared_error(y_test,lasso_grid.predict(X_test)))
# Code ends here


