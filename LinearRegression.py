import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# had to install scikit learn package to use sklearn 
# --break-system-packages (external enviorment) bipass 
# correct way to use Virtualized Enviorment and install packages in there


# Load in dataset 
# Make wine dataframe which is the read csv file , has weird semicolon seperators specify with sep
wine_df = pd.read_csv('winequality-white.csv', sep =';')

# Make linear regression for Alcohol and WineQuality
# create scatterplot using our dataframe as x variable , wine quality as our y variable 
## plt.scatter(wine_df['alcohol'], wine_df['quality']) 

# make pretty with axis labels 
## plt.xlabel('Alcohol')
## plt.ylabel('WineQuality')
## plt.title('Alcohol VS Quality')

# show plot 
## plt.show()

# Manual Linear Regression between Alcohol and Quality 
# Get our variables 
# How many rows we have in our dataframe
# n variable is how many rows in our dataframe (need to use in a lot of sumations)
# get length of winedataframe index 
n =  len(wine_df.index)
#print(n)

# Summary statistics needed are sum of all x variables
# X variable is alcohol  
# what we are going to plug in to get wine quality prediction
# grab that column of the dataframe and just use the sum function/property on that
sum_x = wine_df['alcohol'].sum()

# same process as x 
sum_y = wine_df['quality'].sum()

# print those 
# fstrings another way to make strings, if you do f and a " then you can put variables in brackets inside this string , can be handy to not use multiple print statements
# print(f"{sum_x}{sum_y}")

# Mean of X variables
# taking a straight forward mean: sum of x divided by number of rows
# also a pandas way to do that to
x_mean = sum_x/n
# Mean of Y variables
y_mean = sum_y/n

#print(f"{x_mean}{y_mean}")

# Summary Statistic Shortcut function 
# sum of the x's multiplied by the ys 
# pandas will take care of the dataframe variable multiplication under the hood
# sum of that 
# create a new series in pandas so we can take the sum over that series (handy)
sum_x_times_y = (wine_df['alcohol']*wine_df['quality']).sum() 

# get alcohol variable of wine dataframe take it to the power of two pow(2)
# take sum of that 
# pandas libraries you can chain .pow(2).sum() 
sum_x_squared = wine_df['alcohol'].pow(2).sum()

#print(f'{sum_x_times_y}\n{sum_x_squared}')

# Calculate Numerator and Denominator for m
# Variable Sxy going to be sum_x_times_y minus sumx times sum y divided by n
Sxy = sum_x_times_y - (sum_x*sum_y)/n 
# Variable Sxx is going to be  sum x squared minus sumx times sum x divided by n
Sxx = sum_x_squared - (sum_x*sum_x)/n

print(f'Sxy:{Sxy}\nSxx:{Sxx}') 
