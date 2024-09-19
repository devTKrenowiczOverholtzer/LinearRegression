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
#print(f'Sxy:{Sxy}\nSxx:{Sxx}') 

# Get Line Variables 
m = Sxy/Sxx
b = y_mean - m*x_mean
#print(f'Regression Equation: y={m}x+{b}')

# anytime get a new alcohol quanity we are going to multiply it by .313 and add 2.58 to arrive at the wine quality score
# predict a variable 
row = 5
# try to predict wine quality from alcohol value 
# alcohol value for this row is wine dataframe locate row 5, here is the alcohol score
alcohol_val = wine_df.loc[5, 'alcohol']
# actual quality value is going to be row 5s quality value (zero index so techniqually its actually row 6)
quality_val = wine_df.loc[5, 'quality']
# predictive quality using model , is our alcohol value times m plus b (plugging directly into regression equation 
predicted_quality = alcohol_val*m+b

#print(f'for alcohol value {alcohol_val} the predictive quality was {predicted_quality} and the actual value was {quality_val}')
# not bad , wide range of values that can go from 0-10 not way off only ~.3 off 
# first glance this model looks pretty decent 

# want to check coefficent of determination next 
# create predictions for all our variables now 
# take dataframe add in these brackets this new column that we are calling predictions 
# to create this new column take the entire alcohol column multiply it by our m value and add our b value 
# taking alcohol column and plugging it into our regression equation to create this new prediction column
wine_df["predicitions"] = wine_df['alcohol']*m+b

# create residual squares 
# take quality column which is our ground truth?
# subtract predictions column to power of two 
# ground truth minus what we actually predicted
# square that so they are all positive , take the sum of those 
# residual squares % of variation thats explainable by our model 
# looking at variation becuase we are taking the actual values minus our model predictions ,squaring those and summing them
residual_squares = (wine_df["quality"]-wine_df["predicitions"]).pow(2).sum()

# looking at total variation 
# taking actual scores minus the mean of all the scores (how much everything deviated from the mean) and its squaring that and summing it  
total_squares = (wine_df["quality"]- y_mean).pow(2).sum()

#take residual squares as a percentage er as a proportion subtract it from one and thats going to give us r^2 
r_squared = 1 - (residual_squares/total_squares)
#print(f"Coefficient of Determination: {r_squared}")  
# .189 pretty terrible basically 20% of variation in the actual y variables in comparision to our regression line is explained by our model 
# pretty low percentage so our model isnt that useful, not that good  
# even though for specific example above it looked alright 
# important to check r_squared. went really well on one data point of checking but overall this model doesn't predict most ys very well 

# Run model with scikit learn
# create linear regression model called linearregressionmodel and set that equal to linear regression class
# creating a sklearn linear regression model by invoking that import sklearn linear model
linear_regression_model = LinearRegression()

# Expects a list of x variables since it can handle multiple regression 
# Bracketing , passing in all variables as a list
# take the liner regression model and call the function fit on it
# what we are going to fit is the wine dataframe 
# creating an inner list here for alcohol 
# and winedataframe creating a list of quality 
linear_regression_model.fit(wine_df[["alcohol"]], wine_df[["quality"]])

# store the coefficients , in a structure 
# pull them out 
# coefficient coef_
# coefficient index [0][0]
# can handle multiple regressions so indexing is not straight foward to get this back out
# documention explain structure it is returning it in 
model_m = linear_regression_model.coef_[0][0]
model_b = linear_regression_model.intercept_[0]
# intercept private property of that and add zero

print ("Scikit Learn Regression Model")
print(f"Regression Equation: y = {model_m}x + {model_b}")

# Predict same value as before minus a rounding error at the end 
# predict alcohol value using scikit learn
predicted_model_quality = linear_regression_model.predict([[alcohol_val]])
# have to reindex it 
print(f'for alcohol value {alcohol_val} the predictive quality was {predicted_model_quality} and the actual value was {quality_val}')


# with the scikit model get the coefficieint of determination
# pass in all alcohol values, inputs to our linear regression model 
# pass in our quality as our values
score = linear_regression_model.score(wine_df[["alcohol"]], wine_df[["quality"]])
print(f"Coefficient of Determination: {score}")
