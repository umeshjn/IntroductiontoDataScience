import numpy as np
import pandas
from ggplot import *
import matplotlib.pyplot as plt

"""
In this question, you need to:
1) implement the compute_cost() and gradient_descent() procedures
2) Select features (in the predictions procedure) and make predictions.

"""

def normalize_features(array):
   """
   Normalize the features in the data set.
   """
   array_normalized = (array-array.mean())/array.std()
   mu = array.mean()
   sigma = array.std()

   return array_normalized, mu, sigma

def compute_cost(features, values, theta):
    """
    Compute the cost function given a set of features / values, 
    and the values for our thetas.
    
    This can be the same code as the compute_cost function in the lesson #3 exercises,
    but feel free to implement your own.
    """
    m = len(values)
    total = np.square(np.dot(features,theta)-values).sum()
    cost = total/(2*m)
    return cost

def gradient_descent(features, values, theta, alpha, num_iterations):
    """
    Perform gradient descent given a data set with an arbitrary number of features.
    
    This can be the same gradient descent code as in the lesson #3 exercises,
    but feel free to implement your own.
    """
    
    m = len(values)
    cost_history = []

    for i in range(num_iterations):
        predicted_values = np.dot(features,theta)
        theta = theta - alpha/m * np.dot((predicted_values - values),features)
        cost = compute_cost(features,values,theta)
        cost_history.append(cost)
    return theta, pandas.Series(cost_history)

def predictions():
    '''
    The NYC turnstile data is stored in a pandas dataframe called weather_turnstile.
    Using the information stored in the dataframe, let's predict the ridership of
    the NYC subway using linear regression with gradient descent.
    
    You can download the complete turnstile weather dataframe here:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv    
    
    Your prediction should have a R^2 value of 0.20 or better.
    You need to experiment using various input features contained in the dataframe. 
    We recommend that you don't use the EXITSn_hourly feature as an input to the 
    linear model because we cannot use it as a predictor: we cannot use exits 
    counts as a way to predict entry counts. 
    
    Note: Due to the memory and CPU limitation of our Amazon EC2 instance, we will
    give you a random subet (~15%) of the data contained in 
    turnstile_data_master_with_weather.csv. You are encouraged to experiment with 
    this computer on your own computer, locally. 
    
    
    If you'd like to view a plot of your cost history, uncomment the call to 
    plot_cost_history below. The slowdown from plotting is significant, so if you 
    are timing out, the first thing to do is to comment out the plot command again.
    
    If you receive a "server has encountered an error" message, that means you are 
    hitting the 30-second limit that's placed on running your program. Try using a 
    smaller number for num_iterations if that's the case.
    
    If you are using your own algorithm/models, see if you can optimize your code so 
    that it runs faster.
    '''
    dataframe = pandas.read_csv(r'C:\NanoDegree-Udacity\NanoDegree-Udacity\Introduction to Data Science\turnstile_data_master_with_weather.csv')
    # Select Features (try different features!)
    #features = dataframe[['rain', 'precipi', 'Hour', 'meantempi']] - 0.461129068126
    #features = dataframe[['rain', 'precipi', 'Hour', 'maxtempi']] - 0.460162862585
    #features = dataframe[['rain', 'precipi', 'Hour', 'mintempi']] - 0.465709894864
    #features = dataframe[['rain', 'precipi', 'Hour', 'minpressurei']] - 0.470615558434
    #features = dataframe[['rain','Hour', 'mintempi']] # 0.474320984287
    #features = dataframe[['rain', 'meanwindspdi', 'Hour', 'maxpressurei']] - 0.465709730807
    # Add UNIT to features using dummy variables
   
    dataframe['weekday'] = pandas.DatetimeIndex(dataframe['DATEn']).weekday
    features = dataframe[['rain','Hour','weekday','mintempi']]
    dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')
    #dummy_units = pandas.get_dummies(dataframe['weekday'], prefix='weekday')
    features = features.join(dummy_units)
    
    # Values
    values = dataframe[['ENTRIESn_hourly']]
    m = len(values)

    features, mu, sigma = normalize_features(features)
    features['ones'] = np.ones(m) # Add a column of 1s (y intercept)
    
    # Convert features and values to numpy arrays
    features_array = np.array(features)
    values_array = np.array(values).flatten()

    # Set values for alpha, number of iterations.
    alpha = 0.99 # please feel free to change this value
    num_iterations = 50 # please feel free to change this value

    # Initialize theta, perform gradient descent
    theta_gradient_descent = np.zeros(len(features.columns))
    theta_gradient_descent, cost_history = gradient_descent(features_array, 
                                                            values_array, 
                                                            theta_gradient_descent, 
                                                            alpha, 
                                                            num_iterations)
    
    plot = None
    # -------------------------------------------------
    # Uncomment the next line to see your cost history
    # -------------------------------------------------
    #plot = plot_cost_history(alpha, cost_history)
    # 
    # Please note, there is a possibility that plotting
    # this in addition to your calculation will exceed 
    # the 30 second limit on the compute servers.
    
    predictions = np.dot(features_array, theta_gradient_descent)
    plot_residuals(dataframe, predictions)
    return predictions, plot


def plot_cost_history(alpha, cost_history):
   """This function is for viewing the plot of your cost history.
   You can run it by uncommenting this

       plot_cost_history(alpha, cost_history) 

   call in predictions.
   
   If you want to run this locally, you should print the return value
   from this function.
   """
   cost_df = pandas.DataFrame({
      'Cost_History': cost_history,
      'Iteration': range(len(cost_history))
   })
   return ggplot(cost_df, aes('Iteration', 'Cost_History')) + \
      geom_point() + ggtitle('Cost History for alpha = %.3f' % alpha )


def plot_residuals(turnstile_weather, predictions):
    '''
    Using the same methods that we used to plot a histogram of entries
    per hour for our data, why don't you make a histogram of the residuals
    (that is, the difference between the original hourly entry data and the predicted values).

    Based on this residual histogram, do you have any insight into how our model
    performed?  Reading a bit on this webpage might be useful:

    http://www.itl.nist.gov/div898/handbook/pri/section2/pri24.htm
    '''
    
    plt.figure()
    plt.axis([-15000,15000, 0, 20000])
    plt.xlabel('Residuals')
    plt.ylabel('Relative Frequency')
    plt.title('Residual Plot')
    (turnstile_weather['ENTRIESn_hourly'] - predictions).hist(bins=200)
    return plt