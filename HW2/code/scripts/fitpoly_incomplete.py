## fitpoly_incomplete.py
# Partial script to help get started in ISTA 421/521 Homework 2

# NOTE: In its released form, this script will NOT run
#       You will get a syntax error on line 92 because w has not been defined

import numpy as np
import matplotlib.pyplot as plt

# make plots interactive
plt.ion()

################################################################
## Load the Olympic data and extract the mens 100m data
male100 = np.asmatrix(np.genfromtxt('../data/olympics.csv', delimiter = ',', dtype = None))

x = male100[:, 0] # Olympic years
t = male100[:, 1] # Winning times

# The following rescales x for numerical reasons
# (In hw2 you are asked to fit 4th order polynomials, which involves
# taking the 4th power of the input x values; taking the 4th power
# of years gets large!)
x = x - x[0]  # in the Olympics data, x[0] is 1896
x = x/4

################################################################
# Plot just the data
plt.figure(1)
plt.scatter(np.asarray(x), np.asarray(t), edgecolor = 'b', color = 'w', marker = 'o')
plt.xlabel('Olympic number (note: the years have been scaled!)')
plt.ylabel('Winning time')
plt.title('Just the Olympic data')

################################################################
## Linear model

# Note: You will notice in the following code that there appear
# two be "two versions" of the input x values that we range
# over: (1) x itself (the input, such as the scaled olympic years) and
#       (2) the pair of variables plotx and plotX.
# plotx and plotX are an approximation of running "continuously"
# over the x-axis so that we can get a nice, smooth plot of
# our linear model; this will be particularly noticable when
# we plot higher-order polynomials

# The following two python lines create a large number of equally
# spaced points between just before the input values start and just
# after they end.  This gives us a set of points to use to plot our
# model so that it looks like a smooth curve.
# First,  because x (and t) is currently a column of a matrix object
# we cannot iterate over it, so the following python line 'flattens'
# x into an array we can iterate over.
flatx = np.asarray(x).flatten()
plotx = np.asmatrix(np.linspace(flatx[0]-2, flatx[-1]+2,
                                (flatx[-1]-flatx[0]+4)/0.01 + 1)).conj().transpose()

# The following code generalizes to nth-order polynomial linear models.
# The variable 'model_order' determines the order of the polynomial.
# The default value is set to 1 for a first-order polynomial, i.e., 
# a line embedded in 2-dimensions: y = w_0 + (w_1 * x^1)
model_order = 1 
X_columns = model_order + 1

# Create the second set of values that will be used as the x values for making
# a smooth plot of our model
plotX = np.asmatrix(np.zeros((plotx.shape[0], X_columns)))

# Create X (the 'design' matrix) of appropriate size for the input data
X = np.asmatrix(np.zeros((x.shape[0], X_columns)))

# The following code takes the x input and generates the higher-order
# polynomial values of x (x^0, x^1, x^2, ..., x^k).  If model_order = 1 (so
# that X_columns = 2), then this will only generate x^0 ad x^1 for a
# 1-dimensional input model
for k in range(X_columns):
    # fill in the X matrix with the input values, to the appropriate polynomial power
    X[:,k] = np.power(x, k)
    # fill in the plotX matrix with the finer-grained x-axis values,
    # to the appropriate polynomial power
    plotX[:,k] = np.power(plotx, k)

# Here's the key step!: create the matrix version of the normal equation(s)
# The following python functions will be useful:
#   np.linalg.inv(X) <-- returns the inverse of X
#   X.transpose()    <-- transposes X
#   X.conj()         <-- not strictly required in this case, but this makes your math 
#                        safe for taking the inverse in the case that the input values
#                        include complex numbers
#                        (you do this before transpose)

w = ### YOUR CODE HERE ###

################################################################
## Plot the data and model

# uncomment this line if you want to first close the first figure
# that displays only the data
# plt.close()

# create figure 2
plt.figure(2)
# plot the input (x,t) data
plt.scatter(np.asarray(x), np.asarray(t), edgecolor = 'b', color = 'w', marker = 'o')
# plot the linear model itself as a smooth line
plt.plot(plotx, plotX*w, color = 'r', linewidth = 2)
# add labels to your figure!
plt.xlabel('Olympic number (note: the years have been scaled!)')
plt.ylabel('Winning time')
plt.title('Olympic data with best fit model')

print 'Model paramters:'
i = 0
for param in w:
    print '  w[{0}] = {1}'.format(i,param)
    i += 1

# This last line may or may not be necessary to keep your matplotlib window open
raw_input('Press <ENTER> to quit...')



