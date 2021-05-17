
import math
import numpy as np


def get_delta_decent(X,y,theta,j):
    decent = 0
    m=len(y)
    for i in range(m):
        xi=X[i]
        xij=xi[j]
        hi=hypothesis(theta,xi)
        decent += ((hi-y[i])*xij)

    return decent/m


def gradient_decent(X,y,theta,alpha):
    theta_t = []
    for i in range(len(theta)):
        deivative = alpha*get_delta_decent(X,y,theta,i)
        theta_t_i=theta[i]-deivative
        theta_t.append(theta_t_i)
    return theta_t


def get_cost_function(X,y,theta):
    cost_sum=0.0
    m=len(y)
    for i in range(m):
        xi=X[i]
        hi=hypothesis(theta,xi)

        if y[i] == 1:
            cost=y[i]*math.log(hi)

        elif y[i] == 0:
            cost=(1-y[i])*math.log(1-hi)

        cost_sum = cost_sum + cost

    return -1/m * cost_sum


def Logistic_Regression(X,y,alpha,theta,epochs):
    epochs = int(epochs)
    decent_cost_function = []
    for i in range(epochs):
        theta_t=gradient_decent(X,y,theta,alpha)
        theta=theta_t
        # calculate cost for every 100 steps
        if i % 100 == 0:
            cost_function=get_cost_function(X,y,theta)
            decent_cost_function.append(cost_function)

    return theta,decent_cost_function


#predict model and return the list that the probability tobe 1
def Logistic_Regression_Predict(model,X):
    y_prob = []
    for i in range(len(X)):
        probability = hypothesis(model,X[i])
        y_prob.append(probability)
    return y_prob


def sigmoid(z):
    return 1/(float(1+np.exp(-1*z)))

def hypothesis(theta,xi):
    z = 0
    for i in range(len(theta)):
        z = z + xi[i]*theta[i]
    return sigmoid(z)





