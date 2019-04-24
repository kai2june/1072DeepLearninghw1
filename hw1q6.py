import matplotlib
import scipy.io
from math import exp
from matplotlib import pyplot as plt
import numpy as np

train = scipy.io.loadmat('train.mat')
test = scipy.io.loadmat('test.mat')

def sigmoid(z):
    return 1 / (1 + exp(-z))


### gradient descent
# initial value
theta0 = 0.0
theta1 = 0.0
theta2 = 0.0
lr = 0.0001
iteration = 10000

# iteration
for i in range(iteration):
    theta0_gradient = 0.0
    theta1_gradient = 0.0
    theta2_gradient = 0.0
    for n in range(len(train['x1'])):
        theta0_gradient += (train['y'][n] - sigmoid(theta0 + theta1 * train['x1'][n][0] + theta2 * train['x2'][n][0]))
        theta1_gradient += (train['y'][n] - sigmoid(theta1 * train['x1'][n][0] + theta2 * train['x2'][n][0])) * \
                           train['x1'][n]
        theta2_gradient += (train['y'][n] - sigmoid(theta1 * train['x1'][n][0] + theta2 * train['x2'][n][0])) * \
                           train['x2'][n]

    theta0 += lr * theta0_gradient
    theta1 += lr * theta1_gradient
    theta2 += lr * theta2_gradient

# accuracy
correct_count=0
c1=0
c2=0
for i in range(len(test['x1'])):
    if(theta0 + theta1*test['x1'][i] + theta2*test['x2'][i] > 0 ):
        c1+=1
        if test['y'][i] == 1:
            correct_count+=1
    elif(theta0 + theta1*test['x1'][i] + theta2*test['x2'][i] < 0):
        c2+=1
        if test['y'][i] == 0:
            correct_count+=1
print(c1, c2, correct_count)

print(theta0[0], theta1[0], theta2[0])

X1 = [4,7]
X2 = []
for i in range(len(X1)):
    X2.append( ((-1)*theta1*X1[i] - theta0) / theta2 )

# plot x1,x2-y
colors=['red', 'blue']
plt.scatter(test['x1'], test['x2'], c=test['y'], cmap=matplotlib.colors.ListedColormap(colors))
plt.plot(X1, X2, ms=2, label="line")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Q6 x1-x2 test.mat, accuracy: %f, \n theta0: %f, theta1: %f, theta2: %f" %(correct_count/len(test['y']), theta0, theta1, theta2))

plt.savefig('./picture/hw1q6.png')
plt.grid()
plt.show()