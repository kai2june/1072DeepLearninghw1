import scipy.io
from matplotlib import pyplot as plt
import pandas as pd
mat = scipy.io.loadmat('data.mat')

# randomly picked initial value
theta0 = 2
theta1 = -1
theta2 = 4
lr = 0.0001
iteration = 10000


# record theta0, theta1, theta2 initial value for plotting
theta0_history = [theta0]
theta1_history = [theta1]
theta2_history = [theta2]

# iteration
for i in range(iteration):
    theta0_gradient = 0.0
    theta1_gradient = 0.0
    theta2_gradient = 0.0

    for n in range(len(mat['x'])):
        if (mat['y'][n][0] - (theta0 + mat['x'][n][0] * theta1 + mat['x'][n][0] * mat['x'][n][0] * theta2)) > 0:
            theta0_gradient -= 1
            theta1_gradient -= mat['x'][n][0]
            theta2_gradient -= mat['x'][n][0]*mat['x'][n][0]
        elif (mat['y'][n][0] - (theta0 + mat['x'][n][0] * theta1 + mat['x'][n][0] * mat['x'][n][0] * theta2)) < 0:
            theta0_gradient += 1
            theta1_gradient += mat['x'][n][0]
            theta2_gradient += mat['x'][n][0]*mat['x'][n][0]
        else:
            pass

    # update theta0, theta1, theta2
    theta0 -= lr*theta0_gradient
    theta1 -= lr*theta1_gradient
    theta2 -= lr*theta2_gradient

    # record theta0, theta1, theta2 for plotting
    theta0_history.append(theta0)
    theta1_history.append(theta1)
    theta2_history.append(theta2)

print(theta0, theta1, theta2)

# compute least_square_parabola_error
average_error = 0.0
for i in range(len(mat['x'])):
    average_error += abs(mat['y'][i][0] - (theta0 + mat['x'][i][0]*theta1 + mat['x'][i][0]*mat['x'][i][0]*theta2))
print("average_error: ")
print(average_error)

# plot
plt.plot(mat['x'], mat['y'], 'o', ms=2, alpha=0.5, label='data')
plt.plot(mat['x'], theta0 + mat['x']*theta1 + mat['x']*mat['x']*theta2, 'r', ms=1, alpha=0.8, label='curve')
plt.legend(loc='best')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Q4 fig, average_error: %f, \n theta0: %f, theta1: %f, theta2: %f"%(average_error, theta0, theta1, theta2))

plt.savefig("./picture/hw1q4.png")
plt.grid()
plt.show()
