import scipy.io
import numpy as np
from numpy import ones
from numpy.linalg import inv
from matplotlib import pyplot as plt

mat = scipy.io.loadmat('data.mat')

# At*A*x = At*b, x= [theta0, theta1] is the least square solution to ||Ax-b||
A = np.concatenate((ones([len(mat['x']),1]), mat['x'], mat['x']*mat['x']), axis=1)
At = A.transpose()
At_dot_A = np.dot(At, A)
inverse_At_dot_A = inv(At_dot_A)
At_dot_b = np.dot(At, mat['y'])
x = np.dot(inverse_At_dot_A, At_dot_b)
theta0 = x[0][0]
theta1 = x[1][0]
theta2 = x[2][0]
print(theta0, theta1,theta2)

# compute least_square_parabola_error
least_square_parabola_error = 0
for i in range(len(mat['x'])):
    least_square_parabola_error += pow(abs(mat['y'][i][0] - (theta0 + mat['x'][i][0]*theta1 + mat['x'][i][0]*mat['x'][i][0]*theta2)), 2)
print("least_square_parabola_error: ")
print(least_square_parabola_error)

# y = theta0 + x*theta1 + x^2*theta2, let's plot
plt.plot(mat['x'], mat['y'], 'o', ms=2, alpha=0.5, label='data')
plt.plot(mat['x'], theta0 + theta1*mat['x'] + theta2*mat['x']*mat['x'], 'r', ms=1, alpha=0.8, label = 'polynomial2')

plt.legend(loc='best')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Q3 fig, least_square_parabola_error: %f, \n theta0: %f,  theta1: %f,  theta2: %f"%(least_square_parabola_error, theta0, theta1, theta2))

plt.savefig("./picture/hw1q3.png")
plt.grid()
plt.show()