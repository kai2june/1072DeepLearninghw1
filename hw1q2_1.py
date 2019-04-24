import scipy.io
from matplotlib import pyplot as plt
mat = scipy.io.loadmat('data.mat')

plt.plot(mat['x'], mat['y'], "o", ms=2, alpha=0.5, label='data')
plt.legend(loc='best')
plt.xlabel("x")
plt.ylabel("y")
Title="Q2.1 fig"
plt.title(Title)

plt.savefig("./picture/hw1q2_1.png")
plt.grid()
plt.show()