import matplotlib
import scipy.io
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

train = scipy.io.loadmat('train.mat')
test = scipy.io.loadmat('test.mat')
print(len(train['x1']))
X_train = np.concatenate((train['x1'], train['x2']), axis=1)
X_test = np.concatenate((test['x1'], test['x2']), axis=1)

logreg = LogisticRegression()
logreg.fit(X_train, train['y'])

x_min, x_max = train['x1'][:, 0].min() - 0.5, train['x1'][:, 0].max() + 0.5
y_min, y_max = train['x2'][:, 0].min() - 0.5, train['x2'][:, 0].max() + 0.5
h = 0.001

# determine the area x axis range, yaxis range to draw
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
print(xx, yy)

#  predict the area (xmin~ xmax ,  ymin~ymax) 's  background  color  by  'logreg' classifier, and then assign to Z
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4,3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# predict for test data
predict_testY = logreg.predict(X_test)

# count accuracy
correct_count = 0
for i in range(len(test['y'])):
    if test['y'][i][0] == predict_testY[i]:
        correct_count += 1
accuracy = correct_count/len(test['y'])

# plot x1,x2-y
colors=['red', 'blue']
plt.scatter(test['x1'], test['x2'], c=test['y'], cmap=matplotlib.colors.ListedColormap(colors))
# plt.plot(clf.predict(X_test), ms=2, label="line")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Q5 x1-x2 test.mat, accuracy: %f"%(accuracy))

plt.savefig('./picture/hw1q5.png')
plt.grid()
plt.show()

