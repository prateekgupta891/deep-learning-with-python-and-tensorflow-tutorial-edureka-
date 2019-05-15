import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
dataset = pd.read_csv('D:/Online Course/Neural Network with Python/Social_Network_Ads.csv')
print(dataset.head(5))

X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,4].values
print(X[0:5])

from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)

from sklearn.preprocessing import StandardScaler
sc  = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) #helps give same scale
print(X_train[0:5,])

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm  = confusion_matrix(Y_test,Y_pred)
print(cm)

#visualization
from matplotlib.colors import ListedColormap
X_set,Y_set = X_train, Y_train
x1,x2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step = 0.01),np.arange(start = X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step = 0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha =0.5,cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(Y_set)):
            plt.scatter(X_set[Y_set==j,0],X_set[Y_set==j,1],
            c = ListedColormap(('red','green'))(i),label=j)
plt.title('LogisticRegression')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
    


