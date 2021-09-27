#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[24]:


hd=pd.read_csv("heart.csv")
hd.head()


# In[25]:


hd.target.value_counts()


# In[26]:


sns.countplot(x="target", data=hd)
plt.show()


# In[27]:


# create dummy variables
d1=pd.get_dummies(hd['cp'],prefix="cp")
d2=pd.get_dummies(hd['thal'],prefix="thal")
d3=pd.get_dummies(hd['slope'],prefix="slope")
t=[hd,d1,d2,d3]
hd=pd.concat(t,axis=1)
hd=hd.drop(columns=['cp','thal','slope'])
hd.head()


# In[28]:


y=hd['target']
y
x=hd.drop(columns=['target'])
# normalize data
X=(x-np.min(x))/(np.max(x)-np.min(x))
X


# In[9]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[8]:


def classify(model,x_train,y_train,x_test,y_test):
    from sklearn.metrics import classification_report
    target_names = ['target1', 'target0']
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))


# In[10]:


# knn
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train,y_train)
prediction= knn.predict(x_test)
knn.score(x_test,y_test)


# In[11]:


# try different k
trainscore = []
testscore=[]
np.random.seed(22)
for i in range(1,20):
    knn1 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k
    knn1.fit(x_train, y_train)
    trainscore.append(knn1.score(x_train, y_train))
    testscore.append(knn1.score(x_test, y_test))
plt.figure(figsize=(14,6))
p=sns.lineplot(range(1,20),trainscore,marker='o',label='train score')
p=sns.lineplot(range(1,20),testscore,marker='*',label='test score')
maxacc1=max(trainscore)
maxacc2=max(testscore)
print(maxacc2)
print(testscore)
knn2 = KNeighborsClassifier(n_neighbors =5 )
classify(knn2,x_train,y_train,x_test,y_test)


# In[12]:


#SVM
np.random.seed(92)
from sklearn.svm import SVC
svcresult=[]
svm=SVC(kernel='linear',random_state=1)
svm.fit(x_train,y_train)
svmacc=svm.score(x_test,y_test)
svmacc
svcresult.append(svmacc)
svm1=SVC(kernel='poly',random_state=1)
svm1.fit(x_train,y_train)
svmacc=svm1.score(x_test,y_test)
svmacc
svcresult.append(svmacc)
svm=SVC(kernel='rbf',random_state=1)
svm.fit(x_train,y_train)
svmacc=svm.score(x_test,y_test)
svmacc
svcresult.append(svmacc)
svm=SVC(kernel='sigmoid',random_state=1)
svm.fit(x_train,y_train)
svmacc=svm.score(x_test,y_test)
svmacc
svcresult.append(svmacc)
svcresult
x1=['linear','poly','rbf','sigmoid']
plt.scatter(x1,svcresult,c="#ff1212",marker='*')
plt.xlabel('Kernel function')
plt.ylabel('accuracy')
for x,y in zip(x1,svcresult):
    plt.text(x,y+0.0003,str(y),fontsize=10)
plt.show()
svmt=SVC(kernel='rbf',random_state=1)
svmt.fit(x_train,y_train)
classify(svmt,x_train,y_train,x_test,y_test)


# In[13]:


# Decision tree
np.random.seed(51)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
dtacc=dt.score(x_test,y_test)
dtacc
classify(dt,x_train,y_train,x_test,y_test)


# In[15]:


# max_leaf_nodes
from sklearn.metrics import mean_squared_error
def plot_learning_curve(x_train,x_test,y_train,y_test):
    trainscore=[]
    testscore=[]
    for i in range(1,len(x_train)+1):
        algo=DecisionTreeClassifier()
        algo.fit(x_train[:i],y_train[:i])
        yp=algo.predict(x_train[:i])
        trainscore.append(mean_squared_error(y_train[:i]))
np.random.seed(123)

def dt_error(n,x_train,y_train,x_test,y_test):
    nodes = range(2, n)
    error_rate = []
    for k in nodes:
        model = DecisionTreeClassifier(criterion="entropy",splitter="best",max_leaf_nodes=k)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        error_rate.append(np.mean(y_pred != y_test))
    kloc = error_rate.index(min(error_rate))
    print("Lowest error is %s occurs at n=%s." % (error_rate[kloc], nodes[kloc]))
    plt.plot(nodes, error_rate, color='blue', linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=10)
    plt.xlabel('Tree Size')
    plt.ylabel('Cross-Validated MSE')
    plt.show()
    return nodes[kloc]


# In[16]:


np.random.seed(125)
n=dt_error(20,x_train,y_train,x_test,y_test)


# In[17]:



dt1=DecisionTreeClassifier(criterion="entropy",splitter="best",max_depth=8,
                            max_leaf_nodes=6)
dt1.fit(x_train,y_train)
dtacc=dt1.score(x_test,y_test)
print(dtacc)
classify(dt1,x_train,y_train,x_test,y_test)


# In[18]:


# boosting
np.random.seed(256)
from sklearn.ensemble import GradientBoostingClassifier
gbk = GradientBoostingClassifier(max_leaf_nodes=6)
gbk.fit(x_train, y_train)
gbk.score(x_test,y_test)
classify(gbk,x_train,y_train,x_test,y_test)


# In[ ]:


#draw the learning curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ =         learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")
  
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")
    return plt
fig, axes = plt.subplots(3,2, figsize=(10, 15))
title = "Learning Curves (SVC)"
plot_learning_curve(SVC(kernel='rbf'), title, X, y, axes=axes[:, 1], ylim=(0.4, 1.01),
                    cv=3, n_jobs=4)
plt.show()
# title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# # SVC is more expensive so we do a lower number of CV iterations:
# cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
# estimator = SVC(gamma=0.001)
# plot_learning_curve(estimator, title, X, y, axes=axes[:, 1], ylim=(0.5, 1.01),
#                     cv=cv, n_jobs=4)


# In[ ]:


X = np.array(data.drop(['target'], 1))
y = np.array(data['target'])
mean = X.mean(axis=0)
X -= mean
std = X.std(axis=0)
X /= std


# In[ ]:


from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, stratify=y, random_state=42, test_size = 0.2)


# In[ ]:


from keras.utils.np_utils import to_categorical

Y_train = to_categorical(y_train, num_classes=None)
Y_test = to_categorical(y_test, num_classes=None)
print (Y_train.shape)
print (Y_train[:10])


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout
from keras import regularizers

# define a function to build the keras model
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(16, input_dim=13, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(8, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation='softmax'))
    
    # compile model
    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

model = create_model()

print(model.summary())


# In[ ]:


history=model.fit(X_train, Y_train, validation_data=(X_test, Y_test),epochs=50, batch_size=10)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# In[ ]:


# Model Losss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# In[ ]:


Y_train_binary = y_train.copy()
Y_test_binary = y_test.copy()

Y_train_binary[Y_train_binary > 0] = 1
Y_test_binary[Y_test_binary > 0] = 1

print(Y_train_binary[:20])


# In[ ]:


def create_binary_model():
    # create model
    model = Sequential()
    model.add(Dense(16, input_dim=13, kernel_initializer='normal',  kernel_regularizer=regularizers.l2(0.001),activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(8, kernel_initializer='normal',  kernel_regularizer=regularizers.l2(0.001),activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    adam = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

binary_model = create_binary_model()

print(binary_model.summary())


# In[ ]:


history=binary_model.fit(X_train, Y_train_binary, validation_data=(X_test, Y_test_binary)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

