#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[44]:


gl=pd.read_csv("glass.csv")
gl.head()


# In[45]:


# make sure missing values in the dataset
gl.isna().sum()


# In[46]:


# type
gl["Type"].value_counts()


# In[47]:


plt.style.use('seaborn')
fig, ax = plt.subplots(figsize=(10,8))
plt.pie(x=gl['Type'].value_counts(), 
        labels=['Type 2', 'Type 1', 'Type 7', 'Type 3', 'Type 5', 'Type 6'],
        autopct="%1.2f%%", 
        )
plt.title("Glass Types Distribution",fontsize=15)
plt.show()


# In[48]:


#split data
X=gl.drop(columns=['Type'])
y=gl['Type']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[49]:


def classify(model,x_train,y_train,x_test,y_test):
    from sklearn.metrics import classification_report
    target_names = ['Type 2', 'Type 1', 'Type 7', 'Type 3', 'Type 5', 'Type 6']
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))


# In[51]:


#knn
 np.random.seed(42)
from sklearn.neighbors import KNeighborsClassifier
trainscore = []
testscore=[]
np.random.seed(42)
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
knn2 = KNeighborsClassifier(n_neighbors =11 )
classify(knn2,x_train,y_train,x_test,y_test)


# In[52]:


#SVM
 np.random.seed(22)
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
svcresult
svmt=SVC(kernel='linear',random_state=1,gamma=0.6)
svmt.fit(x_train,y_train)
classify(svmt,x_train,y_train,x_test,y_test)


# In[53]:


# Decision tree
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
dtacc=dt.score(x_test,y_test)
dtacc
classify(dt,x_train,y_train,x_test,y_test)


# In[55]:


# max_leaf_nodes(prune tree)
 np.random.seed(50)
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


# In[56]:


n=dt_error(20,x_train,y_train,x_test,y_test)


# In[57]:


dt1=DecisionTreeClassifier(criterion="entropy",splitter="best",
                            max_leaf_nodes=17)
dt1.fit(x_train,y_train)
dtacc=dt1.score(x_test,y_test)
print(dtacc)
classify(dt1,x_train,y_train,x_test,y_test)


# In[58]:


# boosting
from sklearn.ensemble import GradientBoostingClassifier
gbk = GradientBoostingClassifier(
                            max_leaf_nodes=17)
gbk.fit(x_train, y_train)
print(gbk.score(x_test,y_test))
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


#Neural network
X = dataset.drop('Type', axis = 1).values
y = dataset['Type'].values.reshape(-1,1)


# In[ ]:


# Feature Scaling
sc_X = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.fit_transform(X_test)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)


# In[ ]:


from keras.utils import to_categorical
def encode(data):
    print('Shape of data (BEFORE encode): %s' % str(data.shape))
    encoded = to_categorical(data)
    print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))
    return encoded


# In[ ]:


y_train_encoded = encode(y_train)
y_test_encoded = encode(y_test)
y_train_encoded = np.delete(y_train_encoded, [0,4], axis = 1)
y_test_encoded = np.delete(y_test_encoded, [0,4], axis = 1)
print(y_train_encoded[2])
print(y_test_encoded[2])


# In[ ]:


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the second hidden layer
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
history = classifier.fit(X_train_scaled, y_train_encoded, validation_data=(X_test_scaled, y_test_encoded), batch_size = 100, epochs = 1150)


# In[ ]:


f, axes = plt.subplots(1,2,figsize=(14,4))
axes[0].plot(history.history['loss'])
axes[0].plot(history.history['val_loss'])
axes[0].set_xlabel('Epoch', fontsize=14)
axes[0].set_ylabel('Loss', fontsize=14)
axes[0].yaxis.tick_left()
axes[0].legend(['Train', 'Test'], loc='upper left')

axes[1].plot(history.history['accuracy'])
axes[1].plot(history.history['val_accuracy'])
axes[1].set_xlabel('Epoch', fontsize=14)
axes[1].set_ylabel('Accuracy', fontsize=14)
axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()
axes[1].legend(['Train', 'Test'], loc='upper left')

plt.show()


# In[ ]:


print("Training set: ", history.history.get('accuracy')[-1])
print("Test set: ", history.history.get('val_accuracy')[-1])

