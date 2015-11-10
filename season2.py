import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
warnings.simplefilter(action = "ignore", category = FutureWarning)

import os
os.chdir('/home/bolaka/python-workspace/CVX-timelines/')

import pandas as pd
import numpy as np
import matplotlib as plt

# turn off pandas warning on data frame operations
pd.options.mode.chained_assignment = None  # default='warn'

# Load the data - titanic data from Kaggle
df = pd.read_csv('/home/bolaka/datasets/titanic/train.csv', header=0)        # Load the train file into a dataframe

## first look...
#df.head(10)
#
## five number summary of numericals
#df.describe()
#
## Data Exploration
## Numerical variable analysis
## histograms
## Age is normal - has missing
#fig = plt.pyplot.figure()
#ax = fig.add_subplot(111)
#ax.hist(df['Age'], bins = 10, range = (df['Age'].min(),df['Age'].max()))
#plt.pyplot.title('Age distribution')
#plt.pyplot.xlabel('Age')
#plt.pyplot.ylabel('Count of Passengers')
#plt.pyplot.show()
#
## Fare is skewed - has outliers
#fig = plt.pyplot.figure()
#ax = fig.add_subplot(111)
#ax.hist(df['Fare'], bins = 10, range = (df['Fare'].min(),df['Fare'].max()))
#plt.pyplot.title('Fare distribution')
#plt.pyplot.xlabel('Fare')
#plt.pyplot.ylabel('Count of Passengers')
#plt.pyplot.show()
#
## boxplots
## Fare by itself
#df.boxplot(column='Fare')
#
## Fare by class - class 1 actually has outliers
#df.boxplot(column='Fare', by='Pclass')
#
## Categorical variable analysis
## Bar Charts
## Counts of passengers w.r.t Class
#temp1 = df.groupby('Pclass').Survived.count()
#fig = plt.pyplot.figure(figsize=(8,4))
#ax1 = fig.add_subplot(121)
#ax1.set_xlabel('Pclass')
#ax1.set_ylabel('Count of Passengers')
#ax1.set_title("Passengers by Pclass")
#temp1.plot(kind='bar')
#
## Survivality of passengers w.r.t Class
#temp2 = df.groupby('Pclass').Survived.sum()/df.groupby('Pclass').Survived.count()
#ax2 = fig.add_subplot(122)
#temp2.plot(kind = 'bar')
#ax2.set_xlabel('Pclass')
#ax2.set_ylabel('Probability of Survival')
#ax2.set_title("Probability of survival by class")
#
## Counts of passengers w.r.t Sex
#temp3 = df.groupby('Sex').Survived.count()
#fig = plt.pyplot.figure(figsize=(8,4))
#ax1 = fig.add_subplot(121)
#ax1.set_xlabel('Sex')
#ax1.set_ylabel('Count of Passengers')
#ax1.set_title("Passengers by Sex")
#temp3.plot(kind='bar')
#
## Survivality of passengers w.r.t Sex
#temp4 = df.groupby('Sex').Survived.sum()/df.groupby('Sex').Survived.count()
#ax2 = fig.add_subplot(122)
#temp4.plot(kind = 'bar')
#ax2.set_xlabel('Sex')
#ax2.set_ylabel('Probability of Survival')
#ax2.set_title("Probability of survival by Sex")
#
## Counts of passengers w.r.t Embarked
#temp3 = df.groupby('Embarked').Survived.count()
#fig = plt.pyplot.figure(figsize=(8,4))
#ax1 = fig.add_subplot(121)
#ax1.set_xlabel('Embarked')
#ax1.set_ylabel('Count of Passengers')
#ax1.set_title("Passengers by Embarked")
#temp3.plot(kind='bar')
#
## Survivality of passengers w.r.t Embarked
#temp4 = df.groupby('Embarked').Survived.sum()/df.groupby('Embarked').Survived.count()
#ax2 = fig.add_subplot(122)
#temp4.plot(kind = 'bar')
#ax2.set_xlabel('Embarked')
#ax2.set_ylabel('Probability of Survival')
#ax2.set_title("Probability of survival by Embarked")
#
## Stacked Bar Charts
## basic classification algorithm based on Sex and Class
#temp5 = pd.crosstab([df.Sex, df.Pclass], df.Survived.astype(bool))
#temp5.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
#
## basic classification algorithm based on Sex, Class, Embarked
#temp6 = pd.crosstab([df.Sex, df.Pclass, df.Embarked], df.Survived.astype(bool))
#temp6.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)

# Data Munging
# count missing in Cabin
missingCabins = sum(df['Cabin'].isnull()) # # 687 missing

# drop Cabin & Ticket
df = df.drop(['Ticket','Cabin'], axis=1) 

# count missing in Age
missingAges = ( df.PassengerId.count() - df.Age.count() ) / df.PassengerId.count() 

## fill missing in Age by mean - simplest approach!
#meanAge = np.mean(df.Age)
#df.Age = df.Age.fillna(meanAge)

# The other extreme could be to build a supervised learning model to predict age 
# on the basis of other variables and then use age along with other variables to 
# predict survival - yes i have tried to go down this path many times but unless
# the features well relates to others, this will not be effective!

def name_extract(word):
    return word.split(',')[1].split('.')[0].strip()

df2 = pd.DataFrame({'Salutation':df['Name'].apply(name_extract)})

df = pd.merge(df, df2, left_index = True, right_index = True) # merges on index
temp1 = df.groupby('Salutation').PassengerId.count()

def group_salutation(old_salutation):
    if old_salutation == 'Mr':
        return('Mr')
    elif old_salutation == 'Mrs':
        return('Mrs')
    elif old_salutation == 'Master':
        return('Master')
    elif old_salutation == 'Miss':
        return('Miss')
    else:
        return('Others')

df3 = pd.DataFrame({'New_Salutation':df['Salutation'].apply(group_salutation)})
df = pd.merge(df, df3, left_index = True, right_index = True)
temp1 = df3.groupby('New_Salutation').count()
#df.boxplot(column='Age', by = 'New_Salutation')
#df.boxplot(column='Age', by = [ 'Sex', 'Pclass' ])

table = df.pivot_table(values='Age', index=['New_Salutation'], columns=['Pclass', 'Sex'], aggfunc=np.median)

def fage(x):
    return table[x['Pclass']][x['Sex']][x['New_Salutation']]

df['Age'].fillna(df[df['Age'].isnull()].apply(fage, axis=1), inplace=True)

numeric_variables = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
y = df.pop("Survived")
X = df[numeric_variables]

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Gradient Boosting", "Naive Bayes", "LDA", "QDA"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
#    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    RandomForestClassifier(n_estimators = 100 , random_state = 42),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LDA(),
    QDA()]

# let's find which classifier has best cross validation score
classifier_scores = []
for name, classifier in zip(names, classifiers):
    print('\nCross Validating', name)
    cv = cross_validation.KFold(len(X), 10)
    
    fold = 0
    accuracies = []    
    for trainInd, testInd in cv:
        trainingX = X.loc[trainInd]
        trainingY = y[trainInd]
        testingX = X.loc[testInd]
        testingY = y[testInd]
        
        model = classifier.fit(trainingX, trainingY)
        score_train = round(model.score(trainingX, trainingY), 3)*100 
        score_test = round(model.score(testingX, testingY), 3)*100 
        print((fold + 1), 'training accuracy :', score_train, 'testing accuracy :', score_test)
        accuracies.append(score_test)
        fold += 1
    best = np.array(accuracies).mean()
    print('Results: ', str( best )) 
    classifier_scores.append(best)
    
    model = classifier.fit(X, y)
    accuracy = round(model.score(X, y), 3)*100
    y_hat = model.predict(X)
    
    print('\nTraining accuracy of', name, '=', accuracy) # np.mean(y == y_hat)
    
    print('Confusion matrix :')
    cm = confusion_matrix( y, y_hat )
    class_names = [str(x) for x in np.unique(y)]
    norm_cm = []
    for i in cm:
        a = 0
        tmp_arr = []
        a = sum(i,0)
        for j in i:
            tmp_arr.append(float(j)/float(a) * 100)
        norm_cm.append(tmp_arr)        
    
    fig = plt.pyplot.figure()
    ax = fig.add_subplot(111)
    #cax = ax.matshow(cm)
    cax = ax.imshow(norm_cm, interpolation='nearest')
    for i, cas in enumerate(norm_cm):
        for j, c in enumerate(cas):
            if c>0:
                plt.pyplot.text(j-.2, i+.2, "%.1f" % c, fontsize=14)
    
    fig.colorbar(cax)
    ax.set_xticklabels([''] + class_names)
    ax.set_yticklabels([''] + class_names)
    plt.pyplot.xlabel('Predicted')
    plt.pyplot.ylabel('Actual')
    plt.pyplot.show()    
    
    
ind = np.argmax(classifier_scores)
print("\n\nBest Classifier = ", names[ind])

