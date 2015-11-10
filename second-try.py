# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 16:33:05 2015

@author: bolaka
"""

""" Writing my first code.
Author : Bolaka Mukherjee
Date : 2nd June 2015
please see packages.python.org/milk/randomforests.html for more

""" 

import os
os.chdir('/home/bolaka/python-workspace/CVX-timelines/')

# imports
from cvxtextproject import *
from mlclassificationlibs import *

setPath('/home/bolaka/datasets/titanic')
import pandas as pd

idCol = 'PassengerId'

# Data cleanup
# TRAIN DATA
train_df = pd.read_csv('train.csv', header=0, index_col=idCol)        # Load the train file into a dataframe
# I need to convert all strings to integer classifiers.
# I need to fill in the missing values of the data and make it complete.

# female = 0, Male = 1
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

## Process Cabin
#train_df['cabin_section'] = train_df.Cabin.str.extract('([a-z])', flags=re.IGNORECASE)
#dummies = pd.get_dummies(train_df['cabin_section'])
#train_df = pd.concat([train_df, dummies], axis=1)
#train_df.drop(['cabin_section', 'T', 'G'], axis=1, inplace=True) 
#
## Dummify Embarked for training set
#dummies = pd.get_dummies(train_df.Embarked, prefix='embarked')
#train_df = pd.concat([train_df, dummies], axis=1)
#train_df.drop(['Embarked', 'embarked_S'], axis=1, inplace=True) # , 'S'

## Embarked from 'C', 'Q', 'S'
## Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.
#
## All missing Embarked -> just make them embark from most common place
#if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
#    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values
#
#Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
#Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
#train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

# All the ages with no data -> make the median of all Ages
median_age = train_df['Age'].dropna().median()
if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age 

#train_df['age_cat'] = pd.cut(train_df.Age.values, [0, 12, 70, train_df.Age.max()], labels=[ 0, 1, 2 ])
#dummies = pd.get_dummies(train_df['age_cat'], prefix='age_grp')
#train_df = pd.concat([train_df, dummies], axis=1)
    
#train_df.Age = train_df.Age.fillna(0)

train_df['Ticket'] = pd.Categorical.from_array(train_df['Ticket']).labels

train_df.to_csv('training-features.csv')

# TEST DATA
test_df = pd.read_csv('test.csv', header=0, index_col=idCol)        # Load the test file into a dataframe

# I need to do the same with the test data now, so that the columns are the same as the training data
# I need to convert all strings to integer classifiers:
# female = 0, Male = 1
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

## All the missing Fares -> assume median of their respective class
#if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
#    median_fare = np.zeros(3)
#    for f in range(0,3):                                              # loop 0 to 2
#        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
#    for f in range(0,3):                                              # loop 0 to 2
#        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

## Process Cabin
#test_df['cabin_section'] = test_df.Cabin.str.extract('([a-z])', flags=re.IGNORECASE)
#dummies = pd.get_dummies(test_df['cabin_section'])
#test_df = pd.concat([test_df, dummies], axis=1)
#test_df.drop(['cabin_section', 'G'], axis=1, inplace=True) 
#
## Dummify Embarked for test set
#dummies = pd.get_dummies(test_df.Embarked, prefix='embarked')
#test_df = pd.concat([test_df, dummies], axis=1)
#test_df.drop(['Embarked', 'embarked_S'], axis=1, inplace=True) # , 'S'

## Embarked from 'C', 'Q', 'S'
## All missing Embarked -> just make them embark from most common place
#if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
#    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values
## Again convert all Embarked strings to int
#test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)

# All the ages with no data -> make the median of all Ages
median_age = test_df['Age'].dropna().median()
if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age

#test_df.Age = test_df.Age.fillna(0)

#test_df['age_cat'] = pd.cut(test_df.Age.values, [0, 12, 70, test_df.Age.max()], labels=[ 0, 1, 2 ])
#dummies = pd.get_dummies(test_df['age_cat'], prefix='age_grp')
#test_df = pd.concat([test_df, dummies], axis=1)

test_df['Ticket'] = pd.Categorical.from_array(test_df['Ticket']).labels

featuresUnused = ['Survived', 'Name', 'Sex', 'Cabin', 'Embarked', 'Fare' ] #'Age', 'Ticket', 
code1 = analyzeMetric('Survived', train_df, featuresUnused, 4)
showFeatureImportance(train_df, code1['features'], 'Survived')

tempTest = predict(code1['model'], test_df[code1['features']], 'Survived')
test_df['Survived'] = tempTest['Survived']

print('classification accuracy = ', classificationAccuracy(test_df['actuals'], test_df['Survived']))

#test_df = test_df[['Survived']]
#test_df.to_csv('submission5.csv', sep=',', encoding='utf-8')
