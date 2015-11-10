# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 17:26:45 2015

@author: bolaka
"""

import os
os.chdir('/home/bolaka/python-workspace/CVX-timelines/')

# imports
from cvxtextproject import *
from mlclassificationlibs import *

setPath('/home/bolaka/datasets/titanic')
import pandas as pd
import seaborn as sns
from scipy.stats import boxcox

idCol = 'PassengerId'

# Data cleanup and feature engineering

# TRAIN DATA
train_df = pd.read_csv('train.csv', header=0, index_col=idCol)        # Load the train file into a dataframe

# TEST DATA
test_df = pd.read_csv('test.csv', header=0, index_col=idCol)        # Load the test file into a dataframe
test_df['Survived'] = 0

# merge the training and test sets
trainingLen = len(train_df)
combined = pd.concat([ train_df, test_df ])

# female = 0, Male = 1
combined['Gender'] = combined['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

#d = combined.loc[ combined[] ]
#d = train_df.loc[ train_df.Sex == 'male' ]
d = train_df
d['age_cats'] = pd.cut(d.Age.values, [0, 12, 30, 60, d.Age.max()], labels=[ '0 to 12', 
                        '12 to 30', '30 to 60', '60 to 80' ])
d = d.dropna(subset=['age_cats'])

groups = d.groupby([ 'age_cats', 'Pclass' ])

df = pd.DataFrame(columns=['age_group','class','survivality'])
count = 0
for index, group in groups:
#    print(index[0], index[1], group.Survived.mean())
    df.loc[count] = [index[0], index[1], group.Survived.mean()]
    count += 1

pivot = df.pivot("class", "age_group", "survivality")

#df.set_index('age_group', inplace=True)
sns.heatmap(pivot, annot=True, fmt="f")
#aggregated.to_csv('aggregated-features.csv')


#data = combined.loc[ combined['Pclass'] == 3 ]
##data.plot(kind='scatter', x='Age', y='Fare', c='Pclass', s=50)
#data['Fare'].hist()

#sns.factorplot("Pclass", "Age", "Survived", data=combined, palette="Pastel1")
#sns.set(style="whitegrid")
#sns.factorplot("Pclass", "Fare", data=combined, palette="Pastel1")

# I need to convert all strings to integer classifiers.
# I need to fill in the missing values of the data and make it complete.

#combined['lastname'] = combined.Name.str.extract("^([a-z|'|\s|-]{1,}),.*", flags=re.IGNORECASE)
##combined['lastname'] = pd.Categorical.from_array(combined['lastname']).codes
#
#def modifyNames(group):
#    n = len(group)
#    if (n < 3):
##        print((group['Ticket'].values[0], group['lastname'].values[0], n))
#        group['lastname'] = 'single'
#    return group
#
#groups = combined.groupby([ 'lastname', 'Ticket' ])
#combined = groups.apply(modifyNames)
#
#
#dummies = pd.get_dummies(combined['lastname'])
#combined = pd.concat([combined, dummies], axis=1)
#combined.drop(['lastname', 'single' ], axis=1, inplace=True) #, 'van Melkebeke'
#
#variance = combined.var(axis=0)
#combined.drop(variance.loc[ variance < 0.003 ].index.values, axis=1, inplace=True)

##dummies = pd.get_dummies(combined['Ticket'])
##combined = pd.concat([combined, dummies], axis=1)
##combined.drop(['110152'], axis=1, inplace=True) 

combined['title'] = combined.Name.str.extract('\\,\s+([a-z|\s]{1,20})\\.', flags=re.IGNORECASE)
#combined.loc[ (combined['title'].isin([ 'Dr', 'Rev', 'Col', 'Major', 'Capt', 'Sir', 'Don', 'Jonkheer', 'Lady',
#               'the Countess', 'Dona' ])), 'title' ] = 'Noble' # 
#combined.loc[ combined['title'].isin([ 'Mme' ]), 'title' ] = 'Mrs'
#combined.loc[ combined['title'].isin([ 'Ms', 'Mlle' ]), 'title' ] = 'Miss'
#
###combined['title_code'] = pd.Categorical.from_array(combined['title']).codes
#
#dummies = pd.get_dummies(combined['title'])
#combined = pd.concat([combined, dummies], axis=1)
#combined.drop([ 'title', 'Mr' ], axis=1, inplace=True) #, 'van Melkebeke'

# Process Cabin
combined['cabin_section'] = combined.Cabin.str.extract('([a-z])', flags=re.IGNORECASE)
combined.loc[ combined['cabin_section'] == 'T', 'cabin_section' ] = float('nan')

# All the missing cabins -> assume mode of their respective class
if len(combined.cabin_section[ combined.cabin_section.isnull() ]) > 0:
#    mode_cabin = np.zeros(3)
    mode_cabin = []
    for f in range(0,3):                                              # loop 0 to 2
#        mode_cabin[f] = combined[ combined.Pclass == f+1 ]['cabin_section'].dropna().mode()
        mode_cabin.append(combined[ combined.Pclass == f+1 ]['cabin_section'].value_counts().idxmax())
    for f in range(0,3):                                              # loop 0 to 2
        combined.loc[ (combined.cabin_section.isnull()) & (combined.Pclass == f+1 ), 'cabin_section'] = mode_cabin[f]



#dummies = pd.get_dummies(combined['cabin_section'])
#combined = pd.concat([combined, dummies], axis=1)
#combined.drop(['cabin_section', 'T', 'G'], axis=1, inplace=True) 

# Dummify Embarked for training set
dummies = pd.get_dummies(combined.Embarked, prefix='embarked')
combined = pd.concat([combined, dummies], axis=1)
combined.drop(['Embarked', 'embarked_S'], axis=1, inplace=True) # 

# All the missing ages -> assume median/avg of their respective class
if len(combined.Age[ combined.Age.isnull() ]) > 0:
    median_age = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_age[f] = combined[ combined.Pclass == f+1 ]['Age'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        combined.loc[ (combined.Age.isnull()) & (combined.Pclass == f+1 ), 'Age'] = median_age[f]

## All the missing ages -> assume median/avg of their respective class
#if len(combined.Age[ combined.Age.isnull() ]) > 0:
#    median_age = np.zeros(13)
#    for f in range(0,13):                                              # loop 0 to 2
#        median_age[f] = combined[ combined.title_code == f ]['Age'].dropna().median()
#    for f in range(0,13):                                              # loop 0 to 2
#        combined.loc[ (combined.Age.isnull()) & (combined.title_code == f ), 'Age'] = median_age[f]

#combined['age_cat'] = pd.cut(combined.Age.values, [0, 30, combined.Age.max()], labels=[ 0, 1 ])
combined['age_cat1'] = pd.cut(combined.Age.values, [0, 12, d.Age.max()], labels=[ 0, 
                        1 ])
combined['age_cat2'] = pd.cut(combined.Age.values, [0, 30, d.Age.max()], labels=[ 0, 
1 ])

##dummies = pd.get_dummies(train_df['age_cat'], prefix='age_grp')
##train_df = pd.concat([train_df, dummies], axis=1)
    
#combined.drop(['title', 'Noble' ], axis=1, inplace=True)

#combined['Ticket'] = pd.Categorical.from_array(combined['Ticket']).codes
#combined['Cabin'] = pd.Categorical.from_array(combined['Cabin']).codes

## remove outliers
#outliers = 5
#print('removing fare outliers beyond ', outliers, ' standard deviations...')
##combined.loc[ np.abs(combined.fare_log10 - combined.fare_log10.mean()) > (outliers * combined.fare_log10.std()), 'fare_log10' ] #keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.
#combined['fare_std'] = np.abs( (combined.Fare - combined.Fare.mean()) / combined.Fare.std() )
#combined.loc[ combined['fare_std'] > 5, 'Fare' ] = float('nan')
#
#combined.drop(['fare_std' ], axis=1, inplace=True)
#
## transform fare
#combined['fare_log10'] = np.log10(combined['Fare'].values + 1) 

##fare_box, fare_lambda = boxcox(combined['Fare'].values + 1)
##combined['fare_box'] = fare_box

## All the missing Fares -> assume median of their respective class
#if len(combined.fare_log10[ combined.fare_log10.isnull() ]) > 0:
#    median_fare = np.zeros(3)
#    for f in range(0,3):                                              # loop 0 to 2
#        median_fare[f] = combined[ combined.Pclass == f+1 ]['fare_log10'].dropna().median()
#    for f in range(0,3):                                              # loop 0 to 2
#        combined.loc[ (combined.fare_log10.isnull()) & (combined.Pclass == f+1 ), 'fare_log10'] = median_fare[f]

# All the missing Fares -> assume median of their respective class
if len(combined.Fare[ combined.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = combined[ combined.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        combined.loc[ (combined.Fare.isnull()) & (combined.Pclass == f+1 ), 'Fare'] = median_fare[f]

# separate into training and test sets
train_df = combined.head(trainingLen)
test_df = combined.drop(train_df.index)

train_df.to_csv('training-features.csv')
test_df.to_csv('testing-features.csv')

# drop metrics from the testing set
test_df.drop(['Survived'], axis=1, inplace=True)

# data slicing by class
training_c1 = train_df.loc[ (train_df['Pclass'] == 1) ]
training_c2 = train_df.loc[ (train_df['Pclass'] == 2) ]
training_c3 = train_df.loc[ (train_df['Pclass'] == 3) ]

testing_c1 = test_df.loc[ (test_df['Pclass'] == 1) ]
testing_c2 = test_df.loc[ (test_df['Pclass'] == 2) ]
testing_c3 = test_df.loc[ (test_df['Pclass'] == 3) ]

# class 1
featuresUnused = ['Survived', 'Name', 'Sex', 'Cabin', 'Embarked', 'Age', 'actuals', 'title_code',
                  'Ticket', 'cabin_section', 'Pclass', 'age_cat2' ] #  ,, 'title' , 'Fare'
code1 = analyzeMetric('Survived', training_c1, featuresUnused, 4)
showFeatureImportance(training_c1, code1['features'], 'Survived')

tempTest = predict(code1['model'], testing_c1[code1['features']], 'Survived')
testing_c1['Survived'] = tempTest['Survived']
print('classification accuracy class 1 = ', classificationAccuracy(testing_c1['actuals'], testing_c1['Survived']))


# class 2
featuresUnused = ['Survived', 'Name', 'Sex', 'Cabin', 'Embarked', 'Age', 'actuals', 'title_code',
                  'Ticket', 'cabin_section', 'Pclass', 'age_cat2' ] #  ,, 'title' , 'Fare'
code1 = analyzeMetric('Survived', training_c2, featuresUnused, 4)
showFeatureImportance(training_c2, code1['features'], 'Survived')

tempTest = predict(code1['model'], testing_c2[code1['features']], 'Survived')
testing_c2['Survived'] = tempTest['Survived']
print('classification accuracy class 2 = ', classificationAccuracy(testing_c2['actuals'], testing_c2['Survived']))


# class 3
featuresUnused = ['Survived', 'Name', 'Sex', 'Cabin', 'Embarked', 'Age', 'actuals', 'title_code',
                  'Ticket', 'cabin_section', 'Pclass', 'age_cat1' ] #  ,, 'title' , 'Fare'
code1 = analyzeMetric('Survived', training_c3, featuresUnused, 4)
showFeatureImportance(training_c3, code1['features'], 'Survived')

tempTest = predict(code1['model'], testing_c3[code1['features']], 'Survived')
testing_c3['Survived'] = tempTest['Survived']
print('classification accuracy class 3= ', classificationAccuracy(testing_c3['actuals'], testing_c3['Survived']))

testing_regrouped = pd.concat([ testing_c1, testing_c2, testing_c3 ])
testing_regrouped.sort_index(inplace=True)
print('classification merged accuracy = ', classificationAccuracy(testing_regrouped['actuals'], testing_regrouped['Survived']) )

# overall
featuresUnused = ['Survived', 'Name', 'Sex', 'Cabin', 'Embarked', 'Age', 'actuals', 'title_code',
                  'Ticket', 'cabin_section', 'age_cat2' ] #  ,, 'title' , 'Fare'
code1 = analyzeMetric('Survived', train_df, featuresUnused, 4)
showFeatureImportance(train_df, code1['features'], 'Survived')

tempTest = predict(code1['model'], test_df[code1['features']], 'Survived')
test_df['Survived'] = tempTest['Survived']
print('classification accuracy = ', classificationAccuracy(test_df['actuals'], test_df['Survived']))




#test_df = test_df[['Survived']]
#test_df.to_csv('submission6.csv', sep=',', encoding='utf-8')
