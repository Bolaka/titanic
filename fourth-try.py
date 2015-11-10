# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:39:44 2015

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
test_df['Survived'] = 100

# merge the training and test sets
trainingLen = len(train_df)
combined = pd.concat([ train_df, test_df ])

# female = 0, Male = 1
combined['Gender'] = combined['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# All the missing Fares -> assume median of their respective class
if len(combined.Fare[ combined.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = combined[ combined.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        combined.loc[ (combined.Fare.isnull()) & (combined.Pclass == f+1 ), 'Fare'] = median_fare[f]

#combined['title'] = combined.Name.str.extract('\\,\s+([a-z|\s]{1,20})\\.', flags=re.IGNORECASE)
#combined.loc[ combined['title'].isin([ 'Mlle', 'Ms' ]), 'title' ] = 'Miss'
#combined.loc[ combined['title'].isin([ 'Jonkheer', 'Major', 'Col', 'Don', 'Capt', 'Rev', 'Dr', 'Sir', 'Mr' ]), 'title' ] = 'adult'
#combined.loc[ combined['title'].isin([ 'Mrs', 'Dona', 'Lady', 'Mme', 'the Countess' ]), 'title' ] = 'adult'

#combined.loc[ combined['title'].isin([ 'Mme', 'Mlle', 'Ms' ]), 'title' ] = 'Miss'
#combined.loc[ combined['title'].isin([ 'Jonkheer', 'Major', 'Col', 'Don', 'Capt', 'Rev', 'Dr' ]), 'title' ] = 'Sir'
#combined.loc[ combined['title'].isin([ 'the Countess', 'Dona' ]), 'title' ] = 'Lady'
#
#dummies = pd.get_dummies(combined['title'])
#combined = pd.concat([combined, dummies], axis=1)

## impute missing values in Age by prediction
#age_test = combined.loc[ combined.Age.isnull() ]
#age_train = combined.drop(age_test.index)
#
#featuresUnused = [ 'Age', 'Cabin', 'Name', 'Sex', 'Survived', 'Ticket', 'actuals', 'Embarked', 'Parch', 'SibSp',
#                  'title']
#code1 = analyzeMetricNumerical('Age', age_train, featuresUnused, False)
#showFeatureImportanceNumerical(age_train, code1['features'], 'Age')
#
#temp = predict(code1['model'], age_test[code1['features']], 'Age')
#age_test['Age'] = temp['Age'].values
#
#combined = pd.concat([ age_train, age_test ])

##d = combined.loc[ combined[] ]
##d = train_df.loc[ train_df.Sex == 'male' ]
#d = train_df
#d['age_cats'] = pd.cut(d.Age.values, [0, 12, 30, 60, d.Age.max()], labels=[ '0 to 12', 
#                        '12 to 30', '30 to 60', '60 to 80' ])
#d = d.dropna(subset=['age_cats'])
#
#groups = d.groupby([ 'age_cats', 'Pclass' ])
#
#df = pd.DataFrame(columns=['age_group','class','survivality'])
#count = 0
#for index, group in groups:
##    print(index[0], index[1], group.Survived.mean())
#    df.loc[count] = [index[0], index[1], group.Survived.mean()]
#    count += 1
#
#pivot = df.pivot("class", "age_group", "survivality")
#
##df.set_index('age_group', inplace=True)
#sns.heatmap(pivot, annot=True, fmt="f")

#data = combined.loc[ combined['Pclass'] == 3 ]
##data.plot(kind='scatter', x='Age', y='Fare', c='Pclass', s=50)
#data['Fare'].hist()

#sns.factorplot("Pclass", "Age", "Survived", data=combined, palette="Pastel1")
#sns.set(style="whitegrid")
#sns.factorplot("Pclass", "Fare", data=combined, palette="Pastel1")

# I need to convert all strings to integer classifiers.
# I need to fill in the missing values of the data and make it complete.

combined['lastname'] = combined.Name.str.extract("^([a-z|'|\s|-]{1,}),.*", flags=re.IGNORECASE)

def modifyNames(group):
    n = len(group)
    if (n < 3):
#        print((group['Ticket'].values[0], group['lastname'].values[0], n))
        group['lastname'] = 'single'
    return group

groups = combined.groupby([ 'lastname', 'Ticket' ])
combined = groups.apply(modifyNames)

dummies = pd.get_dummies(combined['lastname'])
variance = dummies.var(axis=0)
dummies.drop(variance.loc[ variance < 0.003 ].index.values, axis=1, inplace=True)
dummies.drop('single', axis=1, inplace=True)

relevant_lastnames = names(dummies)

combined = pd.concat([combined, dummies], axis=1)
#combined.drop(['lastname', 'single' ], axis=1, inplace=True) #, 'van Melkebeke'
combined.drop(['lastname' ], axis=1, inplace=True) #, 'van Melkebeke'


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

# Dummify Embarked for training set
combined.Embarked[ combined.Embarked.isnull() ] = combined.Embarked.dropna().mode().values

dummies = pd.get_dummies(combined.Embarked, prefix='embarked')
combined = pd.concat([combined, dummies], axis=1)
combined.drop(['Embarked', 'embarked_S'], axis=1, inplace=True) # 

#dummies = pd.get_dummies(combined['cabin_section'])
#combined = pd.concat([combined, dummies], axis=1)
#combined.drop(['cabin_section', 'T', 'G'], axis=1, inplace=True) 

# TODO All the missing ages -> assume median/avg of their respective class
# median age of title for each class
#groups = combined.groupby([ 'title', 'Pclass' ])
#
#def fillAgebyClassTitle(group):
#    missing = len(group.Age[ group.Age.isnull() ])
#    average = group.Age.dropna().median() # .median()
##    print('before = ', group.title.values[0], group.Pclass.values[0], average, missing )
#    if missing > 0:
#        group.loc[ (group.Age.isnull()), 'Age'] = average
#    missing = len(group.Age[ group.Age.isnull() ])
##    print('after = ', group.title.values[0], group.Pclass.values[0], average, missing )
#    return group
#
#combined = groups.apply(fillAgebyClassTitle)
#
#df = pd.DataFrame(columns=['title','class','age'])
#count = 0
#for index, group in groups:
#    missing = len(group.Age[ group.Age.isnull() ])
#    average = group.Age.dropna().median()
##    print('before = ', index[0], index[1], average, missing )
#    if missing > 0:
#        group.loc[ (group.Age.isnull()), 'Age'] = average
#    missing = len(group.Age[ group.Age.isnull() ])
##    print('after = ', index[0], index[1], average, missing )
#    
#    df.loc[count] = [index[0], index[1], group.Age.median()]
#    count += 1
#
#pivot = df.pivot("class", "title", "age")
#sns.heatmap(pivot, annot=True, fmt="f")

if len(combined.Age[ combined.Age.isnull() ]) > 0:
    median_age = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_age[f] = combined[ combined.Pclass == f+1 ]['Age'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        combined.loc[ (combined.Age.isnull()) & (combined.Pclass == f+1 ), 'Age'] = median_age[f]

#combined.drop([ 'title' ], axis=1, inplace=True)


## All the missing ages -> assume median/avg of their respective class
#if len(combined.Age[ combined.Age.isnull() ]) > 0:
#    median_age = np.zeros(13)
#    for f in range(0,13):                                              # loop 0 to 2
#        median_age[f] = combined[ combined.title_code == f ]['Age'].dropna().median()
#    for f in range(0,13):                                              # loop 0 to 2
#        combined.loc[ (combined.Age.isnull()) & (combined.title_code == f ), 'Age'] = median_age[f]

combined['age_cat1'] = pd.cut(combined.Age.values, [0, 12, combined.Age.max()], labels=[ 0, 1 ])

#combined['class_age'] = combined.Pclass * combined.Age

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

# size of family
#combined['family_size'] = combined.SibSp + combined.Parch
#combined['has_SibSp'] = 0
#combined.loc[ combined.SibSp > 1, 'has_SibSp' ] = 1
#
#combined['has_Parch'] = 0
#combined.loc[ combined.Parch > 1, 'has_Parch' ] = 1

# separate into training and test sets
train_df = combined.loc[ combined.Survived != 100 ]
test_df = combined.drop(train_df.index)

train_df.to_csv('training-features.csv')
test_df.to_csv('testing-features.csv')

# drop metrics from the testing set
test_df.drop(['Survived'], axis=1, inplace=True)

# data slicing by sex
training_males = train_df.loc[ (train_df['Gender'] == 1) ]
testing_males = test_df.loc[ (test_df['Gender'] == 1) ]
#testing_males['Survived'] = 100
#
#combined_males = pd.concat([ training_males, testing_males ])
#
#combined_males['lastname'] = combined_males.Name.str.extract("^([a-z|'|\s|-]{1,}),.*", flags=re.IGNORECASE)
#
#def modifyNames(group, names):
#    name = group['lastname'].values[0]
##    print(name)  
#    if (name not in names):
##        print(group['Ticket'].values[0], name)
#        group['lastname'] = 'remove'
#    return group
#
#groups = combined_males.groupby([ 'lastname', 'Ticket' ])
#combined_males = groups.apply(modifyNames, relevant_lastnames)
##combined_males.to_csv('combined_males-features.csv')
#
#
#dummies = pd.get_dummies(combined_males['lastname'])
#combined_males = pd.concat([combined_males, dummies], axis=1)
#combined_males.drop(['lastname', 'remove' ], axis=1, inplace=True) #, 'van Melkebeke'
#
##variance = combined_males.var(axis=0)
##combined_males.drop(variance.loc[ variance < 0.003 ].index.values, axis=1, inplace=True)
#
#training_males = combined_males.loc[ combined_males.Survived != 100 ]
#testing_males = combined_males.drop(training_males.index)
#testing_males.drop(['Survived'], axis=1, inplace=True)




training_females = train_df.loc[ (train_df['Gender'] == 0) ]
testing_females = test_df.loc[ (test_df['Gender'] == 0) ]
testing_females['Survived'] = 100


combined_females = pd.concat([ training_females, testing_females ])
combined_females.drop(relevant_lastnames, axis=1, inplace=True) 

# process tickets
combined_females['ticket_cleaned'] = [re.sub('\.', '', re.sub('/', '', ticket) ) for ticket in combined_females.Ticket]
combined_females['ticket_group'] = [c.isalnum() for c in combined_females['ticket_cleaned'] ]
combined_females.loc[ combined_females['ticket_group'] == False, 'ticket_cleaned' ].values

prefixes = []
for ticket in combined_females['ticket_cleaned']:
    alfa = ticket.isalnum()
    if alfa:
        tl = len(ticket)
        l = round(len(ticket)/2)
        prefixes.append(ticket[0:l] ) # + '_' + str(tl)
    else:
        part1 = ticket.split()[0]
        part2 = ticket.split()[1]
        l = round(len(part2)/2)
#        l = len(part2)
#        if l % 2 != 0:
#            l += 1
#        l = int(l/2)
#        print(ticket, '------------', part1 + part2[0:l])
        prefixes.append(part1 + part2[0:l])

combined_females['ticket_prefix'] = prefixes
combined_females.to_csv('combined_females-features.csv')


#def modifyTickets(group):
#    n = len(group)
##    print((group['Ticket'].values[0], group['ticket_prefix'].values[0], n))
#    if (n < 2):
#        group['ticket_prefix'] = 'single'
#    return group
#
#groups = combined_females.groupby( 'ticket_prefix' )
#combined_females = groups.apply(modifyTickets)

#combined_females['ticket_start'] = combined_females['ticket_cleaned'].map( lambda x: x[0:2] )
#def noofclusters(group):
#    length = len(group)
#    print(group.ticket_start.values[0], length)
#    if length < 10:
#        group['ticket_start'] = 'toofew'
#
#g = combined_females.groupby('ticket_start')
##for index, group in g:
##    print(index)
#
#combined_females = g.apply(noofclusters)

dummies = pd.get_dummies(combined_females['ticket_prefix'], prefix='tic_str')
variance = dummies.var(axis=0)
dummies.drop(variance.loc[ variance < 0.003 ].index.values, axis=1, inplace=True)

combined_females = pd.concat([combined_females, dummies], axis=1)
combined_females['ticket_prefix'].value_counts()

combined_females.drop([ 'ticket_prefix', 'ticket_cleaned' ], axis=1, inplace=True) 

training_females = combined_females.loc[ combined_females.Survived != 100 ]
testing_females = combined_females.drop(training_females.index)
testing_females.drop(['Survived'], axis=1, inplace=True)

print('NO OF TICKET CLUSTERS = ', len(names(dummies)))
training_females.to_csv('training_females-features.csv')
training_males.to_csv('training_males-features.csv')

# males
featuresUnused = ['Survived', 'Name', 'Sex', 'Cabin', 'Embarked', 'Age', 'actuals', 'title_code',
                  'Ticket', 'cabin_section', 'Gender', 'family_size' ] #  'SibSp', 'Parch', 
code1 = analyzeMetric('Survived', training_males, featuresUnused, 4)
showFeatureImportance(training_males, code1['features'], 'Survived')

tempTest = predict(code1['model'], testing_males[code1['features']], 'Survived')
testing_males['Survived'] = tempTest['Survived']
print('classification accuracy males = ', classificationAccuracy(testing_males['actuals'], testing_males['Survived']))


# females
featuresUnused = ['Survived', 'Name', 'Sex', 'Cabin', 'Embarked', 'Age', 'actuals', 'title_code',
                  'Ticket', 'cabin_section', 'Gender', 'family_size' ] #  'SibSp', 'Parch', 
code1 = analyzeMetric('Survived', training_females, featuresUnused, 4)
showFeatureImportance(training_females, code1['features'], 'Survived')

tempTest = predict(code1['model'], testing_females[code1['features']], 'Survived')
testing_females['Survived'] = tempTest['Survived']
print('classification accuracy females = ', classificationAccuracy(testing_females['actuals'], testing_females['Survived']))


## class 3
#featuresUnused = ['Survived', 'Name', 'Sex', 'Cabin', 'Embarked', 'Age', 'actuals', 'title_code',
#                  'Ticket', 'cabin_section', 'Pclass', 'age_cat1' ] #  ,, 'title' , 'Fare'
#code1 = analyzeMetric('Survived', training_c3, featuresUnused, 4)
#showFeatureImportance(training_c3, code1['features'], 'Survived')
#
#tempTest = predict(code1['model'], testing_c3[code1['features']], 'Survived')
#testing_c3['Survived'] = tempTest['Survived']
#print('classification accuracy class 3= ', classificationAccuracy(testing_c3['actuals'], testing_c3['Survived']))

testing_regrouped = pd.concat([ testing_females, testing_males ])
testing_regrouped.sort_index(inplace=True)
print('classification merged accuracy = ', classificationAccuracy(testing_regrouped['actuals'], testing_regrouped['Survived']) )

testing_regrouped = testing_regrouped[['Survived']]
testing_regrouped.to_csv('submission9.csv', sep=',', encoding='utf-8')

## overall
#featuresUnused = ['Survived', 'Name', 'Sex', 'Cabin', 'Embarked', 'Age', 'actuals', 'title_code',
#                  'Ticket', 'cabin_section', 'age_cat2' ] #  ,, 'title' , 'Fare'
#code1 = analyzeMetric('Survived', train_df, featuresUnused, 4)
#showFeatureImportance(train_df, code1['features'], 'Survived')
#
#tempTest = predict(code1['model'], test_df[code1['features']], 'Survived')
#test_df['Survived'] = tempTest['Survived']
#print('classification accuracy = ', classificationAccuracy(test_df['actuals'], test_df['Survived']))




#test_df = test_df[['Survived']]
#test_df.to_csv('submission6.csv', sep=',', encoding='utf-8')
