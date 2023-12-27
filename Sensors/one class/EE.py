import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedKFold
from sklearn.covariance import EllipticEnvelope

import warnings

# ignore all warnings about editing a copy of features dataframe
warnings.filterwarnings("ignore")

data = 'h'  # dataset to use
time = 0.2  # authentication time
t = 0  # outlier threshold
relevantFeatures = 60 # number of relevant features
k = 10
contamination = 0.1

if data == 'h':
    data = 'HMOG'
elif data == 'b':
    data = 'BrainRun'

print('INFO:')
print('Elliptic Envelope')
print(f'contamination:___________{contamination}')
print(f'dataset:_________________{data}')
print(f'threshold:_______________{t}')
print(f'signal length:___________{time}')
print(f'number of folds:__________{k}')
print(f'number of features:_______{relevantFeatures}')
print()

# load all sensor data
accelerometer = pd.read_csv(f'./data/{data}/{time} seconds/accelerometer_{t}.csv')
gyroscope = pd.read_csv(f'./data/{data}/{time} seconds/gyroscope_{t}.csv')

gyroscope = gyroscope.drop(columns=['user', 'zeroCrossing', 'peakToPeakAmplitude', 'impulse'])
accelerometer = accelerometer.drop(columns=['zeroCrossing', 'peakToPeakAmplitude', 'impulse'])
accelerometer = accelerometer.add_suffix('_acc')
accelerometer = accelerometer.rename(columns={'user_acc': 'user'})
gyroscope = gyroscope.add_suffix('_gyro')

# concatenate sensor data
features = pd.concat([accelerometer, gyroscope], axis=1)
# selection of the first 20 users for experiments
# features = features[features['user'].isin(features['user'].unique()[:20])]

# getting the unique users in the dataset
uniqueUsers = features['user'].unique()

kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
selector = SelectKBest(score_func=chi2, k=relevantFeatures)

allUserAccuracy = []
allUserEer = []
farl = []
frrl = []
u = 0
for user in uniqueUsers:
    u += 1
    print(f'{u}/{len(uniqueUsers)}')

    # getting the data of the current user
    personOfInterestData = features[features['user'] == user]

    # change the label to 1 for current user (normal) and -1 for the others (anomalies)
    personOfInterestData['user'] = 1
    testData = features[features['user'] != user]
    testData['user'] = -1

    yUser = personOfInterestData.pop('user')
    yOther = testData.pop('user')

    # select the relevantFeatures for the current user
    selected = selector.fit_transform(personOfInterestData, yUser)
    selectedFeaturesIndices = selector.get_support(indices=True)
    selectedFeatures = personOfInterestData.iloc[:, selectedFeaturesIndices]

    columnsToRemove = []
    for column in testData.columns:
        if column not in selectedFeatures.columns:
            columnsToRemove.append(column)

    testData.drop(columns=columnsToRemove, inplace=True)

    accuracyValues = []
    eerValues = []

    for trainIndex, testIndex in kf.split(selectedFeatures, yUser):
        # split the dataset into training and testing
        xTrain, xTest = selectedFeatures.iloc[trainIndex], selectedFeatures.iloc[testIndex]
        yTest = yUser.iloc[testIndex]

        # initialize the model
        oc_env = EllipticEnvelope(contamination=contamination)

        # training phase
        oc_env.fit(xTrain)

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        # make prediction on normal instances
        yPred = oc_env.predict(xTest)
        yTest = yTest.values

        for i in range(0, len(yTest)):
            predVal = yPred[i]
            if predVal != 1:
                fn += 1
            else:
                tp += 1

        # make prediction on anomalies
        yPred = oc_env.predict(testData)
        yOther = pd.Series(yOther).values

        for i in range(0, len(yOther)):
            predVal = yPred[i]
            if predVal == 1:
                fp += 1
            else:
                tn += 1

        # calculate eer and accuracy
        far = fp / (tn + fp)
        frr = fn / (fn + tp)

        farl.append(far)
        frrl.append(frr)

        eer = (far + frr) / 2

        accuracy = (tp + tn) / (tp + tn + fp + fn)

        eerValues.append(eer)
        accuracyValues.append(accuracy)

    allUserAccuracy.append(np.mean(accuracyValues))
    allUserEer.append(np.mean(eerValues))

print(f'Average Accuracy for all users: ________{np.mean(allUserAccuracy)}')
print(f'Average EER for all users: _____________{np.mean(allUserEer)}')
print(f'Average FAR for all users: _____________{np.mean(farl)}')
print(f'Average FRR for all users: _____________{np.mean(frrl)}')
