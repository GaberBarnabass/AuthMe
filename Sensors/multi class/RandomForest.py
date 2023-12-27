import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

# parameters
numberOfSubtree = 60  # number of subtrees of the rf
minLeaf = 1  # minimum number of sample to consider as a leaf node
data = 'b'  # data to use
t = 0  # outlier threshold
time = 2  # length of the signal
k = 10
relevantFeatures = 40  # number of features to use and select

if data == 'h':
    data = 'HMOG'
elif data == 'b':
    data = 'BrainRun'

print('INFO:')
print('Random Forest')
print(f'number of subtrees:_____{numberOfSubtree}')
print(f'min samples per leaf:___{minLeaf}')
print(f'dataset:________________{data}')
print(f'threshold_______________{t}')
print(f'signal length:__________{time}')
print(f'number of folds:________{k}')
print(f'number of features:_____{relevantFeatures}')
print()

# loading data
accelerometer = pd.read_csv(f'./data/{data}/{time} seconds/accelerometer_{t}.csv')
gyroscope = pd.read_csv(f'./data/{data}/{time} seconds/gyroscope_{t}.csv')

gyroscope = gyroscope.drop(columns=['user', 'zeroCrossing', 'peakToPeakAmplitude', 'impulse'])
accelerometer = accelerometer.drop(columns=['zeroCrossing', 'peakToPeakAmplitude', 'impulse'])
accelerometer = accelerometer.add_suffix('_acc')
accelerometer = accelerometer.rename(columns={'user_acc': 'user'})
gyroscope = gyroscope.add_suffix('_gyro')

# concatenate data from the two sensors
features = pd.concat([accelerometer, gyroscope], axis=1)
# selection of the first 20 users for experiments
# features = features[features['user'].isin(features['user'].unique()[:20])]

# extract target variable
users = features.pop('user')

# select the best relevantFeatures
selector = SelectKBest(score_func=chi2, k=relevantFeatures)
selected = selector.fit_transform(features, users)
selectedFeaturesIndices = selector.get_support(indices=True)
selectedFeatures = features.iloc[:, selectedFeaturesIndices]

print('selected features:')
print()
for i in range(0, len(selectedFeatures.columns)):
    if i + 1 < 10:
        print(f' {i + 1}) {selectedFeatures.columns[i]}')
    else:
        print(f'{i + 1}) {selectedFeatures.columns[i]}')
print()

kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

accuracies = []
eer_values = []

for trainIndex, testIndex in kf.split(selectedFeatures, users):
    # split the dataset in training and test
    X_train, X_test = selectedFeatures.iloc[trainIndex], selectedFeatures.iloc[testIndex]
    y_train, y_test = users.iloc[trainIndex], users.iloc[testIndex]

    # initialize the model
    rfClassifier = RandomForestClassifier(n_estimators=numberOfSubtree, min_samples_leaf=minLeaf)

    # model training
    rfClassifier.fit(X_train, y_train)

    # make predictions
    y_pred = rfClassifier.predict(X_test)

    # calculate accuracy and eer
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    y_test_binarized = label_binarize(y_test, classes=np.unique(y_train))

    n_classes = y_test_binarized.shape[1]
    fpr = dict()
    tpr = dict()
    eer = []

    y_scores = rfClassifier.predict_proba(X_test)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_scores[:, i])
        eer_i = brentq(lambda x: 1. - x - interp1d(fpr[i], tpr[i])(x), 0., 1.)
        eer.append(eer_i)

    eer_values.append(np.mean(eer))

average_accuracy = np.mean(accuracies)
average_eer = np.mean(eer_values)

print(f'Average Accuracy:___{average_accuracy}')
print(f'Average EER:________{average_eer}')
