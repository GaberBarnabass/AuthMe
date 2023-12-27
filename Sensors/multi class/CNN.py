"""import tensorflow as tf
# If the terminal support gpu computation, tensorflow will automatically use gpu but, if you want to use cpu,
# in my case so much faster than gpu uncomment this
# Force TensorFlow to use the CPU
tf.config.set_visible_devices([], 'GPU')"""
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization

data = 'h'  # dataset to use
t = 0  # threshold for outlier removal
time = 0.2  # time length
k = 10
relevantFeatures = 40  # number of relevant features
optimizer = 'adam'
loss = 'sparse_categorical_crossentropy'

if data == 'h':
    data = 'HMOG'
elif data == 'b':
    data = 'BrainRun'

print('INFO:')
print('CNN')
print(f'optimizer:______________{optimizer}')
print(f'loss:___________________{loss}')
print(f'dataset:________________{data}')
print(f'threshold_______________{t}')
print(f'signal length:__________{time}')
print(f'number of folds:________{k}')
print(f'number of features:_____{relevantFeatures}')
print()

# load sensor data
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

# transform categorical target variable into numerical 0 to n
label_encoder = LabelEncoder()
features['user'] = label_encoder.fit_transform(features['user'])

# extract the target variable
users = features.pop('user')

# select the best relevantFeatures
selector = SelectKBest(score_func=f_classif, k=relevantFeatures)
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

selectedFeatures = selected

accuracies = []
eer_values = []

for trainIndex, testIndex in kf.split(selectedFeatures, users):
    # split the dataset into training and testing
    xTrain, xTest = selectedFeatures[trainIndex], selectedFeatures[testIndex]
    yTrain, yTest = users.iloc[trainIndex], users.iloc[testIndex]

    # cnn layers
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(relevantFeatures, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(len(label_encoder.classes_), activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    xTrain, xValidation, yTrain, yValidation = train_test_split(xTrain, yTrain, test_size=0.2, random_state=42)

    # model training
    model.fit(xTrain, yTrain, epochs=20, batch_size=32, validation_data=(xValidation, yValidation),
              callbacks=[early_stopping], verbose=0)

    # make predictions
    y_pred_proba = model.predict(xTest)

    # convert probabilities to class labels
    y_pred = np.argmax(y_pred_proba, axis=1)

    # calculate accuracy and eer
    accuracy = accuracy_score(yTest, y_pred)
    accuracies.append(accuracy)

    eer = []
    for i in range(len(np.unique(users))):
        fpr, tpr, _ = roc_curve((yTest == i).astype(int), y_pred_proba[:, i])
        eer_i = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        eer.append(eer_i)

    eer_values.append(np.mean(eer))

average_accuracy = np.mean(accuracies)
average_eer = np.mean(eer_values)

print(f'Average Accuracy:___{average_accuracy}')
print(f'Average EER:________{average_eer}')
