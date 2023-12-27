import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

data = 'b'
t = 2
time = 5
if data == 'h':
    data = 'HMOG'
elif data == 'b':
    data = 'BrainRun'

accelerometer = pd.read_csv(f'./data/{data}/{time} seconds/accelerometer_{t}.csv')
gyroscope = pd.read_csv(f'./data/{data}/{time} seconds/gyroscope_{t}.csv')

gyroscope = gyroscope.drop(columns=['user', 'zeroCrossing', 'peakToPeakAmplitude', 'impulse'])
accelerometer = accelerometer.drop(columns=['zeroCrossing', 'peakToPeakAmplitude', 'impulse'])
accelerometer = accelerometer.add_suffix('_acc')
accelerometer = accelerometer.rename(columns={'user_acc': 'user'})
gyroscope = gyroscope.add_suffix('_gyro')

df = pd.concat([accelerometer, gyroscope], axis=1)

X = df.drop('user', axis=1)
y = df['user']

k_best = SelectKBest(score_func=chi2, k='all')
fit = k_best.fit(X, y)

feature_scores = fit.scores_
feature_names = X.columns

feature_scores_df = pd.DataFrame({'Feature': feature_names, 'Score': feature_scores})

feature_scores_df = feature_scores_df.sort_values(by='Score', ascending=False)

plt.figure(figsize=(20, 7))
plt.bar(feature_scores_df['Feature'], feature_scores_df['Score'])
plt.xlabel('Feature')
plt.ylabel('Score')
plt.title(f'test chi2 - {data}')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
