import glob
import csv
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import json
import os
import shutil
from scipy import signal, stats
from scipy.stats import entropy
from scipy.signal import welch
from sklearn.preprocessing import MinMaxScaler
import multiprocessing


def getUsers(path_):
    """

    :param path_: path to dataset
    :return: a list of all subdirectories representing all users
    """
    return [user for user in os.scandir(path_) if user.is_dir()]


def removeEmpty():
    """
    removes user with missing accelerometer/gyroscope readings
    p.s. empty files of HMOG will be removed in loadAllSensorData()
    """
    users_ = getUsers('./data/BrainRun/raw sensors/raw data')
    for user in users_:
        jsonFiles = [jsn for jsn in os.scandir(user.path) if jsn.name.endswith('.json')]
        for jsn in jsonFiles.copy():
            os.system('cls')
            print()
            print(f'user {user.name}')
            print(f'file {jsn.name}')
            with open(jsn) as jsonFile:
                jsonContent = json.load(jsonFile)
                lenAcc = len(jsonContent['accelerometer'])
                lenGyro = len(jsonContent['gyroscope'])
                if lenGyro == 0 or lenAcc == 0:
                    jsonFile.close()
                    os.remove(jsn)
                else:
                    jsonFile.close()
        # remove the user folder if it is empty after removing files with empty readings
        userDir = os.listdir(user.path)
        print(user.path)
        if len(userDir) == 0:
            os.rmdir(user.path)


def analyzeData():
    """
    For BrainRun dataset calculate the mean number of files (minutes) for all user
    :return: a dataframe containing all user that have a number of files gt or equal to avg
    """
    users_ = getUsers('./data/BrainRun/raw sensors/raw data')
    info_ = pd.DataFrame(columns=['user', 'nFiles'])
    index = 0
    for u in users_:
        df = pd.DataFrame({
            'user': u.name,
            'nFiles': len([jsn for jsn in os.listdir(u.path) if jsn.endswith('.json')])
        }, index=[index])
        index += 1
        info_ = pd.concat([info_, df], ignore_index=True)

    infoN = pd.DataFrame(columns=['n', 'howMany'])

    for n in info_['nFiles'].unique():
        index = 0
        df = pd.DataFrame({
            'n': n,
            'howMany': (info_['nFiles'] == n).sum()
        }, index=[index])
        index += 1
        infoN = pd.concat([infoN, df], ignore_index=True)

    averageNFiles = int(info_["nFiles"].mean())
    print(f'average number of files per user: {averageNFiles}')
    rowsToDelete = (infoN['n'] < averageNFiles)
    infoN = infoN.drop(infoN[rowsToDelete].index)
    print(f'number of user that have a number of files >= average number of files: {(infoN["howMany"]).sum()}')
    rowsToDelete = (info_['nFiles'] < averageNFiles)
    info_ = info_.drop(info_[rowsToDelete].index)
    return averageNFiles, info_['user'].unique()


def removeLTorGTaverage(avg, validUsers):
    """
    Remove user with #files < avg, resample users that have more than avg files
    :param avg: average number of files for all user
    :param validUsers: users who have #files >= avg
    :return:
    """
    users_ = getUsers('./data/BrainRun/raw sensors/raw data')
    # remove user containing less than 71 files each
    for u in users_.copy():
        if u.name not in validUsers:
            print(u.path)
            shutil.rmtree(u.path)
            users_.remove(u)

    for u in users_:
        jsonFiles = [jsn for jsn in os.listdir(u.path) if jsn.endswith('.json')]
        # cut the number of files per user to 71
        if len(jsonFiles) > avg:
            for toRemove in jsonFiles[avg:]:
                os.remove(os.path.join(u.path, toRemove))


def adjustReadingsFrequency(tSeries):
    """
    Take as input a time series with a sample rate different form 10 Hz:
    each file of BrainRun must contain 600 readings
    if #readings > 600 : down-sampling;
    if #readings < 600 : up-sampling
    :param tSeries: time series of an axis (x, y, z) of some sensor with sample rate > 10 Hz or < 10 Hz
    :return: tSeries with the correct sampling rate
    """
    # the frequency must be 10 Hz, 1 sample every 100 ms = 10 per second = 600 per minute

    currentTimestamps = np.linspace(0, 60, len(tSeries))
    x_ = np.array(tSeries)
    desiredFrequency = 10  # Hz

    if len(tSeries) < 600:
        # up-sampling
        desiredTimestamps = np.linspace(0, 60, desiredFrequency * 60)
        interpolation = interp1d(currentTimestamps, x_, kind='linear', fill_value='extrapolate')
        x__ = interpolation(desiredTimestamps)
        return x__
    elif len(tSeries) > 600:
        # down-sampling
        currentSampleRate = len(x_) / 60
        downSamplingFactor = currentSampleRate / desiredFrequency
        # x__ = x_[::downSamplingFactor]
        x__ = signal.resample(x_, int(len(x_) / downSamplingFactor))
        downSampledTimestamps = np.arange(0, len(x__)) / desiredFrequency
        return x__

    else:
        return tSeries


def loadAllSensorData(data_=''):
    """
    takes raw data form sensor S and user U, for each file adjust data frequency and save them into folder S and
    sub-folder U
    :param data_: dataset to use
    :return: create a folder for accelerometer data and one for gyroscope data which will contain sensor data of all
            user
    """
    if data_ == 'BrainRun':
        users_ = getUsers('./data/BrainRun/raw sensors/raw data')
        info_ = pd.DataFrame(columns=['user', 'accelerometerSize', 'gyroscopeSize', 'timestamp'])
        sensorData = pd.DataFrame(
            columns=['x', 'y', 'z', 'user', 'timestamp'])

        uIndex = 0
        dfInfoList = []
        for user in users_:
            uIndex += 1
            jsonFiles = [jsn for jsn in os.scandir(user.path) if jsn.name.endswith('.json')]
            index = 0

            for jsn in jsonFiles:
                index += 1
                os.system('cls')
                print()
                print(f'user {user.name}       {uIndex}/{len(users_)}')
                print(f'file {jsn.name}          {index}/{len(jsonFiles)}')

                with open(jsn) as jsonFile:
                    jsonContent = json.load(jsonFile)
                    jsn_n = jsn.name.replace('.json', '')
                    tmp = jsn_n.split('_')
                    timestamp = tmp[1]

                    x_ = []
                    y_ = []
                    z_ = []
                    for j in jsonContent['accelerometer']:
                        x, y, z = j['x'], j['y'], j['z']
                        x_.append(x)
                        y_.append(y)
                        z_.append(z)
                    x_ = adjustReadingsFrequency(x_)
                    y_ = adjustReadingsFrequency(y_)
                    z_ = adjustReadingsFrequency(z_)
                    sensorData['x'] = x_
                    sensorData['y'] = y_
                    sensorData['z'] = z_
                    sensorData['timestamp'] = timestamp
                    sensorData['user'] = user.name

                    if not os.path.exists(f'./data/BrainRun/raw sensors/accelerometer/{user.name}'):
                        os.makedirs(f'./data/BrainRun/raw sensors/accelerometer/{user.name}')
                    sensorData.to_csv(f'./data/BrainRun/raw sensors/accelerometer/{user.name}/{timestamp}.csv',
                                      index=False)
                    accelerometerSize = len(x_)
                    sensorData = pd.DataFrame(columns=sensorData.columns)

                    x_ = []
                    y_ = []
                    z_ = []
                    for j in jsonContent['gyroscope']:
                        x, y, z = j['x'], j['y'], j['z']
                        x_.append(x)
                        y_.append(y)
                        z_.append(z)
                    x_ = adjustReadingsFrequency(x_)
                    y_ = adjustReadingsFrequency(y_)
                    z_ = adjustReadingsFrequency(z_)
                    sensorData['x'] = x_
                    sensorData['y'] = y_
                    sensorData['z'] = z_
                    sensorData['timestamp'] = timestamp
                    sensorData['user'] = user.name

                    if not os.path.exists(f'./data/BrainRun/raw sensors/gyroscope/{user.name}'):
                        os.makedirs(f'./data/BrainRun/raw sensors/gyroscope/{user.name}')
                    sensorData.to_csv(f'./data/BrainRun/raw sensors/gyroscope/{user.name}/{timestamp}.csv', index=False)
                    gyroSize = len(x_)
                    sensorData = pd.DataFrame(columns=sensorData.columns)

                    df = pd.DataFrame({
                        'user': user.name,
                        'accelerometerSize': accelerometerSize,
                        'gyroscopeSize': gyroSize,
                        'timestamp': timestamp
                    }, index=[index - 1])
                    dfInfoList.append(df)
                jsonFile.close()
        info_ = pd.concat(dfInfoList, ignore_index=True)
        info_.to_csv('./data/BrainRun/raw sensors/info.csv', index=False)
    elif data_ == 'HMOG':
        uIndex = 0
        users_ = getUsers('./data/HMOG/raw sensors/raw data')
        for user in users_:
            uIndex += 1
            os.system('cls')
            print()
            print(f'user {user.name}       {uIndex}/{len(users_)}')
            accelerometerList = []
            gyroscopeList = []
            for session in os.scandir(user.path):
                if not os.path.isdir(session.path):
                    continue
                accelerometer = os.path.join(session.path, 'Accelerometer.csv')
                gyroscope = os.path.join(session.path, 'Gyroscope.csv')
                # remove empty files from HMOG users
                if os.path.getsize(accelerometer) == 0:
                    continue
                if os.path.getsize(gyroscope) == 0:
                    continue
                accelerometerList.append(accelerometer)
                gyroscopeList.append(gyroscope)

                """shutil.copy(accelerometer, os.path.join(pAcc, session.name + '.csv'))
                shutil.copy(gyroscope, os.path.join(pGyro, session.name + '.csv'))"""
            if len(accelerometerList) == 24 and len(gyroscopeList) == 24:
                totalRows = 0
                noPortraitRows = 0
                for i in range(0, len(accelerometerList)):
                    accelerometer = pd.read_csv(accelerometerList[i], index_col=None)
                    gyroscope = pd.read_csv(gyroscopeList[i], index_col=None)
                    accelerometer.columns = ['timestamp', 'eventTime', 'activityId', 'x', 'y', 'z', 'orientation']
                    gyroscope.columns = ['timestamp', 'eventTime', 'activityId', 'x', 'y', 'z', 'orientation']
                    totalRows += len(accelerometer) + len(gyroscope)
                    noPortraitRows += len(accelerometer.loc[accelerometer['orientation'] != 0])
                    noPortraitRows += len(gyroscope.loc[gyroscope['orientation'] != 0])
                perc = (noPortraitRows / totalRows) * 100
                if perc > 10:
                    continue

                pAcc = f'./data/HMOG/raw sensors/accelerometer/{user.name}'
                pGyro = f'./data/HMOG/raw sensors/gyroscope/{user.name}'
                if not os.path.exists(pAcc):
                    os.makedirs(pAcc)
                if not os.path.exists(pGyro):
                    os.makedirs(pGyro)
                for i in range(0, len(accelerometerList)):
                    shutil.copy(accelerometerList[i], os.path.join(pAcc, f'{user.name}_session_{i}.csv'))
                    shutil.copy(gyroscopeList[i], os.path.join(pGyro, f'{user.name}_session_{i}.csv'))


def clean(data_='', threshold_=3):
    """
    utility function to remove pooled data in case of mistake
    :param data_: dataset to use
    :param threshold_: outlier threshold for z score
    :return:
    """
    for dir_ in os.scandir(f'./data/{data_}/raw sensors'):
        if os.path.isdir(dir_) and dir_.name != 'raw data':
            users_ = getUsers(dir_)
            uIndex = 0
            for user in users_:
                uIndex += 1
                os.system('cls')
                print()
                print(dir_.name)
                print(f'user {user.name}       {uIndex}/{len(users_)}')
                csvFiles = os.scandir(user.path)
                for csv_ in csvFiles:
                    if csv_ == user.name + f'_{threshold_}.csv':
                        os.remove(csv_)


def removeOutliers(df, threshold_):
    """
    Remove outliers using the z score method: between 2 and 3 standard deviation from the mean
    :param df: dataframe in which the outliers must be removed
    :param threshold_: 2 standard deviation or 3 standard deviation from the mean
    :return: df with outliers replaced by the mean
    """
    dfCopy = df.copy()

    zScoreX = np.abs(stats.zscore(dfCopy['x']))
    zScoreY = np.abs(stats.zscore(dfCopy['y']))
    zScoreZ = np.abs(stats.zscore(dfCopy['z']))

    df.loc[zScoreX > threshold_, 'x'] = df['x'].mean()
    df.loc[zScoreY > threshold_, 'y'] = df['y'].mean()
    df.loc[zScoreZ > threshold_, 'z'] = df['z'].mean()

    return df


def poolData(data_='', threshold_=3):
    """
    For each sensor and for each user takes the data and pool the together making 3 dataframe for each user:
    one with threshold 0 -> no outlier removal
             threshold 3 -> outlier removal between 3 standard deviation
             threshold 2 -> outlier removal between 2 standard deviation
    :param data_: dataset to use
    :param threshold_: [0, 2, 3]
    :return: a unique dataframe for user manipulated following the threshold
    """
    clean(data_, threshold_)
    for dir_ in os.scandir(f'./data/{data_}/raw sensors'):
        if os.path.isdir(dir_) and dir_.name != 'raw data':
            users_ = getUsers(dir_)
            uIndex = 0
            for user in users_:
                uIndex += 1
                os.system('cls')
                print()
                print(threshold_)
                print(dir_.name)
                print(f'user {user.name}       {uIndex}/{len(users_)}')
                csvFiles = glob.glob(f'{user.path}/*.csv')
                tmp = []
                for csv_ in csvFiles:
                    if (csv_ == os.path.join(user.path, user.name + '_2.csv')
                            or csv_ == os.path.join(user.path, user.name + '_3.csv')
                            or csv_ == os.path.join(user.path, user.name + '_0.csv')):
                        continue
                    tmpDf = pd.read_csv(csv_)
                    if data_ == 'HMOG':
                        # remove all rows containing phone orientation != 0
                        tmpDf.columns = ['timestamp', 'eventTime', 'activityId', 'x', 'y', 'z', 'orientation']
                        tmpDf.drop(columns=['eventTime', 'activityId', 'orientation'])
                        tmpDf = tmpDf[tmpDf['orientation'] == 0]
                        tmpDf['user'] = user.name
                    tmp.append(tmpDf)
                # cancat all data from accelerometer or gyroscope for a single user and save it in sensor/userFolder
                df = pd.concat(tmp, ignore_index=True)

                if threshold_ != 0:
                    df = removeOutliers(df, threshold_)

                df.to_csv(f'{user.path}/{user.name}_{threshold_}.csv', index=False)


def calculateFeatures(magnitudeSignal_):
    """

    :param magnitudeSignal_: the magnitude of the input window
    :return: time domain features and frequency domain features
    """
    mean = np.mean(magnitudeSignal_)
    stdDev = np.std(magnitudeSignal_)
    variance = np.var(magnitudeSignal_)
    coefficientOfVariation = (stdDev / mean) * 100
    skewness = stats.skew(magnitudeSignal_)
    kurtosis = stats.kurtosis(magnitudeSignal_)
    min_ = min(magnitudeSignal_)
    max_ = max(magnitudeSignal_)
    range_ = max_ - min_
    coefficientOfRange = (range_ / mean) * 100
    q1 = np.percentile(magnitudeSignal_, 25)
    q2 = np.median(magnitudeSignal_)  # median
    q3 = np.percentile(magnitudeSignal_, 75)
    interquartileRange = q3 - q1
    meanAbsoluteDeviation = np.mean(np.abs(np.array(magnitudeSignal_) - mean))
    medianAbsoluteDeviation = np.median(np.abs(np.array(magnitudeSignal_) - q2))

    rms = np.sqrt(np.mean(magnitudeSignal_ ** 2))
    zeroCrossing = np.sum(np.diff(np.sign(magnitudeSignal_)) != 0)
    crestFactor = np.max(magnitudeSignal_) / rms
    peakToPeakAmplitude = np.ptp(magnitudeSignal_)

    # Autocorrelation Coefficients (Lags 1 to 5)
    autoCorrelation1 = np.correlate(magnitudeSignal_, magnitudeSignal_, mode='full')[len(magnitudeSignal_) - 1]
    autoCorrelation2 = np.correlate(magnitudeSignal_, np.roll(magnitudeSignal_, 2), mode='full')[
        len(magnitudeSignal_) - 1]
    autoCorrelation3 = np.correlate(magnitudeSignal_, np.roll(magnitudeSignal_, 3), mode='full')[
        len(magnitudeSignal_) - 1]
    autoCorrelation4 = np.correlate(magnitudeSignal_, np.roll(magnitudeSignal_, 4), mode='full')[
        len(magnitudeSignal_) - 1]
    autoCorrelation5 = np.correlate(magnitudeSignal_, np.roll(magnitudeSignal_, 5), mode='full')[
        len(magnitudeSignal_) - 1]

    signalEntropy = entropy(magnitudeSignal_)
    impulse = np.sum(np.diff(magnitudeSignal_) > 0)

    """print()
    print()
    print(f'Mean:________________________________{mean}')
    print(f'Standard Deviation:__________________{stdDev}')
    print(f'Variance:____________________________{variance}')
    print(f'Coefficient of Variation:____________{coefficientOfVariation}')
    print(f'Skewness:____________________________{skewness}')
    print(f'Kurtosis:____________________________{kurtosis}')
    print(f'Maximum:_____________________________{max_}')
    print(f'Minimum:_____________________________{min_}')
    print(f'Range:_______________________________{range_}')
    print(f'Coefficient of Range:________________{coefficientOfRange}')
    print(f'Q1 (25th percentile):________________{q1}')
    print(f'Median (50th percentile):____________{q2}')
    print(f'Q3 (75th percentile):________________{q3}')
    print(f'Interquartile Range (IQR):___________{interquartileRange}')
    print(f'Mean Absolute Deviation (MAD):_______{meanAbsoluteDeviation}')
    print(f'Median Absolute Deviation (MAD):_____{medianAbsoluteDeviation}')
    print(f'RMS:_________________________________{rms}')
    print(f'Zero Crossing Rate:__________________{zeroCrossing}')
    print(f'Crest Factor:________________________{crestFactor}')
    print(f'Peak-to-Peak Amplitude:______________{peakToPeakAmplitude}')
    print(f'Autocorrelation lag 1:_______________{autoCorrelation1}')
    print(f'Autocorrelation lag 2:_______________{autoCorrelation2}')
    print(f'Autocorrelation lag 3:_______________{autoCorrelation3}')
    print(f'Autocorrelation lag 4:_______________{autoCorrelation4}')
    print(f'Autocorrelation lag 5:_______________{autoCorrelation5}')
    print(f'Signal Entropy:______________________{signalEntropy}')
    print(f'Signal Impulse Indicator:____________{impulse}')"""

    td_features = {
        'mean': mean,
        'std': stdDev,
        'var': variance,
        'coefficientOfVariation': coefficientOfVariation,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'max': max_,
        'min': min_,
        'range': range_,
        'coefficientOfRange': coefficientOfRange,
        'q1': q1,
        'q2': q2,
        'q3': q3,
        'iqr': interquartileRange,
        'meanADev': meanAbsoluteDeviation,
        'medianADev': medianAbsoluteDeviation,
        'rms': rms,
        'zeroCrossing': zeroCrossing,
        'crestFactor': crestFactor,
        'peakToPeakAmplitude': peakToPeakAmplitude,
        'autoCorrelation1': autoCorrelation1,
        'autoCorrelation2': autoCorrelation2,
        'autoCorrelation3': autoCorrelation3,
        'autoCorrelation4': autoCorrelation4,
        'autoCorrelation5': autoCorrelation5,
        'signalEntropy': signalEntropy,
        'impulse': impulse
    }

    samplingRate = 10
    f, psd = welch(magnitudeSignal_, fs=samplingRate, nperseg=10)

    totalPower = np.sum(psd)

    # indices of the op 2 peaks excluding dc component
    peakIndices = np.argsort(psd)[-2:][::-1]

    dominantFrequency = f[peakIndices[0]]
    dominantAmplitude = psd[peakIndices[0]]

    secondDominantFrequency = f[peakIndices[1]]
    secondDominantAmplitude = psd[peakIndices[1]]

    spectralEntropy = -np.sum(psd * np.log2(psd + np.finfo(float).eps))
    spectralRollOff = f[np.argmax(np.cumsum(psd) >= 0.85 * totalPower)]

    meanFrequency = np.sum(f * psd) / totalPower

    cumulativeEnergy = np.cumsum(psd)
    medianFrequency = f[np.argmax(cumulativeEnergy >= np.sum(psd) / 2)]

    geometricMean = np.exp(np.mean(np.log(psd + 1e-12)))  # small constant to avoid log(0)
    arithmeticMean = np.mean(psd)
    spectralFlatness = geometricMean / arithmeticMean

    """print()
    print(f'Spectral Entropy:____________________{spectralEntropy}')
    print(f'Total Power:_________________________{totalPower}')
    print(f'Spectral Roll Off (85%):_____________{spectralRollOff} Hz')
    print(f'Spectral Flatness:___________________{spectralFlatness}')
    print(f'Primary Peak Frequency:______________{dominantFrequency} Hz')
    print(f'Primary Peak Amplitude:______________{dominantAmplitude}')
    print(f'Secondary Peak Frequency:____________{secondDominantFrequency} Hz')
    print(f'Secondary Peak Amplitude:____________{secondDominantAmplitude}')
    print(f'Mean Frequency:______________________{meanFrequency} Hz')
    print(f"Median Frequency:____________________{medianFrequency} Hz")"""

    fd_features = {
        'spectralEntropy': spectralEntropy,
        'totalPower': totalPower,
        'spectralRollOffHz': spectralRollOff,
        'spectralFlatness': spectralFlatness,
        'dominantFrequencyHz': dominantFrequency,
        'dominantAmplitude': dominantAmplitude,
        'secondDominantFrequencyHz': secondDominantFrequency,
        'secondDominantAmplitude': secondDominantAmplitude,
        'meanFrequencyHz': meanFrequency,
        'medianFrequencyHz': medianFrequency
    }

    return td_features, fd_features


def dataPreprocessing(data_='', threshold_=3, windowSize_=5, overlap=0.5):
    """
    Using the technique of the sliding window, with an overlap of 50% extract features form the data
    :param data_: dataset to use
    :param threshold_: threshold of outlier removal [0, 2, 3]
    :param windowSize_: size of the window in seconds
    :param overlap: percentual of overlap
    :return: a file sensorT containing all the extracted features
                sensor: accelerometer, gyroscope
                T: threshold of outlier removal
    """
    skipped = []
    frequency = 10  # Hz 10 examples per second
    if data_ == 'HMOG':
        frequency = 100
    examplesPerWindow = int(frequency * windowSize_)  # number of examples per window
    stepSize = int(examplesPerWindow * overlap)  # start of the next window
    for dir_ in [d for d in os.scandir(f'./data/{data_}/raw sensors') if d.is_dir() and d.name != 'raw data']:

        featuresList = []
        # load all user in the current dir_
        users_ = getUsers(dir_)

        for user in users_:
            # read the csv in dir_ for that user
            userDf = pd.read_csv(os.path.join(user.path, user.name + '_' + str(threshold_) + '.csv'))
            userFrequencyDomainFeatures = []
            userTimeDomainFeatures = []

            j = 0
            allInvalidWindow = True
            totalWindows = 1 + (len(userDf) - windowSize_) // stepSize
            for i in range(0, len(userDf), stepSize):
                j = j + 1
                window = userDf.iloc[i:i + examplesPerWindow]
                if len(window) < examplesPerWindow:
                    continue
                # for each window calculate magnitude
                windowMagnitude = []
                windowMagnitude = np.array(np.sqrt(window['x'] ** 2 + window['y'] ** 2 + window['z'] ** 2))

                """for index, row in window.iterrows():
                    x, y, z = row['x'], row['y'], row['z']
                    windowMagnitude.append(np.sqrt(np.square(x) + np.square(y) + np.square(z)))
                windowMagnitude = np.array(windowMagnitude)"""

                """os.system('cls')
                print(f'window size:___________{windowSize_}')
                print(f'threshold:_____________{threshold_}')
                print(dir_.name)
                print(f'{user.name}____________{j}/{totalWindows}')"""

                # if magnitude contains nan or data points are all equal continue, because this may lead to error
                # calculating some features or to normalization
                if np.isnan(windowMagnitude).any() or np.all(windowMagnitude == windowMagnitude[0]):
                    # save for further analysis
                    toSkip = {
                        'user': user.name,
                        'signal': windowMagnitude
                    }
                    skipped.append(toSkip)
                    continue
                else:
                    allInvalidWindow = False

                td, fd = calculateFeatures(windowMagnitude)
                userTimeDomainFeatures.append(pd.DataFrame(td, index=[j - 1]))
                userFrequencyDomainFeatures.append(pd.DataFrame(fd, index=[j - 1]))

            # union in a single dataframe of time domain and frequency domain features
            # if the whole signal is nan or contains all equal data points skip it _n7ml8hx_

            if allInvalidWindow:
                print(user.name + 'not suitable')
                continue
            userFeatures = pd.concat(
                [
                    pd.concat(userTimeDomainFeatures, ignore_index=True),
                    pd.concat(userFrequencyDomainFeatures, ignore_index=True)
                ], axis=1, join='inner')

            # data normalization
            min_maxScaler = MinMaxScaler()
            scaledNumericData = min_maxScaler.fit_transform(userFeatures)
            scaledNumericData = pd.DataFrame(scaledNumericData, columns=userFeatures.columns)
            # add key value user
            scaledNumericData.insert(0, 'user', user.name)

            # append the dataframe created for this user to the other
            featuresList.append(scaledNumericData)

        # create the final dir_ dataframe and save it
        dir_df = pd.concat(featuresList, ignore_index=True)
        dir_df.to_csv(f'./data/{data_}/{str(windowSize_)} seconds/{dir_.name}_{threshold_}.csv', index=False)

        # save skipped
        if skipped:
            if not os.path.exists(f'./data/{data_}/{str(windowSize_)} seconds/skipped'):
                os.makedirs(f'./data/{data_}/{str(windowSize_)} seconds/skipped')
            with open(f'./data/{data_}/{str(windowSize_)} seconds/skipped/skipped_{dir_.name}_{threshold_}.csv',
                      'w', newline='') as csvFile:

                csvWriter = csv.writer(csvFile)
                csvWriter.writerow(skipped)


def adjustData(windowSize_, data_=''):
    for t_ in [0, 2, 3]:
        accelerometer = pd.read_csv(f'./data/{data_}/{windowSize_} seconds/accelerometer_{t_}.csv')
        gyroscope = pd.read_csv(f'./data/{data_}/{windowSize_} seconds/gyroscope_{t_}.csv')
        userLabel = 'user'
        uniqueUsersAccelerometer = accelerometer[userLabel].unique()
        uniqueUsersGyroscope = gyroscope[userLabel].unique()

        uniqueUsersIntersection = set(uniqueUsersAccelerometer).intersection(uniqueUsersGyroscope)
        toRemoveAccelerometer = []
        toRemoveGyroscope = []
        for user in uniqueUsersAccelerometer:
            if user not in uniqueUsersIntersection:
                toRemoveAccelerometer.append(user)
                accelerometer = accelerometer[accelerometer[userLabel] != user]

        for user in uniqueUsersGyroscope:
            if user not in uniqueUsersIntersection:
                toRemoveGyroscope.append(user)
                gyroscope = gyroscope[gyroscope[userLabel] != user]

        print(t_)
        print(f'{len(toRemoveAccelerometer)} users removed from accelerometer')
        for user in toRemoveAccelerometer:
            print(user)

        print(f'{len(toRemoveGyroscope)} users removed from gyroscope')
        for user in toRemoveGyroscope:
            print(user)

        # cut rows where lectures of one sensor are more than the other
        difference = 0
        usersCountAccelerometer = accelerometer[userLabel].value_counts()
        usersCountGyroscope = gyroscope[userLabel].value_counts()

        allSubDfAccelerometer = []
        allSubDfGyroscope = []
        for user in accelerometer[userLabel].unique():
            countAccelerometer = usersCountAccelerometer[user]
            countGyroscope = usersCountGyroscope[user]

            tmpAccelerometer = accelerometer[accelerometer[userLabel] == user]
            tmpGyroscope = gyroscope[gyroscope[userLabel] == user]

            if countGyroscope != countAccelerometer:
                difference += 1
                print(f'user: {user}')
                print(f'accelerometer readings:_____{countAccelerometer}')
                print(f'gyroscope readings:_________{countGyroscope}')

                if len(tmpAccelerometer) > len(tmpGyroscope):
                    tmpAccelerometer = tmpAccelerometer.head(len(tmpGyroscope))
                elif len(tmpGyroscope) > len(tmpAccelerometer):
                    tmpGyroscope = tmpGyroscope.head(len(tmpAccelerometer))

            allSubDfAccelerometer.append(tmpAccelerometer)
            allSubDfGyroscope.append(tmpGyroscope)

        accelerometer = pd.concat(allSubDfAccelerometer, ignore_index=True)
        gyroscope = pd.concat(allSubDfGyroscope, ignore_index=True)

        print(f'number of users with different readings: {difference}')

        if difference == 0:
            continue

        print('after equalization:')
        difference = 0
        usersCountAccelerometer = accelerometer[userLabel].value_counts()
        usersCountGyroscope = gyroscope[userLabel].value_counts()
        for user in accelerometer[userLabel].unique():
            countAccelerometer = usersCountAccelerometer[user]
            countGyroscope = usersCountGyroscope[user]

            tmpAccelerometer = accelerometer[accelerometer[userLabel] == user]
            tmpGyroscope = gyroscope[gyroscope[userLabel] == user]

            if countGyroscope != countAccelerometer:
                difference += 1
                print(f'user: {user}')
                print(f'accelerometer readings:_____{countAccelerometer}')
                print(f'gyroscope readings:_________{countGyroscope}')

        print(f'number of users with different readings: {difference}')
        accelerometer.to_csv(f'./data/{data_}/{windowSize_} seconds/accelerometer_{t_}.csv', index=False)
        gyroscope.to_csv(f'./data/{data_}/{windowSize_} seconds/gyroscope_{t_}.csv', index=False)


if __name__ == "__main__":
    data = 'h'  # h or b
    if data == 'h':
        data = 'HMOG'
        print('loading all sensor data')
        loadAllSensorData(data_=data)
        processes = []
        # 3 outlier removal between 3 standard deviation
        # 2                         2
        # 0 no outlier removal
        for threshold in [0, 2, 3]:
            process = multiprocessing.Process(target=poolData, args=(data, threshold))
            processes.append(process)
            process.start()
            if threshold != 0:
                print(f'pooling data and outlier removal within {threshold} standard deviations')
            else:
                print('pooling data with no outlier removal')
        for process in processes:
            process.join()
        print('pooling data and outlier removal within 3 standard deviations')
        poolData(data_=data, threshold_=3)
        print('pooling data and outlier removal within 2 standard deviations')
        poolData(data_=data, threshold_=2)
        print('pooling data with no outlier removal')
        poolData(data_=data, threshold_=0)
        for windowSize in [5, 4, 3, 2, 1, 0.5, 0.2]:
            processes = []
            print(f'window size: {windowSize}')
            for threshold in [0, 2, 3]:
                process = multiprocessing.Process(target=dataPreprocessing, args=(data, threshold, windowSize))
                processes.append(process)
                process.start()
                print(f'subprocess with threshold {threshold} started.')

            for process in processes:
                process.join()
            print(f'finished for {windowSize} seconds long signal')
            print('adjusting data')
            adjustData(windowSize, data)
    elif data == 'b':
        data = 'BrainRun'
        print('empty file removal')
        removeEmpty()

        print('data analysis')
        avg, valid = analyzeData()
        # average number of files per user: 71
        # number of user that have a number of files >= average number of files: 158

        print('balance data')
        removeLTorGTaverage(avg, valid)

        print('data loading')
        loadAllSensorData(data_=data)

        print('pooling data and outlier removal within 3 standard deviations')
        poolData(data_=data, threshold_=3)
        print('pooling data and outlier removal within 2 standard deviations')
        poolData(data_=data, threshold_=2)
        print('pooling data with no outlier removal')
        poolData(data_=data, threshold_=0)
        for windowSize in [5, 4, 3, 2]:
            processes = []
            print(f'window size: {windowSize}')
            for threshold in [0, 2, 3]:
                process = multiprocessing.Process(target=dataPreprocessing, args=(data, threshold, windowSize))
                processes.append(process)
                process.start()
                print(f'subprocess with threshold {threshold} started.')
                # dataPreprocessing(data_=data, threshold_=threshold, windowSize_=windowSize)

            for process in processes:
                process.join()
            print(f'finished for {windowSize} seconds long signal')
            print('adjusting data')
            adjustData(windowSize, data)
        # the user n7ml8hx has been removed from all t, all windows contain the same value for the magnitude signal
        # some rows have been skipped for nan values or same data point, making feature calculation impossible
        # the number of skipped rows is different for each t."""
        print('finish')
