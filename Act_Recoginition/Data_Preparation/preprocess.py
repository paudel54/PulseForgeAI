import pandas as pd
import numpy as np

columns = ['timestamp', 'activity_id', 'heart_rate']
imu_parts = ['temp', 'acc_x1', 'acc_y1', 'acc_z1',
             'acc_x2', 'acc_y2', 'acc_z2',
             'gyro_x', 'gyro_y', 'gyro_z',
             'mag_x', 'mag_y', 'mag_z',
             'orientation1', 'orientation2',
             'orientation3', 'orientation4']
for sensor in ['chest', 'hand', 'ankle']:
    for part in imu_parts:
        columns.append(f'{sensor}_{part}')

ACC_COLS = ['hand_acc_x1', 'hand_acc_y1', 'hand_acc_z1']
SAMPLE_RATE = 100
TARGET_RATE = 30
WINDOW_SIZE = TARGET_RATE * 10

activity_labels = {
    1: 'lying', 2: 'sitting', 3: 'standing',
    4: 'walking', 5: 'running', 6: 'cycling',
    7: 'nordic_walking'
}

def load_subject(filepath):
    df = pd.read_csv(filepath, sep=' ', header=None, names=columns)
    df = df[df['activity_id'] != 0]
    df = df[df['activity_id'].isin(activity_labels.keys())]
    df = df[['timestamp', 'activity_id'] + ACC_COLS]
    df = df.dropna()
    return df

def resample_to_30hz(df):
    step = int(SAMPLE_RATE / TARGET_RATE)
    return df.iloc[::step].reset_index(drop=True)

def create_windows(df):
    X, y = [], []
    data = df[ACC_COLS].values
    labels = df['activity_id'].values
    for start in range(0, len(data) - WINDOW_SIZE, WINDOW_SIZE):
        end = start + WINDOW_SIZE
        window = data[start:end]
        window_labels = labels[start:end]
        if len(set(window_labels)) == 1:
            X.append(window)
            y.append(window_labels[0])
    return np.array(X), np.array(y)

DATA_PATH = 'data/PAMAP_DATA/PAMAP2_Dataset/Protocol/'
all_X, all_y, all_subjects = [], [], []

print("Processing subjects...")
for i in range(101, 110):
    filepath = f'{DATA_PATH}subject{i}.dat'
    print(f"  Loading subject{i}...")
    df = load_subject(filepath)
    df = resample_to_30hz(df)
    X, y = create_windows(df)
    if len(y) == 0:
        print(f"  Subject{i}: skipped (no windows)")
        continue
    all_X.append(X)
    all_y.append(y)
    all_subjects.extend([i] * len(y))
    print(f"  Subject{i}: {len(y)} windows")

X = np.concatenate(all_X)
y = np.concatenate(all_y)
subjects = np.array(all_subjects)

print(f"\nTotal windows: {X.shape}")
print("\nActivity distribution:")
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {activity_labels[u]}: {c} windows")

np.save('data/X.npy', X)
np.save('data/y.npy', y)
np.save('data/subjects.npy', subjects)
print("\nData saved!")
