import wfdb
import numpy as np
import os

DATA_PATH = r'C:\Users\rumon\Downloads\wearable-exercise-frailty\acc'

ACTIVITY_MAP = {
    'STAIR': 1, '6MWT': 2,
    'GAIT_ANALYSIS': 3, 'TUG': 4, 'VELO': 5,
}

ACTIVITY_NAMES = {
    0: 'sitting', 1: 'stair_climbing', 2: 'walking_6mwt',
    3: 'treadmill_walking', 4: 'timed_up_and_go', 5: 'cycling'
}

SOURCE_RATE = 200
TARGET_RATE = 30
WINDOW_SIZE = TARGET_RATE * 10

def resample(signal, from_rate, to_rate):
    step  = from_rate / to_rate
    idx   = np.arange(0, len(signal), step).astype(int)
    idx   = idx[idx < len(signal)]
    return signal[idx]

def extract_windows(signal, start_sec, duration_sec, fs):
    start  = int(start_sec * fs)
    end    = min(int((start_sec + duration_sec) * fs), len(signal))
    seg    = resample(signal[start:end], fs, TARGET_RATE)
    return [seg[i:i+WINDOW_SIZE]
            for i in range(0, len(seg) - WINDOW_SIZE, WINDOW_SIZE)]

hea_files = sorted([f.replace('.hea','')
                    for f in os.listdir(DATA_PATH)
                    if f.endswith('.hea')])

all_X, all_y, all_pids = [], [], []
skipped = 0

print(f"Processing {len(hea_files)} records...")
for fname in hea_files:
    fpath      = os.path.join(DATA_PATH, fname)
    patient_id = fname.split('_')[0]   # e.g. '001', '012'

    try:
        record    = wfdb.rdrecord(fpath)
        signal    = record.p_signal
        fs        = record.fs
        total_sec = len(signal) / fs

        try:
            ann = wfdb.rdann(fpath, 'atr')
        except:
            skipped += 1
            continue

        ann_times = [(s/fs, n.strip())
                     for s, n in zip(ann.sample, ann.aux_note)
                     if n.strip() in ACTIVITY_MAP]

        if len(ann_times) == 0:
            skipped += 1
            continue

        # Sitting windows before first activity
        first_t = ann_times[0][0]
        if first_t > 60:
            for w in extract_windows(signal, first_t-60, 60, fs):
                all_X.append(w)
                all_y.append(0)
                all_pids.append(patient_id)

        # Exercise windows
        for i, (start_t, note) in enumerate(ann_times):
            act_id   = ACTIVITY_MAP[note]
            duration = min(
                (ann_times[i+1][0] - start_t
                 if i+1 < len(ann_times)
                 else total_sec - start_t),
                300
            )
            for w in extract_windows(signal, start_t, duration, fs):
                all_X.append(w)
                all_y.append(act_id)
                all_pids.append(patient_id)

        print(f"  ✅ {fname} (patient {patient_id}): "
              f"{len(ann_times)} activities")

    except Exception as e:
        print(f"  ❌ {fname}: {e}")
        skipped += 1

X       = np.array(all_X)
y       = np.array(all_y)
pids    = np.array(all_pids)

print(f"\n{'='*50}")
print(f"Total windows   : {X.shape[0]}")
print(f"Unique patients : {len(np.unique(pids))}")
print(f"Skipped         : {skipped}")
print(f"\nActivity distribution:")
for aid, aname in ACTIVITY_NAMES.items():
    print(f"  {aname:<22}: {np.sum(y==aid):4d} windows")

np.save('data/physionet_X.npy',    X)
np.save('data/physionet_y.npy',    y)
np.save('data/physionet_pids.npy', pids)
print("\n✅ Saved with patient IDs!")
print(f"Unique patient IDs: {sorted(np.unique(pids))}")