import scipy.io as sio
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── 1. LOAD ANNOTATIONS ──────────────────────────────────────────
print("Loading .mat file... (this takes ~30 seconds)")
mat = sio.loadmat(
    'mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat',
    struct_as_record=False,
    squeeze_me=True
)
release = mat['RELEASE']
print("Loaded.")

# ── 2. JOINT ID MAP ──────────────────────────────────────────────
JOINT_NAMES = {
    0: 'r_ankle',    1: 'r_knee',      2: 'r_hip',
    3: 'l_hip',      4: 'l_knee',      5: 'l_ankle',
    6: 'pelvis',     7: 'thorax',      8: 'upper_neck',
    9: 'head_top',   10: 'r_wrist',    11: 'r_elbow',
    12: 'r_shoulder', 13: 'l_shoulder', 14: 'l_elbow',
    15: 'l_wrist'
}

SITTING_KEYWORDS = [
    'sitting', 'computer', 'desk', 'office',
    'typing', 'reading', 'writing', 'working'
]

# ── 3. HELPER FUNCTIONS ───────────────────────────────────────────
def get_image_name(release, i):
    try:
        return str(release.annolist[i].image.name)
    except:
        return None

def is_train(release, i):
    try:
        return bool(release.img_train[i])
    except:
        return False

def get_activity(release, i):
    try:
        act = release.act[i]
        return {
            'act_name': str(act.act_name),
            'cat_name': str(act.cat_name),
        }
    except:
        return None

def is_sitting(act):
    if act is None:
        return False
    text = act['act_name'].lower() + ' ' + act['cat_name'].lower()
    return any(kw in text for kw in SITTING_KEYWORDS)

def get_joints(release, i):
    try:
        annorect = release.annolist[i].annorect
        if not hasattr(annorect, '__len__'):
            annorect = [annorect]
        points = annorect[0].annopoints.point
        if not hasattr(points, '__len__'):
            points = [points]
        joints = {}
        for p in points:
            jid = int(p.id)
            joints[JOINT_NAMES[jid]] = np.array([float(p.x), float(p.y)])
        return joints
    except:
        return None

def compute_angle(v1, v2):
    denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
    cos = np.dot(v1, v2) / denom
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

def label_posture(joints):
    needed = ['upper_neck', 'head_top', 'r_shoulder', 'l_shoulder', 'thorax', 'pelvis']
    if not all(k in joints for k in needed):
        return None

    up = np.array([0, -1])

    neck_vec  = joints['head_top']   - joints['upper_neck']
    spine_vec = joints['thorax']     - joints['pelvis']

    neck_angle    = compute_angle(neck_vec, up)
    spine_angle   = compute_angle(spine_vec, up)
    shoulder_tilt = abs(joints['r_shoulder'][1] - joints['l_shoulder'][1])

    if spine_angle > 20:
        return 'slouching'
    elif neck_angle > 20:
        return 'neck_forward'
    elif shoulder_tilt > 15:
        return 'leaning'
    else:
        return 'correct'

# ── 4. MAIN LOOP ──────────────────────────────────────────────────
print("Parsing annotations...")
n = len(release.annolist)
rows = []

for i in range(n):
    if not is_train(release, i):
        continue
    act = get_activity(release, i)
    if not is_sitting(act):
        continue
    joints = get_joints(release, i)
    if joints is None:
        continue
    label = label_posture(joints)
    if label is None:
        continue

    up = np.array([0, -1])
    neck_angle = compute_angle(
        joints.get('head_top', np.zeros(2)) - joints.get('upper_neck', np.zeros(2)), up)
    spine_angle = compute_angle(
        joints.get('thorax', np.zeros(2)) - joints.get('pelvis', np.zeros(2)), up)
    shoulder_tilt = abs(
        joints.get('r_shoulder', np.zeros(2))[1] - joints.get('l_shoulder', np.zeros(2))[1])

    rows.append({
        'img_path':      os.path.join('images', get_image_name(release, i)),
        'act_name':      act['act_name'],
        'neck_angle':    round(neck_angle, 2),
        'spine_angle':   round(spine_angle, 2),
        'shoulder_tilt': round(shoulder_tilt, 2),
        'label':         label
    })

df = pd.DataFrame(rows)
df.to_csv('mpii_posture_labeled.csv', index=False)

print(f"\nDone. Total samples: {len(df)}")
print(df['label'].value_counts())

# ── 5. TRAIN CLASSIFIER ───────────────────────────────────────────
print("\nTraining classifier...")
X = df[['neck_angle', 'spine_angle', 'shoulder_tilt']].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

print(f"\nAccuracy: {clf.score(X_test, y_test):.2%}")
print(classification_report(y_test, clf.predict(X_test)))

joblib.dump(clf, 'posture_classifier.pkl')
print("\nSaved: posture_classifier.pkl")