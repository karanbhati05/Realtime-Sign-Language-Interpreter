import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from ensemble import EnsembleModel
import joblib

print("Loading dataset...")
df    = pd.read_csv("dataset.csv")
X_raw = np.nan_to_num(df.iloc[:, :-1].values, nan=0.0)
y     = df.iloc[:, -1].values
print(f"Shape: {X_raw.shape} | Classes: {len(set(y))}")

# Dataset has 252 features:
# 0:63   = mean x,y,z per landmark
# 63:126 = std x,y,z per landmark
# 126:189 = motion (last - first frame)
# 189:252 = mid frame

mean_f = X_raw[:, :63]
mean_r = mean_f.reshape(-1, 21, 3)

# Fingertip distances from wrist
fingertips = [4, 8, 12, 16, 20]
wrist      = mean_r[:, 0, :]
tip_dists  = np.array([
    np.linalg.norm(mean_r[:, t, :] - wrist, axis=1)
    for t in fingertips
]).T  # (N,5)

# Finger curl angles
def angle_batch(a, b, c):
    ba  = a - b; bc = c - b
    cos = np.einsum('ij,ij->i', ba, bc) / (
          np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-8)
    return np.arccos(np.clip(cos, -1, 1))

finger_joints = [(1,2,3),(5,6,7),(9,10,11),(13,14,15),(17,18,19)]
curl_angles   = np.column_stack([
    angle_batch(mean_r[:,a,:], mean_r[:,b,:], mean_r[:,c,:])
    for a,b,c in finger_joints
])  # (N,5)

# Inter-fingertip distances
pairs = [(4,8),(8,12),(12,16),(16,20),(4,20),(4,12),(8,20)]
inter = np.column_stack([
    np.linalg.norm(mean_r[:,p,:] - mean_r[:,q,:], axis=1)
    for p,q in pairs
])  # (N,7)

# Palm dimensions
palm_w = np.linalg.norm(mean_r[:,5,:]  - mean_r[:,17,:], axis=1, keepdims=True)
palm_h = np.linalg.norm(mean_r[:,0,:]  - mean_r[:,9,:],  axis=1, keepdims=True)
spread = mean_r[:, fingertips, :].std(axis=1)  # (N,3)

X = np.hstack([X_raw, tip_dists, curl_angles, inter, palm_w, palm_h, spread])
print(f"Total features after engineering: {X.shape[1]}")

le    = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# MLP
print("\nTraining MLP...")
mlp = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation='relu',
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        learning_rate='adaptive',
        n_iter_no_change=20,
        verbose=False
    ))
])
mlp.fit(X_train, y_train)
mlp_acc = (mlp.predict(X_test) == y_test).mean() * 100
print(f"MLP accuracy:  {mlp_acc:.1f}%")

# Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=500,
    max_features='sqrt',
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_acc = (rf.predict(X_test) == y_test).mean() * 100
print(f"RF accuracy:   {rf_acc:.1f}%")

# Ensemble
print("Building ensemble...")
ensemble = EnsembleModel(mlp, rf, w1=0.6, w2=0.4)
y_pred   = ensemble.predict(X_test)
acc      = (y_pred == y_test).mean() * 100

print("\n--- Final Evaluation ---")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print(f"Ensemble accuracy: {acc:.1f}%")

joblib.dump(ensemble, "sign_model.pkl")
joblib.dump(le,       "label_encoder.pkl")
with open("model_config.txt", "w") as f:
    f.write(str(X.shape[1]))

print("\nSaved: sign_model.pkl | label_encoder.pkl | model_config.txt")
