# temp.py

import pandas as pd
import joblib
import os
from pyproj import Transformer

# ─── Load & concat your 2017–2020 data (same as in app) ────────────
files = [
    'DataSet/Accident 2017.xls',
    'DataSet/accident 2018.xls',
    'DataSet/Accidents 2019.xlsx',
    'DataSet/Calderdale Collisions 2020.xlsx'
]
dfs = []
for f in files:
    engine = 'xlrd' if f.lower().endswith('.xls') else 'openpyxl'
    print(f"Loading {f} …")
    dfs.append(pd.read_excel(f, engine=engine))
df = pd.concat(dfs, ignore_index=True)
print(f"▶ Total rows: {len(df)}")

# ─── Preprocess exactly as during training ──────────────────────────
df['Accident Date'] = pd.to_datetime(df['Accident Date'], dayfirst=True, errors='coerce')
df['Hour'] = df['Time (24hr)'].fillna(0).astype(int) // 100
for col in ['Weather Conditions','1st Road Class','Road Surface','Lighting Conditions']:
    df[col + '_code'] = df.get(col, pd.Series()).astype('category').cat.codes

# ─── Load your model.pkl ────────────────────────────────────────────
model_path = 'model.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"{model_path} not found")
model = joblib.load(model_path)
print(f"✅ Loaded {model_path}")

# ─── Inspect exactly what feature names (and order) the model expects ─
print("⚙️  model.feature_names_in_ =", list(model.feature_names_in_))

# ─── Now build a toy input using those exact names & order ──────────
#    Here’s your “hard‑coded” example:
raw = {
    'Hour': 14,
    'Weather Conditions_code': 0,
    '1st Road Class_code': 3,
    'Road Surface_code': 0,
    'Lighting Conditions_code': 1
}

# Turn into DataFrame, then reorder to match feature_names_in_
X = pd.DataFrame([raw])
X = X[model.feature_names_in_]

print("🔍 Test input (reordered):")
print(X)

pred = model.predict(X)[0]
print("🔮 Prediction:", pred)
