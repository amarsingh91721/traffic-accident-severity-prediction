import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings

# 1) Load all Excel files
data_dir = Path('DataSet')
files = list(data_dir.glob('*.xls*'))  # .xls or .xlsx

dfs = []
for f in files:
    ext = f.suffix.lower()
    engine = 'xlrd' if ext == '.xls' else 'openpyxl'
    print(f"Loading {f.name} …")
    dfs.append(pd.read_excel(f, engine=engine))

df = pd.concat(dfs, ignore_index=True)
print(f"Total rows: {len(df)}")

# 2) Parse date and extract hour
df['Accident Date'] = pd.to_datetime(df['Accident Date'], dayfirst=True, errors='coerce')
df['Hour'] = pd.to_numeric(df['Time (24hr)'], errors='coerce').fillna(0).astype(int) // 100

# 3) Encode categorical features
for col in ['Weather Conditions', '1st Road Class', 'Road Surface', 'Lighting Conditions']:
    if col in df.columns:
        df[col + '_code'] = df[col].astype('category').cat.codes
    else:
        df[col + '_code'] = 0  # fallback if missing

# 4) Find and rename severity column
sev_cols = [c for c in df.columns if c == 'Casualty Severity']
if not sev_cols:
    sev_cols = [c for c in df.columns if c.startswith('Casualty Severity ')]
if not sev_cols:
    raise KeyError("No 'Casualty Severity' column found")

label_col = sev_cols[0]
df = df.dropna(subset=[label_col]).rename(columns={label_col: 'Accident_Severity'})

print("Label distribution:\n", df['Accident_Severity'].value_counts())

# 5) Define features and label
X = df[[
    'Weather Conditions_code',
    '1st Road Class_code',
    'Road Surface_code',
    'Lighting Conditions_code',
    'Hour'
]]
y = df['Accident_Severity']

# 6) Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.06, random_state=42, stratify=y
)

# 7) Train model
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 8) Evaluate
y_pred = model.predict(X_test)

# Suppress warnings about undefined metrics
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

print("\nClassification report on hold‑out set:")
print(classification_report(y_test, y_pred, zero_division=0))

# 9) Save model
joblib.dump(model, 'model.pkl')
print("\n✅  Saved RandomForest to model.pkl")
