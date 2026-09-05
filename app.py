from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import joblib
import os
from pyproj import Transformer
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# ─── Mappings for display ───────────────────────────────────────────
WEATHER_MAP = {
    0: "Fine without high winds",
    1: "Fine with high winds",
    2: "Raining with high winds",
    3: "Fog or mist",
    4: "Snowing with high winds",
    5: "Raining"
}
ROAD_CLASS_MAP = {
    0: "Unclassified",
    1: "A",
    2: "B",
    3: "Motorway"
}

# ─── CRS transformer ─────────────────────────────────────────────────
transformer = Transformer.from_crs('EPSG:27700', 'EPSG:4326', always_xy=True)

# ─── Load & preprocess 2017–2020 data ─────────────────────────────────
files = [
    os.path.join('DataSet', 'Accident 2017.xls'),
    os.path.join('DataSet', 'accident 2018.xls'),
    os.path.join('DataSet', 'Accidents 2019.xlsx'),
    os.path.join('DataSet', 'Calderdale Collisions 2020.xlsx')
]
dfs = []
for f in files:
    print(f"Loading {f} …")
    dfs.append(pd.read_excel(f, engine='xlrd' if f.endswith('.xls') else 'openpyxl'))
df = pd.concat(dfs, ignore_index=True)
print(f"Total rows: {len(df)}")

df['Accident Date'] = pd.to_datetime(
    df['Accident Date'], dayfirst=True, errors='coerce'
)
df['Hour'] = df['Time (24hr)'].fillna(0).astype(int) // 100

for col in ['Weather Conditions','1st Road Class','Road Surface','Lighting Conditions']:
    df[col + '_code'] = df.get(col, pd.Series()).astype('category').cat.codes

if 'Grid Ref: Easting' in df and 'Grid Ref: Northing' in df:
    east  = pd.to_numeric(df['Grid Ref: Easting'], errors='coerce').fillna(0).to_numpy()
    north = pd.to_numeric(df['Grid Ref: Northing'], errors='coerce').fillna(0).to_numpy()
    lons, lats = transformer.transform(east, north)
    df['lng'], df['lat'] = lons, lats
print(f"▶ Valid coords: {df['lat'].notnull().sum()}/{len(df)} rows")

# ─── Load the trained model ───────────────────────────────────────────
def load_model():
    if os.path.exists('model.pkl'):
        print("✅ Loaded model.pkl")
        return joblib.load('model.pkl')
    print("⚠️  Warning: model.pkl not found")
    return None

model = load_model()

# ─── Routes ──────────────────────────────────────────────────────────

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/years')
def years():
    ys = df['Accident Date'].dt.year.dropna().astype(int).unique().tolist()
    ys.sort()
    return jsonify(ys)

@app.route('/hotspots')
def hotspots():
    sub = df.copy()
    year = request.args.get('year', type=int)
    wc   = request.args.get('weather_code', type=int)
    rc   = request.args.get('road_class_code', type=int)
    if year is not None:
        sub = sub[sub['Accident Date'].dt.year == year]
    if wc is not None:
        sub = sub[sub['Weather Conditions_code'] == wc]
    if rc is not None:
        sub = sub[sub['1st Road Class_code'] == rc]
    pts = sub[['lat','lng']].dropna().to_dict(orient='records')
    return jsonify(pts)

@app.route('/details')
def details():
    lat = request.args.get('lat', type=float)
    lng = request.args.get('lng', type=float)
    mask = ((df['lat'] - lat)**2 + (df['lng'] - lng)**2) < (0.01**2)
    near = df[mask]
    return jsonify({
        'total_accidents': len(near),
        'road_surface':    near['Road Surface'].fillna('Unknown').value_counts().to_dict(),
        'casualty_by_severity': {
            **pd.concat([near[f'Casualty Severity {i}'] for i in range(1,8)])
                  .dropna().astype(str).value_counts().to_dict()
        }
    })

@app.route('/predict')
def predict():
    # require lat & lng
    lat = request.args.get('lat', type=float)
    lng = request.args.get('lng', type=float)
    if lat is None or lng is None:
        return jsonify({ 'error': 'lat and lng are required' }), 400

    if model is None:
        return jsonify({ 'error': 'model not loaded' }), 500

    # find nearby accidents
    mask = ((df['lat'] - lat)**2 + (df['lng'] - lng)**2) < (0.01**2)
    near = df[mask]

    # get filters & hour
    wc = request.args.get('weather_code', type=int)
    rc = request.args.get('road_class_code', type=int)
    hr = request.args.get('hour', type=int, default=datetime.datetime.now().hour)

    # derive model features
    surf_code  = int(near['Road Surface_code'].mode()[0])        if not near.empty else 0
    light_code = int(near['Lighting Conditions_code'].mode()[0]) if not near.empty else 0

    raw = {
        'Weather Conditions_code':   wc if wc is not None else
                                      (int(near['Weather Conditions_code'].mode()[0]) if not near.empty else 0),
        '1st Road Class_code':       rc if rc is not None else
                                      (int(near['1st Road Class_code'].mode()[0])     if not near.empty else 0),
        'Road Surface_code':         surf_code,
        'Lighting Conditions_code':  light_code,
        'Hour':                      hr,
    }

    # logging
    print("🔍 [PREDICT] raw features:", raw)

    X = pd.DataFrame([raw])
    try:
        X = X[model.feature_names_in_]
    except Exception as e:
        print("❌ [PREDICT] feature reorder failed:", e)
        return jsonify({ 'error': f'Bad features: {e}' }), 400

    print("🔧 [PREDICT] feature matrix:", X.to_dict(orient='records')[0])

    pred = model.predict(X)[0]
    print("✅ [PREDICT] prediction:", pred)

    # human‑readable labels
    weather_label     = WEATHER_MAP.get(raw['Weather Conditions_code'],
                                        (near['Weather Conditions'].mode()[0] if not near.empty else "Unknown"))
    road_class_label  = ROAD_CLASS_MAP.get(raw['1st Road Class_code'],
                                           (near['1st Road Class'].mode()[0] if not near.empty else "Unknown"))
    road_surface_lbl  = near['Road Surface'].mode()[0]       if not near.empty else "Unknown"
    lighting_lbl      = near['Lighting Conditions'].mode()[0] if not near.empty else "Unknown"

    return jsonify({
        'prediction': str(pred),
        'conditions': {
            'weather':     weather_label,
            'road_class':  road_class_label,
            'road_surface':road_surface_lbl,
            'lighting':    lighting_lbl,
            'hour':        raw['Hour']
        }
    })

def make_png(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

@app.route('/plot/severity.png')
def plot_severity():
    lat = request.args.get('lat', type=float)
    lng = request.args.get('lng', type=float)
    mask = ((df['lat'] - lat)**2 + (df['lng'] - lng)**2) < (0.01**2)
    near = df[mask]
    sev = pd.concat([near[f'Casualty Severity {i}'] for i in range(1,8)]).dropna().astype(str)
    counts = sev.value_counts()
    fig, ax = plt.subplots(figsize=(6,3))
    counts.plot(kind='bar', ax=ax, color=['#4caf50','#ff9800','#f44336'])
    ax.set_ylabel('# of casualties')
    ax.set_title('Casualty Severity Breakdown')
    return send_file(make_png(fig), mimetype='image/png')

@app.route('/plot/road.png')
def plot_road():
    lat = request.args.get('lat', type=float)
    lng = request.args.get('lng', type=float)
    mask = ((df['lat'] - lat)**2 + (df['lng'] - lng)**2) < (0.01**2)
    near = df[mask]
    counts = near['Road Surface'].fillna('Unknown').value_counts()
    fig, ax = plt.subplots(figsize=(6,4))
    counts.plot(kind='pie', ax=ax, autopct='%1.1f%%')
    ax.set_ylabel('')
    ax.set_title('Road Surface Breakdown')
    return send_file(make_png(fig), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
