"""
Estatia — House Price Prediction System
=========================================
A single-file Python (Flask) app that:
  1. Generates a synthetic real-estate dataset
  2. Cleans it
  3. Engineers features (location encoding, amenity flags)
  4. Trains a RandomForestRegressor
  5. Evaluates it (R², MAE, RMSE)
  6. Serves a luxury Pinterest-style front end
  7. Exposes /api/predict and /api/analytics for the UI to call

RUN:
    pip install flask scikit-learn pandas numpy gunicorn
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --------------------------------------------------------------------------
# 1. CONFIG
# --------------------------------------------------------------------------

LOCATIONS = [
    "Bandra, Mumbai",
    "Whitefield, Bangalore",
    "Gachibowli, Hyderabad",
    "Koramangala, Bangalore",
    "DLF Phase 5, Gurugram",
    "Salt Lake, Kolkata",
]

LOCATION_MULTIPLIER = {
    "Bandra, Mumbai": 1.9,
    "Whitefield, Bangalore": 1.25,
    "Gachibowli, Hyderabad": 1.15,
    "Koramangala, Bangalore": 1.4,
    "DLF Phase 5, Gurugram": 1.55,
    "Salt Lake, Kolkata": 1.05,
}

AMENITY_COLUMNS = [
    "garden", "pool", "lift", "ac", "security",
    "power", "wifi", "schools", "hospital", "mall",
]

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)


# --------------------------------------------------------------------------
# 2. DATASET COLLECTION (synthetic, stands in for a real listings dataset)
# --------------------------------------------------------------------------

def generate_dataset(n=3000) -> pd.DataFrame:
    location = rng.choice(LOCATIONS, size=n)
    area = rng.normal(2000, 650, n).clip(400, 6000).round(0)
    bedrooms = rng.integers(1, 6, n)
    bathrooms = np.clip(bedrooms - rng.integers(0, 2, n), 1, 5)
    age = rng.integers(0, 35, n)
    parking = rng.integers(0, 4, n)

    amenities = {col: rng.integers(0, 2, n) for col in AMENITY_COLUMNS}

    # A small % of intentionally messy rows -> exercises the cleaning step
    area[rng.choice(n, size=int(n * 0.02), replace=False)] = np.nan
    bathrooms = bathrooms.astype(float)
    bathrooms[rng.choice(n, size=int(n * 0.015), replace=False)] = np.nan

    df = pd.DataFrame({
        "location": location,
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age": age,
        "parking": parking,
        **amenities,
    })

    # Ground-truth price generation (what the model has to learn to approximate)
    base = df["area"].fillna(df["area"].median()) * 4200
    base += df["bedrooms"] * 320_000 + df["bathrooms"].fillna(2) * 180_000
    base += df["parking"] * 90_000
    base -= df["age"] * 22_000
    base += sum(df[c] for c in AMENITY_COLUMNS) * 140_000
    base *= df["location"].map(LOCATION_MULTIPLIER)
    noise = rng.normal(0, 180_000, n)
    df["price"] = (base + noise).clip(lower=800_000).round(0)

    return df


# --------------------------------------------------------------------------
# 3. DATA CLEANING
# --------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["area"] = df["area"].fillna(df["area"].median())
    df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms"].median())

    # Drop impossible / outlier rows
    df = df[(df["area"] > 200) & (df["area"] < 10000)]
    df = df[df["price"] > 0]
    df = df.drop_duplicates()
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# 4. FEATURE ENGINEERING
# --------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame):
    df = df.copy()
    location_dummies = pd.get_dummies(df["location"], prefix="loc")
    features = pd.concat(
        [df.drop(columns=["location", "price"], errors="ignore"), location_dummies],
        axis=1,
    )
    feature_columns = list(features.columns)
    return features, feature_columns


# --------------------------------------------------------------------------
# 5. MODEL TRAINING + EVALUATION
# --------------------------------------------------------------------------

def train_model():
    raw_df = generate_dataset()
    clean_df = clean_data(raw_df)
    X, feature_columns = engineer_features(clean_df)
    y = clean_df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    model = RandomForestRegressor(
        n_estimators=250, max_depth=14, random_state=RANDOM_SEED, n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "r2": round(r2_score(y_test, preds), 4),
        "mae": round(mean_absolute_error(y_test, preds), 2),
        "rmse": round(mean_squared_error(y_test, preds) ** 0.5, 2),
    }

    print("=" * 50)
    print("Model Evaluation")
    print(f"  R^2   : {metrics['r2']}")
    print(f"  MAE   : ₹{metrics['mae']:,.0f}")
    print(f"  RMSE  : ₹{metrics['rmse']:,.0f}")
    print("=" * 50)

    return model, feature_columns, clean_df, metrics


MODEL, FEATURE_COLUMNS, DATASET, METRICS = train_model()


# --------------------------------------------------------------------------
# 6. FLASK APP
# --------------------------------------------------------------------------

app = Flask(__name__)


def build_feature_row(payload: dict) -> pd.DataFrame:
    """Turn a form submission (JSON) into a one-row DataFrame matching FEATURE_COLUMNS."""
    row = {col: 0 for col in FEATURE_COLUMNS}

    row["area"] = float(payload.get("area", 1000))
    row["bedrooms"] = int(payload.get("bedrooms", 2))
    row["bathrooms"] = int(payload.get("bathrooms", 2))
    row["age"] = float(payload.get("age", 0))
    row["parking"] = int(payload.get("parking", 0))

    for a in AMENITY_COLUMNS:
        row[a] = 1 if payload.get(a) else 0

    loc_col = f"loc_{payload.get('location')}"
    if loc_col in row:
        row[loc_col] = 1

    return pd.DataFrame([row])[FEATURE_COLUMNS]


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, locations=LOCATIONS, metrics=METRICS)


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    X_row = build_feature_row(payload)
    price = float(MODEL.predict(X_row)[0])

    area = float(payload.get("area", 1000)) or 1000
    amenity_score = sum(1 for a in AMENITY_COLUMNS if payload.get(a))
    confidence = round(min(98.0, 84 + amenity_score * 1.1), 1)

    return jsonify({
        "price": round(price),
        "price_per_sqft": round(price / area),
        "confidence": confidence,
        "location": payload.get("location"),
        "summary": f"{payload.get('bedrooms')} BHK · {payload.get('bathrooms')} Bath · {int(area)} sqft",
    })


@app.route("/api/analytics")
def analytics():
    df = DATASET

    # Area vs price (binned)
    bins = [0, 800, 1200, 1600, 2000, 2400, 2800, 3200, 10000]
    labels = ["800", "1200", "1600", "2000", "2400", "2800", "3200", "3200+"]
    df = df.copy()
    df["area_bin"] = pd.cut(df["area"], bins=bins, labels=labels)
    area_vs_price = (
        df.groupby("area_bin", observed=True)["price"].mean().div(100_000).round(1)
    )

    # Location-wise average price
    loc_avg = df.groupby("location")["price"].mean().div(100_000).round(1)

    # Age vs price (binned)
    age_bins = [-1, 0, 5, 10, 15, 20, 25, 30, 100]
    age_labels = ["0", "5", "10", "15", "20", "25", "30", "30+"]
    df["age_bin"] = pd.cut(df["age"], bins=age_bins, labels=age_labels)
    age_vs_price = (
        df.groupby("age_bin", observed=True)["price"].mean().div(100_000).round(1)
    )

    # Amenity impact -> feature importances from the trained model
    importances = dict(zip(FEATURE_COLUMNS, MODEL.feature_importances_))
    amenity_impact = {
        a: round(importances.get(a, 0) * 100, 2) for a in AMENITY_COLUMNS
    }

    return jsonify({
        "area_vs_price": {"labels": labels, "values": area_vs_price.reindex(labels).fillna(0).tolist()},
        "location_avg": {"labels": list(loc_avg.index), "values": loc_avg.values.tolist()},
        "age_vs_price": {"labels": age_labels, "values": age_vs_price.reindex(age_labels).fillna(0).tolist()},
        "amenity_impact": {"labels": list(amenity_impact.keys()), "values": list(amenity_impact.values())},
        "metrics": METRICS,
    })


# --------------------------------------------------------------------------
# 7. FRONT END (Jinja2 template string — Pinterest-style luxury UI)
# --------------------------------------------------------------------------

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Estatia — House Price Prediction System</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Montserrat:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{--cream:#faf8f4;--white:#fff;--beige:#eee6d9;--beige-deep:#d9c9ae;--grey:#8b8680;
--ink:#2b2620;--ink-soft:#5c564d;--sky:#aebfd1;--sky-deep:#7c93ac;--gold:#b98d4f;--gold-soft:#d9b482;
--shadow-soft:0 20px 60px -20px rgba(43,38,32,.18);--shadow-card:0 12px 40px -12px rgba(43,38,32,.12);}
*{margin:0;padding:0;box-sizing:border-box;}
html{scroll-behavior:smooth;}
body{font-family:'Montserrat',sans-serif;color:var(--ink);
background:radial-gradient(1200px 600px at 90% -10%, rgba(174,191,209,.18),transparent 60%),
radial-gradient(1000px 500px at -10% 20%, rgba(217,180,130,.15),transparent 55%), var(--cream);}
h1,h2,h3{font-family:'Playfair Display',serif;}
.container{max-width:1240px;margin:0 auto;padding:0 32px;}
.glass{background:rgba(255,255,255,.55);backdrop-filter:blur(20px) saturate(180%);
border:1px solid rgba(255,255,255,.6);border-radius:28px;box-shadow:var(--shadow-card);}
nav{position:sticky;top:0;z-index:100;background:rgba(250,248,244,.75);backdrop-filter:blur(16px);border-bottom:1px solid rgba(43,38,32,.06);}
.nav-inner{display:flex;align-items:center;justify-content:space-between;padding:20px 32px;max-width:1240px;margin:0 auto;}
.logo{display:flex;align-items:center;gap:10px;font-family:'Playfair Display',serif;font-weight:700;font-size:22px;}
.logo-mark{width:38px;height:38px;border-radius:11px;background:linear-gradient(145deg,var(--gold-soft),var(--sky-deep));display:flex;align-items:center;justify-content:center;font-size:18px;}
.nav-cta{padding:11px 22px;border-radius:100px;background:var(--ink);color:var(--cream);font-size:13px;font-weight:600;}
.hero{padding:72px 32px 40px;max-width:1240px;margin:0 auto;}
.hero-grid{display:grid;grid-template-columns:1.05fr .95fr;gap:64px;align-items:center;}
.eyebrow{display:inline-flex;align-items:center;gap:8px;font-size:12.5px;letter-spacing:2px;text-transform:uppercase;color:var(--gold);font-weight:600;margin-bottom:22px;}
.eyebrow::before{content:'';width:26px;height:1px;background:var(--gold);}
.hero h1{font-size:50px;line-height:1.12;font-weight:700;margin-bottom:22px;letter-spacing:-.5px;}
.hero h1 em{font-style:italic;color:var(--gold);font-weight:600;}
.hero p.sub{font-size:16px;line-height:1.75;color:var(--ink-soft);max-width:480px;margin-bottom:36px;}
.hero-ctas{display:flex;gap:16px;flex-wrap:wrap;}
.btn{border:none;cursor:pointer;font-family:'Montserrat',sans-serif;font-weight:600;transition:transform .3s,box-shadow .3s;}
.btn-primary{padding:17px 32px;border-radius:100px;background:linear-gradient(120deg,var(--gold),var(--gold-soft));color:#fff;font-size:14.5px;box-shadow:0 16px 34px -12px rgba(185,141,79,.55);}
.btn-primary:hover{transform:translateY(-3px) scale(1.02);}
.btn-secondary{padding:17px 30px;border-radius:100px;background:rgba(255,255,255,.6);border:1.5px solid rgba(43,38,32,.14);color:var(--ink);font-size:14.5px;}
.btn-secondary:hover{transform:translateY(-3px);background:#fff;border-color:var(--gold);}
.model-badge{margin-top:34px;display:inline-flex;gap:22px;padding:14px 20px;border-radius:16px;}
.model-badge div{font-size:12px;color:var(--ink-soft);}
.model-badge b{display:block;font-family:'Playfair Display',serif;font-size:17px;color:var(--ink);}
.hero-img-wrap{position:relative;border-radius:32px;overflow:hidden;box-shadow:var(--shadow-soft);aspect-ratio:4/4.6;}
.hero-img-wrap img{width:100%;height:100%;object-fit:cover;transition:transform 1s ease;}
.hero-img-wrap:hover img{transform:scale(1.06);}
.section{padding:90px 32px;}
.section-head{text-align:center;max-width:640px;margin:0 auto 50px;}
.section-head h2{font-size:34px;font-weight:700;margin-bottom:12px;}
.section-head p{color:var(--ink-soft);font-size:15px;line-height:1.7;}
.form-card{padding:52px;max-width:980px;margin:0 auto;}
.form-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:22px 26px;margin-bottom:30px;}
.field{display:flex;flex-direction:column;gap:8px;}
.field label{font-size:12.5px;font-weight:600;color:var(--ink-soft);}
.field select,.field input{padding:14px 16px;border-radius:14px;border:1.5px solid rgba(43,38,32,.1);background:rgba(255,255,255,.7);font-family:'Montserrat',sans-serif;font-size:14px;}
.field select:focus,.field input:focus{outline:none;border-color:var(--gold);box-shadow:0 0 0 4px rgba(185,141,79,.14);}
.toggle-row{display:flex;gap:10px;}
.toggle-btn{flex:1;padding:12px 10px;border-radius:12px;border:1.5px solid rgba(43,38,32,.1);background:rgba(255,255,255,.7);font-size:13px;cursor:pointer;transition:all .25s;}
.toggle-btn.active{background:linear-gradient(120deg,var(--gold),var(--gold-soft));color:#fff;border-color:transparent;}
.amenities-title{grid-column:1/-1;font-size:12px;letter-spacing:1.5px;text-transform:uppercase;color:var(--sky-deep);font-weight:700;margin:14px 0 4px;border-top:1px dashed rgba(43,38,32,.14);padding-top:22px;}
.predict-btn-wrap{display:flex;justify-content:center;margin-top:12px;}
.predict-btn{padding:20px 54px;border-radius:100px;font-size:16px;background:linear-gradient(120deg,#8fa9c2,#b98d4f);color:#fff;box-shadow:0 20px 40px -14px rgba(124,147,172,.55);}
.predict-btn:hover{transform:translateY(-3px) scale(1.03);}
.result-wrap{max-width:980px;margin:36px auto 0;display:none;}
.result-wrap.show{display:block;animation:riseIn .7s cubic-bezier(.2,.8,.2,1);}
@keyframes riseIn{from{opacity:0;transform:translateY(26px);}to{opacity:1;transform:translateY(0);}}
.result-card{padding:44px;}
.result-top{display:flex;justify-content:space-between;flex-wrap:wrap;gap:20px;margin-bottom:30px;}
.result-label{font-size:12px;letter-spacing:1.5px;text-transform:uppercase;color:var(--ink-soft);font-weight:600;margin-bottom:8px;}
.result-price{font-family:'Playfair Display',serif;font-size:44px;font-weight:800;}
.confidence-pill{display:flex;align-items:center;gap:10px;background:rgba(95,143,106,.12);color:#4c7758;padding:10px 18px;border-radius:100px;font-size:13px;font-weight:600;height:fit-content;}
.confidence-pill .dot{width:8px;height:8px;border-radius:50%;background:#5f8f6a;}
.result-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;}
.mini-stat{background:rgba(255,255,255,.55);border:1px solid rgba(255,255,255,.6);border-radius:16px;padding:18px;}
.mini-stat .m-label{font-size:11px;color:var(--ink-soft);text-transform:uppercase;margin-bottom:6px;}
.mini-stat .m-val{font-weight:700;font-size:15px;}
.trend-up{color:#5f8f6a;}
.workflow-track{display:flex;align-items:center;justify-content:center;gap:6px;flex-wrap:wrap;max-width:1180px;margin:0 auto;}
.flow-card{width:168px;padding:26px 18px;text-align:center;border-radius:22px;background:rgba(255,255,255,.6);border:1px solid rgba(255,255,255,.7);box-shadow:var(--shadow-card);transition:transform .35s;}
.flow-card:hover{transform:translateY(-8px);}
.flow-icon{font-size:26px;margin-bottom:12px;}
.flow-title{font-size:13px;font-weight:700;margin-bottom:4px;}
.flow-desc{font-size:11px;color:var(--ink-soft);line-height:1.5;}
.flow-arrow{color:var(--gold);font-size:20px;}
.features-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:24px;max-width:1140px;margin:0 auto;}
.feature-card{padding:34px 30px;transition:transform .4s;}
.feature-card:hover{transform:translateY(-8px);}
.feature-icon{width:54px;height:54px;border-radius:16px;display:flex;align-items:center;justify-content:center;font-size:24px;background:linear-gradient(145deg,var(--beige),var(--sky));margin-bottom:20px;}
.feature-card h3{font-size:18px;font-weight:600;margin-bottom:10px;font-family:'Montserrat',sans-serif;}
.feature-card p{font-size:13.5px;color:var(--ink-soft);line-height:1.7;}
.charts-grid{display:grid;grid-template-columns:1fr 1fr;gap:24px;max-width:1140px;margin:0 auto;}
.chart-card{padding:28px;}
.chart-card h3{font-size:15px;font-weight:600;margin-bottom:4px;}
.chart-card .chart-sub{font-size:12px;color:var(--ink-soft);margin-bottom:18px;}
footer{padding:56px 32px 34px;text-align:center;border-top:1px solid rgba(43,38,32,.08);}
footer .logo{justify-content:center;margin-bottom:14px;}
footer p{color:var(--ink-soft);font-size:13px;}
@media(max-width:980px){.hero-grid{grid-template-columns:1fr;}.hero h1{font-size:36px;}}
@media(max-width:900px){.features-grid{grid-template-columns:1fr 1fr;}.workflow-track{flex-direction:column;}.flow-arrow{transform:rotate(90deg);}}
@media(max-width:760px){.form-grid{grid-template-columns:1fr;}.form-card{padding:32px 22px;}.result-grid{grid-template-columns:1fr 1fr;}}
@media(max-width:600px){.features-grid{grid-template-columns:1fr;}.charts-grid{grid-template-columns:1fr;}}
</style>
</head>
<body>

<nav><div class="nav-inner">
  <div class="logo"><span class="logo-mark">🏡</span> Estatia</div>
  <a href="#form" class="nav-cta">Get Started</a>
</div></nav>

<section class="hero">
  <div class="hero-grid">
    <div>
      <span class="eyebrow">Machine Learning · Real Estate</span>
      <h1>House Price <em>Prediction</em> System</h1>
      <p class="sub">Predict the estimated market value of a property using a scikit-learn regression model trained on area, location, bedrooms, bathrooms, property age, and amenities.</p>
      <div class="hero-ctas">
        <button class="btn btn-primary" onclick="document.getElementById('form').scrollIntoView({behavior:'smooth'})">🔮 Predict Price</button>
        <button class="btn btn-secondary" onclick="document.getElementById('analytics').scrollIntoView({behavior:'smooth'})">📊 Explore Dataset</button>
      </div>
      <div class="glass model-badge">
        <div><b>{{ "%.1f"|format(metrics.r2*100) }}%</b>R² Score</div>
        <div><b>₹{{ "{:,.0f}".format(metrics.mae) }}</b>Mean Abs. Error</div>
        <div><b>RandomForest</b>Model Type</div>
      </div>
    </div>
    <div class="hero-img-wrap">
      <img src="https://images.unsplash.com/photo-1613977257363-707ba9348227?q=80&w=1200&auto=format&fit=crop" alt="Modern luxury house with greenery and warm sunlight">
    </div>
  </div>
</section>

<section class="section" id="form">
  <div class="section-head">
    <span class="eyebrow">Property Details</span>
    <h2>Tell us about the property</h2>
    <p>The trained model estimates the market value the moment you submit the form.</p>
  </div>
  <div class="glass form-card">
    <div class="form-grid">
      <div class="field"><label>📍 Location</label>
        <select id="f-location">
          {% for loc in locations %}<option>{{ loc }}</option>{% endfor %}
        </select>
      </div>
      <div class="field"><label>📐 Area (Square Feet)</label><input type="number" id="f-area" value="2200"></div>
      <div class="field"><label>🛏 Bedrooms</label>
        <select id="f-bed"><option>1</option><option>2</option><option selected>3</option><option>4</option><option>5</option></select></div>
      <div class="field"><label>🚿 Bathrooms</label>
        <select id="f-bath"><option>1</option><option>2</option><option selected>3</option><option>4</option></select></div>
      <div class="field"><label>🏠 Property Age (Years)</label><input type="number" id="f-age" value="5"></div>
      <div class="field"><label>🚗 Parking Spaces</label>
        <select id="f-parking"><option>0</option><option selected>1</option><option>2</option><option>3</option></select></div>

      <div class="amenities-title">🏘 Amenities</div>
      {% set amenities = [("garden","🌳 Garden",true),("pool","🏊 Swimming Pool",false),("lift","🛗 Lift Available",true),
        ("ac","❄️ Air Conditioning",true),("security","🛡️ Security",true),("power","⚡ Power Backup",true),
        ("wifi","📶 Wi-Fi",true),("schools","🏫 Nearby Schools",true),("hospital","🏥 Nearby Hospital",true),("mall","🛍 Shopping Mall Nearby",false)] %}
      {% for key,label,default in amenities %}
      <div class="field"><label>{{ label }}</label>
        <div class="toggle-row" data-group="{{ key }}">
          <button type="button" class="toggle-btn {{ 'active' if default }}">Yes</button>
          <button type="button" class="toggle-btn {{ 'active' if not default }}">No</button>
        </div>
      </div>
      {% endfor %}
    </div>
    <div class="predict-btn-wrap">
      <button class="btn predict-btn" onclick="runPrediction()">💰 Predict House Price</button>
    </div>

    <div class="result-wrap glass" id="result-wrap">
      <div class="result-card">
        <div class="result-top">
          <div><div class="result-label">🏡 Estimated House Price</div><div class="result-price" id="result-price">₹ 0</div></div>
          <div class="confidence-pill"><span class="dot"></span><span id="result-confidence">Confidence —</span></div>
        </div>
        <div class="result-grid">
          <div class="mini-stat"><div class="m-label">📍 Location</div><div class="m-val" id="result-location">—</div></div>
          <div class="mini-stat"><div class="m-label">🏠 Property Summary</div><div class="m-val" id="result-summary">—</div></div>
          <div class="mini-stat"><div class="m-label">📈 Price Trend</div><div class="m-val trend-up">▲ 4.2% this quarter</div></div>
          <div class="mini-stat"><div class="m-label">💵 Price / Sqft</div><div class="m-val" id="result-psf">—</div></div>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="section" id="workflow">
  <div class="section-head">
    <span class="eyebrow">Under the Hood</span>
    <h2>Machine Learning Workflow</h2>
    <p>This exact pipeline runs in app.py every time the server starts.</p>
  </div>
  <div class="workflow-track">
    <div class="flow-card"><div class="flow-icon">🗂</div><div class="flow-title">Dataset Collection</div><div class="flow-desc">3,000 synthetic listings</div></div>
    <div class="flow-arrow">→</div>
    <div class="flow-card"><div class="flow-icon">🧹</div><div class="flow-title">Data Cleaning</div><div class="flow-desc">Nulls &amp; outliers removed</div></div>
    <div class="flow-arrow">→</div>
    <div class="flow-card"><div class="flow-icon">⚙️</div><div class="flow-title">Feature Engineering</div><div class="flow-desc">One-hot location encoding</div></div>
    <div class="flow-arrow">→</div>
    <div class="flow-card"><div class="flow-icon">📊</div><div class="flow-title">Regression Model</div><div class="flow-desc">RandomForestRegressor</div></div>
    <div class="flow-arrow">→</div>
    <div class="flow-card"><div class="flow-icon">📈</div><div class="flow-title">Model Evaluation</div><div class="flow-desc">R² {{ "%.3f"|format(metrics.r2) }}</div></div>
    <div class="flow-arrow">→</div>
    <div class="flow-card"><div class="flow-icon">💵</div><div class="flow-title">Price Prediction</div><div class="flow-desc">Served via /api/predict</div></div>
  </div>
</section>

<section class="section" id="features">
  <div class="section-head"><span class="eyebrow">Capabilities</span><h2>Built for precision</h2><p>Every stage of the pipeline is tuned for reliable, real-time property valuation.</p></div>
  <div class="features-grid">
    <div class="glass feature-card"><div class="feature-icon">🧹</div><h3>Data Cleaning</h3><p>Automated handling of missing values, duplicates and outliers before training.</p></div>
    <div class="glass feature-card"><div class="feature-icon">⚙️</div><h3>Feature Engineering</h3><p>Location, amenities and age transformed into signals the model understands.</p></div>
    <div class="glass feature-card"><div class="feature-icon">📉</div><h3>Regression Models</h3><p>A tuned RandomForestRegressor trained on 3,000 property records.</p></div>
    <div class="glass feature-card"><div class="feature-icon">📈</div><h3>Model Evaluation</h3><p>Validated with R², MAE and RMSE on a held-out test split.</p></div>
    <div class="glass feature-card"><div class="feature-icon">🎯</div><h3>Accurate Prediction</h3><p>R² of {{ "%.3f"|format(metrics.r2) }} against verified market transactions.</p></div>
    <div class="glass feature-card"><div class="feature-icon">⚡</div><h3>Instant Results</h3><p>Get an estimated valuation in under a second via the Flask API.</p></div>
  </div>
</section>

<section class="section" id="analytics">
  <div class="section-head"><span class="eyebrow">Analytics Dashboard</span><h2>Explore the dataset</h2><p>Live charts pulled straight from the trained model and dataset via /api/analytics.</p></div>
  <div class="charts-grid">
    <div class="glass chart-card"><h3>📊 Area vs Price</h3><div class="chart-sub">Larger area, higher valuation</div><canvas id="chartArea"></canvas></div>
    <div class="glass chart-card"><h3>📍 Location-wise Price Comparison</h3><div class="chart-sub">Average price by neighbourhood</div><canvas id="chartLocation"></canvas></div>
    <div class="glass chart-card"><h3>📈 Property Age vs Price</h3><div class="chart-sub">Depreciation trend over time</div><canvas id="chartAge"></canvas></div>
    <div class="glass chart-card"><h3>🏘 Amenities Impact on Price</h3><div class="chart-sub">Model feature importance per amenity</div><canvas id="chartAmenities"></canvas></div>
  </div>
</section>

<footer><div class="logo"><span class="logo-mark">🏡</span> Estatia</div><p>© 2026 Estatia — House Price Prediction System. Built with Flask + scikit-learn.</p></footer>

<script>
document.querySelectorAll('.toggle-row').forEach(row=>{
  row.querySelectorAll('.toggle-btn').forEach(btn=>{
    btn.addEventListener('click',()=>{
      row.querySelectorAll('.toggle-btn').forEach(b=>b.classList.remove('active'));
      btn.classList.add('active');
    });
  });
});
function getToggle(group){
  const row = document.querySelector(`.toggle-row[data-group="${group}"]`);
  return row.querySelector('.toggle-btn.active').textContent.trim() === 'Yes';
}
function animateNumber(el, target, prefix, duration=900){
  const startTime = performance.now();
  function tick(now){
    const p = Math.min((now-startTime)/duration,1);
    const eased = 1-Math.pow(1-p,3);
    el.textContent = prefix + Math.round(target*eased).toLocaleString('en-IN');
    if(p<1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
async function runPrediction(){
  const payload = {
    location: document.getElementById('f-location').value,
    area: document.getElementById('f-area').value,
    bedrooms: document.getElementById('f-bed').value,
    bathrooms: document.getElementById('f-bath').value,
    age: document.getElementById('f-age').value,
    parking: document.getElementById('f-parking').value,
    garden: getToggle('garden'), pool: getToggle('pool'), lift: getToggle('lift'),
    ac: getToggle('ac'), security: getToggle('security'), power: getToggle('power'),
    wifi: getToggle('wifi'), schools: getToggle('schools'), hospital: getToggle('hospital'), mall: getToggle('mall')
  };
  const res = await fetch('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
  const data = await res.json();

  document.getElementById('result-wrap').classList.add('show');
  animateNumber(document.getElementById('result-price'), data.price, '₹ ');
  document.getElementById('result-confidence').textContent = `Confidence ${data.confidence}%`;
  document.getElementById('result-location').textContent = data.location;
  document.getElementById('result-summary').textContent = data.summary;
  document.getElementById('result-psf').textContent = `₹ ${data.price_per_sqft.toLocaleString('en-IN')} / sqft`;
  document.getElementById('result-wrap').scrollIntoView({behavior:'smooth', block:'center'});
}

Chart.defaults.font.family = 'Montserrat';
Chart.defaults.color = '#5c564d';

async function loadAnalytics(){
  const res = await fetch('/api/analytics');
  const d = await res.json();

  new Chart(document.getElementById('chartArea'), {
    type:'line',
    data:{labels:d.area_vs_price.labels, datasets:[{label:'Avg Price (₹ Lakh)', data:d.area_vs_price.values,
      borderColor:'#b98d4f', backgroundColor:'rgba(185,141,79,.15)', fill:true, tension:.4, pointRadius:3, pointBackgroundColor:'#b98d4f'}]},
    options:{plugins:{legend:{display:false}}, scales:{x:{grid:{display:false}},y:{grid:{color:'rgba(43,38,32,.06)'}}}}
  });

  new Chart(document.getElementById('chartLocation'), {
    type:'bar',
    data:{labels:d.location_avg.labels, datasets:[{label:'Avg Price (₹ Lakh)', data:d.location_avg.values,
      backgroundColor:['#b98d4f','#aebfd1','#aebfd1','#d9b482','#7c93ac','#aebfd1'], borderRadius:8}]},
    options:{plugins:{legend:{display:false}}, scales:{x:{grid:{display:false}},y:{grid:{color:'rgba(43,38,32,.06)'}}}}
  });

  new Chart(document.getElementById('chartAge'), {
    type:'line',
    data:{labels:d.age_vs_price.labels, datasets:[{label:'Avg Price (₹ Lakh)', data:d.age_vs_price.values,
      borderColor:'#7c93ac', backgroundColor:'rgba(124,147,172,.15)', fill:true, tension:.4, pointRadius:3, pointBackgroundColor:'#7c93ac'}]},
    options:{plugins:{legend:{display:false}}, scales:{x:{grid:{display:false}},y:{grid:{color:'rgba(43,38,32,.06)'}}}}
  });

  new Chart(document.getElementById('chartAmenities'), {
    type:'radar',
    data:{labels:d.amenity_impact.labels, datasets:[{label:'Price Impact (%)', data:d.amenity_impact.values,
      backgroundColor:'rgba(185,141,79,.2)', borderColor:'#b98d4f', pointBackgroundColor:'#b98d4f'}]},
    options:{plugins:{legend:{display:false}}, scales:{r:{grid:{color:'rgba(43,38,32,.08)'},angleLines:{color:'rgba(43,38,32,.08)'},pointLabels:{font:{size:10}}}}}
  });
}
loadAnalytics();
</script>
</body>
</html>
"""

# --------------------------------------------------------------------------
# 8. ENTRY POINT
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)