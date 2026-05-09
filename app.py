from flask import Flask, render_template, request, jsonify
import pickle, sqlite3, os, json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

app = Flask(__name__)

# ── Load model & encoders ──────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

model            = pickle.load(open(os.path.join(BASE, "cold_start_model.pkl"), "rb"))
gender_encoder   = pickle.load(open(os.path.join(BASE, "gender_encoder.pkl"),   "rb"))
device_encoder   = pickle.load(open(os.path.join(BASE, "device_encoder.pkl"),   "rb"))
position_encoder = pickle.load(open(os.path.join(BASE, "position_encoder.pkl"), "rb"))
time_encoder     = pickle.load(open(os.path.join(BASE, "time_encoder.pkl"),     "rb"))
category_encoder = pickle.load(open(os.path.join(BASE, "category_encoder.pkl"), "rb"))

def safe_encode(encoder, value):
    try:
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        else:
            return -1
    except:
        return -1

# ── DB setup ───────────────────────────────────────────────
DB = os.path.join(BASE, "predictions.db")

def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

with get_db() as conn:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age INTEGER,
            gender TEXT,
            device_type TEXT,
            ad_position TEXT,
            time_of_day TEXT,
            ad_category TEXT,
            prediction INTEGER,
            probability REAL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()

# ── GRAPH ─────────────────────────────────────────────────
def make_dark_graph():
    conn = get_db()
    clicks    = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction=1").fetchone()[0]
    no_clicks = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction=0").fetchone()[0]
    conn.close()
    plt.figure()
    plt.bar(["Clicks", "No Clicks"], [clicks, no_clicks])
    click_patch = mpatches.Patch(color='blue', label='Click')
    no_click_patch = mpatches.Patch(color='gray', label='No Click')
    plt.legend(handles=[click_patch, no_click_patch])
    plt.title("Ad Click Prediction Results")
    plt.ylabel("Users")
    plt.savefig(os.path.join(BASE, "static", "graph.png"))
    plt.close()

# ── REAL-TIME RECOMMENDATION: Best Ad Category ─────────────
def get_best_category(age, gender_enc, device_enc, position_enc, time_enc):
    best_category = None
    best_prob = -1
    for cat in category_encoder.classes_:
        cat_enc = safe_encode(category_encoder, cat)
        if cat_enc == -1:
            continue
        prob = model.predict_proba([[age, gender_enc, device_enc, position_enc, time_enc, cat_enc]])[0][1]
        if prob > best_prob:
            best_prob = prob
            best_category = cat
    return best_category, round(best_prob * 100, 2)

# ── TIME-BASED OPTIMIZATION: Best Time to Show Ads ─────────
def get_best_time(age, gender_enc, device_enc, position_enc, category_enc):
    valid_times = [t for t in time_encoder.classes_ if isinstance(t, str)]
    best_time = None
    best_prob = -1
    for t in valid_times:
        t_enc = safe_encode(time_encoder, t)
        if t_enc == -1:
            continue
        prob = model.predict_proba([[age, gender_enc, device_enc, position_enc, t_enc, category_enc]])[0][1]
        if prob > best_prob:
            best_prob = prob
            best_time = t
    return best_time, round(best_prob * 100, 2)

# ── Routes ─────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    age      = int(request.form["age"])
    gender   = request.form["gender"]
    device   = request.form["device_type"]
    position = request.form["ad_position"]
    time     = request.form["time_of_day"]
    category = request.form["ad_category"]

    gender_enc   = safe_encode(gender_encoder, gender)
    device_enc   = safe_encode(device_encoder, device)
    position_enc = safe_encode(position_encoder, position)
    time_enc     = safe_encode(time_encoder, time)
    category_enc = safe_encode(category_encoder, category)

    features    = [[age, gender_enc, device_enc, position_enc, time_enc, category_enc]]
    prediction  = model.predict(features)[0]
    probability = round(model.predict_proba(features)[0][1] * 100, 2)

    with get_db() as conn:
        conn.execute(
            "INSERT INTO predictions(age,gender,device_type,ad_position,time_of_day,ad_category,prediction,probability) VALUES(?,?,?,?,?,?,?,?)",
            (age, gender, device, position, time, category, int(prediction), probability)
        )
        conn.commit()

    make_dark_graph()

    importances = model.feature_importances_
    feat_names  = ['Age', 'Gender', 'Device', 'Position', 'Time', 'Category']
    top_feat    = feat_names[int(np.argmax(importances))]

    best_category, best_cat_prob = get_best_category(age, gender_enc, device_enc, position_enc, time_enc)
    best_time, best_time_prob    = get_best_time(age, gender_enc, device_enc, position_enc, category_enc)

    return render_template("dashboard.html",
        prediction=int(prediction),
        probability=probability,
        age=age, gender=gender, device=device,
        position=position, time=time, category=category,
        top_feature=top_feat,
        best_category=best_category,
        best_cat_prob=best_cat_prob,
        best_time=best_time,
        best_time_prob=best_time_prob,
    )

@app.route("/analytics")
def analytics():
    conn = get_db()
    total    = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    clicks   = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction=1").fetchone()[0]
    avg_prob = conn.execute("SELECT AVG(probability) FROM predictions").fetchone()[0]
    avg_prob = round(avg_prob, 2) if avg_prob else 0
    ctr      = round(clicks / total * 100, 1) if total > 0 else 0
    by_cat    = [dict(r) for r in conn.execute("SELECT ad_category, COUNT(*) as cnt, SUM(prediction) as clicks FROM predictions GROUP BY ad_category ORDER BY clicks DESC").fetchall()]
    by_device = [dict(r) for r in conn.execute("SELECT device_type, COUNT(*) as cnt, SUM(prediction) as clicks FROM predictions GROUP BY device_type ORDER BY clicks DESC").fetchall()]
    by_time   = [dict(r) for r in conn.execute("SELECT time_of_day, COUNT(*) as cnt, SUM(prediction) as clicks FROM predictions GROUP BY time_of_day ORDER BY clicks DESC").fetchall()]
    conn.close()
    make_dark_graph()
    return render_template("analytics.html", total=total, clicks=clicks, no_clicks=total-clicks, avg_prob=avg_prob, ctr=ctr, by_cat=by_cat, by_device=by_device, by_time=by_time)

@app.route("/history")
def history():
    conn = get_db()
    rows = conn.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 50").fetchall()
    conn.close()
    return render_template("history.html", rows=[dict(r) for r in rows])

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    try:
        age          = int(data['age'])
        gender_enc   = safe_encode(gender_encoder, data['gender'])
        device_enc   = safe_encode(device_encoder, data['device_type'])
        position_enc = safe_encode(position_encoder, data['ad_position'])
        time_enc     = safe_encode(time_encoder, data['time_of_day'])
        category_enc = safe_encode(category_encoder, data['ad_category'])

        features = [[age, gender_enc, device_enc, position_enc, time_enc, category_enc]]
        pred = model.predict(features)[0]
        prob = round(model.predict_proba(features)[0][1] * 100, 2)

        best_category, best_cat_prob = get_best_category(age, gender_enc, device_enc, position_enc, time_enc)
        best_time, best_time_prob    = get_best_time(age, gender_enc, device_enc, position_enc, category_enc)

        return jsonify({
            "prediction": int(pred),
            "probability": prob,
            "will_click": bool(pred),
            "recommendations": {
                "best_ad_category": best_category,
                "best_category_probability": best_cat_prob,
                "best_time_to_show": best_time,
                "best_time_probability": best_time_prob,
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
