"""
SABOTAGE ML Backend v1.0
FastAPI + scikit-learn + SQLite
Railway deployment
"""

import os, json, sqlite3, hashlib, math
from datetime import datetime, timezone
from typing import Optional, List
import httpx
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── ML imports ────────────────────────────────────────────────────
try:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import cross_val_score
    import pickle
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  scikit-learn no disponible — modo heurístico activo")

# ── Config ────────────────────────────────────────────────────────
TWELVE_KEY    = os.environ.get("TWELVE_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
DB_PATH       = os.environ.get("DB_PATH", "/data/sabotage.db")
MIN_OPS_ML    = 20   # mínimo de ops cerradas para activar ML real

app = FastAPI(title="SABOTAGE ML Backend", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET","POST","PUT","PATCH","DELETE","OPTIONS"],
    allow_headers=["*"],
)

# ── Mapas ─────────────────────────────────────────────────────────
SYMBOL_MAP = {
    "XAUUSD":"XAU/USD","XAGUSD":"XAG/USD","EURUSD":"EUR/USD","GBPUSD":"GBP/USD",
    "USDJPY":"USD/JPY","USDCHF":"USD/CHF","AUDUSD":"AUD/USD","USDCAD":"USD/CAD",
    "US30":"US30","US500":"SPX","NAS100":"NDX","GER40":"DAX",
    "BTCUSD":"BTC/USD","ETHUSD":"ETH/USD","USOIL":"WTI","UKOIL":"BRENT",
}
DECIMALS = {
    "XAUUSD":2,"XAGUSD":3,"BTCUSD":0,"ETHUSD":1,
    "US30":0,"US500":1,"NAS100":0,"GER40":1,"USOIL":2,"UKOIL":2,
}
TF_INTERVAL = {
    "M5":"5min","M15":"15min","M30":"30min",
    "H1":"1h","H4":"4h","D1":"1day","W1":"1week",
}
DIR_NUM = {"BULL":1,"NEUTRAL":0,"BEAR":-1}

# ════════════════════════════════════════════════════════════════
#  DATABASE
# ════════════════════════════════════════════════════════════════
def get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS operations (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        uid         TEXT NOT NULL,
        symbol      TEXT NOT NULL,
        direction   TEXT NOT NULL,
        status      TEXT NOT NULL DEFAULT 'OPEN',
        result      TEXT,
        entry       REAL,
        sl          REAL,
        tp          REAL,
        risk_pct    REAL,
        pnl         REAL,
        notes       TEXT,
        -- Contexto técnico capturado automáticamente al abrir
        score       INTEGER,
        tf_m5       TEXT, tf_m15 TEXT, tf_m30 TEXT,
        tf_h1       TEXT, tf_h4  TEXT, tf_d1  TEXT, tf_w1 TEXT,
        price_open  REAL,
        -- Features adicionales
        hour_open   INTEGER,
        dow_open    INTEGER,
        atr_h1      REAL,
        -- Timestamps
        open_time   TEXT,
        close_time  TEXT,
        created_at  TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS models (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        uid         TEXT NOT NULL,
        symbol      TEXT NOT NULL,
        model_type  TEXT NOT NULL,
        model_data  BLOB,
        n_samples   INTEGER,
        accuracy    REAL,
        updated_at  TEXT DEFAULT (datetime('now')),
        UNIQUE(uid, symbol, model_type)
    );
    CREATE TABLE IF NOT EXISTS users (
        uid         TEXT PRIMARY KEY,
        display_name TEXT,
        pwd_hash    TEXT,
        created_at  TEXT DEFAULT (datetime('now'))
    );
    """)
    conn.commit()
    conn.close()

init_db()

# ════════════════════════════════════════════════════════════════
#  ANÁLISIS TÉCNICO
# ════════════════════════════════════════════════════════════════
def ema(data, period):
    k = 2 / (period + 1)
    e = data[0]
    for v in data[1:]: e = v * k + e * (1 - k)
    return e

def compute_direction(candles):
    if not candles or len(candles) < 51: return "NEUTRAL"
    closes = [float(c["close"]) for c in candles]
    ema20 = ema(closes[-20:], 20)
    ema50 = ema(closes[-50:], 50)
    price = closes[-1]
    bull = (1 if price > ema20 else 0) + (1 if ema20 > ema50 else 0)
    bear = (1 if price < ema20 else 0) + (1 if ema20 < ema50 else 0)
    return "BULL" if bull >= 2 else "BEAR" if bear >= 2 else "NEUTRAL"

def compute_zone(candles):
    if not candles or len(candles) < 20: return None
    closes = sorted([float(c["close"]) for c in candles[-20:]])
    return [round(closes[len(closes)//4], 2), round(closes[3*len(closes)//4], 2)]

def compute_atr(candles, period=14):
    if not candles or len(candles) < period+1: return None
    trs = []
    for i in range(1, min(period+1, len(candles))):
        h = float(candles[i]["high"]) if "high" in candles[i] else float(candles[i]["close"])
        l = float(candles[i]["low"]) if "low" in candles[i] else float(candles[i]["close"])
        pc = float(candles[i-1]["close"])
        trs.append(max(h-l, abs(h-pc), abs(l-pc)))
    return round(sum(trs)/len(trs), 4) if trs else None

def compute_score(analysis):
    dirs = [v["direction"] for v in analysis.values()]
    if not dirs: return 0
    bull = dirs.count("BULL")
    bear = dirs.count("BEAR")
    return round((max(bull, bear) / len(dirs)) * 100)

async def fetch_tf_data(client, twelve_sym, tf):
    interval = TF_INTERVAL.get(tf, "1h")
    try:
        r = await client.get(
            "https://api.twelvedata.com/time_series",
            params={"symbol": twelve_sym, "interval": interval, "outputsize": 60, "apikey": TWELVE_KEY},
            timeout=10
        )
        d = r.json()
        return d.get("values", [])
    except:
        return []

# ════════════════════════════════════════════════════════════════
#  ML ENGINE
# ════════════════════════════════════════════════════════════════
def build_features(op: dict) -> Optional[list]:
    """Convierte una operación en vector de features para el ML."""
    try:
        feats = [
            DIR_NUM.get(op.get("tf_h1","NEUTRAL"), 0),
            DIR_NUM.get(op.get("tf_h4","NEUTRAL"), 0),
            DIR_NUM.get(op.get("tf_d1","NEUTRAL"), 0),
            DIR_NUM.get(op.get("tf_m15","NEUTRAL"), 0),
            DIR_NUM.get(op.get("tf_m30","NEUTRAL"), 0),
            1 if op.get("direction") == "BULL" else -1,
            int(op.get("score", 0)),
            int(op.get("hour_open", 12)),
            int(op.get("dow_open", 0)),
            float(op.get("risk_pct", 1) or 1),
            float(op.get("atr_h1", 0) or 0),
        ]
        return feats
    except:
        return None

def train_model(uid: str, symbol: str):
    """Entrena o reentrena el modelo para un usuario+símbolo."""
    if not ML_AVAILABLE:
        return None, 0, 0.0

    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM operations WHERE uid=? AND symbol=? AND status='CLOSED' AND result IN ('WIN','LOSS','BE')",
        (uid, symbol)
    ).fetchall()
    conn.close()

    ops = [dict(r) for r in rows]
    if len(ops) < MIN_OPS_ML:
        return None, len(ops), 0.0

    X, y_result, y_dir = [], [], []
    for op in ops:
        feats = build_features(op)
        if feats and op.get("result") and op.get("direction"):
            X.append(feats)
            y_result.append(op["result"])
            y_dir.append(op["direction"])

    if len(X) < MIN_OPS_ML:
        return None, len(X), 0.0

    X = np.array(X)

    # Modelo 1: predice resultado WIN/LOSS/BE
    clf_result = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    clf_result.fit(X, y_result)

    # Modelo 2: predice dirección BULL/BEAR
    clf_dir = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf_dir.fit(X, y_dir)

    # Accuracy con cross-validation
    try:
        acc = cross_val_score(clf_result, X, y_result, cv=min(3, len(X)//5)).mean()
    except:
        acc = 0.0

    model_bundle = {"result": clf_result, "direction": clf_dir}
    model_bytes = pickle.dumps(model_bundle)

    # Guardar en DB
    conn = get_db()
    conn.execute(
        "INSERT OR REPLACE INTO models (uid, symbol, model_type, model_data, n_samples, accuracy, updated_at) VALUES (?,?,?,?,?,?,?)",
        (uid, symbol, "rf_v1", model_bytes, len(X), float(acc), datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()

    return model_bundle, len(X), float(acc)

def load_model(uid: str, symbol: str):
    """Carga el modelo guardado para un usuario+símbolo."""
    if not ML_AVAILABLE:
        return None
    conn = get_db()
    row = conn.execute(
        "SELECT model_data FROM models WHERE uid=? AND symbol=?",
        (uid, symbol)
    ).fetchone()
    conn.close()
    if not row or not row["model_data"]:
        return None
    try:
        return pickle.loads(row["model_data"])
    except:
        return None

def predict(uid: str, symbol: str, features: list) -> dict:
    """Hace una predicción con el modelo guardado o usa heurísticas."""
    model = load_model(uid, symbol)

    if model and ML_AVAILABLE:
        X = np.array([features])
        # Probabilidades de resultado
        proba_result = model["result"].predict_proba(X)[0]
        classes_result = model["result"].classes_
        result_probs = {c: round(float(p)*100, 1) for c, p in zip(classes_result, proba_result)}

        # Probabilidades de dirección
        proba_dir = model["direction"].predict_proba(X)[0]
        classes_dir = model["direction"].classes_
        dir_probs = {c: round(float(p)*100, 1) for c, p in zip(classes_dir, proba_dir)}

        best_result = max(result_probs, key=result_probs.get)
        best_dir    = max(dir_probs, key=dir_probs.get)

        return {
            "mode": "ML",
            "result": best_result,
            "result_probs": result_probs,
            "direction": best_dir,
            "dir_probs": dir_probs,
            "confidence": round(result_probs.get(best_result, 0)),
        }
    else:
        # Heurístico basado en alineación de TFs
        h1  = features[0]
        h4  = features[1]
        d1  = features[2]
        score = features[6]
        aligned = sum(1 for x in [h1, h4, d1] if x != 0 and x == h1)
        win_prob = 30 + (aligned * 15) + (score * 0.2)
        win_prob = min(max(win_prob, 10), 85)
        direction = "BULL" if h1+h4+d1 > 0 else "BEAR" if h1+h4+d1 < 0 else "NEUTRAL"
        return {
            "mode": "HEURISTIC",
            "result": "WIN" if win_prob > 55 else "NEUTRAL",
            "result_probs": {"WIN": round(win_prob,1), "LOSS": round(100-win_prob,1)},
            "direction": direction,
            "dir_probs": {"BULL": 50, "BEAR": 50},
            "confidence": round(win_prob),
        }

# ════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ════════════════════════════════════════════════════════════════

@app.get("/")
def health():
    conn = get_db()
    total_ops = conn.execute("SELECT COUNT(*) FROM operations").fetchone()[0]
    conn.close()
    return {
        "status": "ok",
        "service": "SABOTAGE ML Backend v1.0",
        "ml_available": ML_AVAILABLE,
        "total_ops": total_ops,
    }

# ── Análisis de mercado ───────────────────────────────────────────
@app.get("/analysis/{symbol}")
async def analysis(
    symbol: str,
    timeframes: str = Query(default="H1,H4,D1"),
    uid: Optional[str] = Query(default=None),
):
    sym_upper   = symbol.upper()
    twelve_sym  = SYMBOL_MAP.get(sym_upper, sym_upper)
    tfs         = [t.strip().upper() for t in timeframes.split(",") if t.strip()]
    dec         = DECIMALS.get(sym_upper, 4)

    price = None
    candles_by_tf = {}

    async with httpx.AsyncClient() as client:
        # Precio spot
        try:
            r = await client.get(
                "https://api.twelvedata.com/price",
                params={"symbol": twelve_sym, "apikey": TWELVE_KEY},
                timeout=8
            )
            d = r.json()
            if "price" in d: price = round(float(d["price"]), dec)
        except: pass

        # Velas por TF
        for tf in tfs:
            candles_by_tf[tf] = await fetch_tf_data(client, twelve_sym, tf)

        # H1 para ATR aunque no esté en los TFs seleccionados
        if "H1" not in candles_by_tf:
            candles_by_tf["H1"] = await fetch_tf_data(client, twelve_sym, "H1")

    tf_analysis = {}
    for tf, candles in candles_by_tf.items():
        if tf in tfs:
            tf_analysis[tf] = {
                "direction": compute_direction(candles),
                "zone": compute_zone(candles),
            }

    atr_h1 = compute_atr(candles_by_tf.get("H1", []))
    score  = compute_score(tf_analysis)

    if score >= 80:   message = f"Triple confluencia {sym_upper}. Condiciones institucionales activas."
    elif score >= 60: message = f"Confluencia parcial {sym_upper}. Espera confirmacion en TF menor."
    elif score <= 20 and any(v["direction"] != "NEUTRAL" for v in tf_analysis.values()):
                      message = f"Conflicto de estructura en {sym_upper}. No operar."
    else:             message = f"Estructura neutral en {sym_upper}. Modo observacion."

    # Predicción ML si hay uid
    ml_prediction = None
    if uid:
        now = datetime.now(timezone.utc)
        features = [
            DIR_NUM.get(tf_analysis.get("H1",{}).get("direction","NEUTRAL"), 0),
            DIR_NUM.get(tf_analysis.get("H4",{}).get("direction","NEUTRAL"), 0),
            DIR_NUM.get(tf_analysis.get("D1",{}).get("direction","NEUTRAL"), 0),
            DIR_NUM.get(tf_analysis.get("M15",{}).get("direction","NEUTRAL"), 0),
            DIR_NUM.get(tf_analysis.get("M30",{}).get("direction","NEUTRAL"), 0),
            1,  # dirección placeholder
            score,
            now.hour,
            now.weekday(),
            1.0,
            atr_h1 or 0,
        ]
        ml_prediction = predict(uid, sym_upper, features)

        # Info del modelo
        conn = get_db()
        model_row = conn.execute(
            "SELECT n_samples, accuracy, updated_at FROM models WHERE uid=? AND symbol=?",
            (uid, sym_upper)
        ).fetchone()
        n_ops = conn.execute(
            "SELECT COUNT(*) FROM operations WHERE uid=? AND symbol=? AND status='CLOSED'",
            (uid, sym_upper)
        ).fetchone()[0]
        conn.close()

        if model_row:
            ml_prediction["n_samples"] = model_row["n_samples"]
            ml_prediction["accuracy"]  = round(float(model_row["accuracy"]) * 100, 1)
            ml_prediction["updated_at"] = model_row["updated_at"]
        ml_prediction["n_ops_closed"] = n_ops
        ml_prediction["ml_active"] = n_ops >= MIN_OPS_ML

    return {
        "symbol": sym_upper, "price": price, "prev_price": None,
        "analysis": tf_analysis, "alignment_score": score,
        "message": message, "atr_h1": atr_h1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "LIVE" if price else "OFFLINE",
        "ml": ml_prediction,
    }

# ── Guardar operación ─────────────────────────────────────────────
class OperationIn(BaseModel):
    uid:        str
    symbol:     str
    direction:  str
    status:     str = "OPEN"
    result:     Optional[str] = None
    entry:      Optional[float] = None
    sl:         Optional[float] = None
    tp:         Optional[float] = None
    risk_pct:   Optional[float] = None
    pnl:        Optional[float] = None
    notes:      Optional[str]   = None
    open_time:  Optional[str]   = None
    close_time: Optional[str]   = None
    # Contexto técnico (el frontend lo envía desde data actual)
    score:      Optional[int]   = None
    tf_m5:      Optional[str]   = None
    tf_m15:     Optional[str]   = None
    tf_m30:     Optional[str]   = None
    tf_h1:      Optional[str]   = None
    tf_h4:      Optional[str]   = None
    tf_d1:      Optional[str]   = None
    tf_w1:      Optional[str]   = None
    price_open: Optional[float] = None
    atr_h1:     Optional[float] = None

@app.post("/operations")
async def save_operation(op: OperationIn):
    now = datetime.now(timezone.utc)
    hour_open = now.hour
    dow_open  = now.weekday()

    if op.open_time:
        try:
            dt = datetime.fromisoformat(op.open_time.replace("Z",""))
            hour_open = dt.hour
            dow_open  = dt.weekday()
        except: pass

    conn = get_db()
    cur = conn.execute("""
        INSERT INTO operations
        (uid,symbol,direction,status,result,entry,sl,tp,risk_pct,pnl,notes,
         score,tf_m5,tf_m15,tf_m30,tf_h1,tf_h4,tf_d1,tf_w1,
         price_open,hour_open,dow_open,atr_h1,open_time,close_time)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        op.uid, op.symbol.upper(), op.direction, op.status,
        op.result, op.entry, op.sl, op.tp, op.risk_pct, op.pnl, op.notes,
        op.score, op.tf_m5, op.tf_m15, op.tf_m30, op.tf_h1, op.tf_h4, op.tf_d1, op.tf_w1,
        op.price_open, hour_open, dow_open, op.atr_h1,
        op.open_time or now.isoformat(), op.close_time,
    ))
    op_id = cur.lastrowid
    conn.commit()
    conn.close()

    # Si está cerrada, reentrenar el modelo
    retrain_info = None
    if op.status == "CLOSED" and op.result in ("WIN","LOSS","BE"):
        _, n_samples, accuracy = train_model(op.uid, op.symbol.upper())
        retrain_info = {"n_samples": n_samples, "accuracy": round(accuracy*100,1)}

    return {"id": op_id, "status": "saved", "retrain": retrain_info}

# ── Obtener operaciones de un usuario ─────────────────────────────
@app.get("/operations/{uid}")
async def get_operations(uid: str, symbol: Optional[str] = None):
    conn = get_db()
    if symbol:
        rows = conn.execute(
            "SELECT * FROM operations WHERE uid=? AND symbol=? ORDER BY created_at DESC",
            (uid, symbol.upper())
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM operations WHERE uid=? ORDER BY created_at DESC",
            (uid,)
        ).fetchall()
    conn.close()
    return {"operations": [dict(r) for r in rows]}

# ── Cerrar operación existente ────────────────────────────────────
class CloseOp(BaseModel):
    result:     str
    pnl:        float
    close_time: Optional[str] = None

@app.patch("/operations/{op_id}/close")
async def close_operation(op_id: int, body: CloseOp):
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    conn.execute(
        "UPDATE operations SET status='CLOSED', result=?, pnl=?, close_time=? WHERE id=?",
        (body.result, body.pnl, body.close_time or now, op_id)
    )
    conn.commit()
    # Obtener uid y symbol para reentrenar
    row = conn.execute("SELECT uid, symbol FROM operations WHERE id=?", (op_id,)).fetchone()
    conn.close()

    retrain_info = None
    if row and body.result in ("WIN","LOSS","BE"):
        _, n_samples, accuracy = train_model(row["uid"], row["symbol"])
        retrain_info = {"n_samples": n_samples, "accuracy": round(accuracy*100,1)}

    return {"status": "closed", "retrain": retrain_info}

# ── Borrar memoria ML de un usuario ─────────────────────────────
@app.post("/operations/{uid}/clear")
async def delete_operations(uid: str, symbol: Optional[str] = None):
    conn = get_db()
    if symbol:
        conn.execute("DELETE FROM operations WHERE uid=? AND symbol=?", (uid, symbol.upper()))
        conn.execute("DELETE FROM models WHERE uid=? AND symbol=?", (uid, symbol.upper()))
    else:
        conn.execute("DELETE FROM operations WHERE uid=?", (uid,))
        conn.execute("DELETE FROM models WHERE uid=?", (uid,))
    conn.commit()
    conn.close()
    return {"status": "deleted", "uid": uid, "symbol": symbol}

# ── Stats del modelo ML por usuario ──────────────────────────────
@app.get("/ml/stats/{uid}")
async def ml_stats(uid: str, symbol: str = "XAUUSD"):
    conn = get_db()
    model_row = conn.execute(
        "SELECT n_samples, accuracy, updated_at FROM models WHERE uid=? AND symbol=?",
        (uid, symbol.upper())
    ).fetchone()
    ops = conn.execute(
        """SELECT result, direction,
           AVG(pnl) as avg_pnl, COUNT(*) as total
           FROM operations
           WHERE uid=? AND symbol=? AND status='CLOSED'
           GROUP BY result, direction""",
        (uid, symbol.upper())
    ).fetchall()
    total_closed = conn.execute(
        "SELECT COUNT(*) FROM operations WHERE uid=? AND symbol=? AND status='CLOSED'",
        (uid, symbol.upper())
    ).fetchone()[0]
    wins = conn.execute(
        "SELECT COUNT(*) FROM operations WHERE uid=? AND symbol=? AND result='WIN'",
        (uid, symbol.upper())
    ).fetchone()[0]
    conn.close()

    return {
        "uid": uid, "symbol": symbol.upper(),
        "total_closed": total_closed,
        "wins": wins,
        "win_rate": round(wins/total_closed*100, 1) if total_closed > 0 else 0,
        "ml_active": total_closed >= MIN_OPS_ML,
        "min_ops_needed": MIN_OPS_ML,
        "ops_remaining": max(0, MIN_OPS_ML - total_closed),
        "model": dict(model_row) if model_row else None,
        "breakdown": [dict(r) for r in ops],
    }

# ── Copilot (Charli) ──────────────────────────────────────────────
class CopilotRequest(BaseModel):
    question: str
    system:   str

@app.post("/copilot")
async def copilot(req: CopilotRequest):
    if not ANTHROPIC_KEY:
        raise HTTPException(500, "ANTHROPIC_API_KEY no configurada")
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 800,
                "system": req.system,
                "messages": [{"role": "user", "content": req.question}],
            }
        )
        d = r.json()
        if "error" in d: raise HTTPException(500, d["error"]["message"])
        return {"text": d["content"][0]["text"]}

# ── Macro news ────────────────────────────────────────────────────
class MacroNewsRequest(BaseModel):
    date: Optional[str] = None

@app.post("/macro/news")
async def macro_news(req: MacroNewsRequest):
    if not ANTHROPIC_KEY:
        return {"news": [], "error": "ANTHROPIC_API_KEY no configurada"}
    date_str = req.date or datetime.now().strftime("%d/%m/%Y")
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 1500,
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                "system": 'Eres analista macro. Responde UNICAMENTE con array JSON valido sin texto ni markdown. Formato: [{"id":1,"titulo":"...","fuente":"...","categoria":"FED|BCE|TRUMP|MACRO","impacto":"ALTO|MEDIO|BAJO","resumen":"...","hora":"HH:MM"}] Maximo 8 noticias.',
                "messages": [{"role": "user", "content": f"Busca noticias de {date_str} que afecten al oro, divisas e indices."}],
            }
        )
        d = r.json()
        txt = "".join(b["text"] for b in d.get("content", []) if b.get("type") == "text")
        txt = txt.replace("```json","").replace("```","").strip()
        try: news = json.loads(txt or "[]")
        except: news = []
        return {"news": news}

# ── Macro events ──────────────────────────────────────────────────
@app.post("/macro/events")
async def macro_events():
    if not ANTHROPIC_KEY:
        return {"events": [], "error": "ANTHROPIC_API_KEY no configurada"}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 800,
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                "system": 'Eres monitor del calendario economico. Responde UNICAMENTE con JSON valido sin texto ni markdown. Formato: [{"id":"evt_1","title":"...","category":"FED|BCE|TRUMP|MACRO","impact":"ALTO|MEDIO|BAJO","time":"ISO8601_UTC","description":"..."}] Solo proximas 4 horas. Si no hay eventos responde [].',
                "messages": [{"role": "user", "content": f"Ahora son {datetime.now(timezone.utc).isoformat()}. Busca proximos eventos macro en las proximas 4 horas."}],
            }
        )
        d = r.json()
        txt = "".join(b["text"] for b in d.get("content", []) if b.get("type") == "text")
        txt = txt.replace("```json","").replace("```","").strip()
        try: events = json.loads(txt or "[]")
        except: events = []
        return {"events": events}
