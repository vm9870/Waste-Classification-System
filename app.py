import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib, json, io, base64
from pathlib import Path

st.set_page_config(page_title="Waste Classifier", page_icon="♻️", layout="wide")
# --- Custom CSS for a modern look ---
st.markdown("""
<style>
:root {
  --bg: #0f172a;
  --card: #111827;
  --muted: #94a3b8;
  --accent: #22c55e;
  --accent-2: #06b6d4;
  --rounded: 18px;
}
html, body, [class*="css"]  {
  background: var(--bg);
  color: #e5e7eb;
}
.sidebar .sidebar-content { background: var(--card); }
.block-container { padding-top: 2rem; }
.stButton>button {
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  border: none; color: white; padding: 0.7rem 1rem; border-radius: var(--rounded);
  font-weight: 600; box-shadow: 0 6px 20px rgba(0,0,0,0.3);
}
.stDownloadButton>button {
  background: transparent; border: 1px solid #334155; color: #e2e8f0;
  padding: 0.6rem 1rem; border-radius: var(--rounded);
}
.stProgress > div > div { background: linear-gradient(90deg, var(--accent), var(--accent-2)); }
.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
  border: 1px solid rgba(148,163,184,0.15);
  border-radius: var(--rounded);
  padding: 1rem 1.2rem;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}
.badge {
  display: inline-block;
  padding: 0.25rem 0.6rem;
  border-radius: 999px;
  border: 1px solid rgba(148,163,184,0.2);
  color: #cbd5e1;
  font-size: 0.8rem;
}
.pred {
  font-size: 1.8rem;
  font-weight: 800;
  letter-spacing: 0.5px;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
small { color: var(--muted); }
hr { border-color: rgba(148,163,184,0.15); }

.pred.organic { background: linear-gradient(45deg, #00c853, #b2ff59); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.pred.hazardous { background: linear-gradient(45deg, #ff1744, #ff9100); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.pred.recyclable { background: linear-gradient(45deg, #8E2DE2, #FF6EC7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

</style>
""", unsafe_allow_html=True)

# --- Load model ---
MODEL_PATH = Path("models/model.pkl")
LABELS_PATH = Path("labels.json")
IMG_SIZE = 128

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        classes = json.load(f)
    return model, classes

def img_to_vec_cv2(img_bgr):
    # expects BGR image from cv2
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    arr = img.astype(np.float32) / 255.0
    return arr.reshape(1, -1)

model, classes = load_assets()

# --- Header ---
col1, col2 = st.columns([1, 2], gap="small")
with col1:
    st.markdown("### ♻️ Waste Classification&nbsp;System")
    st.markdown("<small>Project → EiSystems → Vishal</small>", unsafe_allow_html=True)

st.markdown("---")

# --- Tabs for Upload and Camera ---
tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Use Camera"])

def predict_and_show_cv2(img_bgr):
    vec = img_to_vec_cv2(img_bgr)
    probs = model.predict_proba(vec)[0]
    top_idx = int(np.argmax(probs))
    pred = classes[top_idx]
    conf = float(probs[top_idx])
    # UI
    c1, c2 = st.columns([1,1], gap="large")
    with c1:
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Input image", use_container_width=True)
    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        pred_cap = pred.capitalize()
        pred_key = pred.lower().strip()
        cls_map = {'organic':'organic','hazardous':'hazardous','recyclable':'recyclable'}
        css_cls = cls_map.get(pred_key, 'recyclable')
        st.markdown(f"<div class='pred {css_cls}'>{pred_cap}</div>", unsafe_allow_html=True)
        st.markdown(f"<small>Confidence: {conf*100:.1f}%</small>", unsafe_allow_html=True)
        st.progress(conf)
        st.markdown("**All class probabilities:**")
        for cls, p in sorted(zip(classes, probs), key=lambda x: -x[1]):
            st.write(f"- {cls}: **{p*100:.1f}%**")
        st.markdown("</div>", unsafe_allow_html=True)

with tab1:
    uploaded = st.file_uploader("Drop an image here", type=["png","jpg","jpeg","bmp","webp"])
    if uploaded is not None:
        import numpy as _np
        _bytes = uploaded.getvalue()
        _nparr = _np.frombuffer(_bytes, dtype=_np.uint8)
        img_bgr = cv2.imdecode(_nparr, cv2.IMREAD_COLOR)
        predict_and_show_cv2(img_bgr)

with tab2:
    camera_img = st.camera_input("Take a photo")
    if camera_img is not None:
        import numpy as _np
        _bytes = camera_img.getvalue()
        _nparr = _np.frombuffer(_bytes, dtype=_np.uint8)
        img_bgr = cv2.imdecode(_nparr, cv2.IMREAD_COLOR)
        predict_and_show_cv2(img_bgr)

st.markdown("---")
st.caption("Tip: Place the object centered with plain background for best results.")
