from pathlib import Path
import numpy as np, json, joblib, os, cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

IMG_SIZE = 128
DATASET_DIR = Path("dataset")
MODEL_DIR = Path("models"); MODEL_DIR.mkdir(parents=True, exist_ok=True)
LABELS_PATH = Path("labels.json")

def cv2_read_image(path, size=IMG_SIZE):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR
    if img is None:
        return None
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # to RGB for consistency
    arr = img.astype(np.float32) / 255.0
    return arr.reshape(-1)  # flattened

def load_dataset():
    classes = sorted([d.name for d in DATASET_DIR.iterdir() if d.is_dir()])
    X_list, y_list = [], []
    for cls in classes:
        for p in (DATASET_DIR/cls).glob("*"):
            if not p.is_file():
                continue
            vec = cv2_read_image(p)
            if vec is None:
                continue
            X_list.append(vec)
            y_list.append(cls)
    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list)
    return X, y, classes

def main():
    print("Loading dataset from", DATASET_DIR)
    X, y, classes = load_dataset()
    print("Samples:", X.shape, "Classes:", classes)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print("Val acc:", round(acc,4))
    print(classification_report(y_val, y_pred))

    joblib.dump(model, MODEL_DIR / "model.pkl")
    with open(LABELS_PATH, "w") as f:
        json.dump(classes, f, indent=2)
    print("Saved:", MODEL_DIR/'model.pkl', "and", LABELS_PATH)

if __name__ == "__main__":
    main()
