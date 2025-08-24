# ♻️ Waste Classifier (Streamlit)

Modern, fast, and simple waste classification app with **camera + upload** support.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Your dataset should be in:

```
dataset/
  hazardous/
  organic/
  recyclable/
```

To retrain the model:

```bash
python train.py
```

Trained model and labels are saved to:

```
models/model.pkl
labels.json
```

## Notes

- Model is a `RandomForestClassifier` on normalized 128×128 RGB pixels with light **data augmentation** (flip, rotate, brightness/contrast, zoom).
- Webcam uses `st.camera_input` (no OpenCV install required).
- UI uses custom CSS for a **modern look**.

1. Code & Execution

All source code files (app.py, train.py, labels.json, model.pkl, requirements.txt) are present and organized.

Project runs successfully without errors in a fresh environment.

Dependencies install correctly using requirements.txt.

2. Model & Accuracy

Model (model.pkl) is trained on provided dataset.

Model is able to predict all three classes: Organic, Hazardous, Recyclable.

Predictions are reasonably accurate and consistent with dataset.

3. User Interface (Streamlit)

Application launches with command:

\*\*```python
streamlit run app.py

```**


User can upload image or capture using camera.

Prediction is displayed in gradient styled text:

Organic → Green gradient

Hazardous → Red/Orange gradient

Recyclable → Purple/Pink gradient

Confidence percentage and probability distribution for all classes are shown.

Modern card-based UI with responsive layout.

4. Usability

Instructions are provided for users (upload/camera).

Labels are clearly visible with first-letter capitalized.

Output is visually clear and professional.

5. Documentation

README.md includes:

Project objective

How to install dependencies

How to run the project

Example screenshots

“EiSystems” name updated in project description.

6. Quality Standards

No unused code/files.

Consistent naming conventions followed.

Proper comments for key functions (train.py, app.py).

Code pushed in final ZIP/GitHub repo.

# Acceptance Criteria

If all above checklist items are satisfied, the project is considered “Done”.

The system should be demo-ready: teacher can upload an image and immediately see results.
```
