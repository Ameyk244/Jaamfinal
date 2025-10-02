import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from huggingface_hub import hf_hub_download
from flask import Flask, request, render_template_string, send_from_directory, url_for
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

MODEL_REPO = "subx24/ml-alzheimer-models"
CLASS_LABELS = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
CONFIDENCE_THRESHOLD = 0.7

def load_models():
    models = {}
    for name, fname in {"vgg19":"vgg19.h5","resnet":"resnet.keras","densenet":"densenet.keras"}.items():
        path = hf_hub_download(MODEL_REPO, fname, token=HF_TOKEN)
        models[name] = load_model(path)
        print(f"✅ Loaded {name}")
    return models

MODELS = load_models()

def preprocess(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    arr = image.img_to_array(img)/255.0
    arr = np.expand_dims(arr,0)
    return img, arr

def predict_single(img_array, model_name):
    model = MODELS[model_name]
    preds = model.predict(img_array)
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))
    label = CLASS_LABELS[idx] if conf >= CONFIDENCE_THRESHOLD else "Unknown / Invalid MRI"
    return label, conf*100

def predict_ensemble(img_path):
    _, arr = preprocess(img_path)
    preds = [predict_single(arr, m) for m in MODELS]
    labels, confs = zip(*preds)
    valid = [l for l in labels if l!="Unknown / Invalid MRI"]
    if len(valid)!=len(MODELS) or len(set(valid))!=1:
        return "Unknown / Invalid MRI",0.0
    return valid[0], np.mean([c for l,c in zip(labels,confs) if l==valid[0]])

UPLOAD_HTML = '''
<h1>Upload MRI Image</h1>
<form action="/upload" method="POST" enctype="multipart/form-data">
<input type="file" name="file" required>
<button type="submit">Upload</button>
</form>
'''

RESULT_HTML = '''
<h1>Prediction Result</h1>
<h2>Prediction: {{ label }}</h2>
<h3>Confidence: {{ confidence }}%</h3>
<img src="{{ img_url }}" style="max-width:500px;">
<br><br>
<a href="/">Upload another image</a>
'''

@app.route('/')
def home(): return render_template_string(UPLOAD_HTML)

@app.route('/upload', methods=['POST'])
def upload_file():
    f = request.files['file']
    path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(path)
    label, conf = predict_ensemble(path)
    result_path = os.path.join(RESULTS_FOLDER,"result_"+f.filename)
    Image.open(path).save(result_path)
    img_url = url_for('uploaded_file',filename="result_"+f.filename)
    return render_template_string(RESULT_HTML,label=label,confidence=round(conf,2),img_url=img_url)

@app.route('/results/<filename>')
def uploaded_file(filename):
    return send_from_directory(RESULTS_FOLDER,filename)

if __name__=="__main__":
    print("Running locally: http://127.0.0.1:5021")
    app.run(port=5021)
