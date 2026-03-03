from flask import Flask, render_template_string, request, jsonify
import torch
import numpy as np
from transformers import AutoFeatureExtractor, AutoModel
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import os
import uuid
import joblib
import base64
import io
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class SupervisedDeepBoostingClassifier:
    def __init__(self, model_name="google/vit-base-patch16-224"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.deep_model = AutoModel.from_pretrained(model_name).to(self.device)
        self.deep_model.eval()
        
        self.boosting = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=42
        )
        
        self.label_encoder = LabelEncoder()
        self.classes = []
        self.is_trained = False
    
    def extract_deep_features(self, image_path):
        image = Image.open(image_path).convert('RGB')
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.deep_model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)
        
        return features.cpu().numpy().flatten()
    
    def train_supervised(self, image_paths, labels):
        print("Training started...")
        features = []
        valid_labels = []
        
        for img_path, label in zip(image_paths, labels):
            if os.path.exists(img_path):
                feat = self.extract_deep_features(img_path)
                features.append(feat)
                valid_labels.append(label)
        
        X = np.array(features)
        y = self.label_encoder.fit_transform(valid_labels)
        self.classes = self.label_encoder.classes_
        
        self.boosting.fit(X, y)
        
        train_pred = self.boosting.predict(X)
        accuracy = np.mean(train_pred == y)
        
        self.is_trained = True
        
        return {
            'accuracy': float(accuracy),
            'samples': len(features),
            'classes': list(self.classes)
        }
    
    def predict(self, image_path, top_k=3):
        if not self.is_trained:
            return [{'class': 'Model not trained', 'confidence': 0, 'probability': '0%'}]
        
        features = self.extract_deep_features(image_path).reshape(1, -1)
        probabilities = self.boosting.predict_proba(features)[0]
        
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        predictions = []
        for idx in top_indices:
            predictions.append({
                'class': self.classes[idx],
                'confidence': float(probabilities[idx]),
                'probability': f"{probabilities[idx]:.2%}"
            })
        
        return predictions
    
    def save(self, path='model.pkl'):
        joblib.dump({
            'boosting': self.boosting,
            'encoder': self.label_encoder,
            'classes': self.classes,
            'is_trained': self.is_trained
        }, path)
    
    def load(self, path='model.pkl'):
        if os.path.exists(path):
            data = joblib.load(path)
            self.boosting = data['boosting']
            self.label_encoder = data['encoder']
            self.classes = data['classes']
            self.is_trained = data['is_trained']
            return True
        return False

model = SupervisedDeepBoostingClassifier()
model.load()

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>ML Image Classifier</title>
    <style>
        body { font-family: Arial; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; text-align: center; }
        .container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
        .upload-area { border: 3px dashed #ddd; padding: 40px; text-align: center; border-radius: 10px; margin: 20px 0; cursor: pointer; }
        .upload-area:hover { border-color: #667eea; }
        .btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 12px 24px; border-radius: 5px; cursor: pointer; width: 100%; margin: 10px 0; }
        .btn:hover { transform: translateY(-2px); }
        .result-item { display: flex; justify-content: space-between; padding: 10px; background: #f0f0f0; margin: 5px 0; border-radius: 5px; }
        .image-row { display: flex; gap: 10px; margin-bottom: 10px; }
        #preview { max-width: 100%; max-height: 200px; display: none; margin: 10px 0; }
        .status { background: #e3f2fd; padding: 10px; border-radius: 5px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 ML Image Classifier</h1>
        <p>Deep Learning (ViT) + XGBoost Boosting</p>
    </div>
    
    <div class="container">
        <div class="card">
            <h2>📚 Train Model</h2>
            <p>Upload labeled images</p>
            <div id="imageRows"></div>
            <button class="btn" onclick="addRow()">+ Add Image</button>
            <button class="btn" onclick="trainModel()">🚀 Start Training</button>
            <div id="trainingResult" class="status" style="display:none;"></div>
        </div>
        
        <div class="card">
            <h2>🎯 Classify Image</h2>
            <p>Upload image to classify</p>
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <input type="file" id="fileInput" accept="image/*" style="display:none;">
                <p>📸 Click to select image</p>
            </div>
            <img id="preview" src="#">
            <button class="btn" onclick="predictImage()">🔍 Classify</button>
            <div id="predictionResult"></div>
        </div>
    </div>
    
    <script>
        let rowCount = 0;
        
        function addRow() {
            const container = document.getElementById('imageRows');
            const row = document.createElement('div');
            row.className = 'image-row';
            row.id = `row_${rowCount}`;
            row.innerHTML = `
                <input type="file" accept="image/*" style="flex:2;" onchange="handleImage(this, ${rowCount})">
                <input type="text" id="label_${rowCount}" placeholder="Label (e.g., cat)" style="flex:1;">
                <button onclick="removeRow(${rowCount})" style="background:#f44336; color:white; border:none; padding:5px 10px;">✕</button>
            `;
            container.appendChild(row);
            rowCount++;
        }
        
        function removeRow(id) {
            const row = document.getElementById(`row_${id}`);
            if (row) row.remove();
        }
        
        function handleImage(input, rowId) {
            const file = input.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const preview = document.getElementById(`preview_${rowId}`);
                    if (!preview) {
                        const img = document.createElement('img');
                        img.id = `preview_${rowId}`;
                        img.style.maxWidth = '50px';
                        img.style.maxHeight = '50px';
                        img.src = e.target.result;
                        input.parentNode.appendChild(img);
                    }
                }
                reader.readAsDataURL(file);
            }
        }
        
        async function trainModel() {
            const formData = new FormData();
            const rows = document.querySelectorAll('[id^="row_"]');
            
            for (let row of rows) {
                const fileInput = row.querySelector('input[type="file"]');
                const labelInput = row.querySelector('input[type="text"]');
                
                if (fileInput.files[0] && labelInput.value) {
                    formData.append('images', fileInput.files[0]);
                    formData.append('labels', labelInput.value);
                }
            }
            
            document.getElementById('trainingResult').style.display = 'block';
            document.getElementById('trainingResult').innerHTML = '⏳ Training...';
            
            const response = await fetch('/train', { method: 'POST', body: formData });
            const data = await response.json();
            
            if (data.success) {
                document.getElementById('trainingResult').innerHTML = `
                    ✅ Training Complete!<br>
                    Accuracy: ${(data.metrics.accuracy * 100).toFixed(2)}%<br>
                    Classes: ${data.metrics.classes.join(', ')}
                `;
            } else {
                document.getElementById('trainingResult').innerHTML = `❌ Error: ${data.error}`;
            }
        }
        
        document.getElementById('fileInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const preview = document.getElementById('preview');
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                }
                reader.readAsDataURL(file);
            }
        });
        
        async function predictImage() {
            const fileInput = document.getElementById('fileInput');
            if (!fileInput.files[0]) return;
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            document.getElementById('predictionResult').innerHTML = '⏳ Classifying...';
            
            const response = await fetch('/predict', { method: 'POST', body: formData });
            const data = await response.json();
            
            if (data.success) {
                let html = '<h3>Results:</h3>';
                data.predictions.forEach(p => {
                    html += `<div class="result-item"><span>${p.class}</span><span>${p.probability}</span></div>`;
                });
                document.getElementById('predictionResult').innerHTML = html;
            }
        }
        
        addRow();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/train', methods=['POST'])
def train():
    files = request.files.getlist('images')
    labels = request.form.getlist('labels')
    
    image_paths = []
    for file in files:
        if file:
            filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_paths.append(filepath)
    
    try:
        metrics = model.train_supervised(image_paths, labels)
        model.save()
        return jsonify({'success': True, 'metrics': metrics})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        for path in image_paths:
            if os.path.exists(path):
                os.remove(path)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    file = request.files['file']
    filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        predictions = model.predict(filepath)
        return jsonify({'success': True, 'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
