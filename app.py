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
import time
import tempfile

app = Flask(__name__)
app.secret_key = 'your-secret-key'

# Use /tmp for all file operations (Hugging Face compatible)
UPLOAD_FOLDER = '/tmp/uploads'
CACHE_DIR = '/tmp/model_cache'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

class SupervisedDeepBoostingClassifier:
    def __init__(self, model_name="google/vit-base-patch16-224"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use cache_dir to avoid re-downloading
        print(f"Loading model {model_name}...")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name, 
            cache_dir=CACHE_DIR
        )
        self.deep_model = AutoModel.from_pretrained(
            model_name, 
            cache_dir=CACHE_DIR
        ).to(self.device)
        self.deep_model.eval()
        print("Model loaded successfully!")
        
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
    
    def extract_deep_features(self, image_data):
        """Extract features from image bytes or PIL Image"""
        try:
            if isinstance(image_data, str):
                # It's a file path
                image = Image.open(image_data).convert('RGB')
            else:
                # It's bytes or PIL Image
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.deep_model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1)
            
            return features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
    
    def train_supervised(self, image_data_list, labels):
        print(f"Training started with {len(image_data_list)} images...")
        features = []
        valid_labels = []
        
        for img_data, label in zip(image_data_list, labels):
            feat = self.extract_deep_features(img_data)
            if feat is not None:
                features.append(feat)
                valid_labels.append(label)
                print(f"✓ Processed: {label}")
            else:
                print(f"✗ Failed to process image")
        
        if len(features) < 2:
            raise ValueError(f"Need at least 2 valid images for training. Got {len(features)}")
        
        unique_classes = set(valid_labels)
        if len(unique_classes) < 2:
            raise ValueError(f"Need at least 2 different classes. Got: {unique_classes}")
        
        X = np.array(features)
        y = self.label_encoder.fit_transform(valid_labels)
        self.classes = self.label_encoder.classes_
        
        print(f"Training XGBoost with {X.shape[0]} samples, {X.shape[1]} features...")
        self.boosting.fit(X, y)
        
        train_pred = self.boosting.predict(X)
        accuracy = np.mean(train_pred == y)
        
        self.is_trained = True
        
        return {
            'accuracy': float(accuracy),
            'samples': len(features),
            'classes': list(self.classes)
        }
    
    def predict(self, image_data, top_k=3):
        if not self.is_trained:
            return [{'class': 'Model not trained', 'confidence': 0, 'probability': '0%'}]
        
        features = self.extract_deep_features(image_data)
        if features is None:
            return [{'class': 'Error extracting features', 'confidence': 0, 'probability': '0%'}]
        
        features = features.reshape(1, -1)
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
    
    def save(self, path='/tmp/model.pkl'):
        joblib.dump({
            'boosting': self.boosting,
            'encoder': self.label_encoder,
            'classes': self.classes,
            'is_trained': self.is_trained
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path='/tmp/model.pkl'):
        if os.path.exists(path):
            data = joblib.load(path)
            self.boosting = data['boosting']
            self.label_encoder = data['encoder']
            self.classes = data['classes']
            self.is_trained = data['is_trained']
            print(f"Model loaded from {path}")
            return True
        return False

# Initialize model
print("Initializing model...")
model = SupervisedDeepBoostingClassifier()
model.load()
print("Ready!")

HTML = """<!DOCTYPE html>
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
        .btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 12px 24px; border-radius: 5px; cursor: pointer; width: 100%; margin: 10px 0; font-size: 16px; }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
        .result-item { display: flex; justify-content: space-between; padding: 10px; background: #f0f0f0; margin: 5px 0; border-radius: 5px; }
        .image-row { display: flex; gap: 10px; margin-bottom: 10px; align-items: center; }
        #preview { max-width: 100%; max-height: 200px; display: none; margin: 10px 0; border-radius: 5px; }
        .status { background: #e3f2fd; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .loading { text-align: center; padding: 20px; }
        .loading::after { content: "⏳"; animation: spin 1s linear infinite; display: inline-block; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .thumbnail { max-width: 50px; max-height: 50px; border-radius: 5px; }
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
            <p>Upload labeled images (need at least 2 different classes)</p>
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
                <input type="file" accept="image/*" style="flex:2;" onchange="previewImage(this, ${rowCount})">
                <input type="text" id="label_${rowCount}" placeholder="Label (e.g., cat)" style="flex:1;">
                <button onclick="removeRow(${rowCount})" style="background:#f44336; color:white; border:none; padding:5px 10px; border-radius:5px;">✕</button>
                <img id="preview_${rowCount}" class="thumbnail" style="display:none;">
            `;
            container.appendChild(row);
            rowCount++;
        }
        
        function removeRow(id) {
            const row = document.getElementById(`row_${id}`);
            if (row) row.remove();
        }
        
        function previewImage(input, rowId) {
            const file = input.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const preview = document.getElementById(`preview_${rowId}`);
                    preview.src = e.target.result;
                    preview.style.display = 'inline';
                }
                reader.readAsDataURL(file);
            }
        }
        
        async function trainModel() {
            const formData = new FormData();
            const rows = document.querySelectorAll('[id^="row_"]');
            let hasData = false;
            
            for (let row of rows) {
                const fileInput = row.querySelector('input[type="file"]');
                const labelInput = row.querySelector('input[type="text"]');
                
                if (fileInput.files[0] && labelInput.value) {
                    formData.append('images', fileInput.files[0]);
                    formData.append('labels', labelInput.value);
                    hasData = true;
                }
            }
            
            if (!hasData) {
                alert('Please add at least one image with a label');
                return;
            }
            
            document.getElementById('trainingResult').style.display = 'block';
            document.getElementById('trainingResult').innerHTML = '<div class="loading">Training in progress... ⏳</div>';
            
            try {
                const response = await fetch('/train', { 
                    method: 'POST', 
                    body: formData
                });
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || 'Training failed');
                }
                
                const data = await response.json();
                
                if (data.success) {
                    document.getElementById('trainingResult').innerHTML = `
                        <div style="background:#4CAF50; color:white; padding:10px; border-radius:5px;">
                            ✅ Training Complete!<br>
                            Accuracy: ${(data.metrics.accuracy * 100).toFixed(2)}%<br>
                            Classes: ${data.metrics.classes.join(', ')}<br>
                            Samples: ${data.metrics.samples}
                        </div>
                    `;
                }
            } catch (error) {
                document.getElementById('trainingResult').innerHTML = `
                    <div style="background:#f44336; color:white; padding:10px; border-radius:5px;">
                        ❌ Error: ${error.message}
                    </div>
                `;
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
            if (!fileInput.files[0]) {
                alert('Please select an image first');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            document.getElementById('predictionResult').innerHTML = '<div class="loading">Classifying... ⏳</div>';
            
            try {
                const response = await fetch('/predict', { 
                    method: 'POST', 
                    body: formData 
                });
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || 'Prediction failed');
                }
                
                const data = await response.json();
                
                if (data.success) {
                    let html = '<h3>Results:</h3>';
                    data.predictions.forEach(p => {
                        html += `<div class="result-item"><span>${p.class}</span><span>${p.probability}</span></div>`;
                    });
                    document.getElementById('predictionResult').innerHTML = html;
                }
            } catch (error) {
                document.getElementById('predictionResult').innerHTML = `
                    <div style="background:#f44336; color:white; padding:10px; border-radius:5px;">
                        ❌ Error: ${error.message}
                    </div>
                `;
            }
        }
        
        // Initialize with one row
        addRow();
    </script>
</body>
</html>"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/train', methods=['POST'])
def train():
    files = request.files.getlist('images')
    labels = request.form.getlist('labels')
    
    if len(files) < 2:
        return jsonify({'error': 'Need at least 2 images'}), 400
    
    # Read image data into memory instead of saving to disk first
    image_data_list = []
    valid_labels = []
    
    for file in files:
        if file:
            # Read file bytes immediately
            img_bytes = file.read()
            image_data_list.append(img_bytes)
    
    # Match labels with files
    for label in labels:
        if label:
            valid_labels.append(label.strip())
    
    if len(valid_labels) != len(image_data_list):
        return jsonify({'error': 'Labels count mismatch'}), 400
    
    try:
        metrics = model.train_supervised(image_data_list, valid_labels)
        model.save()
        return jsonify({'success': True, 'metrics': metrics})
    except Exception as e:
        print(f"Training error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        # Read file bytes directly
        img_bytes = file.read()
        predictions = model.predict(img_bytes)
        return jsonify({'success': True, 'predictions': predictions})
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
