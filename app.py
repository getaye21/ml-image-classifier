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
import io
import base64
from werkzeug.utils import secure_filename
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Use /tmp for all file operations (Hugging Face compatible)
UPLOAD_FOLDER = '/tmp/uploads'
CACHE_DIR = '/tmp/model_cache'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

class SupervisedDeepBoostingClassifier:
    def __init__(self, model_name="google/vit-base-patch16-224"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Use cache_dir to avoid re-downloading
        logger.info(f"Loading model {model_name}...")
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                model_name, 
                cache_dir=CACHE_DIR
            )
            self.deep_model = AutoModel.from_pretrained(
                model_name, 
                cache_dir=CACHE_DIR
            ).to(self.device)
            self.deep_model.eval()
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        self.boosting = None
        self.label_encoder = LabelEncoder()
        self.classes = []
        self.is_trained = False
    
    def extract_deep_features(self, image_data):
        """Extract features from image bytes or PIL Image"""
        try:
            if isinstance(image_data, str):
                # It's a file path
                if not os.path.exists(image_data):
                    logger.error(f"File not found: {image_data}")
                    return None
                image = Image.open(image_data).convert('RGB')
            elif isinstance(image_data, bytes):
                # It's bytes
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
            else:
                # Assume it's a PIL Image
                image = image_data.convert('RGB')
            
            # Resize image if needed (ViT expects 224x224)
            if image.size != (224, 224):
                image = image.resize((224, 224))
            
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.deep_model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1)
            
            return features.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_supervised(self, image_data_list, labels):
        logger.info(f"Training started with {len(image_data_list)} images...")
        
        if len(image_data_list) < 2:
            raise ValueError(f"Need at least 2 images for training. Got {len(image_data_list)}")
        
        features = []
        valid_labels = []
        
        for i, (img_data, label) in enumerate(zip(image_data_list, labels)):
            logger.info(f"Processing image {i+1}/{len(image_data_list)} with label: {label}")
            feat = self.extract_deep_features(img_data)
            if feat is not None:
                features.append(feat)
                valid_labels.append(label)
                logger.info(f"✓ Successfully processed: {label}")
            else:
                logger.error(f"✗ Failed to process image {i+1}")
        
        if len(features) < 2:
            raise ValueError(f"Need at least 2 valid images for training. Got {len(features)}")
        
        unique_classes = set(valid_labels)
        logger.info(f"Unique classes detected: {unique_classes}")
        
        if len(unique_classes) < 2:
            raise ValueError(f"Need at least 2 different classes. Got: {unique_classes}")
        
        X = np.array(features)
        logger.info(f"Feature matrix shape: {X.shape}")
        
        y = self.label_encoder.fit_transform(valid_labels)
        self.classes = self.label_encoder.classes_
        num_classes = len(self.classes)
        logger.info(f"Encoded labels: {y}")
        logger.info(f"Classes: {self.classes}, num_classes: {num_classes}")
        
        # FIX: Use XGBoost native API to avoid sklearn wrapper issues
        logger.info(f"Training XGBoost classifier with native API...")
        
        # Convert to DMatrix for native API
        dtrain = xgb.DMatrix(X, label=y)
        
        # Set parameters explicitly
        params = {
            'objective': 'multi:softprob',
            'num_class': num_classes,
            'max_depth': 8,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'mlogloss',
            'seed': 42
        }
        
        # Train the model
        bst = xgb.train(params, dtrain, num_boost_round=200)
        
        # Create a sklearn-compatible wrapper for predictions
        from xgboost import XGBClassifier
        self.boosting = XGBClassifier()
        self.boosting._Booster = bst
        self.boosting.classes_ = self.classes
        
        # Make predictions for accuracy calculation
        train_pred_proba = bst.predict(dtrain)
        train_pred = np.argmax(train_pred_proba, axis=1)
        accuracy = np.mean(train_pred == y)
        logger.info(f"Training accuracy: {accuracy:.4f}")
        
        self.is_trained = True
        
        return {
            'accuracy': float(accuracy),
            'samples': len(features),
            'classes': list(self.classes)
        }
    
    def predict(self, image_data, top_k=3):
        if not self.is_trained or self.boosting is None:
            logger.warning("Model not trained yet")
            return [{'class': 'Model not trained', 'confidence': 0, 'probability': '0%'}]
        
        features = self.extract_deep_features(image_data)
        if features is None:
            return [{'class': 'Error extracting features', 'confidence': 0, 'probability': '0%'}]
        
        features = features.reshape(1, -1)
        
        # Use the booster directly for prediction
        dtest = xgb.DMatrix(features)
        probabilities = self.boosting._Booster.predict(dtest)[0]
        
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
        if self.boosting is not None:
            joblib.dump({
                'booster': self.boosting._Booster,
                'encoder': self.label_encoder,
                'classes': self.classes,
                'is_trained': self.is_trained
            }, path)
            logger.info(f"Model saved to {path}")
    
    def load(self, path='/tmp/model.pkl'):
        if os.path.exists(path):
            data = joblib.load(path)
            from xgboost import XGBClassifier
            self.boosting = XGBClassifier()
            self.boosting._Booster = data['booster']
            self.label_encoder = data['encoder']
            self.classes = data['classes']
            self.is_trained = data['is_trained']
            self.boosting.classes_ = self.classes
            logger.info(f"Model loaded from {path}")
            return True
        return False

# Initialize model
logger.info("Initializing model...")
model = SupervisedDeepBoostingClassifier()
model.load()
logger.info("Ready!")

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
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .result-item { display: flex; justify-content: space-between; padding: 10px; background: #f0f0f0; margin: 5px 0; border-radius: 5px; }
        .image-row { display: flex; gap: 10px; margin-bottom: 10px; align-items: center; background: #f9f9f9; padding: 10px; border-radius: 5px; }
        #preview { max-width: 100%; max-height: 200px; display: none; margin: 10px 0; border-radius: 5px; }
        .status { background: #e3f2fd; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .loading { text-align: center; padding: 20px; }
        .loading-spinner { display: inline-block; width: 20px; height: 20px; border: 3px solid #f3f3f3; border-top: 3px solid #667eea; border-radius: 50%; animation: spin 1s linear infinite; margin-right: 10px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .thumbnail { max-width: 50px; max-height: 50px; border-radius: 5px; margin-left: 10px; }
        .error { background: #ffebee; color: #c62828; padding: 10px; border-radius: 5px; margin: 10px 0; }
        .success { background: #e8f5e9; color: #2e7d32; padding: 10px; border-radius: 5px; margin: 10px 0; }
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
            <p>Upload labeled images <strong>(need at least 2 images with DIFFERENT labels)</strong></p>
            <div id="imageRows"></div>
            <button class="btn" onclick="addRow()" id="addBtn">+ Add Image</button>
            <button class="btn" onclick="trainModel()" id="trainBtn">🚀 Start Training</button>
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
            <button class="btn" onclick="predictImage()" id="predictBtn">🔍 Classify</button>
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
                <input type="file" accept="image/*" style="flex:2;" onchange="previewImage(this, ${rowCount})" required>
                <input type="text" id="label_${rowCount}" placeholder="Label (e.g., cat)" style="flex:1;" required>
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
        
        function validateTrainingData() {
            const rows = document.querySelectorAll('[id^="row_"]');
            const labels = new Set();
            let hasImages = false;
            
            for (let row of rows) {
                const fileInput = row.querySelector('input[type="file"]');
                const labelInput = row.querySelector('input[type="text"]');
                
                if (fileInput.files[0] && labelInput.value) {
                    hasImages = true;
                    labels.add(labelInput.value.trim());
                }
            }
            
            return {
                isValid: hasImages && labels.size >= 2,
                numClasses: labels.size,
                hasImages: hasImages
            };
        }
        
        async function trainModel() {
            const validation = validateTrainingData();
            
            if (!validation.hasImages) {
                alert('Please add at least one image with a label');
                return;
            }
            
            if (validation.numClasses < 2) {
                alert(`Need at least 2 DIFFERENT classes. You have ${validation.numClasses} class(es). Please add images with different labels (e.g., 'cat' and 'dog')`);
                return;
            }
            
            const formData = new FormData();
            const rows = document.querySelectorAll('[id^="row_"]');
            
            for (let row of rows) {
                const fileInput = row.querySelector('input[type="file"]');
                const labelInput = row.querySelector('input[type="text"]');
                
                if (fileInput.files[0] && labelInput.value) {
                    formData.append('images', fileInput.files[0]);
                    formData.append('labels', labelInput.value.trim());
                }
            }
            
            // Disable buttons during training
            document.getElementById('trainBtn').disabled = true;
            document.getElementById('addBtn').disabled = true;
            
            document.getElementById('trainingResult').style.display = 'block';
            document.getElementById('trainingResult').innerHTML = '<div class="loading"><span class="loading-spinner"></span>Training in progress... ⏳</div>';
            
            try {
                const response = await fetch('/train', { 
                    method: 'POST', 
                    body: formData
                });
                
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.error || `Server error: ${response.status}`);
                }
                
                if (data.success) {
                    document.getElementById('trainingResult').innerHTML = `
                        <div class="success">
                            ✅ <strong>Training Complete!</strong><br>
                            Accuracy: ${(data.metrics.accuracy * 100).toFixed(2)}%<br>
                            Classes: ${data.metrics.classes.join(', ')}<br>
                            Samples: ${data.metrics.samples}
                        </div>
                    `;
                }
            } catch (error) {
                document.getElementById('trainingResult').innerHTML = `
                    <div class="error">
                        ❌ <strong>Error:</strong> ${error.message}
                    </div>
                `;
            } finally {
                // Re-enable buttons
                document.getElementById('trainBtn').disabled = false;
                document.getElementById('addBtn').disabled = false;
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
            
            // Disable predict button
            document.getElementById('predictBtn').disabled = true;
            
            document.getElementById('predictionResult').innerHTML = '<div class="loading"><span class="loading-spinner"></span>Classifying... ⏳</div>';
            
            try {
                const response = await fetch('/predict', { 
                    method: 'POST', 
                    body: formData 
                });
                
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.error || `Server error: ${response.status}`);
                }
                
                if (data.success) {
                    let html = '<h3>Results:</h3>';
                    data.predictions.forEach(p => {
                        html += `<div class="result-item"><span>${p.class}</span><span>${p.probability}</span></div>`;
                    });
                    document.getElementById('predictionResult').innerHTML = html;
                }
            } catch (error) {
                document.getElementById('predictionResult').innerHTML = `
                    <div class="error">
                        ❌ <strong>Error:</strong> ${error.message}
                    </div>
                `;
            } finally {
                // Re-enable predict button
                document.getElementById('predictBtn').disabled = false;
            }
        }
        
        // Initialize with two rows to encourage proper training
        addRow(); // First row
        addRow(); // Second row
    </script>
</body>
</html>"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model.deep_model is not None,
        'model_trained': model.is_trained,
        'classes': list(model.classes) if model.classes else []
    })

@app.route('/train', methods=['POST'])
def train():
    logger.info("=" * 50)
    logger.info("TRAIN REQUEST RECEIVED")
    logger.info("=" * 50)
    
    try:
        # Debug: print all request parts
        logger.debug(f"Files in request: {list(request.files.keys())}")
        logger.debug(f"Form data: {list(request.form.keys())}")
        
        files = request.files.getlist('images')
        labels = request.form.getlist('labels')
        
        logger.info(f"Number of files received: {len(files)}")
        logger.info(f"Number of labels received: {len(labels)}")
        logger.info(f"Labels: {labels}")
        
        # Print details of each file
        for i, file in enumerate(files):
            logger.info(f"File {i}: {file.filename}, content_type: {file.content_type}")
        
        if len(files) < 2:
            logger.error("Less than 2 files received")
            return jsonify({'error': 'Need at least 2 images'}), 400
        
        # Check if we have matching files and labels
        if len(files) != len(labels):
            logger.error(f"Mismatch - {len(files)} files vs {len(labels)} labels")
            return jsonify({'error': f'Number of files ({len(files)}) does not match number of labels ({len(labels)})'}), 400
        
        # Read image data into memory
        image_data_list = []
        valid_labels = []
        
        for i, file in enumerate(files):
            if file and file.filename:
                # Read file bytes
                file.seek(0)  # Ensure we're at the start of the file
                img_bytes = file.read()
                logger.info(f"File {i} ({file.filename}) size: {len(img_bytes)} bytes")
                
                if len(img_bytes) == 0:
                    logger.warning(f"File {i} ({file.filename}) is empty")
                    continue
                
                # Get corresponding label
                if i < len(labels):
                    label = labels[i].strip()
                    if label:
                        image_data_list.append(img_bytes)
                        valid_labels.append(label)
                        logger.info(f"✓ Added: {file.filename} with label '{label}'")
                    else:
                        logger.warning(f"Empty label for file {file.filename}")
                else:
                    logger.warning(f"No label for file {file.filename}")
        
        logger.info(f"Final - Valid images: {len(image_data_list)}, Valid labels: {len(valid_labels)}")
        logger.info(f"Unique labels: {set(valid_labels)}")
        
        if len(image_data_list) < 2:
            logger.error(f"Less than 2 valid images after processing: {len(image_data_list)}")
            return jsonify({'error': f'Need at least 2 valid images. Got {len(image_data_list)}'}), 400
        
        unique_classes = set(valid_labels)
        if len(unique_classes) < 2:
            logger.error(f"Need at least 2 different classes. Got: {unique_classes}")
            return jsonify({'error': f'Need at least 2 different classes. Got: {unique_classes}. Please use different labels like "cat" and "dog"'}), 400
        
        logger.info("Starting model training...")
        metrics = model.train_supervised(image_data_list, valid_labels)
        model.save()
        logger.info(f"Training successful: {metrics}")
        return jsonify({'success': True, 'metrics': metrics})
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("=" * 50)
    logger.info("PREDICT REQUEST RECEIVED")
    logger.info("=" * 50)
    
    try:
        if 'file' not in request.files:
            logger.error("No file in request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({'error': 'Empty filename'}), 400
        
        logger.info(f"File: {file.filename}, content_type: {file.content_type}")
        
        # Read file bytes directly
        file.seek(0)
        img_bytes = file.read()
        logger.info(f"File size: {len(img_bytes)} bytes")
        
        if len(img_bytes) == 0:
            logger.error("Empty file")
            return jsonify({'error': 'Empty file'}), 400
        
        predictions = model.predict(img_bytes)
        logger.info(f"Predictions: {predictions}")
        
        return jsonify({'success': True, 'predictions': predictions})
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
