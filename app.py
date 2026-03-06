from flask import Flask, render_template_string, request, jsonify, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import torch
import numpy as np
from transformers import AutoFeatureExtractor, AutoModel
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import os
import joblib
import io
import logging
from functools import wraps
import secrets

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(32)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///aau_ml.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('/tmp/model_cache', exist_ok=True)

# Database
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

with app.app_context():
    db.create_all()

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ML Model Class
class SupervisedDeepBoostingClassifier:
    def __init__(self, model_name="google/vit-base-patch16-224"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading model {model_name}...")
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                model_name, cache_dir='/tmp/model_cache'
            )
            self.deep_model = AutoModel.from_pretrained(
                model_name, cache_dir='/tmp/model_cache'
            ).to(self.device)
            self.deep_model.eval()
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        self.booster = None
        self.label_encoder = LabelEncoder()
        self.classes = []
        self.is_trained = False
    
    def extract_deep_features(self, image_data):
        try:
            if isinstance(image_data, str):
                image = Image.open(image_data).convert('RGB')
            elif isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
            else:
                image = image_data.convert('RGB')
            
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
            return None
    
    def train(self, image_data_list, labels):
        logger.info(f"Training with {len(image_data_list)} images...")
        
        features = []
        valid_labels = []
        
        for img_data, label in zip(image_data_list, labels):
            feat = self.extract_deep_features(img_data)
            if feat is not None:
                features.append(feat)
                valid_labels.append(label)
        
        if len(features) < 2:
            raise ValueError(f"Need at least 2 valid images. Got {len(features)}")
        
        unique_classes = set(valid_labels)
        if len(unique_classes) < 2:
            raise ValueError(f"Need at least 2 different classes. Got: {unique_classes}")
        
        X = np.array(features)
        y = self.label_encoder.fit_transform(valid_labels)
        self.classes = self.label_encoder.classes_
        num_classes = len(self.classes)
        
        dtrain = xgb.DMatrix(X, label=y)
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
        
        self.booster = xgb.train(params, dtrain, num_boost_round=200)
        
        train_pred_proba = self.booster.predict(dtrain)
        train_pred = np.argmax(train_pred_proba, axis=1)
        accuracy = np.mean(train_pred == y)
        
        self.is_trained = True
        return {
            'accuracy': float(accuracy),
            'samples': len(features),
            'classes': list(self.classes)
        }
    
    def predict(self, image_data, top_k=3):
        if not self.is_trained or self.booster is None:
            return [{'class': 'Model not trained', 'probability': '0%'}]
        
        features = self.extract_deep_features(image_data)
        if features is None:
            return [{'class': 'Error extracting features', 'probability': '0%'}]
        
        features = features.reshape(1, -1)
        dtest = xgb.DMatrix(features)
        probabilities = self.booster.predict(dtest)[0]
        
        probabilities = probabilities / np.sum(probabilities)
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        predictions = []
        for idx in top_indices:
            predictions.append({
                'class': self.classes[idx],
                'probability': f"{probabilities[idx]:.2%}"
            })
        
        return predictions

# Initialize model
model = SupervisedDeepBoostingClassifier()

# ==================== GLOBAL STYLES ====================
GLOBAL_STYLES = """
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    .aau-header {
        background: linear-gradient(135deg, #002B5C 0%, #1a4a7a 100%);
        color: white;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .aau-header h1 {
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .aau-header h2 {
        font-size: 1.2rem;
        font-weight: 400;
        opacity: 0.9;
    }
    .aau-header .program {
        background: rgba(255,255,255,0.2);
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        margin-top: 1rem;
        font-size: 0.9rem;
        letter-spacing: 1px;
    }
    .aau-footer {
        background: #002B5C;
        color: white;
        text-align: center;
        padding: 1.5rem;
        margin-top: 2rem;
        font-size: 0.9rem;
    }
    .container {
        max-width: 1400px;
        margin: 2rem auto;
        padding: 0 2rem;
    }
    .auth-container {
        max-width: 400px;
        margin: 3rem auto;
        background: white;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        padding: 2.5rem;
    }
    .auth-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .form-group {
        margin-bottom: 1.5rem;
    }
    .form-group label {
        display: block;
        margin-bottom: 0.5rem;
        color: #333;
        font-weight: 500;
    }
    .form-group input {
        width: 100%;
        padding: 0.8rem;
        border: 2px solid #e1e5e9;
        border-radius: 8px;
        font-size: 1rem;
        transition: border-color 0.3s;
    }
    .form-group input:focus {
        border-color: #002B5C;
        outline: none;
    }
    .btn {
        background: linear-gradient(135deg, #002B5C 0%, #1a4a7a 100%);
        color: white;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: transform 0.3s, box-shadow 0.3s;
        width: 100%;
        display: inline-block;
        text-decoration: none;
        text-align: center;
    }
    .btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,43,92,0.3);
    }
    .btn-secondary {
        background: #6c757d;
    }
    .btn-small {
        width: auto;
        padding: 0.5rem 1rem;
        font-size: 0.9rem;
    }
    .alert {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .alert-success {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .alert-danger {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .alert-warning {
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
    }
    .grid-2 {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2rem;
    }
    .card {
        background: white;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        padding: 2rem;
    }
    .image-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
        align-items: center;
    }
    .result-item {
        display: flex;
        justify-content: space-between;
        padding: 1rem;
        background: #f8f9fa;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 4px solid #002B5C;
    }
    .loading {
        text-align: center;
        padding: 2rem;
    }
    .loading-spinner {
        display: inline-block;
        width: 40px;
        height: 40px;
        border: 4px solid #f3f3f3;
        border-top: 4px solid #002B5C;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    .thumbnail { max-width: 50px; max-height: 50px; border-radius: 5px; margin-left: 10px; }
    .status-badge {
        background: #28a745;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 50px;
        font-size: 0.8rem;
        display: inline-block;
    }
    .flex-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
    }
    .upload-area {
        border: 3px dashed #002B5C;
        padding: 2rem;
        text-align: center;
        border-radius: 10px;
        cursor: pointer;
        margin-bottom: 1rem;
    }
    .upload-area:hover {
        background: #f0f4f8;
    }
    @media (max-width: 768px) {
        .grid-2 { grid-template-columns: 1fr; }
    }
</style>
"""

# ==================== GLOBAL SCRIPTS ====================
GLOBAL_SCRIPTS = """
<script>
    function showLoading(elementId) {
        document.getElementById(elementId).innerHTML = '<div class="loading"><div class="loading-spinner"></div><p style="margin-top:1rem;">Processing...</p></div>';
    }
    
    function showError(elementId, message) {
        document.getElementById(elementId).innerHTML = `<div class="alert alert-danger">❌ ${message}</div>`;
    }
    
    function showSuccess(elementId, message) {
        document.getElementById(elementId).innerHTML = `<div class="alert alert-success">✅ ${message}</div>`;
    }
</script>
"""

# ==================== PAGE TEMPLATES ====================

INDEX_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AAU ML Image Classifier</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    """ + GLOBAL_STYLES + """
</head>
<body>
    <div class="aau-header">
        <h1>🎯 Addis Ababa University</h1>
        <h2>College of Natural and Computational Sciences</h2>
        <div class="program">Department of Computer Science | MSc in Network & Security</div>
    </div>
    
    <div class="container">
        <div style="text-align: center; padding: 3rem; background: white; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
            <h1 style="color: #002B5C; font-size: 2.5rem; margin-bottom: 1.5rem;">🔬 ML Image Classifier</h1>
            <p style="font-size: 1.2rem; color: #666; margin-bottom: 2rem; max-width: 800px; margin-left: auto; margin-right: auto;">
                A supervised learning solution combining Vision Transformers (ViT) and XGBoost boosting algorithm
            </p>
            
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem; margin: 3rem 0;">
                <div style="padding: 2rem; background: #f8f9fa; border-radius: 15px;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">🧠</div>
                    <h3>ViT Feature Extraction</h3>
                    <p style="color: #666;">Google's Vision Transformer extracts 768-dimensional features</p>
                </div>
                <div style="padding: 2rem; background: #f8f9fa; border-radius: 15px;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">⚡</div>
                    <h3>XGBoost Classification</h3>
                    <p style="color: #666;">Gradient boosting for accurate multi-class predictions</p>
                </div>
                <div style="padding: 2rem; background: #f8f9fa; border-radius: 15px;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">🔐</div>
                    <h3>Secure Authentication</h3>
                    <p style="color: #666;">Student ID-based access for AAU community</p>
                </div>
            </div>
            
            <div style="display: flex; gap: 1rem; justify-content: center;">
                <a href="/login" class="btn" style="width: auto; padding: 1rem 3rem;">🔑 Login</a>
                <a href="/register" class="btn btn-secondary" style="width: auto; padding: 1rem 3rem;">📝 Register</a>
            </div>
        </div>
    </div>
    
    <div class="aau-footer">
        <p>© 2026 Addis Ababa University | ML Image Classifier | Supervised Learning with ViT + XGBoost</p>
        <p style="font-size:0.8rem; opacity:0.8; margin-top:0.5rem;">Developed for Research Method Course</p>
    </div>
</body>
</html>
"""

LOGIN_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AAU ML Image Classifier - Login</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    """ + GLOBAL_STYLES + """
</head>
<body>
    <div class="aau-header">
        <h1>🎯 Addis Ababa University</h1>
        <h2>College of Natural and Computational Sciences</h2>
        <div class="program">Department of Computer Science | MSc in Network & Security</div>
    </div>
    
    <div class="container">
        <div class="auth-container">
            <div class="auth-header">
                <h2 style="color: #002B5C;">🔐 Student Login</h2>
                <p style="color: #666; margin-top: 0.5rem;">Access your ML workspace</p>
            </div>
            
            {% with messages = get_flashed_messages(with_categories=true) %}
                {% if messages %}
                    {% for category, message in messages %}
                        <div class="alert alert-{{ category }}">{{ message }}</div>
                    {% endfor %}
                {% endif %}
            {% endwith %}
            
            <form method="POST">
                <div class="form-group">
                    <label>🎓 Student ID</label>
                    <input type="text" name="student_id" required placeholder="e.g., UGR/1234/12">
                </div>
                <div class="form-group">
                    <label>🔑 Password</label>
                    <input type="password" name="password" required>
                </div>
                <button type="submit" class="btn">Login</button>
            </form>
            
            <p style="text-align: center; margin-top: 1.5rem;">
                Don't have an account? <a href="/register" style="color: #002B5C;">Register here</a>
            </p>
        </div>
    </div>
    
    <div class="aau-footer">
        <p>© 2026 Addis Ababa University | ML Image Classifier</p>
    </div>
</body>
</html>
"""

REGISTER_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AAU ML Image Classifier - Register</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    """ + GLOBAL_STYLES + """
</head>
<body>
    <div class="aau-header">
        <h1>🎯 Addis Ababa University</h1>
        <h2>College of Natural and Computational Sciences</h2>
        <div class="program">Department of Computer Science | MSc in Network & Security</div>
    </div>
    
    <div class="container">
        <div class="auth-container">
            <div class="auth-header">
                <h2 style="color: #002B5C;">📝 Student Registration</h2>
                <p style="color: #666; margin-top: 0.5rem;">Join the AAU ML community</p>
            </div>
            
            {% with messages = get_flashed_messages(with_categories=true) %}
                {% if messages %}
                    {% for category, message in messages %}
                        <div class="alert alert-{{ category }}">{{ message }}</div>
                    {% endfor %}
                {% endif %}
            {% endwith %}
            
            <form method="POST">
                <div class="form-group">
                    <label>👤 Full Name</label>
                    <input type="text" name="full_name" required>
                </div>
                <div class="form-group">
                    <label>🎓 Student ID</label>
                    <input type="text" name="student_id" required placeholder="e.g., UGR/1234/12">
                </div>
                <div class="form-group">
                    <label>📧 Email</label>
                    <input type="email" name="email" required placeholder="@student.aau.edu.et">
                </div>
                <div class="form-group">
                    <label>🔑 Password</label>
                    <input type="password" name="password" required>
                </div>
                <div class="form-group">
                    <label>✓ Confirm Password</label>
                    <input type="password" name="confirm_password" required>
                </div>
                <button type="submit" class="btn">Register</button>
            </form>
            
            <p style="text-align: center; margin-top: 1.5rem;">
                Already registered? <a href="/login" style="color: #002B5C;">Login here</a>
            </p>
        </div>
    </div>
    
    <div class="aau-footer">
        <p>© 2026 Addis Ababa University | ML Image Classifier</p>
    </div>
</body>
</html>
"""

DASHBOARD_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AAU ML Image Classifier - Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    """ + GLOBAL_STYLES + """
</head>
<body>
    <div class="aau-header">
        <h1>🎯 Addis Ababa University</h1>
        <h2>College of Natural and Computational Sciences</h2>
        <div class="program">Department of Computer Science | MSc in Network & Security</div>
    </div>
    
    <div class="container">
        <div class="flex-row">
            <h2 style="color: #002B5C;">👋 Welcome, {{ session.user_name }}</h2>
            <div>
                <span class="status-badge">Model Trained: {{ 'Yes' if model_trained else 'No' }}</span>
                <a href="/logout" class="btn btn-small" style="margin-left: 1rem;">🚪 Logout</a>
            </div>
        </div>

        <div class="grid-2">
            <!-- Training Section -->
            <div class="card">
                <h3 style="margin-bottom: 1.5rem;">📚 Train Model</h3>
                <p style="color: #666; margin-bottom: 1rem;">Upload labeled images <strong>(minimum 3 images per class, 2+ classes)</strong></p>
                
                <div id="imageRows"></div>
                <button class="btn" onclick="addRow()" style="margin-bottom: 1rem;">+ Add Image</button>
                <button class="btn" onclick="trainModel()" id="trainBtn">🚀 Start Training</button>
                <div id="trainingResult" style="margin-top: 1rem;"></div>
            </div>
            
            <!-- Classification Section -->
            <div class="card">
                <h3 style="margin-bottom: 1.5rem;">🎯 Classify Image</h3>
                
                <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                    <input type="file" id="fileInput" accept="image/*" style="display:none;">
                    <p style="font-size: 3rem; margin-bottom: 0.5rem;">📸</p>
                    <p>Click to select image</p>
                </div>
                
                <img id="preview" src="#" style="max-width: 100%; max-height: 200px; display: none; margin: 1rem 0; border-radius: 10px;">
                <button class="btn" onclick="predictImage()" id="predictBtn">🔍 Classify</button>
                <div id="predictionResult" style="margin-top: 1rem;"></div>
            </div>
        </div>
    </div>
    
    <div class="aau-footer">
        <p>© 2026 Addis Ababa University | ML Image Classifier</p>
    </div>

    """ + GLOBAL_SCRIPTS + """
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
                <button onclick="removeRow(${rowCount})" style="background:#dc3545; color:white; border:none; padding:5px 10px; border-radius:5px;">✕</button>
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
            let totalImages = 0;
            
            for (let row of rows) {
                const fileInput = row.querySelector('input[type="file"]');
                const labelInput = row.querySelector('input[type="text"]');
                
                if (fileInput.files[0] && labelInput.value) {
                    totalImages++;
                    labels.add(labelInput.value.trim());
                }
            }
            
            return {
                totalImages: totalImages,
                numClasses: labels.size
            };
        }
        
        async function trainModel() {
            const validation = validateTrainingData();
            
            if (validation.totalImages < 4) {
                showError('trainingResult', 'Need at least 4 total images (3+ per class recommended)');
                return;
            }
            
            if (validation.numClasses < 2) {
                showError('trainingResult', 'Need at least 2 different classes');
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
            
            document.getElementById('trainBtn').disabled = true;
            showLoading('trainingResult');
            
            try {
                const response = await fetch('/train', { method: 'POST', body: formData });
                const data = await response.json();
                
                if (data.success) {
                    showSuccess('trainingResult', 
                        `Training Complete!<br>Accuracy: ${(data.metrics.accuracy * 100).toFixed(2)}%<br>Classes: ${data.metrics.classes.join(', ')}<br>Samples: ${data.metrics.samples}`
                    );
                } else {
                    showError('trainingResult', data.error);
                }
            } catch (error) {
                showError('trainingResult', error.message);
            } finally {
                document.getElementById('trainBtn').disabled = false;
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
            
            document.getElementById('predictBtn').disabled = true;
            showLoading('predictionResult');
            
            try {
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const data = await response.json();
                
                if (data.success) {
                    let html = '<h4 style="margin-bottom:1rem;">Results:</h4>';
                    data.predictions.forEach(p => {
                        html += `<div class="result-item"><span>${p.class}</span><span>${p.probability}</span></div>`;
                    });
                    document.getElementById('predictionResult').innerHTML = html;
                } else {
                    showError('predictionResult', data.error);
                }
            } catch (error) {
                showError('predictionResult', error.message);
            } finally {
                document.getElementById('predictBtn').disabled = false;
            }
        }
        
        // Initialize with 3 rows
        addRow(); addRow(); addRow();
    </script>
</body>
</html>
"""

# ==================== ROUTES ====================

@app.route('/')
def index():
    return render_template_string(INDEX_PAGE)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        student_id = request.form.get('student_id')
        password = request.form.get('password')
        
        user = User.query.filter_by(student_id=student_id).first()
        
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['user_name'] = user.full_name
            session.permanent = True
            flash(f'Welcome back, {user.full_name}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid Student ID or Password', 'danger')
    
    return render_template_string(LOGIN_PAGE)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        student_id = request.form.get('student_id')
        email = request.form.get('email')
        full_name = request.form.get('full_name')
        password = request.form.get('password')
        confirm = request.form.get('confirm_password')
        
        if password != confirm:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('register'))
        
        existing = User.query.filter_by(student_id=student_id).first()
        if existing:
            flash('Student ID already registered', 'danger')
            return redirect(url_for('register'))
        
        user = User(student_id=student_id, email=email, full_name=full_name)
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template_string(REGISTER_PAGE)

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template_string(
        DASHBOARD_PAGE,
        model_trained=model.is_trained,
        session=session
    )

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'success')
    return redirect(url_for('index'))

@app.route('/train', methods=['POST'])
@login_required
def train():
    try:
        files = request.files.getlist('images')
        labels = request.form.getlist('labels')
        
        if len(files) < 4:
            return jsonify({'error': 'Need at least 4 images (3+ per class recommended)'}), 400
        
        image_data_list = []
        valid_labels = []
        
        for i, file in enumerate(files):
            if file and file.filename and i < len(labels):
                img_bytes = file.read()
                label = labels[i].strip()
                if len(img_bytes) > 0 and label:
                    image_data_list.append(img_bytes)
                    valid_labels.append(label)
        
        if len(image_data_list) < 4:
            return jsonify({'error': f'Only {len(image_data_list)} valid images. Need at least 4'}), 400
        
        metrics = model.train(image_data_list, valid_labels)
        
        return jsonify({'success': True, 'metrics': metrics})
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        img_bytes = file.read()
        
        if len(img_bytes) == 0:
            return jsonify({'error': 'Empty file'}), 400
        
        predictions = model.predict(img_bytes)
        return jsonify({'success': True, 'predictions': predictions})
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
