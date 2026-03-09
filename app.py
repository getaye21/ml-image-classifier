"""
====================================================================
ADDIS ABABA UNIVERSITY - College of Natural and Computational Sciences
Department of Computer Science
====================================================================

Project: Hybrid Deep Learning-Boosting Classifier 
Student Name: Getaye Fiseha
Student ID: GSE/6132/18
Program: MSc in Computer Science
Stream: Network & Security
Course: COSC 6041 - Machine Learning
Supervisor: Dr. Yaregal A.
Submission Date: March 2026

FOCUS: REAL Image Classification - Animals, Plants, Objects
====================================================================
"""

from flask import Flask, render_template_string, request, jsonify, session, redirect, url_for, flash
import torch
import numpy as np
from transformers import AutoFeatureExtractor, AutoModel
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import os
import uuid
import joblib
import hashlib
import secrets
from datetime import datetime, timedelta
from functools import wraps
import re
import json
import glob
from werkzeug.utils import secure_filename
import requests
from io import BytesIO

# ============================================================================
# Application Configuration
# ============================================================================

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['MAX_LOGIN_ATTEMPTS'] = 5
app.config['LOCKOUT_TIME'] = timedelta(minutes=15)

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('users', exist_ok=True)
os.makedirs('training_data', exist_ok=True)

# ============================================================================
# Secure User Authentication System
# ============================================================================

class SecureUserManager:
    def __init__(self):
        self.users_file = 'users/users.json'
        self.login_attempts = {}
        
    def hash_password(self, password):
        salt = "AAU_SALT"
        return hashlib.sha256((password + salt).encode()).hexdigest()
    
    def verify_password(self, password, hashed):
        return self.hash_password(password) == hashed
    
    def check_login_attempts(self, username, ip_address):
        key = f"{username}_{ip_address}"
        if key in self.login_attempts:
            attempts, first_attempt = self.login_attempts[key]
            if attempts >= app.config['MAX_LOGIN_ATTEMPTS']:
                if datetime.now() - first_attempt < app.config['LOCKOUT_TIME']:
                    return False, "Too many failed attempts. Account locked for 15 minutes."
                else:
                    del self.login_attempts[key]
        return True, "OK"
    
    def record_failed_attempt(self, username, ip_address):
        key = f"{username}_{ip_address}"
        if key in self.login_attempts:
            attempts, first_attempt = self.login_attempts[key]
            self.login_attempts[key] = (attempts + 1, first_attempt)
        else:
            self.login_attempts[key] = (1, datetime.now())
    
    def login_user(self, username, password, ip_address):
        allowed, msg = self.check_login_attempts(username, ip_address)
        if not allowed:
            return False, msg
        
        default_users = {
            'getaye': {
                'password_hash': self.hash_password('Getaye@6132'),
                'full_name': 'Getaye Fiseha',
                'role': 'student',
                'student_id': 'GSE/6132/18',
                'email': 'getaye.fiseha@aau.edu.et'
            },
            'guest': {
                'password_hash': self.hash_password('Guest@2026'),
                'full_name': 'Guest User',
                'role': 'guest',
                'student_id': 'GUEST/001',
                'email': 'guest@aau.edu.et'
            }
        }
        
        if username in default_users:
            if self.verify_password(password, default_users[username]['password_hash']):
                return True, default_users[username]
        
        self.record_failed_attempt(username, ip_address)
        return False, "Invalid username or password"

user_manager = SecureUserManager()

# ============================================================================
# Login Required Decorator
# ============================================================================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Please login to access this page', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ============================================================================
# REAL IMAGE CLASSIFIER - Actually Learns from Images
# ============================================================================

class RealImageClassifier:
    """
    REAL image classifier that actually learns from training data
    Supports: Animals, Plants, Objects, People, Vehicles, etc.
    """
    
    def __init__(self, model_name="google/vit-base-patch16-224"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing on device: {self.device}")
        
        # Load Hugging Face ViT
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.cnn = AutoModel.from_pretrained(model_name).to(self.device)
        self.cnn.eval()
        
        # Initialize XGBoost (will be trained later)
        self.boosting = None
        self.label_encoder = LabelEncoder()
        self.classes = []
        self.is_trained = False
        
        # Try to load existing model
        self.load_model()
    
    def extract_features(self, image):
        """Extract deep features from image"""
        try:
            # Handle both file paths and PIL Images
            if isinstance(image, str):
                img = Image.open(image).convert('RGB')
            else:
                img = image.convert('RGB')
            
            inputs = self.feature_extractor(images=img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.cnn(**inputs)
                features = outputs.last_hidden_state.mean(dim=1)
            
            return features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
    
    def train(self, image_paths, labels):
        """
        Train the classifier on real images
        """
        print(f"\n📚 Training on {len(image_paths)} images...")
        
        # Extract features from all images
        features = []
        valid_labels = []
        
        for img_path, label in zip(image_paths, labels):
            if os.path.exists(img_path):
                feat = self.extract_features(img_path)
                if feat is not None:
                    features.append(feat)
                    valid_labels.append(label)
                    print(f"   ✓ Processed: {os.path.basename(img_path)} -> {label}")
        
        if len(features) == 0:
            raise ValueError("No valid features extracted!")
        
        X = np.array(features)
        y = self.label_encoder.fit_transform(valid_labels)
        self.classes = self.label_encoder.classes_
        
        print(f"\n📊 Training Data:")
        print(f"   Features shape: {X.shape}")
        print(f"   Classes: {list(self.classes)}")
        
        # Split into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize and train XGBoost
        self.boosting = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        )
        
        print("\n🚀 Training XGBoost...")
        self.boosting.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Evaluate
        train_pred = self.boosting.predict(X_train)
        val_pred = self.boosting.predict(X_val)
        
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        
        self.is_trained = True
        
        print(f"\n✅ Training Complete!")
        print(f"   Training Accuracy: {train_acc:.2%}")
        print(f"   Validation Accuracy: {val_acc:.2%}")
        
        # Save the model
        self.save_model()
        
        return {
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'classes': list(self.classes),
            'num_classes': len(self.classes),
            'num_samples': len(features)
        }
    
    def predict(self, image_path, top_k=3):
        """Predict class for a new image"""
        if not self.is_trained or self.boosting is None:
            return self._get_training_needed_message()
        
        features = self.extract_features(image_path)
        if features is None:
            return [{'class': 'Error extracting features', 'confidence': 0, 'probability': '0%'}]
        
        features = features.reshape(1, -1)
        
        # Get probabilities
        probabilities = self.boosting.predict_proba(features)[0]
        
        # Get top-k predictions
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        predictions = []
        for idx in top_indices:
            predictions.append({
                'class': self.classes[idx],
                'confidence': float(probabilities[idx]),
                'probability': f"{probabilities[idx]:.2%}"
            })
        
        return predictions
    
    def _get_training_needed_message(self):
        """Return message when model needs training"""
        return [{
            'class': '⚠️ Model Not Trained',
            'confidence': 0,
            'probability': 'Please upload training images first!'
        }]
    
    def save_model(self, path='real_classifier_model.pkl'):
        """Save trained model"""
        if self.boosting is not None:
            joblib.dump({
                'boosting': self.boosting,
                'encoder': self.label_encoder,
                'classes': self.classes,
                'is_trained': self.is_trained
            }, path)
            print(f"💾 Model saved to {path}")
    
    def load_model(self, path='real_classifier_model.pkl'):
        """Load trained model"""
        if os.path.exists(path):
            data = joblib.load(path)
            self.boosting = data['boosting']
            self.label_encoder = data['encoder']
            self.classes = data['classes']
            self.is_trained = data['is_trained']
            print(f"📂 Loaded model with {len(self.classes)} classes")
            return True
        return False

# Initialize the REAL classifier
classifier = RealImageClassifier()

# ============================================================================
# Routes - Authentication
# ============================================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        ip_address = request.remote_addr
        
        success, result = user_manager.login_user(username, password, ip_address)
        
        if success:
            session['username'] = username
            session['full_name'] = result['full_name']
            session['role'] = result['role']
            session['student_id'] = result['student_id']
            session.permanent = True
            flash(f'Welcome, {result["full_name"]}!', 'success')
            return redirect(url_for('index'))
        else:
            error = result
    
    return render_template_string(LOGIN_HTML, error=error)

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

# ============================================================================
# Main Routes
# ============================================================================

@app.route('/')
def index():
    """Home page - Classification Interface"""
    return render_template_string(INDEX_HTML, session=session, 
                                 is_trained=classifier.is_trained,
                                 classes=classifier.classes)

@app.route('/train', methods=['GET', 'POST'])
@login_required
def train():
    """Training interface - Upload labeled images"""
    if request.method == 'POST':
        files = request.files.getlist('images')
        labels = request.form.getlist('labels')
        
        if len(files) != len(labels):
            return jsonify({'error': 'Number of images and labels must match'}), 400
        
        # Save uploaded files
        image_paths = []
        for file in files:
            if file and file.filename:
                filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                image_paths.append(filepath)
        
        try:
            # Train the classifier
            metrics = classifier.train(image_paths, labels)
            return jsonify({'success': True, 'metrics': metrics})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            # Cleanup
            for path in image_paths:
                if os.path.exists(path):
                    os.remove(path)
    
    return render_template_string(TRAIN_HTML, session=session)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict endpoint"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save temp file
    filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Get predictions
        predictions = classifier.predict(filepath)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'filename': file.filename,
            'is_trained': classifier.is_trained
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/model-status')
def model_status():
    """Get model status"""
    return jsonify({
        'is_trained': classifier.is_trained,
        'classes': list(classifier.classes) if classifier.is_trained else [],
        'num_classes': len(classifier.classes) if classifier.is_trained else 0
    })

@app.route('/profile')
@login_required
def profile():
    return render_template_string(PROFILE_HTML, session=session)

@app.route('/sample-dataset')
@login_required
def sample_dataset():
    """Download and prepare a sample dataset for testing"""
    try:
        # Create sample directories
        sample_dir = 'sample_training'
        os.makedirs(sample_dir, exist_ok=True)
        
        # Sample image URLs (you can add more)
        samples = {
            'cat': ['https://images.pexels.com/photos/45201/kitty-cat-kitten-pet-45201.jpeg'],
            'dog': ['https://images.pexels.com/photos/1805164/pexels-photo-1805164.jpeg'],
            'bird': ['https://images.pexels.com/photos/326900/pexels-photo-326900.jpeg'],
            'elephant': ['https://images.pexels.com/photos/66898/elephant-calf-african-elephant-africa-66898.jpeg'],
            'flower': ['https://images.pexels.com/photos/736230/pexels-photo-736230.jpeg']
        }
        
        image_paths = []
        labels = []
        
        for class_name, urls in samples.items():
            class_dir = os.path.join(sample_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for i, url in enumerate(urls):
                try:
                    response = requests.get(url)
                    img_path = os.path.join(class_dir, f"{i}.jpg")
                    with open(img_path, 'wb') as f:
                        f.write(response.content)
                    image_paths.append(img_path)
                    labels.append(class_name)
                except:
                    pass
        
        if image_paths:
            metrics = classifier.train(image_paths, labels)
            return jsonify({'success': True, 'metrics': metrics})
        else:
            return jsonify({'error': 'Could not download samples'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# HTML Templates
# ============================================================================

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AAU - Real Image Classifier | Getaye Fiseha GSE/6132/18</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }
        
        .aau-header { 
            background: linear-gradient(135deg, #006B3F 0%, #FCD116 100%); 
            padding: 20px; 
            color: white; 
            text-align: center; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .aau-header h1 { font-size: 32px; margin-bottom: 5px; }
        .aau-header h2 { font-size: 18px; font-weight: normal; opacity: 0.9; }
        
        .navbar { 
            background: #006B3F; 
            padding: 15px 30px; 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            color: white; 
        }
        .nav-links a { 
            color: white; 
            text-decoration: none; 
            margin-left: 20px; 
            padding: 8px 16px; 
            border-radius: 5px; 
            transition: all 0.3s;
        }
        .nav-links a:hover { 
            background: #FCD116; 
            color: #006B3F; 
        }
        .student-badge { 
            background: #FCD116; 
            color: #006B3F; 
            padding: 5px 15px; 
            border-radius: 20px; 
            font-weight: bold;
        }
        
        .container { max-width: 1200px; margin: 40px auto; padding: 0 20px; }
        
        .model-status {
            background: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 5px solid #006B3F;
        }
        
        .status-trained {
            color: #006B3F;
            font-weight: bold;
        }
        
        .status-untrained {
            color: #FCD116;
            font-weight: bold;
        }
        
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }
        
        .card {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        
        .upload-area { 
            border: 3px dashed #006B3F; 
            border-radius: 20px; 
            padding: 40px; 
            text-align: center; 
            cursor: pointer; 
            transition: all 0.3s; 
            margin: 20px 0;
        }
        .upload-area:hover { 
            border-color: #FCD116; 
            background: #fff9e6;
        }
        
        .btn { 
            background: linear-gradient(135deg, #006B3F, #FCD116); 
            color: white; 
            border: none; 
            padding: 15px 30px; 
            border-radius: 10px; 
            font-size: 16px; 
            font-weight: 600; 
            cursor: pointer; 
            transition: all 0.3s; 
            width: 100%;
            margin: 10px 0;
        }
        .btn:hover { 
            transform: translateY(-2px); 
            box-shadow: 0 10px 20px rgba(0,107,63,0.3);
        }
        
        .btn-secondary {
            background: #f0f0f0;
            color: #333;
        }
        
        #preview { 
            max-width: 100%; 
            max-height: 300px; 
            display: none; 
            margin: 20px auto; 
            border-radius: 10px; 
            border: 4px solid #FCD116;
        }
        
        .prediction-item { 
            display: flex; 
            justify-content: space-between; 
            padding: 15px; 
            background: #f8f9fa; 
            margin: 10px 0; 
            border-radius: 10px; 
        }
        .top-prediction { 
            border-left: 8px solid #FCD116; 
            background: #fff9e6; 
        }
        
        .classes-list {
            margin-top: 20px;
            padding: 15px;
            background: #f0f0f0;
            border-radius: 10px;
        }
        
        .class-tag {
            display: inline-block;
            padding: 5px 15px;
            background: #006B3F;
            color: white;
            border-radius: 20px;
            margin: 5px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="aau-header">
        <h1>🎓 ADDIS ABABA UNIVERSITY</h1>
        <h2>College of Natural and Computational Sciences - Department of Computer Science</h2>
    </div>
    
    <div class="navbar">
        <div>
            <span class="student-badge">GSE/6132/18 | Getaye Fiseha</span>
        </div>
        <div class="nav-links">
            <a href="/">Home</a>
            {% if session.username %}
                <a href="/train">Train Model</a>
                <a href="/profile">{{ session.full_name }}</a>
                <a href="/logout">Logout</a>
            {% else %}
                <a href="/login">Login</a>
            {% endif %}
        </div>
    </div>
    
    <div class="container">
        <div class="model-status">
            <h3>🤖 Model Status: 
                {% if is_trained %}
                <span class="status-trained">✓ TRAINED ({{ classes|length }} classes)</span>
                {% else %}
                <span class="status-untrained">⚠ NOT TRAINED - Please train first</span>
                {% endif %}
            </h3>
        </div>
        
        <div class="grid-2">
            <!-- Classification Card -->
            <div class="card">
                <h2 style="color: #006B3F;">🔍 Classify Image</h2>
                <p>Upload any image to classify (animals, plants, objects, etc.)</p>
                
                <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                    <input type="file" id="fileInput" accept="image/*" style="display: none;">
                    <p style="font-size: 24px;">📸</p>
                    <p>Click to select image</p>
                </div>
                
                <img id="preview" src="#" alt="Preview">
                
                <button class="btn" onclick="classifyImage()">🔍 Classify Image</button>
                
                <div id="results" style="margin-top: 20px;"></div>
            </div>
            
            <!-- Training Card -->
            <div class="card">
                <h2 style="color: #006B3F;">📚 Train Model</h2>
                <p>Upload labeled images to teach the model</p>
                
                {% if session.username %}
                <div id="trainingArea">
                    <div id="imageRows"></div>
                    <button class="btn btn-secondary" onclick="addRow()">+ Add Image</button>
                    <button class="btn" onclick="trainModel()">🚀 Start Training</button>
                </div>
                <div id="trainingResult" style="margin-top: 20px;"></div>
                {% else %}
                <p>Please <a href="/login">login</a> to train the model</p>
                {% endif %}
            </div>
        </div>
        
        {% if is_trained %}
        <div class="classes-list">
            <h4>📋 Trained Classes:</h4>
            {% for class in classes %}
            <span class="class-tag">{{ class }}</span>
            {% endfor %}
        </div>
        {% endif %}
    </div>
    
    <script>
        let rowCount = 0;
        
        function addRow() {
            const container = document.getElementById('imageRows');
            const row = document.createElement('div');
            row.id = `row_${rowCount}`;
            row.style.marginBottom = '10px';
            row.innerHTML = `
                <input type="file" accept="image/*" style="margin-right: 10px;">
                <input type="text" id="label_${rowCount}" placeholder="Label (e.g., cat, dog, flower)" style="padding: 5px;">
                <button onclick="removeRow(${rowCount})" style="background: #dc3545; color: white; border: none; padding: 5px 10px;">✕</button>
            `;
            container.appendChild(row);
            rowCount++;
        }
        
        function removeRow(id) {
            const row = document.getElementById(`row_${id}`);
            if (row) row.remove();
        }
        
        document.getElementById('fileInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('preview').src = e.target.result;
                    document.getElementById('preview').style.display = 'block';
                }
                reader.readAsDataURL(file);
            }
        });
        
        async function classifyImage() {
            const fileInput = document.getElementById('fileInput');
            if (!fileInput.files[0]) {
                alert('Please select an image');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '⏳ Classifying...';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    let html = '<h3>Results:</h3>';
                    data.predictions.forEach((pred, i) => {
                        const bgClass = i === 0 ? 'top-prediction' : '';
                        html += `
                            <div class="prediction-item ${bgClass}">
                                <span><strong>${i === 0 ? '🏆 ' : ''}${pred.class}</strong></span>
                                <span style="background: #006B3F; color: white; padding: 5px 15px; border-radius: 20px;">${pred.probability}</span>
                            </div>
                        `;
                    });
                    resultsDiv.innerHTML = html;
                } else {
                    resultsDiv.innerHTML = `❌ Error: ${data.error}`;
                }
            } catch (error) {
                resultsDiv.innerHTML = `❌ Error: ${error.message}`;
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
            
            const resultDiv = document.getElementById('trainingResult');
            resultDiv.innerHTML = '⏳ Training...';
            
            try {
                const response = await fetch('/train', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    resultDiv.innerHTML = `
                        <div style="background: #d4edda; color: #155724; padding: 15px; border-radius: 10px;">
                            <strong>✅ Training Complete!</strong><br>
                            Training Accuracy: ${(data.metrics.train_accuracy * 100).toFixed(2)}%<br>
                            Validation Accuracy: ${(data.metrics.val_accuracy * 100).toFixed(2)}%<br>
                            Classes: ${data.metrics.classes.join(', ')}
                        </div>
                    `;
                    setTimeout(() => location.reload(), 2000);
                } else {
                    resultDiv.innerHTML = `<div style="background: #f8d7da; color: #721c24; padding: 15px;">❌ Error: ${data.error}</div>`;
                }
            } catch (error) {
                resultDiv.innerHTML = `<div style="background: #f8d7da; color: #721c24; padding: 15px;">❌ Error: ${error.message}</div>`;
            }
        }
        
        // Add initial row
        {% if session.username %}
        addRow();
        {% endif %}
    </script>
</body>
</html>
"""

TRAIN_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AAU - Train Classifier | Getaye Fiseha</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; }
        .container { max-width: 800px; margin: 40px auto; padding: 20px; }
        .card { background: white; border-radius: 15px; padding: 30px; box-shadow: 0 5px 20px rgba(0,0,0,0.1); }
        .btn { background: linear-gradient(135deg, #006B3F, #FCD116); color: white; border: none; 
               padding: 12px 24px; border-radius: 5px; cursor: pointer; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h2 style="color: #006B3F;">📚 Train Image Classifier</h2>
            <p>Upload labeled images to teach the model to recognize:</p>
            <ul>
                <li>🐱 Animals (cat, dog, elephant, etc.)</li>
                <li>🌿 Plants (flower, tree, etc.)</li>
                <li>📦 Objects (car, book, phone, etc.)</li>
                <li>👥 People</li>
                <li>🌍 Anything you want!</li>
            </ul>
            <a href="/" class="btn">← Back to Classifier</a>
        </div>
    </div>
</body>
</html>
"""

PROFILE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AAU - Profile | Getaye Fiseha</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; }
        .container { max-width: 600px; margin: 40px auto; padding: 20px; }
        .card { background: white; border-radius: 15px; padding: 30px; box-shadow: 0 5px 20px rgba(0,0,0,0.1); }
        .info-item { padding: 15px; background: #f8f9fa; margin: 10px 0; border-radius: 8px; 
                    border-left: 4px solid #FCD116; }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h2 style="color: #006B3F;">🎓 Student Profile</h2>
            <div class="info-item"><strong>Name:</strong> {{ session.full_name }}</div>
            <div class="info-item"><strong>ID:</strong> {{ session.student_id }}</div>
            <div class="info-item"><strong>Email:</strong> {{ session.email }}</div>
            <div class="info-item"><strong>Role:</strong> {{ session.role }}</div>
            <a href="/" style="display: inline-block; margin-top: 20px;">← Back</a>
        </div>
    </div>
</body>
</html>
"""

LOGIN_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AAU - Login</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
               background: linear-gradient(135deg, #006B3F 0%, #FCD116 100%); 
               min-height: 100vh; display: flex; align-items: center; justify-content: center; }
        .login-container { background: white; padding: 40px; border-radius: 15px; 
                          box-shadow: 0 20px 60px rgba(0,0,0,0.3); width: 100%; max-width: 400px; }
        .aau-header { text-align: center; margin-bottom: 30px; color: #006B3F; }
        .form-group { margin-bottom: 20px; }
        .form-group input { width: 100%; padding: 12px; border: 2px solid #e1e1e1; 
                           border-radius: 8px; }
        .btn { width: 100%; padding: 14px; background: linear-gradient(135deg, #006B3F 0%, #FCD116 100%); 
               color: white; border: none; border-radius: 8px; cursor: pointer; }
        .info { text-align: center; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="aau-header">
            <h2>🎓 ADDIS ABABA UNIVERSITY</h2>
            <p>MSc Computer Science</p>
        </div>
        {% if error %}<div style="color: red; margin-bottom: 20px;">{{ error }}</div>{% endif %}
        <form method="POST">
            <div class="form-group">
                <input type="text" name="username" placeholder="Username" required>
            </div>
            <div class="form-group">
                <input type="password" name="password" placeholder="Password" required>
            </div>
            <button type="submit" class="btn">Login</button>
        </form>
        <div class="info">
            <p>Demo: getaye / Getaye@6132</p>
        </div>
    </div>
</body>
</html>
"""

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
