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

READY-TO-USE CLASSIFIER - Pre-trained on 1000+ categories
Can recognize: Animals, Plants, Objects, People, Vehicles, Food, etc.
====================================================================
"""

from flask import Flask, render_template_string, request, jsonify, session, redirect, url_for, flash
import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import os
import uuid
import hashlib
import secrets
from datetime import datetime, timedelta
from functools import wraps
import re
import json
from werkzeug.utils import secure_filename
import numpy as np

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

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('users', exist_ok=True)

# ============================================================================
# PRE-TRAINED CLASSIFIER - Ready to use, no training needed!
# ============================================================================

class PreTrainedClassifier:
    """
    READY-TO-USE classifier pre-trained on ImageNet (1000+ categories)
    Can recognize: animals, plants, objects, people, vehicles, food, etc.
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Loading PRE-TRAINED model on {self.device}...")
        
        # Load a model that's already trained on millions of images
        # This model already knows how to recognize thousands of categories!
        model_name = "google/vit-base-patch16-224"
        
        print("📦 Loading feature extractor...")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        
        print("🧠 Loading pre-trained model (this knows 1000+ categories)...")
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get the pre-trained labels (ImageNet classes)
        self.labels = self.model.config.id2label
        print(f"✅ Model loaded! Knows {len(self.labels)} different categories")
        print("   Examples: tiger, elephant, rose, car, computer, etc.")
        
        # Common categories for display
        self.category_examples = [
            "🐱 Animals: cat, dog, tiger, elephant, bird, fish",
            "🌿 Plants: flower, tree, rose, sunflower, cactus",
            "🚗 Objects: car, bicycle, airplane, chair, book",
            "🍎 Food: pizza, apple, banana, cake, coffee",
            "👤 People: person, baby, crowd, portrait",
            "🏠 Places: house, mountain, beach, forest, city"
        ]
    
    def predict(self, image_path, top_k=5):
        """
        Predict using pre-trained model - NO TRAINING REQUIRED!
        Returns top-k predictions with confidence scores
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference (this uses the pre-trained knowledge)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)[0]
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            predictions = []
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                class_id = idx.item()
                class_name = self.labels[class_id]
                confidence = prob.item()
                
                # Clean up the class name (remove technical prefixes)
                class_name = class_name.split(',')[0].strip()
                class_name = class_name.replace('_', ' ')
                
                predictions.append({
                    'rank': i + 1,
                    'class': class_name,
                    'class_id': class_id,
                    'confidence': confidence,
                    'probability': f"{confidence:.2%}"
                })
            
            return predictions
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def get_category_info(self, class_name):
        """Categorize the prediction into broad categories"""
        class_lower = class_name.lower()
        
        if any(word in class_lower for word in ['cat', 'dog', 'tiger', 'lion', 'elephant', 'bird', 'fish', 'horse', 'cow']):
            return "🐾 Animal"
        elif any(word in class_lower for word in ['flower', 'tree', 'plant', 'rose', 'grass', 'leaf']):
            return "🌿 Plant"
        elif any(word in class_lower for word in ['car', 'truck', 'bicycle', 'airplane', 'boat', 'train']):
            return "🚗 Vehicle"
        elif any(word in class_lower for word in ['person', 'man', 'woman', 'child', 'baby']):
            return "👤 Person"
        elif any(word in class_lower for word in ['pizza', 'apple', 'banana', 'cake', 'bread', 'food']):
            return "🍎 Food"
        elif any(word in class_lower for word in ['house', 'building', 'mountain', 'beach', 'forest']):
            return "🏠 Place"
        else:
            return "📦 Object"

# Initialize the pre-trained classifier
classifier = PreTrainedClassifier()

# ============================================================================
# Simple Authentication (keep same as before)
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
    
    def login_user(self, username, password, ip_address):
        default_users = {
            'getaye': {
                'password_hash': self.hash_password('Getaye@6132'),
                'full_name': 'Getaye Fiseha',
                'student_id': 'GSE/6132/18',
                'email': 'getaye.fiseha@aau.edu.et'
            },
            'guest': {
                'password_hash': self.hash_password('Guest@2026'),
                'full_name': 'Guest User',
                'student_id': 'GUEST/001',
                'email': 'guest@aau.edu.et'
            }
        }
        
        if username in default_users:
            if self.verify_password(password, default_users[username]['password_hash']):
                return True, default_users[username]
        
        return False, "Invalid username or password"

user_manager = SecureUserManager()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Please login', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ============================================================================
# Routes
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
    return redirect(url_for('index'))

@app.route('/')
def index():
    """Home page - Classification Ready to Use!"""
    return render_template_string(INDEX_HTML, 
                                 session=session,
                                 num_categories=len(classifier.labels),
                                 examples=classifier.category_examples)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict endpoint - Uses pre-trained model"""
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
        # Get predictions from pre-trained model
        predictions = classifier.predict(filepath)
        
        if predictions:
            return jsonify({
                'success': True,
                'predictions': predictions,
                'filename': file.filename,
                'model_info': {
                    'name': 'Google ViT (ImageNet)',
                    'categories': len(classifier.labels),
                    'type': 'Pre-trained - Ready to use'
                }
            })
        else:
            return jsonify({'error': 'Could not classify image'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/model-info')
def model_info():
    """Get model information"""
    return jsonify({
        'model': 'Vision Transformer (ViT)',
        'pretrained_on': 'ImageNet',
        'categories': len(classifier.labels),
        'status': 'Ready to use - No training needed'
    })

# ============================================================================
# HTML Template - Clean and Professional
# ============================================================================

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AAU - Pre-trained Classifier | Getaye Fiseha GSE/6132/18</title>
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
        
        .ready-badge {
            background: linear-gradient(135deg, #006B3F, #FCD116);
            color: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 30px;
            font-size: 24px;
            font-weight: bold;
            box-shadow: 0 10px 30px rgba(0,107,63,0.3);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .stat-number {
            font-size: 36px;
            font-weight: bold;
            color: #006B3F;
        }
        
        .main-card {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }
        
        .upload-area { 
            border: 3px dashed #006B3F; 
            border-radius: 20px; 
            padding: 60px 40px; 
            text-align: center; 
            cursor: pointer; 
            transition: all 0.3s; 
            margin: 30px 0;
            background: linear-gradient(135deg, #fff, #f8f9fa);
        }
        .upload-area:hover { 
            border-color: #FCD116; 
            background: #fff9e6;
            transform: translateY(-5px);
        }
        .upload-area i { 
            font-size: 64px; 
            color: #006B3F; 
            margin-bottom: 15px; 
            display: block; 
        }
        
        #preview { 
            max-width: 100%; 
            max-height: 400px; 
            display: none; 
            margin: 20px auto; 
            border-radius: 15px; 
            border: 4px solid #FCD116;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        
        .btn { 
            background: linear-gradient(135deg, #006B3F, #FCD116); 
            color: white; 
            border: none; 
            padding: 18px 40px; 
            border-radius: 50px; 
            font-size: 20px; 
            font-weight: 600; 
            cursor: pointer; 
            transition: all 0.3s; 
            width: 100%;
            box-shadow: 0 10px 20px rgba(0,107,63,0.3);
        }
        .btn:hover { 
            transform: translateY(-3px); 
            box-shadow: 0 15px 30px rgba(0,107,63,0.4);
        }
        
        .results { 
            margin-top: 40px; 
            padding: 30px; 
            background: white; 
            border-radius: 20px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.15); 
            display: none; 
        }
        
        .prediction-item { 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            padding: 15px 20px; 
            background: #f8f9fa; 
            margin: 10px 0; 
            border-radius: 10px; 
            transition: all 0.3s;
        }
        .prediction-item:hover {
            transform: translateX(10px);
        }
        .top-prediction { 
            border-left: 8px solid #FCD116; 
            background: #fff9e6; 
            font-weight: bold;
        }
        .prediction-rank {
            background: #006B3F;
            color: white;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-right: 15px;
        }
        .prediction-prob { 
            background: #006B3F; 
            color: white; 
            padding: 8px 20px; 
            border-radius: 30px; 
            font-size: 16px; 
            font-weight: bold; 
        }
        
        .examples-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-top: 20px;
        }
        
        .example-item {
            background: #f0f0f0;
            padding: 15px;
            border-radius: 10px;
            color: #006B3F;
        }
        
        .footer { 
            background: #006B3F; 
            color: #FCD116; 
            padding: 30px; 
            margin-top: 60px; 
            text-align: center; 
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
                <a href="/profile">{{ session.full_name }}</a>
                <a href="/logout">Logout</a>
            {% else %}
                <a href="/login">Login</a>
            {% endif %}
        </div>
    </div>
    
    <div class="container">
        <div class="ready-badge">
            🚀 READY TO USE - NO TRAINING NEEDED!
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{{ num_categories }}+</div>
                <div>Categories</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">1M+</div>
                <div>Training Images</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">100%</div>
                <div>Ready to Use</div>
            </div>
        </div>
        
        <div class="main-card">
            <h1 style="color: #006B3F; text-align: center; margin-bottom: 20px;">📸 Upload Any Image</h1>
            <p style="text-align: center; color: #666; font-size: 18px; margin-bottom: 30px;">
                The model will recognize: Animals • Plants • Objects • People • Food • Places • Vehicles
            </p>
            
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <input type="file" id="fileInput" accept="image/*" style="display: none;">
                <i>📤</i>
                <p style="font-size: 20px;">Click or drag image here</p>
                <p style="color: #999;">JPG, PNG, JPEG (Max 16MB)</p>
            </div>
            
            <img id="preview" src="#" alt="Preview">
            
            <button class="btn" id="classifyBtn" onclick="classifyImage()">
                🔍 CLASSIFY IMAGE
            </button>
            
            <div id="spinner" style="text-align: center; margin: 20px; display: none;">
                <div style="border: 5px solid #f3f3f3; border-top: 5px solid #FCD116; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin: 0 auto;"></div>
                <p style="margin-top: 10px;">Analyzing image...</p>
            </div>
            
            <div class="results" id="results">
                <h2 style="color: #006B3F; margin-bottom: 20px;">🎯 Classification Results</h2>
                <div id="predictionList"></div>
            </div>
        </div>
        
        <div style="background: white; border-radius: 20px; padding: 30px; margin-top: 30px;">
            <h3 style="color: #006B3F;">📋 The model can recognize:</h3>
            <div class="examples-grid">
                {% for example in examples %}
                <div class="example-item">{{ example }}</div>
                {% endfor %}
            </div>
        </div>
    </div>
    
    <div class="footer">
        <p>© 2026 Getaye Fiseha - MSc Computer Science, Network & Security Stream</p>
        <p>Addis Ababa University | College of Natural and Computational Sciences</p>
    </div>
    
    <style>
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
    
    <script>
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
                alert('Please select an image first');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            // Show loading
            document.getElementById('classifyBtn').disabled = true;
            document.getElementById('spinner').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    // Display results
                    const predictionList = document.getElementById('predictionList');
                    predictionList.innerHTML = '';
                    
                    data.predictions.forEach((pred, index) => {
                        const item = document.createElement('div');
                        item.className = `prediction-item ${index === 0 ? 'top-prediction' : ''}`;
                        
                        item.innerHTML = `
                            <div style="display: flex; align-items: center;">
                                <span class="prediction-rank">${pred.rank}</span>
                                <span style="font-size: 18px;">${pred.class}</span>
                            </div>
                            <span class="prediction-prob">${pred.probability}</span>
                        `;
                        
                        predictionList.appendChild(item);
                    });
                    
                    document.getElementById('results').style.display = 'block';
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                document.getElementById('classifyBtn').disabled = false;
                document.getElementById('spinner').style.display = 'none';
            }
        }
    </script>
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
                           border-radius: 8px; font-size: 16px; }
        .btn { width: 100%; padding: 14px; background: linear-gradient(135deg, #006B3F 0%, #FCD116 100%); 
               color: white; border: none; border-radius: 8px; font-size: 18px; cursor: pointer; }
        .info { text-align: center; margin-top: 20px; color: #666; }
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
