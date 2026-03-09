"""
====================================================================
ADDIS ABABA UNIVERSITY - College of Natural and Computational Sciences
Department of Computer Science
====================================================================

Project: Image Classifier with User Management
Student Name: Getaye Fiseha
Student ID: GSE/6132/18
Program: MSc in Computer Science
Stream: Network & Security
Course: COSC 6041 - Machine Learning
Supervisor: Dr. Yaregal A.
Submission Date: March 2026
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
import json
from werkzeug.utils import secure_filename

# ============================================================================
# Application Configuration
# ============================================================================

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['USER_DATA'] = 'users.json'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================================================
# User Management System
# ============================================================================

class UserManager:
    def __init__(self):
        self.users_file = app.config['USER_DATA']
        self._init_default_users()
    
    def _init_default_users(self):
        if not os.path.exists(self.users_file):
            users = {
                'getaye': {
                    'password_hash': self._hash_password('Getaye@2827'),
                    'full_name': 'Getaye Fiseha',
                    'role': 'admin',
                    'student_id': 'GSE/6132/18',
                    'created_at': datetime.now().isoformat()
                }
            }
            self._save_users(users)
    
    def _hash_password(self, password):
        salt = "AAU_SECURE_SALT_2026"
        return hashlib.sha256((password + salt).encode()).hexdigest()
    
    def _save_users(self, users):
        with open(self.users_file, 'w') as f:
            json.dump(users, f, indent=2)
    
    def _load_users(self):
        if os.path.exists(self.users_file):
            with open(self.users_file, 'r') as f:
                return json.load(f)
        return {}
    
    def create_user(self, username, password, full_name, role='user', created_by=None):
        users = self._load_users()
        if username in users:
            return False, "Username exists"
        
        users[username] = {
            'password_hash': self._hash_password(password),
            'full_name': full_name,
            'role': role,
            'created_at': datetime.now().isoformat(),
            'created_by': created_by
        }
        self._save_users(users)
        return True, "User created"
    
    def verify_login(self, username, password):
        users = self._load_users()
        if username in users and self._hash_password(password) == users[username]['password_hash']:
            return True, users[username]
        return False, "Invalid credentials"
    
    def get_all_users(self):
        users = self._load_users()
        for u in users.values():
            u.pop('password_hash', None)
        return users
    
    def delete_user(self, username, admin_user):
        if username == 'getaye':
            return False, "Cannot delete main admin"
        users = self._load_users()
        if username in users:
            del users[username]
            self._save_users(users)
            return True, "User deleted"
        return False, "User not found"

user_manager = UserManager()

# ============================================================================
# Login Decorator
# ============================================================================

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session or session.get('role') != 'admin':
            flash('Admin access required', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated

# ============================================================================
# Image Classifier - Shows Actual Object Names
# ============================================================================

class ImageClassifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model on {self.device}...")
        
        # Load pre-trained model
        model_name = "google/vit-base-patch16-224"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get ImageNet labels
        self.labels = self.model.config.id2label
        print(f"Model loaded! Knows {len(self.labels)} objects")
    
    def predict(self, image_path, top_k=5):
        """Return top-k object predictions"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = F.softmax(outputs.logits, dim=-1)[0]
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            results = []
            for prob, idx in zip(top_probs, top_indices):
                class_id = idx.item()
                class_name = self.labels[class_id]
                # Clean up class name
                class_name = class_name.split(',')[0].strip().replace('_', ' ')
                confidence = prob.item()
                
                results.append({
                    'object': class_name.title(),
                    'confidence': confidence,
                    'percentage': f"{confidence:.1%}"
                })
            
            return results
            
        except Exception as e:
            print(f"Error: {e}")
            return None

classifier = ImageClassifier()

# ============================================================================
# Authentication Routes
# ============================================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        success, result = user_manager.verify_login(username, password)
        
        if success:
            session['username'] = username
            session['full_name'] = result['full_name']
            session['role'] = result['role']
            session.permanent = True
            return redirect(url_for('index'))
        else:
            error = "Invalid username or password"
    
    return render_template_string(LOGIN_HTML, error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ============================================================================
# User Management Routes
# ============================================================================

@app.route('/users')
@admin_required
def user_list():
    users = user_manager.get_all_users()
    return render_template_string(USERS_HTML, users=users, current_user=session['username'])

@app.route('/users/create', methods=['POST'])
@admin_required
def create_user():
    username = request.form.get('username')
    password = request.form.get('password')
    full_name = request.form.get('full_name')
    role = request.form.get('role', 'user')
    
    success, message = user_manager.create_user(username, password, full_name, role, session['username'])
    flash(message, 'success' if success else 'error')
    return redirect(url_for('user_list'))

@app.route('/users/delete/<username>', methods=['POST'])
@admin_required
def delete_user(username):
    success, message = user_manager.delete_user(username, session['username'])
    flash(message, 'success' if success else 'error')
    return redirect(url_for('user_list'))

# ============================================================================
# Main Routes
# ============================================================================

@app.route('/')
@login_required
def index():
    return render_template_string(INDEX_HTML, session=session)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save temp file
    filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        predictions = classifier.predict(filepath)
        
        if predictions:
            return jsonify({
                'success': True,
                'predictions': predictions
            })
        else:
            return jsonify({'error': 'Could not classify'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

# ============================================================================
# HTML Templates
# ============================================================================

LOGIN_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AAU - Login</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #006B3F, #FCD116);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .login-box {
            background: white;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            width: 100%;
            max-width: 380px;
        }
        h2 {
            color: #006B3F;
            text-align: center;
            margin-bottom: 30px;
            font-size: 28px;
        }
        .input-group {
            margin-bottom: 20px;
        }
        .input-group input {
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #e1e1e1;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        .input-group input:focus {
            outline: none;
            border-color: #FCD116;
        }
        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #006B3F, #FCD116);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
        }
        .error {
            background: #fee;
            color: #c33;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="login-box">
        <h2>🎓 AAU Image Classifier</h2>
        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
        <form method="POST">
            <div class="input-group">
                <input type="text" name="username" placeholder="Username" required>
            </div>
            <div class="input-group">
                <input type="password" name="password" placeholder="Password" required>
            </div>
            <button type="submit">Login</button>
        </form>
    </div>
</body>
</html>
"""

USERS_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>User Management</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; }
        .navbar {
            background: #006B3F;
            padding: 15px 30px;
            color: white;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .nav-links a {
            color: white;
            text-decoration: none;
            margin-left: 20px;
        }
        .container { max-width: 1200px; margin: 40px auto; padding: 0 20px; }
        .card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .btn {
            background: #006B3F;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
        }
        .btn-danger { background: #dc3545; }
        table { width: 100%; border-collapse: collapse; }
        th { background: #006B3F; color: white; padding: 12px; text-align: left; }
        td { padding: 12px; border-bottom: 1px solid #ddd; }
        .form-group { margin-bottom: 15px; }
        .form-group input, .form-group select {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="navbar">
        <h2>👑 User Management</h2>
        <div class="nav-links">
            <a href="/">Home</a>
            <a href="/logout">Logout</a>
        </div>
    </div>
    
    <div class="container">
        <div class="card">
            <h3 style="margin-bottom: 20px;">Create New User</h3>
            <form action="/users/create" method="POST">
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
                    <input type="text" name="username" placeholder="Username" required>
                    <input type="password" name="password" placeholder="Password" required>
                    <input type="text" name="full_name" placeholder="Full Name" required>
                    <select name="role">
                        <option value="user">User</option>
                        <option value="admin">Admin</option>
                    </select>
                    <button type="submit" class="btn">Create</button>
                </div>
            </form>
        </div>
        
        <div class="card">
            <h3 style="margin-bottom: 20px;">Users</h3>
            <table>
                <thead>
                    <tr>
                        <th>Username</th>
                        <th>Full Name</th>
                        <th>Role</th>
                        <th>Created</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {% for username, info in users.items() %}
                    <tr>
                        <td>{{ username }}</td>
                        <td>{{ info.full_name }}</td>
                        <td>{{ info.role }}</td>
                        <td>{{ info.created_at[:10] }}</td>
                        <td>
                            {% if username != current_user and username != 'getaye' %}
                            <form action="/users/delete/{{ username }}" method="POST" style="display:inline;">
                                <button type="submit" class="btn btn-danger" onclick="return confirm('Delete?')">Delete</button>
                            </form>
                            {% endif %}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Image Classifier</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }
        
        .navbar {
            background: #006B3F;
            padding: 15px 30px;
            color: white;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .user-info {
            background: #FCD116;
            color: #006B3F;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }
        
        .nav-links a {
            color: white;
            text-decoration: none;
            margin-left: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 50px auto;
            padding: 0 20px;
        }
        
        .upload-card {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .upload-area {
            border: 3px dashed #006B3F;
            border-radius: 20px;
            padding: 60px 40px;
            margin: 30px 0;
            cursor: pointer;
            transition: all 0.3s;
            background: #f8f9fa;
        }
        
        .upload-area:hover {
            border-color: #FCD116;
            background: #fff9e6;
            transform: translateY(-5px);
        }
        
        .upload-icon {
            font-size: 64px;
            color: #006B3F;
            margin-bottom: 15px;
        }
        
        #preview {
            max-width: 100%;
            max-height: 400px;
            display: none;
            margin: 20px auto;
            border-radius: 15px;
            border: 4px solid #FCD116;
        }
        
        .btn {
            background: linear-gradient(135deg, #006B3F, #FCD116);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 50px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            transition: transform 0.3s;
        }
        
        .btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 20px rgba(0,107,63,0.3);
        }
        
        .results {
            margin-top: 40px;
            display: none;
        }
        
        .prediction-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 20px;
            margin: 10px 0;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 8px solid #006B3F;
            animation: slideIn 0.5s ease;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        .prediction-number {
            background: #006B3F;
            color: white;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 15px;
        }
        
        .prediction-name {
            font-size: 20px;
            font-weight: 600;
            color: #333;
        }
        
        .prediction-confidence {
            background: #006B3F;
            color: white;
            padding: 8px 20px;
            border-radius: 30px;
            font-weight: bold;
        }
        
        .spinner {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #FCD116;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 30px auto;
            display: none;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .footer {
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="navbar">
        <div>
            <span class="user-info">{{ session.full_name }} ({{ session.role }})</span>
        </div>
        <div class="nav-links">
            <a href="/">Home</a>
            {% if session.role == 'admin' %}
            <a href="/users">Manage Users</a>
            {% endif %}
            <a href="/logout">Logout</a>
        </div>
    </div>
    
    <div class="container">
        <div class="upload-card">
            <h1 style="color: #006B3F; margin-bottom: 20px;">📸 Image Classifier</h1>
            <p style="color: #666; margin-bottom: 30px;">Upload any image to identify objects</p>
            
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <input type="file" id="fileInput" accept="image/*" style="display: none;">
                <div class="upload-icon">📤</div>
                <p style="font-size: 18px;">Click or drag image here</p>
                <p style="color: #999;">JPG, PNG, JPEG (Max 16MB)</p>
            </div>
            
            <img id="preview" src="#" alt="Preview">
            
            <div class="spinner" id="spinner"></div>
            
            <button class="btn" onclick="classifyImage()">🔍 Classify Image</button>
            
            <div class="results" id="results">
                <h3 style="color: #006B3F; margin-bottom: 20px;">🎯 Detected Objects</h3>
                <div id="predictionList"></div>
            </div>
        </div>
        
        <div class="footer">
            <p>Getaye Fiseha (GSE/6132/18) - MSc Computer Science</p>
        </div>
    </div>
    
    <script>
        let currentFile = null;
        
        document.getElementById('fileInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                currentFile = file;
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('preview').src = e.target.result;
                    document.getElementById('preview').style.display = 'block';
                }
                reader.readAsDataURL(file);
            }
        });

        async function classifyImage() {
            if (!currentFile) {
                alert('Please select an image first');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', currentFile);
            
            document.getElementById('spinner').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayResults(data.predictions);
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                document.getElementById('spinner').style.display = 'none';
            }
        }
        
        function displayResults(predictions) {
            const resultsDiv = document.getElementById('results');
            const predictionList = document.getElementById('predictionList');
            
            predictionList.innerHTML = '';
            
            predictions.forEach((pred, index) => {
                const item = document.createElement('div');
                item.className = 'prediction-item';
                item.innerHTML = `
                    <div style="display: flex; align-items: center;">
                        <span class="prediction-number">${index + 1}</span>
                        <span class="prediction-name">${pred.object}</span>
                    </div>
                    <span class="prediction-confidence">${pred.percentage}</span>
                `;
                predictionList.appendChild(item);
            });
            
            resultsDiv.style.display = 'block';
        }
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
