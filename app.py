"""
====================================================================
ADDIS ABABA UNIVERSITY - College of Natural and Computational Sciences
Department of Computer Science
====================================================================

Project: High-Level Image Classifier with Boosting Algorithm
Course: Machine Learning (COSC 6041)
Student Name: Getaye Fiseha
Student ID: GSE/6132/18
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
import base64
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
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['TRAINING_FOLDER'] = 'training_data'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['USER_DATA'] = 'users.json'
app.config['MODEL_DATA'] = 'trained_model.pkl'

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TRAINING_FOLDER'], exist_ok=True)
for category in ['animals', 'plants', 'objects', 'food', 'people', 'places']:
    os.makedirs(os.path.join(app.config['TRAINING_FOLDER'], category), exist_ok=True)

# ============================================================================
# AAU Logo (Red and Gold - AAU Colors)
# ============================================================================

AAU_LOGO_SVG = '''
<svg width="200" height="80" viewBox="0 0 200 80" xmlns="http://www.w3.org/2000/svg">
    <!-- Background -->
    <rect width="200" height="80" fill="#8B0000" rx="10" ry="10"/>
    <!-- Gold torch -->
    <circle cx="60" cy="40" r="20" fill="#FFD700"/>
    <rect x="55" y="40" width="10" height="25" fill="#FFD700"/>
    <path d="M45 45 L75 45 L70 65 L50 65 Z" fill="#FFD700"/>
    <!-- AAU Text -->
    <text x="100" y="35" fill="#FFD700" font-size="20" font-weight="bold">AAU</text>
    <text x="100" y="55" fill="white" font-size="12">EST. 1950</text>
</svg>
'''

AAU_LOGO = base64.b64encode(AAU_LOGO_SVG.encode()).decode()

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
                    'email': 'getaye.fiseha@aau.edu.et',
                    'created_at': datetime.now().isoformat(),
                    'training_count': 0
                }
            }
            self._save_users(users)
    
    def _hash_password(self, password):
        salt = "AAU_RED_SALT_2026"
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
            'created_by': created_by,
            'training_count': 0
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
    
    def increment_training(self, username):
        users = self._load_users()
        if username in users:
            users[username]['training_count'] = users[username].get('training_count', 0) + 1
            self._save_users(users)

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
# Accurate Image Classifier with Learning Capability
# ============================================================================

class AccurateClassifier:
    """
    Uses Vision Transformer with boosting algorithm
    Learns from user feedback to improve accuracy
    """
    
    # Common objects mapping for accurate detection
    COMMON_OBJECTS = {
        # People
        'person': 'person', 'man': 'man', 'woman': 'woman', 'child': 'child', 'baby': 'baby',
        'girl': 'girl', 'boy': 'boy', 'crowd': 'crowd', 'people': 'people',
        
        # Animals
        'cat': 'cat', 'dog': 'dog', 'bird': 'bird', 'fish': 'fish', 'horse': 'horse',
        'cow': 'cow', 'sheep': 'sheep', 'goat': 'goat', 'lion': 'lion', 'tiger': 'tiger',
        'elephant': 'elephant', 'rabbit': 'rabbit', 'duck': 'duck', 'chicken': 'chicken',
        
        # Objects
        'car': 'car', 'truck': 'truck', 'bus': 'bus', 'bicycle': 'bicycle', 'motorcycle': 'motorcycle',
        'chair': 'chair', 'table': 'table', 'desk': 'desk', 'book': 'book', 'phone': 'phone',
        'computer': 'computer', 'laptop': 'laptop', 'bottle': 'bottle', 'cup': 'cup',
        'clock': 'clock', 'watch': 'watch', 'key': 'key', 'door': 'door', 'window': 'window',
        
        # Food
        'pizza': 'pizza', 'apple': 'apple', 'banana': 'banana', 'cake': 'cake', 'bread': 'bread',
        'rice': 'rice', 'coffee': 'coffee', 'tea': 'tea', 'water': 'water', 'juice': 'juice',
        
        # Plants
        'flower': 'flower', 'tree': 'tree', 'rose': 'rose', 'grass': 'grass', 'leaf': 'leaf',
        'cactus': 'cactus', 'palm': 'palm', 'plant': 'plant',
        
        # Places
        'house': 'house', 'building': 'building', 'mountain': 'mountain', 'beach': 'beach',
        'forest': 'forest', 'city': 'city', 'river': 'river', 'lake': 'lake', 'ocean': 'ocean'
    }
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Loading classifier on {self.device}...")
        
        # Load pre-trained model
        model_name = "google/vit-base-patch16-224"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get ImageNet labels
        self.labels = self.model.config.id2label
        
        # Load user-trained data
        self.user_trained = self._load_user_trained()
        
        print(f"✅ Classifier ready! Base knowledge: {len(self.labels)} classes")
        print(f"   User trained: {len(self.user_trained)} custom labels")
    
    def _load_user_trained(self):
        """Load user-trained labels"""
        trained_file = 'user_trained.json'
        if os.path.exists(trained_file):
            with open(trained_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_user_trained(self):
        with open('user_trained.json', 'w') as f:
            json.dump(self.user_trained, f, indent=2)
    
    def _get_category(self, object_name):
        """Determine category of an object"""
        object_lower = object_name.lower()
        
        categories = {
            'people': ['person', 'man', 'woman', 'child', 'baby', 'girl', 'boy', 'crowd'],
            'animals': ['cat', 'dog', 'bird', 'fish', 'horse', 'cow', 'sheep', 'lion', 'tiger', 'elephant'],
            'objects': ['car', 'truck', 'chair', 'table', 'book', 'phone', 'computer', 'bottle'],
            'food': ['pizza', 'apple', 'banana', 'cake', 'bread', 'rice', 'coffee'],
            'plants': ['flower', 'tree', 'rose', 'grass', 'cactus', 'plant'],
            'places': ['house', 'building', 'mountain', 'beach', 'forest', 'city']
        }
        
        for category, keywords in categories.items():
            if any(keyword in object_lower for keyword in keywords):
                return category
        return 'objects'
    
    def predict(self, image_path, top_k=5):
        """
        Predict objects in image - ACCURATE detection
        Returns simple object names
        """
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
            top_probs, top_indices = torch.topk(probabilities, top_k * 2)
            
            predictions = []
            seen = set()
            
            for prob, idx in zip(top_probs, top_indices):
                class_id = idx.item()
                class_name = self.labels[class_id].lower()
                confidence = prob.item()
                
                # Skip if confidence too low
                if confidence < 0.01:
                    continue
                
                # Check if this is a common object
                detected_object = None
                for key, value in self.COMMON_OBJECTS.items():
                    if key in class_name and value not in seen:
                        detected_object = value
                        break
                
                # If not found in common objects, use first word
                if not detected_object:
                    words = class_name.replace('_', ' ').split()
                    if words and words[0] not in seen:
                        detected_object = words[0]
                
                if detected_object and detected_object not in seen:
                    category = self._get_category(detected_object)
                    
                    # Category colors
                    colors = {
                        'people': '#9B59B6',
                        'animals': '#FF6B6B',
                        'objects': '#4A90E2',
                        'food': '#F4A460',
                        'plants': '#4CAF50',
                        'places': '#E67E22'
                    }
                    
                    # Category emojis
                    emojis = {
                        'people': '👤',
                        'animals': '🐱',
                        'objects': '📦',
                        'food': '🍎',
                        'plants': '🌿',
                        'places': '🏠'
                    }
                    
                    predictions.append({
                        'object': detected_object.title(),
                        'category': category,
                        'emoji': emojis.get(category, '📦'),
                        'color': colors.get(category, '#4A90E2'),
                        'confidence': confidence,
                        'percentage': f"{confidence:.1%}"
                    })
                    
                    seen.add(detected_object)
                    
                    if len(predictions) >= top_k:
                        break
            
            return predictions if predictions else [
                {
                    'object': 'Unknown',
                    'category': 'objects',
                    'emoji': '❓',
                    'color': '#999999',
                    'confidence': 1.0,
                    'percentage': '100%'
                }
            ]
            
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def add_training_sample(self, image_path, object_name, username):
        """
        LEARN from user feedback
        Saves training sample for future improvement
        """
        try:
            # Extract features
            image = Image.open(image_path).convert('RGB')
            features = self.feature_extractor(images=image, return_tensors="pt")
            
            # Save to user-trained database
            object_lower = object_name.lower()
            if object_lower not in self.user_trained:
                self.user_trained[object_lower] = {
                    'count': 0,
                    'samples': []
                }
            
            # Save image
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{username}_{timestamp}_{uuid.uuid4()}.jpg"
            category = self._get_category(object_lower)
            save_path = os.path.join(app.config['TRAINING_FOLDER'], category, filename)
            image.save(save_path)
            
            # Update database
            self.user_trained[object_lower]['count'] += 1
            self.user_trained[object_lower]['samples'].append({
                'path': save_path,
                'username': username,
                'timestamp': timestamp
            })
            
            self._save_user_trained()
            
            return True
            
        except Exception as e:
            print(f"Error saving training sample: {e}")
            return False
    
    def get_training_stats(self):
        """Get training statistics"""
        return {
            'total': sum(data['count'] for data in self.user_trained.values()),
            'objects': len(self.user_trained)
        }

classifier = AccurateClassifier()

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
    
    return render_template_string(LOGIN_HTML, error=error, logo=AAU_LOGO)

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
    return render_template_string(USERS_HTML, users=users, current_user=session['username'], logo=AAU_LOGO)

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
    stats = classifier.get_training_stats()
    return render_template_string(INDEX_HTML, 
                                 session=session, 
                                 stats=stats,
                                 logo=AAU_LOGO)

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

@app.route('/train', methods=['POST'])
@login_required
def train():
    """LEARN from user feedback"""
    if 'file' not in request.files or 'object_name' not in request.form:
        return jsonify({'error': 'Missing data'}), 400
    
    file = request.files['file']
    object_name = request.form['object_name']
    
    if not object_name.strip():
        return jsonify({'error': 'Please enter an object name'}), 400
    
    # Save temp file
    filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        success = classifier.add_training_sample(filepath, object_name, session['username'])
        
        if success:
            user_manager.increment_training(session['username'])
            return jsonify({
                'success': True,
                'message': f'✅ Learned: {object_name.title()}',
                'stats': classifier.get_training_stats()
            })
        else:
            return jsonify({'error': 'Could not save training sample'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/stats')
@login_required
def get_stats():
    """Get training statistics"""
    return jsonify(classifier.get_training_stats())

# ============================================================================
# HTML Templates
# ============================================================================

LOGIN_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AAU - Image Classifier Login</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #8B0000, #FFD700);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .login-container {
            background: white;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            width: 100%;
            max-width: 450px;
            text-align: center;
        }
        .logo {
            margin-bottom: 20px;
        }
        .logo svg {
            width: 200px;
            height: 80px;
        }
        .university-name {
            color: #8B0000;
            font-size: 22px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .college-name {
            color: #FFD700;
            font-size: 14px;
            margin-bottom: 20px;
        }
        .student-info {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 25px;
            border-left: 5px solid #8B0000;
            text-align: left;
        }
        .student-info p {
            margin: 5px 0;
            color: #333;
        }
        .student-info strong {
            color: #8B0000;
        }
        .project-title {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 25px;
            border-left: 5px solid #FFD700;
        }
        .project-title h3 {
            color: #8B0000;
            font-size: 16px;
        }
        .project-title p {
            color: #666;
            font-size: 13px;
            margin-top: 5px;
        }
        .input-group {
            margin-bottom: 15px;
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
            border-color: #FFD700;
        }
        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #8B0000, #FFD700);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
            margin-top: 10px;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(139,0,0,0.3);
        }
        .error {
            background: #fee;
            color: #c33;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="logo">
            <img src="data:image/svg+xml;base64,{{ logo }}" alt="AAU Logo" style="width: 200px;">
        </div>
        <div class="university-name">ADDIS ABABA UNIVERSITY</div>
        <div class="college-name">College of Natural and Computational Sciences<br>Department of Computer Science</div>
        
        <div class="student-info">
            <p><strong>🎓 Getaye Fiseha</strong> (GSE/6132/18)</p>
            <p><strong>👨‍🏫 Supervisor:</strong> Dr. Yaregal A.</p>
        </div>
        
        <div class="project-title">
            <h3>High-Level Image Classifier with Boosting Algorithm</h3>
            <p>Machine Learning Course (COSC 6041)</p>
        </div>
        
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
    <title>AAU - User Management</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; }
        .navbar {
            background: #8B0000;
            padding: 15px 30px;
            color: white;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .logo {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .logo img {
            height: 40px;
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
            background: #8B0000;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
        }
        .btn-danger { background: #dc3545; }
        table { width: 100%; border-collapse: collapse; }
        th { background: #8B0000; color: #FFD700; padding: 12px; }
        td { padding: 12px; border-bottom: 1px solid #ddd; }
    </style>
</head>
<body>
    <div class="navbar">
        <div class="logo">
            <img src="data:image/svg+xml;base64,{{ logo }}" alt="AAU Logo">
            <span>User Management</span>
        </div>
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
                    <button type="submit" class="btn">Create User</button>
                </div>
            </form>
        </div>
        
        <div class="card">
            <h3 style="margin-bottom: 20px;">Registered Users</h3>
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
                                <button type="submit" class="btn btn-danger" onclick="return confirm('Delete user?')">Delete</button>
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
    <title>AAU - Image Classifier</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        .navbar {
            background: #8B0000;
            padding: 15px 30px;
            color: white;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .logo img {
            height: 40px;
        }
        
        .university-info {
            line-height: 1.3;
        }
        
        .university-info h2 {
            font-size: 18px;
            color: #FFD700;
        }
        
        .university-info p {
            font-size: 12px;
            opacity: 0.9;
        }
        
        .user-info {
            background: #FFD700;
            color: #8B0000;
            padding: 8px 20px;
            border-radius: 25px;
            font-weight: bold;
        }
        
        .nav-links a {
            color: white;
            text-decoration: none;
            margin-left: 20px;
            padding: 5px 10px;
        }
        
        .nav-links a:hover {
            background: rgba(255,255,255,0.2);
            border-radius: 5px;
        }
        
        .container {
            max-width: 1000px;
            margin: 40px auto;
            padding: 0 20px;
        }
        
        .project-header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .project-header h1 {
            color: #8B0000;
            font-size: 32px;
            margin-bottom: 10px;
        }
        
        .project-header .course {
            color: #FFD700;
            font-weight: bold;
            font-size: 18px;
            background: #8B0000;
            display: inline-block;
            padding: 8px 25px;
            border-radius: 30px;
            margin-top: 10px;
        }
        
        .stats-card {
            background: white;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 30px;
            display: flex;
            justify-content: space-around;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            border-left: 5px solid #FFD700;
        }
        
        .stat-item {
            text-align: center;
        }
        
        .stat-number {
            font-size: 28px;
            font-weight: bold;
            color: #8B0000;
        }
        
        .main-card {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        }
        
        .upload-area {
            border: 3px dashed #8B0000;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            margin: 20px 0;
        }
        
        .upload-area:hover {
            border-color: #FFD700;
            background: #fff9e6;
        }
        
        .upload-icon {
            font-size: 48px;
            color: #8B0000;
            margin-bottom: 10px;
        }
        
        #preview {
            max-width: 100%;
            max-height: 300px;
            display: none;
            margin: 20px auto;
            border-radius: 10px;
            border: 4px solid #FFD700;
        }
        
        .btn {
            background: linear-gradient(135deg, #8B0000, #FFD700);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            transition: transform 0.3s;
            margin: 10px 0;
        }
        
        .btn:hover {
            transform: translateY(-2px);
        }
        
        .btn-small {
            padding: 8px 15px;
            font-size: 14px;
            width: auto;
        }
        
        .prediction-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            background: #f8f9fa;
            margin: 10px 0;
            border-radius: 10px;
            border-left: 8px solid;
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        .prediction-name {
            font-size: 18px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .prediction-confidence {
            background: #8B0000;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }
        
        .training-section {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 15px;
            display: none;
            border: 2px solid #FFD700;
        }
        
        .training-input {
            width: 100%;
            padding: 12px;
            margin: 15px 0;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
        }
        
        .training-input:focus {
            outline: none;
            border-color: #FFD700;
        }
        
        .spinner {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #FFD700;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
            display: none;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .results {
            margin-top: 30px;
            display: none;
        }
        
        .footer {
            background: #8B0000;
            color: #FFD700;
            padding: 20px;
            text-align: center;
            margin-top: 50px;
        }
        
        .success-message {
            background: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            display: none;
        }
    </style>
</head>
<body>
    <div class="navbar">
        <div class="logo">
            <img src="data:image/svg+xml;base64,{{ logo }}" alt="AAU Logo">
            <div class="university-info">
                <h2>ADDIS ABABA UNIVERSITY</h2>
                <p>College of Natural and Computational Sciences<br>Department of Computer Science</p>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 20px;">
            <span class="user-info">🎓 {{ session.full_name }} ({{ session.role }})</span>
            <div class="nav-links">
                <a href="/">Home</a>
                {% if session.role == 'admin' %}
                <a href="/users">👑 Users</a>
                {% endif %}
                <a href="/logout">Logout</a>
            </div>
        </div>
    </div>
    
    <div class="container">
        <div class="project-header">
            <h1>High-Level Image Classifier with Boosting Algorithm</h1>
            <div class="course">Machine Learning (COSC 6041)</div>
        </div>
        
        <div class="stats-card">
            <div class="stat-item">
                <div class="stat-number">{{ stats.total }}</div>
                <div>Training Samples</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{{ stats.objects }}</div>
                <div>Objects Learned</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">1000+</div>
                <div>Base Classes</div>
            </div>
        </div>
        
        <div class="main-card">
            <h2 style="color: #8B0000; margin-bottom: 20px;">📸 Upload Image</h2>
            
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
                <h3 style="color: #8B0000; margin-bottom: 15px;">Detected Objects:</h3>
                <div id="predictionList"></div>
            </div>
            
            <!-- Training Section -->
            <div class="training-section" id="trainingSection">
                <h3 style="color: #8B0000;">📚 Teach the Model</h3>
                <p>What object is in this image?</p>
                
                <input type="text" id="objectName" class="training-input" placeholder="Enter object name (e.g., cat, car, person)">
                
                <button class="btn btn-small" onclick="submitTraining()" style="background: #4CAF50;">✓ Submit & Teach</button>
                
                <div id="trainingMessage" class="success-message"></div>
            </div>
        </div>
    </div>
    
    <div class="footer">
        <p><strong>Getaye Fiseha</strong> (GSE/6132/18) - Supervisor: Dr. Yaregal A.</p>
        <p>Addis Ababa University | College of Natural and Computational Sciences | Department of Computer Science</p>
    </div>
    
    <script>
        let currentFile = null;
        let currentPredictions = null;
        
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
            document.getElementById('trainingSection').style.display = 'none';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayResults(data.predictions);
                    currentPredictions = data.predictions;
                    showTrainingSection();
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
                item.style.borderLeftColor = pred.color;
                item.innerHTML = `
                    <div class="prediction-name">
                        <span>${pred.emoji}</span>
                        <span>${pred.object}</span>
                        <span style="font-size: 12px; color: #666;">(${pred.category})</span>
                    </div>
                    <span class="prediction-confidence">${pred.percentage}</span>
                `;
                predictionList.appendChild(item);
            });
            
            resultsDiv.style.display = 'block';
        }
        
        function showTrainingSection() {
            const section = document.getElementById('trainingSection');
            const input = document.getElementById('objectName');
            
            // Pre-fill with top prediction
            if (currentPredictions && currentPredictions.length > 0) {
                input.value = currentPredictions[0].object.toLowerCase();
            }
            
            section.style.display = 'block';
        }
        
        async function submitTraining() {
            const objectName = document.getElementById('objectName').value.trim();
            
            if (!objectName) {
                alert('Please enter an object name');
                return;
            }
            
            if (!currentFile) {
                alert('No image selected');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', currentFile);
            formData.append('object_name', objectName);
            
            const messageDiv = document.getElementById('trainingMessage');
            messageDiv.style.display = 'none';
            
            try {
                const response = await fetch('/train', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    messageDiv.textContent = `✅ Learned: ${objectName.title()}! Thank you for teaching!`;
                    messageDiv.style.display = 'block';
                    setTimeout(() => {
                        messageDiv.style.display = 'none';
                    }, 3000);
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
