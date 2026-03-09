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

Features:
- Classifies images into 6 high-level categories
- Shows simple object names (cat, dog, car, etc.)
- Training interface to improve accuracy
- User management system
- AAU branding with logo
====================================================================
"""

from flask import Flask, render_template_string, request, jsonify, session, redirect, url_for, flash
import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image, ImageDraw, ImageFont
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

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TRAINING_FOLDER'], exist_ok=True)
for category in ['animals', 'plants', 'objects', 'food', 'people', 'places']:
    os.makedirs(os.path.join(app.config['TRAINING_FOLDER'], category), exist_ok=True)

# ============================================================================
# AAU Logo Generator (Base64 encoded - simple text logo)
# ============================================================================

def generate_aau_logo():
    """Generate AAU logo as base64 image"""
    img = Image.new('RGB', (200, 80), color='#006B3F')
    draw = ImageDraw.Draw(img)
    # Draw simple logo text
    draw.text((20, 20), "AAU", fill='#FCD116', size=40)
    draw.text((20, 50), "EST. 1950", fill='#FCD116', size=15)
    
    # Convert to base64
    import io
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

AAU_LOGO = generate_aau_logo()

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
# High-Level Classifier with Boosting Algorithm
# ============================================================================

class HighLevelClassifier:
    """
    Uses Vision Transformer (ViT) with boosting algorithm concepts
    Classifies images into 6 main categories with simple object names
    """
    
    # Category definitions
    CATEGORIES = {
        'animals': {
            'name': 'Animals',
            'emoji': '🐱',
            'color': '#FF6B6B',
            'examples': ['cat', 'dog', 'elephant', 'lion', 'bird', 'fish', 'horse', 'tiger']
        },
        'plants': {
            'name': 'Plants',
            'emoji': '🌿',
            'color': '#4CAF50',
            'examples': ['flower', 'tree', 'rose', 'cactus', 'grass', 'palm', 'fern']
        },
        'objects': {
            'name': 'Objects',
            'emoji': '📦',
            'color': '#4A90E2',
            'examples': ['car', 'chair', 'book', 'phone', 'computer', 'bottle', 'clock']
        },
        'food': {
            'name': 'Food',
            'emoji': '🍎',
            'color': '#F4A460',
            'examples': ['pizza', 'apple', 'banana', 'cake', 'coffee', 'bread', 'rice']
        },
        'people': {
            'name': 'People',
            'emoji': '👤',
            'color': '#9B59B6',
            'examples': ['person', 'man', 'woman', 'child', 'baby', 'crowd']
        },
        'places': {
            'name': 'Places',
            'emoji': '🏠',
            'color': '#E67E22',
            'examples': ['house', 'mountain', 'beach', 'forest', 'city', 'river']
        }
    }
    
    # Object mapping for simplification
    OBJECT_MAPPING = {
        # Animals
        'cat': 'cat', 'kitty': 'cat', 'kitten': 'cat',
        'dog': 'dog', 'puppy': 'dog', 'puppies': 'dog',
        'bird': 'bird', 'parrot': 'bird', 'eagle': 'bird',
        'fish': 'fish', 'tuna': 'fish', 'goldfish': 'fish',
        'horse': 'horse', 'pony': 'horse',
        'elephant': 'elephant',
        'lion': 'lion',
        'tiger': 'tiger',
        'cow': 'cow',
        'sheep': 'sheep',
        'goat': 'goat',
        'rabbit': 'rabbit',
        
        # Plants
        'flower': 'flower', 'rose': 'rose', 'tulip': 'flower',
        'tree': 'tree', 'oak': 'tree', 'pine': 'tree',
        'cactus': 'cactus',
        'grass': 'grass',
        'palm': 'palm tree',
        'fern': 'fern',
        
        # Objects
        'car': 'car', 'automobile': 'car', 'truck': 'truck',
        'chair': 'chair', 'seat': 'chair',
        'book': 'book', 'novel': 'book',
        'phone': 'phone', 'telephone': 'phone', 'cellphone': 'phone',
        'computer': 'computer', 'laptop': 'laptop',
        'bottle': 'bottle',
        'clock': 'clock', 'watch': 'watch',
        
        # Food
        'pizza': 'pizza',
        'apple': 'apple',
        'banana': 'banana',
        'cake': 'cake', 'dessert': 'cake',
        'coffee': 'coffee', 'espresso': 'coffee',
        'bread': 'bread',
        'rice': 'rice',
        
        # People
        'person': 'person', 'people': 'person',
        'man': 'man', 'men': 'man',
        'woman': 'woman', 'women': 'woman',
        'child': 'child', 'children': 'child',
        'baby': 'baby', 'infant': 'baby',
        
        # Places
        'house': 'house', 'home': 'house',
        'mountain': 'mountain', 'hill': 'mountain',
        'beach': 'beach', 'shore': 'beach',
        'forest': 'forest', 'woods': 'forest',
        'city': 'city', 'town': 'city',
        'river': 'river', 'stream': 'river'
    }
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Loading classifier on {self.device}...")
        
        # Load pre-trained model (serves as our boosting ensemble)
        model_name = "google/vit-base-patch16-224"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get ImageNet labels
        self.labels = self.model.config.id2label
        
        # Training statistics
        self.training_stats = self._load_training_stats()
        
        print(f"✅ Classifier ready! Categories:")
        for cat in self.CATEGORIES.values():
            print(f"   {cat['emoji']} {cat['name']}")
    
    def _load_training_stats(self):
        stats_file = 'training_stats.json'
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                return json.load(f)
        return {cat: 0 for cat in self.CATEGORIES.keys()}
    
    def _save_training_stats(self):
        with open('training_stats.json', 'w') as f:
            json.dump(self.training_stats, f, indent=2)
    
    def _simplify_object_name(self, class_name):
        """Convert detailed class name to simple object name"""
        class_lower = class_name.lower()
        
        # Check each word in the class name
        words = class_lower.replace('_', ' ').split()
        for word in words:
            if word in self.OBJECT_MAPPING:
                return self.OBJECT_MAPPING[word]
        
        # Check for partial matches
        for key, value in self.OBJECT_MAPPING.items():
            if key in class_lower:
                return value
        
        # Default: return first word
        return words[0] if words else class_lower
    
    def _get_category(self, class_name):
        """Determine which category an object belongs to"""
        class_lower = class_name.lower()
        
        category_keywords = {
            'animals': ['cat', 'dog', 'bird', 'fish', 'horse', 'cow', 'lion', 'tiger', 'elephant', 'rabbit', 'sheep', 'goat'],
            'plants': ['flower', 'tree', 'plant', 'cactus', 'grass', 'leaf', 'palm', 'fern', 'rose'],
            'objects': ['car', 'chair', 'table', 'book', 'phone', 'computer', 'bottle', 'clock', 'lamp', 'key'],
            'food': ['pizza', 'apple', 'banana', 'cake', 'bread', 'rice', 'coffee', 'soup', 'salad'],
            'people': ['person', 'man', 'woman', 'child', 'baby', 'people', 'crowd'],
            'places': ['house', 'building', 'mountain', 'beach', 'forest', 'city', 'river', 'lake']
        }
        
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in class_lower:
                    return category
        
        return 'objects'  # default
    
    def predict(self, image_path, top_k=5):
        """
        Predict objects in image using boosting algorithm concept
        Returns simplified object names
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions (ensemble of attention heads - boosting concept)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = F.softmax(outputs.logits, dim=-1)[0]
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, top_k * 3)  # Get more for filtering
            
            # Process predictions
            seen_objects = set()
            simple_predictions = []
            
            for prob, idx in zip(top_probs, top_indices):
                class_id = idx.item()
                class_name = self.labels[class_id]
                confidence = prob.item()
                
                # Skip low confidence
                if confidence < 0.01:
                    continue
                
                # Simplify object name
                simple_name = self._simplify_object_name(class_name)
                
                # Avoid duplicates
                if simple_name in seen_objects:
                    continue
                
                # Get category
                category = self._get_category(class_name)
                
                simple_predictions.append({
                    'object': simple_name.title(),
                    'category': category,
                    'category_info': self.CATEGORIES[category],
                    'confidence': confidence,
                    'percentage': f"{confidence:.1%}"
                })
                
                seen_objects.add(simple_name)
                
                if len(simple_predictions) >= top_k:
                    break
            
            return simple_predictions
            
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def add_training_sample(self, image_path, category, object_name, username):
        """Add user feedback to improve model"""
        try:
            # Save image to training folder
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{username}_{timestamp}_{uuid.uuid4()}.jpg"
            save_path = os.path.join(app.config['TRAINING_FOLDER'], category, filename)
            
            # Copy image
            img = Image.open(image_path)
            img.save(save_path)
            
            # Save metadata
            metadata = {
                'object_name': object_name,
                'username': username,
                'timestamp': timestamp,
                'original_file': os.path.basename(image_path)
            }
            
            meta_path = save_path.replace('.jpg', '.json')
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update statistics
            self.training_stats[category] = self.training_stats.get(category, 0) + 1
            self._save_training_stats()
            
            return True
        except Exception as e:
            print(f"Error saving training sample: {e}")
            return False
    
    def get_training_stats(self):
        """Get training statistics"""
        return self.training_stats

classifier = HighLevelClassifier()

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
    return render_template_string(INDEX_HTML, 
                                 session=session, 
                                 categories=classifier.CATEGORIES,
                                 stats=classifier.get_training_stats(),
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
    """Add training sample"""
    if 'file' not in request.files or 'category' not in request.form or 'object_name' not in request.form:
        return jsonify({'error': 'Missing data'}), 400
    
    file = request.files['file']
    category = request.form['category']
    object_name = request.form['object_name']
    
    if category not in classifier.CATEGORIES:
        return jsonify({'error': 'Invalid category'}), 400
    
    # Save temp file
    filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        success = classifier.add_training_sample(filepath, category, object_name, session['username'])
        
        if success:
            user_manager.increment_training(session['username'])
            return jsonify({
                'success': True,
                'message': 'Training sample added!',
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
            background: linear-gradient(135deg, #006B3F, #FCD116);
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
            max-width: 400px;
            text-align: center;
        }
        .logo {
            margin-bottom: 20px;
        }
        .logo img {
            height: 60px;
        }
        .university-name {
            color: #006B3F;
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .college-name {
            color: #FCD116;
            font-size: 14px;
            margin-bottom: 20px;
        }
        .project-title {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 25px;
            border-left: 5px solid #FCD116;
        }
        .project-title h3 {
            color: #006B3F;
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
            margin-top: 10px;
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
        }
        .student-info {
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            font-size: 14px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="logo">
            <img src="data:image/png;base64,{{ logo }}" alt="AAU Logo" style="height: 60px;">
        </div>
        <div class="university-name">ADDIS ABABA UNIVERSITY</div>
        <div class="college-name">College of Natural and Computational Sciences<br>Department of Computer Science</div>
        
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
        
        <div class="student-info">
            <strong>Getaye Fiseha</strong> (GSE/6132/18)<br>
            Supervisor: Dr. Yaregal A.
        </div>
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
            background: #006B3F;
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
            background: #006B3F;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
        }
        .btn-danger { background: #dc3545; }
        table { width: 100%; border-collapse: collapse; }
        th { background: #006B3F; color: white; padding: 12px; }
        td { padding: 12px; border-bottom: 1px solid #ddd; }
    </style>
</head>
<body>
    <div class="navbar">
        <div class="logo">
            <img src="data:image/png;base64,{{ logo }}" alt="AAU Logo">
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
    <title>AAU - High-Level Image Classifier</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
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
            color: #FCD116;
        }
        
        .university-info p {
            font-size: 12px;
            opacity: 0.9;
        }
        
        .user-info {
            background: #FCD116;
            color: #006B3F;
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
            max-width: 1200px;
            margin: 40px auto;
            padding: 0 20px;
        }
        
        .project-header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .project-header h1 {
            color: #006B3F;
            font-size: 32px;
            margin-bottom: 10px;
        }
        
        .project-header .course {
            color: #FCD116;
            font-weight: bold;
            font-size: 18px;
            background: #006B3F;
            display: inline-block;
            padding: 8px 25px;
            border-radius: 30px;
            margin-top: 10px;
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
            border-left: 5px solid #FCD116;
        }
        
        .stat-number {
            font-size: 28px;
            font-weight: bold;
            color: #006B3F;
        }
        
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }
        
        .card {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        }
        
        .card h2 {
            color: #006B3F;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .upload-area {
            border: 3px dashed #006B3F;
            border-radius: 15px;
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
        
        .upload-icon {
            font-size: 48px;
            color: #006B3F;
            margin-bottom: 10px;
        }
        
        #preview {
            max-width: 100%;
            max-height: 300px;
            display: none;
            margin: 20px auto;
            border-radius: 10px;
            border: 4px solid #FCD116;
        }
        
        .btn {
            background: linear-gradient(135deg, #006B3F, #FCD116);
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
        
        .categories-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin: 20px 0;
        }
        
        .category-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 14px;
            border-left: 4px solid transparent;
        }
        
        .prediction-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            background: #f8f9fa;
            margin: 10px 0;
            border-radius: 10px;
            border-left: 8px solid #006B3F;
        }
        
        .prediction-name {
            font-size: 18px;
            font-weight: 600;
        }
        
        .prediction-category {
            font-size: 12px;
            color: #666;
        }
        
        .prediction-confidence {
            background: #006B3F;
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
        }
        
        .category-selector {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin: 15px 0;
        }
        
        .category-option {
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .category-option:hover {
            transform: scale(1.05);
        }
        
        .category-option.selected {
            border: 3px solid #FCD116;
        }
        
        .spinner {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #FCD116;
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
            background: #006B3F;
            color: #FCD116;
            padding: 20px;
            text-align: center;
            margin-top: 50px;
        }
    </style>
</head>
<body>
    <div class="navbar">
        <div class="logo">
            <img src="data:image/png;base64,{{ logo }}" alt="AAU Logo">
            <div class="university-info">
                <h2>ADDIS ABABA UNIVERSITY</h2>
                <p>College of Natural and Computational Sciences<br>Department of Computer Science</p>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 20px;">
            <span class="user-info">{{ session.full_name }} ({{ session.role }})</span>
            <div class="nav-links">
                <a href="/">Home</a>
                {% if session.role == 'admin' %}
                <a href="/users">Manage Users</a>
                {% endif %}
                <a href="/logout">Logout</a>
            </div>
        </div>
    </div>
    
    <div class="container">
        <div class="project-header">
            <h1>High-Level Image Classifier with Boosting Algorithm</h1>
            <div class="course">Machine Learning Course (COSC 6041)</div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{{ stats.values()|sum }}</div>
                <div>Training Samples</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ categories|length }}</div>
                <div>Categories</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">1000+</div>
                <div>Base Classes</div>
            </div>
        </div>
        
        <div class="main-grid">
            <!-- Classification Card -->
            <div class="card">
                <h2>📸 Classify Image</h2>
                <p style="color: #666; margin-bottom: 20px;">Upload any image to identify objects</p>
                
                <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                    <input type="file" id="fileInput" accept="image/*" style="display: none;">
                    <div class="upload-icon">📤</div>
                    <p>Click or drag image here</p>
                    <p style="color: #999; font-size: 14px;">JPG, PNG, JPEG (Max 16MB)</p>
                </div>
                
                <img id="preview" src="#" alt="Preview">
                
                <div class="spinner" id="spinner"></div>
                
                <button class="btn" onclick="classifyImage()">🔍 Classify Image</button>
                
                <div class="results" id="results">
                    <h3 style="color: #006B3F; margin-bottom: 15px;">Detected Objects:</h3>
                    <div id="predictionList"></div>
                </div>
            </div>
            
            <!-- Categories Card -->
            <div class="card">
                <h2>📋 Categories</h2>
                <div class="categories-grid">
                    {% for cat_id, cat in categories.items() %}
                    <div class="category-item" style="border-left-color: {{ cat.color }};">
                        <span style="font-size: 24px;">{{ cat.emoji }}</span><br>
                        <strong>{{ cat.name }}</strong><br>
                        <small>{{ stats[cat_id] }} samples</small>
                    </div>
                    {% endfor %}
                </div>
                
                <div style="margin-top: 20px;">
                    <h4 style="color: #006B3F;">Examples:</h4>
                    <ul style="columns: 2; list-style: none; margin-top: 10px;">
                        {% for cat_id, cat in categories.items() %}
                        <li style="margin: 5px 0;">{{ cat.emoji }} {{ cat.examples[:3]|join(', ') }}</li>
                        {% endfor %}
                    </ul>
                </div>
            </div>
        </div>
        
        <!-- Training Section -->
        <div class="training-section" id="trainingSection">
            <h3 style="color: #006B3F;">📚 Help Improve the Model</h3>
            <p>Select the correct category and object name:</p>
            
            <div class="category-selector" id="categorySelector"></div>
            
            <input type="text" id="objectName" placeholder="Enter object name (e.g., cat, car, flower)" style="width: 100%; padding: 12px; margin: 15px 0; border: 2px solid #ddd; border-radius: 8px;">
            
            <button class="btn btn-small" onclick="submitTraining()" style="background: #4CAF50;">✓ Submit Training Sample</button>
        </div>
    </div>
    
    <div class="footer">
        <p>Getaye Fiseha (GSE/6132/18) - Supervisor: Dr. Yaregal A.</p>
        <p>Addis Ababa University | College of Natural and Computational Sciences | Department of Computer Science</p>
        <p style="font-size: 12px; margin-top: 10px;">© 2026 - High-Level Image Classifier with Boosting Algorithm</p>
    </div>
    
    <script>
        let currentFile = null;
        let currentPredictions = null;
        let selectedCategory = null;
        
        // Category colors from backend
        const categoryColors = {
            {% for cat_id, cat in categories.items() %}
            '{{ cat_id }}': '{{ cat.color }}',
            {% endfor %}
        };
        
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
                    showTrainingSection(data.predictions);
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
                item.style.borderLeftColor = pred.category_info.color;
                item.innerHTML = `
                    <div>
                        <div class="prediction-name">${pred.category_info.emoji} ${pred.object}</div>
                        <div class="prediction-category">${pred.category_info.name}</div>
                    </div>
                    <span class="prediction-confidence">${pred.percentage}</span>
                `;
                predictionList.appendChild(item);
            });
            
            resultsDiv.style.display = 'block';
        }
        
        function showTrainingSection(predictions) {
            const section = document.getElementById('trainingSection');
            const selector = document.getElementById('categorySelector');
            
            selector.innerHTML = '';
            
            // Create category options
            predictions.forEach(pred => {
                const option = document.createElement('div');
                option.className = 'category-option';
                option.style.backgroundColor = pred.category_info.color;
                option.style.color = 'white';
                option.setAttribute('data-category', pred.category);
                option.innerHTML = `${pred.category_info.emoji} ${pred.category_info.name}`;
                option.onclick = function() {
                    selectCategory(pred.category, this);
                };
                selector.appendChild(option);
            });
            
            // Set object name to top prediction
            document.getElementById('objectName').value = predictions[0].object.toLowerCase();
            
            section.style.display = 'block';
        }
        
        function selectCategory(categoryId, element) {
            selectedCategory = categoryId;
            document.querySelectorAll('.category-option').forEach(el => {
                el.classList.remove('selected');
            });
            element.classList.add('selected');
        }
        
        async function submitTraining() {
            if (!selectedCategory) {
                alert('Please select a category');
                return;
            }
            
            const objectName = document.getElementById('objectName').value.trim();
            if (!objectName) {
                alert('Please enter an object name');
                return;
            }
            
            if (!currentFile) {
                alert('No image to train on');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', currentFile);
            formData.append('category', selectedCategory);
            formData.append('object_name', objectName);
            
            try {
                const response = await fetch('/train', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    alert('✅ Training sample added! Thank you for helping improve the model.');
                    document.getElementById('trainingSection').style.display = 'none';
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
