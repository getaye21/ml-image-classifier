"""
====================================================================
ADDIS ABABA UNIVERSITY - College of Natural and Computational Sciences
Department of Computer Science
====================================================================

Project: High-Level Image Classifier with User Management
Student Name: Getaye Fiseha
Student ID: GSE/6132/18
Program: MSc in Computer Science
Stream: Network & Security
Course: COSC 6041 - Machine Learning
Supervisor: Dr. Yaregal A.
Submission Date: March 2026

CATEGORIES:
🐱 Animals: cat, dog, tiger, elephant, bird, fish, lion, horse, cow, monkey
🌿 Plants: flower, tree, rose, sunflower, cactus, grass, leaf, palm, fern
🚗 Objects: car, bicycle, airplane, chair, book, phone, computer, bottle, clock
🍎 Food: pizza, apple, banana, cake, coffee, bread, rice, sandwich, chocolate
👤 People: person, man, woman, baby, crowd, portrait, child, family, group
🏠 Places: house, mountain, beach, forest, city, river, ocean, desert, building
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
import numpy as np
from werkzeug.utils import secure_filename
import joblib

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
app.config['MODEL_DATA'] = 'category_model.pkl'

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('training_data', exist_ok=True)

# ============================================================================
# User Management System
# ============================================================================

class UserManager:
    def __init__(self):
        self.users_file = app.config['USER_DATA']
        self.login_attempts = {}
        self._init_default_users()
    
    def _init_default_users(self):
        """Create default admin user"""
        if not os.path.exists(self.users_file):
            users = {
                'getaye': {
                    'password_hash': self._hash_password('Getaye@2827'),
                    'full_name': 'Getaye Fiseha',
                    'role': 'admin',
                    'student_id': 'GSE/6132/18',
                    'email': 'getaye.fiseha@aau.edu.et',
                    'created_at': datetime.now().isoformat(),
                    'can_create_users': True
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
    
    def create_user(self, username, password, full_name, email, role='user', created_by=None):
        users = self._load_users()
        
        if username in users:
            return False, "Username already exists"
        
        if any(u['email'] == email for u in users.values()):
            return False, "Email already registered"
        
        users[username] = {
            'password_hash': self._hash_password(password),
            'full_name': full_name,
            'email': email,
            'role': role,
            'created_at': datetime.now().isoformat(),
            'created_by': created_by,
            'last_login': None,
            'can_create_users': (role == 'admin')
        }
        
        self._save_users(users)
        return True, "User created successfully"
    
    def verify_login(self, username, password, ip_address):
        users = self._load_users()
        
        if username not in users:
            return False, "Invalid username or password"
        
        if self._hash_password(password) == users[username]['password_hash']:
            # Update last login
            users[username]['last_login'] = datetime.now().isoformat()
            self._save_users(users)
            return True, users[username]
        
        return False, "Invalid username or password"
    
    def get_all_users(self):
        users = self._load_users()
        # Remove password hashes for security
        for u in users.values():
            u.pop('password_hash', None)
        return users
    
    def delete_user(self, username, admin_user):
        if username == 'getaye':  # Can't delete main admin
            return False, "Cannot delete primary admin"
        
        users = self._load_users()
        if username in users:
            del users[username]
            self._save_users(users)
            return True, "User deleted"
        return False, "User not found"

user_manager = UserManager()

# ============================================================================
# Login Decorators
# ============================================================================

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            flash('Please login first', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            flash('Admin access required', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated

# ============================================================================
# Category Mapper - Maps 1000+ ImageNet classes to 6 main categories
# ============================================================================

class CategoryMapper:
    """Maps specific objects to high-level categories"""
    
    # Category definitions with keywords
    CATEGORIES = {
        'animals': {
            'name': '🐱 Animals',
            'emoji': '🐱',
            'color': '#FF6B6B',
            'keywords': [
                'cat', 'dog', 'tiger', 'lion', 'elephant', 'bird', 'fish', 'horse', 
                'cow', 'monkey', 'bear', 'rabbit', 'sheep', 'goat', 'chicken', 'duck',
                'eagle', 'hawk', 'snake', 'lizard', 'frog', 'whale', 'dolphin', 'shark',
                'zebra', 'giraffe', 'deer', 'fox', 'wolf', 'mouse', 'rat', 'squirrel',
                'kangaroo', 'panda', 'koala', 'penguin', 'swan', 'goose', 'turkey'
            ]
        },
        'plants': {
            'name': '🌿 Plants',
            'emoji': '🌿',
            'color': '#4CAF50',
            'keywords': [
                'flower', 'tree', 'rose', 'sunflower', 'cactus', 'grass', 'leaf', 
                'palm', 'fern', 'bush', 'plant', 'weed', 'vine', 'moss', 'mushroom',
                'daisy', 'tulip', 'lily', 'orchid', 'bamboo', 'oak', 'pine', 'maple',
                'willow', 'ivy', 'clover', 'algae', 'fern', 'coral'
            ]
        },
        'objects': {
            'name': '🚗 Objects',
            'emoji': '🚗',
            'color': '#4A90E2',
            'keywords': [
                'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'airplane', 'train',
                'boat', 'ship', 'chair', 'table', 'desk', 'book', 'phone', 'computer',
                'laptop', 'bottle', 'cup', 'glass', 'plate', 'bowl', 'fork', 'spoon',
                'knife', 'clock', 'watch', 'umbrella', 'bag', 'backpack', 'wallet',
                'key', 'door', 'window', 'bed', 'sofa', 'lamp', 'television', 'radio'
            ]
        },
        'food': {
            'name': '🍎 Food',
            'emoji': '🍎',
            'color': '#F4A460',
            'keywords': [
                'pizza', 'apple', 'banana', 'cake', 'coffee', 'bread', 'rice', 'sandwich',
                'chocolate', 'pasta', 'burger', 'fries', 'salad', 'soup', 'steak', 'fish',
                'chicken', 'egg', 'cheese', 'milk', 'juice', 'water', 'wine', 'beer',
                'ice cream', 'cookie', 'donut', 'pie', 'candy', 'fruit', 'vegetable',
                'tomato', 'potato', 'carrot', 'onion', 'garlic', 'lemon', 'orange'
            ]
        },
        'people': {
            'name': '👤 People',
            'emoji': '👤',
            'color': '#9B59B6',
            'keywords': [
                'person', 'man', 'woman', 'child', 'baby', 'crowd', 'people', 'human',
                'boy', 'girl', 'teenager', 'adult', 'senior', 'family', 'group',
                'portrait', 'face', 'head', 'hand', 'foot', 'body', 'couple', 'friends',
                'worker', 'doctor', 'teacher', 'student', 'athlete', 'singer', 'dancer'
            ]
        },
        'places': {
            'name': '🏠 Places',
            'emoji': '🏠',
            'color': '#E67E22',
            'keywords': [
                'house', 'building', 'mountain', 'beach', 'forest', 'city', 'river',
                'ocean', 'desert', 'lake', 'park', 'garden', 'farm', 'field', 'hill',
                'valley', 'cave', 'island', 'waterfall', 'volcano', 'glacier', 'coast',
                'sky', 'cloud', 'sunset', 'sunrise', 'night', 'day', 'street', 'road',
                'bridge', 'tunnel', 'tower', 'castle', 'temple', 'church', 'school'
            ]
        }
    }
    
    @classmethod
    def get_category(cls, class_name):
        """Map a specific class name to one of the 6 categories"""
        class_lower = class_name.lower()
        
        for category_id, category_info in cls.CATEGORIES.items():
            for keyword in category_info['keywords']:
                if keyword in class_lower:
                    return category_id, category_info
        
        # Default to objects if no match
        return 'objects', cls.CATEGORIES['objects']
    
    @classmethod
    def get_all_categories(cls):
        return cls.CATEGORIES

# ============================================================================
# High-Level Category Classifier
# ============================================================================

class HighLevelClassifier:
    """
    Classifies images into 6 main categories:
    Animals, Plants, Objects, Food, People, Places
    """
    
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
        self.imagenet_labels = self.model.config.id2label
        
        # Category mapper
        self.categories = CategoryMapper.get_all_categories()
        
        # Training data storage
        self.training_data = self._load_training_data()
        
        print(f"✅ Classifier ready! Categories:")
        for cat in self.categories.values():
            print(f"   {cat['emoji']} {cat['name']}")
    
    def _load_training_data(self):
        """Load additional training data if exists"""
        if os.path.exists('training_stats.json'):
            with open('training_stats.json', 'r') as f:
                return json.load(f)
        return {cat: 0 for cat in self.categories.keys()}
    
    def _save_training_data(self):
        with open('training_stats.json', 'w') as f:
            json.dump(self.training_data, f, indent=2)
    
    def predict(self, image_path):
        """
        Predict which of the 6 categories the image belongs to
        Returns probabilities for each category
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)[0]
            
            # Map to categories
            category_scores = {cat_id: 0.0 for cat_id in self.categories.keys()}
            category_details = {cat_id: [] for cat_id in self.categories.keys()}
            
            # Aggregate probabilities by category
            top_probs, top_indices = torch.topk(probabilities, 50)
            
            for prob, idx in zip(top_probs, top_indices):
                class_id = idx.item()
                class_name = self.imagenet_labels[class_id].lower()
                confidence = prob.item()
                
                # Find which category this belongs to
                category_id, category_info = CategoryMapper.get_category(class_name)
                
                # Add to category score
                category_scores[category_id] += confidence
                category_details[category_id].append({
                    'class': class_name,
                    'confidence': confidence
                })
            
            # Normalize scores to sum to 1
            total = sum(category_scores.values())
            if total > 0:
                for cat_id in category_scores:
                    category_scores[cat_id] /= total
            
            # Sort categories by score
            sorted_categories = sorted(
                category_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Prepare results
            results = []
            for cat_id, score in sorted_categories:
                cat_info = self.categories[cat_id]
                results.append({
                    'category_id': cat_id,
                    'category': cat_info['name'],
                    'emoji': cat_info['emoji'],
                    'color': cat_info['color'],
                    'confidence': score,
                    'probability': f"{score:.1%}",
                    'top_matches': category_details[cat_id][:3]  # Top 3 specific items
                })
            
            return results
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def add_training_sample(self, image_path, category_id):
        """Add a training sample to improve category recognition"""
        try:
            # Extract features
            image = Image.open(image_path).convert('RGB')
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            
            # Save for future fine-tuning
            train_dir = f'training_data/{category_id}'
            os.makedirs(train_dir, exist_ok=True)
            
            filename = f"{uuid.uuid4()}.jpg"
            filepath = os.path.join(train_dir, filename)
            image.save(filepath)
            
            # Update stats
            self.training_data[category_id] = self.training_data.get(category_id, 0) + 1
            self._save_training_data()
            
            return True
        except Exception as e:
            print(f"Error saving training sample: {e}")
            return False

# Initialize the classifier
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
        ip_address = request.remote_addr
        
        success, result = user_manager.verify_login(username, password, ip_address)
        
        if success:
            session['username'] = username
            session['full_name'] = result['full_name']
            session['role'] = result['role']
            session.permanent = True
            flash(f'Welcome, {result["full_name"]}!', 'success')
            return redirect(url_for('index'))
        else:
            error = result
    
    return render_template_string(LOGIN_HTML, error=error)

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'info')
    return redirect(url_for('login'))

# ============================================================================
# User Management Routes (Admin Only)
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
    email = request.form.get('email')
    role = request.form.get('role', 'user')
    
    success, message = user_manager.create_user(
        username, password, full_name, email, role, session['username']
    )
    
    if success:
        flash(message, 'success')
    else:
        flash(message, 'error')
    
    return redirect(url_for('user_list'))

@app.route('/users/delete/<username>', methods=['POST'])
@admin_required
def delete_user(username):
    success, message = user_manager.delete_user(username, session['username'])
    if success:
        flash(message, 'success')
    else:
        flash(message, 'error')
    return redirect(url_for('user_list'))

# ============================================================================
# Main Classification Routes
# ============================================================================

@app.route('/')
@login_required
def index():
    """Main classification page"""
    return render_template_string(INDEX_HTML, 
                                 session=session,
                                 categories=classifier.categories,
                                 training_stats=classifier.training_data)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Predict category of uploaded image"""
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
        # Get predictions
        predictions = classifier.predict(filepath)
        
        if predictions:
            return jsonify({
                'success': True,
                'predictions': predictions,
                'filename': file.filename
            })
        else:
            return jsonify({'error': 'Could not classify image'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/train', methods=['POST'])
@login_required
def train():
    """Add training sample to improve category"""
    if 'file' not in request.files or 'category' not in request.form:
        return jsonify({'error': 'Missing data'}), 400
    
    file = request.files['file']
    category_id = request.form['category']
    
    if category_id not in classifier.categories:
        return jsonify({'error': 'Invalid category'}), 400
    
    # Save temp file
    filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Add training sample
        success = classifier.add_training_sample(filepath, category_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Added to {classifier.categories[category_id]["name"]} training set',
                'stats': classifier.training_data
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
    return jsonify(classifier.training_data)

# ============================================================================
# HTML Templates
# ============================================================================

LOGIN_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AAU - Login | Image Classifier</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #006B3F 0%, #FCD116 100%); 
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
        }
        .aau-header { 
            text-align: center; 
            margin-bottom: 30px; 
        }
        .aau-header h1 { 
            color: #006B3F; 
            font-size: 24px; 
            margin-bottom: 5px; 
        }
        .aau-header p { 
            color: #FCD116; 
            font-weight: bold; 
        }
        .form-group { 
            margin-bottom: 20px; 
        }
        .form-group input { 
            width: 100%; 
            padding: 12px; 
            border: 2px solid #e1e1e1; 
            border-radius: 8px; 
            font-size: 16px; 
            transition: border-color 0.3s; 
        }
        .form-group input:focus { 
            outline: none; 
            border-color: #FCD116; 
        }
        .btn { 
            width: 100%; 
            padding: 14px; 
            background: linear-gradient(135deg, #006B3F, #FCD116); 
            color: white; 
            border: none; 
            border-radius: 8px; 
            font-size: 18px; 
            font-weight: 600; 
            cursor: pointer; 
            transition: transform 0.3s; 
        }
        .btn:hover { 
            transform: translateY(-2px); 
            box-shadow: 0 10px 20px rgba(0,107,63,0.3); 
        }
        .error { 
            background: #fee; 
            color: #c33; 
            padding: 12px; 
            border-radius: 8px; 
            margin-bottom: 20px; 
            text-align: center; 
        }
        .info { 
            text-align: center; 
            margin-top: 20px; 
            padding: 15px; 
            background: #f5f5f5; 
            border-radius: 8px; 
            font-size: 14px; 
        }
        .info p { 
            margin: 5px 0; 
            color: #666; 
        }
        .student-info {
            background: #e8f5e9;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
            border-left: 4px solid #FCD116;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="aau-header">
            <h1>🎓 ADDIS ABABA UNIVERSITY</h1>
            <p>MSc Computer Science - Network & Security</p>
        </div>
        
        <div class="student-info">
            <strong>Getaye Fiseha</strong><br>
            GSE/6132/18 | Dr. Yaregal A.
        </div>
        
        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
        
        <form method="POST">
            <div class="form-group">
                <input type="text" name="username" placeholder="Username" required>
            </div>
            <div class="form-group">
                <input type="password" name="password" placeholder="Password" required>
            </div>
            <button type="submit" class="btn">🔐 Login to Classifier</button>
        </form>
        
        <div class="info">
            <p><strong>Admin:</strong> getaye / Getaye@2827</p>
            <p><strong>Demo Users:</strong> Ask admin to create</p>
        </div>
    </div>
</body>
</html>
"""

USERS_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AAU - User Management | Admin</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; }
        .navbar { 
            background: linear-gradient(135deg, #006B3F, #FCD116); 
            padding: 15px 30px; 
            color: white; 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
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
            background: linear-gradient(135deg, #006B3F, #FCD116); 
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
        th { background: #006B3F; color: #FCD116; padding: 12px; text-align: left; }
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
        <h2>👑 Admin Panel - User Management</h2>
        <div>
            <a href="/" style="color: white; margin-right: 20px;">Home</a>
            <a href="/logout" style="color: white;">Logout</a>
        </div>
    </div>
    
    <div class="container">
        <div class="card">
            <h2 style="color: #006B3F; margin-bottom: 20px;">➕ Create New User</h2>
            <form action="/users/create" method="POST">
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
                    <div class="form-group">
                        <input type="text" name="username" placeholder="Username" required>
                    </div>
                    <div class="form-group">
                        <input type="password" name="password" placeholder="Password" required>
                    </div>
                    <div class="form-group">
                        <input type="text" name="full_name" placeholder="Full Name" required>
                    </div>
                    <div class="form-group">
                        <input type="email" name="email" placeholder="Email" required>
                    </div>
                    <div class="form-group">
                        <select name="role">
                            <option value="user">User</option>
                            <option value="admin">Admin</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <button type="submit" class="btn">Create User</button>
                    </div>
                </div>
            </form>
        </div>
        
        <div class="card">
            <h2 style="color: #006B3F; margin-bottom: 20px;">📋 Registered Users</h2>
            <table>
                <thead>
                    <tr>
                        <th>Username</th>
                        <th>Full Name</th>
                        <th>Email</th>
                        <th>Role</th>
                        <th>Created</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {% for username, info in users.items() %}
                    <tr>
                        <td><strong>{{ username }}</strong></td>
                        <td>{{ info.full_name }}</td>
                        <td>{{ info.email }}</td>
                        <td>{% if info.role == 'admin' %}👑 Admin{% else %}👤 User{% endif %}</td>
                        <td>{{ info.created_at[:10] }}</td>
                        <td>
                            {% if username != current_user %}
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
    <title>AAU - Category Classifier | Getaye Fiseha</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }
        
        .navbar { 
            background: linear-gradient(135deg, #006B3F, #FCD116); 
            padding: 15px 30px; 
            color: white; 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.1); 
        }
        .nav-links a { 
            color: white; 
            text-decoration: none; 
            margin-left: 20px; 
            padding: 8px 16px; 
            border-radius: 5px; 
            transition: background 0.3s; 
        }
        .nav-links a:hover { 
            background: rgba(255,255,255,0.2); 
        }
        .student-badge { 
            background: #FCD116; 
            color: #006B3F; 
            padding: 5px 15px; 
            border-radius: 20px; 
            font-weight: bold; 
        }
        
        .container { max-width: 1200px; margin: 40px auto; padding: 0 20px; }
        
        .hero {
            text-align: center;
            margin-bottom: 40px;
        }
        .hero h1 {
            color: #006B3F;
            font-size: 36px;
        }
        
        .category-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .category-card {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            text-align: center;
            font-size: 18px;
            font-weight: bold;
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
            padding: 60px; 
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
            margin: 10px 0;
        }
        .btn:hover { 
            transform: translateY(-3px); 
            box-shadow: 0 15px 30px rgba(0,107,63,0.4);
        }
        
        .results { 
            margin-top: 40px; 
            display: none; 
        }
        
        .result-bar {
            margin: 15px 0;
            padding: 15px;
            border-radius: 10px;
            color: white;
            position: relative;
            overflow: hidden;
        }
        
        .result-bar-inner {
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            background: rgba(255,255,255,0.2);
            transition: width 0.5s;
            z-index: 1;
        }
        
        .result-content {
            position: relative;
            z-index: 2;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .result-emoji {
            font-size: 24px;
            margin-right: 15px;
        }
        
        .result-percentage {
            font-size: 24px;
            font-weight: bold;
        }
        
        .training-section {
            background: white;
            border-radius: 20px;
            padding: 30px;
            margin-top: 30px;
        }
        
        .category-badge {
            display: inline-block;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 25px;
            color: white;
            cursor: pointer;
            transition: transform 0.3s;
        }
        .category-badge:hover {
            transform: scale(1.05);
        }
        .category-badge.selected {
            transform: scale(1.1);
            box-shadow: 0 0 0 3px #FCD116;
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
        <div>
            <span class="student-badge">GSE/6132/18 | {{ session.full_name }}</span>
        </div>
        <div class="nav-links">
            <a href="/">Home</a>
            {% if session.role == 'admin' %}
            <a href="/users">👑 Manage Users</a>
            {% endif %}
            <a href="/logout">Logout</a>
        </div>
    </div>
    
    <div class="container">
        <div class="hero">
            <h1>🎯 High-Level Image Classifier</h1>
            <p>Upload any image - I'll tell you which category it belongs to!</p>
        </div>
        
        <div class="category-grid">
            {% for cat_id, cat in categories.items() %}
            <div class="category-card" style="border-left: 8px solid {{ cat.color }};">
                {{ cat.emoji }} {{ cat.name }}
            </div>
            {% endfor %}
        </div>
        
        <div class="main-card">
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <input type="file" id="fileInput" accept="image/*" style="display: none;">
                <i>📸</i>
                <p style="font-size: 20px;">Click or drag image here</p>
                <p style="color: #999;">JPG, PNG, JPEG (Max 16MB)</p>
            </div>
            
            <img id="preview" src="#" alt="Preview">
            
            <div id="spinner" style="text-align: center; margin: 20px; display: none;">
                <div style="border: 5px solid #f3f3f3; border-top: 5px solid #FCD116; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin: 0 auto;"></div>
                <p style="margin-top: 10px;">Analyzing image...</p>
            </div>
            
            <button class="btn" onclick="classifyImage()">🔍 CLASSIFY IMAGE</button>
            
            <div id="results" class="results"></div>
        </div>
        
        <div class="training-section" id="trainingSection" style="display: none;">
            <h2 style="color: #006B3F; margin-bottom: 20px;">📚 Help Improve the Model</h2>
            <p>Select the correct category for this image:</p>
            <div id="trainingCategories" style="margin: 20px 0;"></div>
            <button class="btn" onclick="submitTraining()" style="background: #4CAF50;">✓ Submit Training Sample</button>
        </div>
    </div>
    
    <div class="footer">
        <p>Getaye Fiseha (GSE/6132/18) - MSc Computer Science, Network & Security</p>
        <p>Addis Ababa University | COSC 6041 - Machine Learning</p>
    </div>
    
    <style>@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }</style>
    
    <script>
        let currentImageFile = null;
        let selectedCategory = null;
        let currentPredictions = null;
        
        document.getElementById('fileInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                currentImageFile = file;
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('preview').src = e.target.result;
                    document.getElementById('preview').style.display = 'block';
                }
                reader.readAsDataURL(file);
            }
        });

        async function classifyImage() {
            if (!currentImageFile) {
                alert('Please select an image first');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', currentImageFile);
            
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
            resultsDiv.innerHTML = '<h2 style="color: #006B3F; margin-bottom: 20px;">📊 Classification Results</h2>';
            
            predictions.forEach(pred => {
                const barWidth = pred.confidence * 100;
                const resultDiv = document.createElement('div');
                resultDiv.className = 'result-bar';
                resultDiv.style.backgroundColor = pred.color;
                resultDiv.innerHTML = `
                    <div class="result-bar-inner" style="width: ${barWidth}%;"></div>
                    <div class="result-content">
                        <div>
                            <span class="result-emoji">${pred.emoji}</span>
                            <span style="font-size: 18px;">${pred.category}</span>
                        </div>
                        <span class="result-percentage">${pred.probability}</span>
                    </div>
                `;
                resultsDiv.appendChild(resultDiv);
            });
            
            // Show top matches
            const topPred = predictions[0];
            if (topPred && topPred.top_matches && topPred.top_matches.length > 0) {
                const detailsDiv = document.createElement('div');
                detailsDiv.style.marginTop = '20px';
                detailsDiv.style.padding = '15px';
                detailsDiv.style.background = '#f8f9fa';
                detailsDiv.style.borderRadius = '10px';
                detailsDiv.innerHTML = `
                    <p><strong>Top detected items in ${topPred.category}:</strong></p>
                    <ul style="margin-left: 20px;">
                        ${topPred.top_matches.map(m => `<li>${m.class} (${(m.confidence*100).toFixed(1)}%)</li>`).join('')}
                    </ul>
                `;
                resultsDiv.appendChild(detailsDiv);
            }
            
            resultsDiv.style.display = 'block';
        }
        
        function showTrainingSection(predictions) {
            const section = document.getElementById('trainingSection');
            const categoriesDiv = document.getElementById('trainingCategories');
            
            categoriesDiv.innerHTML = '';
            
            predictions.forEach(pred => {
                const badge = document.createElement('span');
                badge.className = 'category-badge';
                badge.style.backgroundColor = pred.color;
                badge.textContent = `${pred.emoji} ${pred.category}`;
                badge.onclick = () => selectTrainingCategory(pred.category_id, badge);
                categoriesDiv.appendChild(badge);
            });
            
            section.style.display = 'block';
        }
        
        function selectTrainingCategory(categoryId, element) {
            selectedCategory = categoryId;
            document.querySelectorAll('.category-badge').forEach(el => {
                el.classList.remove('selected');
            });
            element.classList.add('selected');
        }
        
        async function submitTraining() {
            if (!selectedCategory) {
                alert('Please select a category');
                return;
            }
            
            if (!currentImageFile) {
                alert('No image to train on');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', currentImageFile);
            formData.append('category', selectedCategory);
            
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

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
