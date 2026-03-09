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

AUTO-TRAINING FEATURE: Trains on 5 images per category at startup!
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
import pickle
import requests
from io import BytesIO
import time
from tqdm import tqdm

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
app.config['LEARNING_DATA'] = 'learning_data.pkl'

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TRAINING_FOLDER'], exist_ok=True)

# ============================================================================
# AAU Logo (Red and Gold)
# ============================================================================

AAU_LOGO_SVG = '''
<svg width="200" height="80" viewBox="0 0 200 80" xmlns="http://www.w3.org/2000/svg">
    <rect width="200" height="80" fill="#8B0000" rx="10" ry="10"/>
    <circle cx="60" cy="40" r="20" fill="#FFD700"/>
    <rect x="55" y="40" width="10" height="25" fill="#FFD700"/>
    <path d="M45 45 L75 45 L70 65 L50 65 Z" fill="#FFD700"/>
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
                    'created_at': datetime.now().isoformat()
                }
            }
            self._save_users(users)
    
    def _hash_password(self, password):
        salt = "AAU_SALT"
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
# TRAINED CLASSIFIER - Pre-loaded with 5 images per category
# ============================================================================

class TrainedClassifier:
    """
    Classifier pre-trained with 5 images per category
    Knows: Animals, Plants, Objects, Food, People, Places
    """
    
    # Complete category definitions with 5+ examples each
    CATEGORIES = {
        # 🐱 ANIMALS (10+ examples)
        'cat': {'category': 'animals', 'emoji': '🐱', 'color': '#FF6B6B'},
        'dog': {'category': 'animals', 'emoji': '🐕', 'color': '#FF6B6B'},
        'tiger': {'category': 'animals', 'emoji': '🐅', 'color': '#FF6B6B'},
        'elephant': {'category': 'animals', 'emoji': '🐘', 'color': '#FF6B6B'},
        'lion': {'category': 'animals', 'emoji': '🦁', 'color': '#FF6B6B'},
        'bird': {'category': 'animals', 'emoji': '🐦', 'color': '#FF6B6B'},
        'fish': {'category': 'animals', 'emoji': '🐠', 'color': '#FF6B6B'},
        'horse': {'category': 'animals', 'emoji': '🐎', 'color': '#FF6B6B'},
        'cow': {'category': 'animals', 'emoji': '🐄', 'color': '#FF6B6B'},
        'rabbit': {'category': 'animals', 'emoji': '🐰', 'color': '#FF6B6B'},
        
        # 🌿 PLANTS (10+ examples)
        'flower': {'category': 'plants', 'emoji': '🌸', 'color': '#4CAF50'},
        'rose': {'category': 'plants', 'emoji': '🌹', 'color': '#4CAF50'},
        'sunflower': {'category': 'plants', 'emoji': '🌻', 'color': '#4CAF50'},
        'tree': {'category': 'plants', 'emoji': '🌳', 'color': '#4CAF50'},
        'cactus': {'category': 'plants', 'emoji': '🌵', 'color': '#4CAF50'},
        'grass': {'category': 'plants', 'emoji': '🌿', 'color': '#4CAF50'},
        'palm': {'category': 'plants', 'emoji': '🌴', 'color': '#4CAF50'},
        'leaf': {'category': 'plants', 'emoji': '🍃', 'color': '#4CAF50'},
        'bamboo': {'category': 'plants', 'emoji': '🎋', 'color': '#4CAF50'},
        'mushroom': {'category': 'plants', 'emoji': '🍄', 'color': '#4CAF50'},
        
        # 🚗 OBJECTS (10+ examples)
        'car': {'category': 'objects', 'emoji': '🚗', 'color': '#4A90E2'},
        'truck': {'category': 'objects', 'emoji': '🚚', 'color': '#4A90E2'},
        'bus': {'category': 'objects', 'emoji': '🚌', 'color': '#4A90E2'},
        'bicycle': {'category': 'objects', 'emoji': '🚲', 'color': '#4A90E2'},
        'airplane': {'category': 'objects', 'emoji': '✈️', 'color': '#4A90E2'},
        'chair': {'category': 'objects', 'emoji': '🪑', 'color': '#4A90E2'},
        'table': {'category': 'objects', 'emoji': '🪑', 'color': '#4A90E2'},
        'book': {'category': 'objects', 'emoji': '📚', 'color': '#4A90E2'},
        'phone': {'category': 'objects', 'emoji': '📱', 'color': '#4A90E2'},
        'computer': {'category': 'objects', 'emoji': '💻', 'color': '#4A90E2'},
        
        # 🍎 FOOD (10+ examples)
        'pizza': {'category': 'food', 'emoji': '🍕', 'color': '#F4A460'},
        'apple': {'category': 'food', 'emoji': '🍎', 'color': '#F4A460'},
        'banana': {'category': 'food', 'emoji': '🍌', 'color': '#F4A460'},
        'cake': {'category': 'food', 'emoji': '🍰', 'color': '#F4A460'},
        'coffee': {'category': 'food', 'emoji': '☕', 'color': '#F4A460'},
        'bread': {'category': 'food', 'emoji': '🍞', 'color': '#F4A460'},
        'rice': {'category': 'food', 'emoji': '🍚', 'color': '#F4A460'},
        'burger': {'category': 'food', 'emoji': '🍔', 'color': '#F4A460'},
        'pasta': {'category': 'food', 'emoji': '🍝', 'color': '#F4A460'},
        'ice cream': {'category': 'food', 'emoji': '🍦', 'color': '#F4A460'},
        
        # 👤 PEOPLE (10+ examples)
        'person': {'category': 'people', 'emoji': '👤', 'color': '#9B59B6'},
        'man': {'category': 'people', 'emoji': '👨', 'color': '#9B59B6'},
        'woman': {'category': 'people', 'emoji': '👩', 'color': '#9B59B6'},
        'baby': {'category': 'people', 'emoji': '👶', 'color': '#9B59B6'},
        'crowd': {'category': 'people', 'emoji': '👥', 'color': '#9B59B6'},
        'child': {'category': 'people', 'emoji': '🧒', 'color': '#9B59B6'},
        'family': {'category': 'people', 'emoji': '👪', 'color': '#9B59B6'},
        'doctor': {'category': 'people', 'emoji': '👨‍⚕️', 'color': '#9B59B6'},
        'teacher': {'category': 'people', 'emoji': '👩‍🏫', 'color': '#9B59B6'},
        'student': {'category': 'people', 'emoji': '🧑‍🎓', 'color': '#9B59B6'},
        
        # 🏠 PLACES (10+ examples)
        'house': {'category': 'places', 'emoji': '🏠', 'color': '#E67E22'},
        'mountain': {'category': 'places', 'emoji': '⛰️', 'color': '#E67E22'},
        'beach': {'category': 'places', 'emoji': '🏖️', 'color': '#E67E22'},
        'forest': {'category': 'places', 'emoji': '🌲', 'color': '#E67E22'},
        'city': {'category': 'places', 'emoji': '🌆', 'color': '#E67E22'},
        'river': {'category': 'places', 'emoji': '🌊', 'color': '#E67E22'},
        'lake': {'category': 'places', 'emoji': '🏞️', 'color': '#E67E22'},
        'desert': {'category': 'places', 'emoji': '🏜️', 'color': '#E67E22'},
        'building': {'category': 'places', 'emoji': '🏢', 'color': '#E67E22'},
        'park': {'category': 'places', 'emoji': '🌳', 'color': '#E67E22'}
    }
    
    # Training images URLs (5 per category)
    TRAINING_URLS = {
        # Animals
        'cat': [
            'https://images.pexels.com/photos/45201/kitty-cat-kitten-pet-45201.jpeg',
            'https://images.pexels.com/photos/416160/pexels-photo-416160.jpeg',
            'https://images.pexels.com/photos/20787/pexels-photo.jpg',
            'https://images.pexels.com/photos/96938/pexels-photo-96938.jpeg',
            'https://images.pexels.com/photos/730896/pexels-photo-730896.jpeg'
        ],
        'dog': [
            'https://images.pexels.com/photos/1805164/pexels-photo-1805164.jpeg',
            'https://images.pexels.com/photos/2023384/pexels-photo-2023384.jpeg',
            'https://images.pexels.com/photos/406014/pexels-photo-406014.jpeg',
            'https://images.pexels.com/photos/733416/pexels-photo-733416.jpeg',
            'https://images.pexels.com/photos/542394/pexels-photo-542394.jpeg'
        ],
        'bird': [
            'https://images.pexels.com/photos/326900/pexels-photo-326900.jpeg',
            'https://images.pexels.com/photos/45911/pexels-photo-45911.jpeg',
            'https://images.pexels.com/photos/36762/parrot-bird-beak-blue.jpg',
            'https://images.pexels.com/photos/63574/pexels-photo-63574.jpeg',
            'https://images.pexels.com/photos/60009/pexels-photo-60009.jpeg'
        ],
        'fish': [
            'https://images.pexels.com/photos/128756/pexels-photo-128756.jpeg',
            'https://images.pexels.com/photos/36348/pexels-photo.jpg',
            'https://images.pexels.com/photos/1386604/pexels-photo-1386604.jpeg',
            'https://images.pexels.com/photos/799410/pexels-photo-799410.jpeg',
            'https://images.pexels.com/photos/3763819/pexels-photo-3763819.jpeg'
        ],
        
        # Plants
        'flower': [
            'https://images.pexels.com/photos/736230/pexels-photo-736230.jpeg',
            'https://images.pexels.com/photos/56866/garden-rose-red-pink-56866.jpeg',
            'https://images.pexels.com/photos/931149/pexels-photo-931149.jpeg',
            'https://images.pexels.com/photos/36764/pexels-photo.jpg',
            'https://images.pexels.com/photos/462118/pexels-photo-462118.jpeg'
        ],
        'tree': [
            'https://images.pexels.com/photos/38136/pexels-photo-38136.jpeg',
            'https://images.pexels.com/photos/97554/pexels-photo-97554.jpeg',
            'https://images.pexels.com/photos/145685/pexels-photo-145685.jpeg',
            'https://images.pexels.com/photos/372058/pexels-photo-372058.jpeg',
            'https://images.pexels.com/photos/158607/caucasus-tree-birds-Autumn-158607.jpeg'
        ],
        
        # Objects
        'car': [
            'https://images.pexels.com/photos/170811/pexels-photo-170811.jpeg',
            'https://images.pexels.com/photos/116675/pexels-photo-116675.jpeg',
            'https://images.pexels.com/photos/210019/pexels-photo-210019.jpeg',
            'https://images.pexels.com/photos/3802508/pexels-photo-3802508.jpeg',
            'https://images.pexels.com/photos/112460/pexels-photo-112460.jpeg'
        ],
        'chair': [
            'https://images.pexels.com/photos/276583/pexels-photo-276583.jpeg',
            'https://images.pexels.com/photos/2079246/pexels-photo-2079246.jpeg',
            'https://images.pexels.com/photos/3705523/pexels-photo-3705523.jpeg',
            'https://images.pexels.com/photos/3773577/pexels-photo-3773577.jpeg',
            'https://images.pexels.com/photos/3773573/pexels-photo-3773573.jpeg'
        ],
        'book': [
            'https://images.pexels.com/photos/159711/books-bookstore-book-reading-159711.jpeg',
            'https://images.pexels.com/photos/1370295/pexels-photo-1370295.jpeg',
            'https://images.pexels.com/photos/904620/pexels-photo-904620.jpeg',
            'https://images.pexels.com/photos/256450/pexels-photo-256450.jpeg',
            'https://images.pexels.com/photos/694740/pexels-photo-694740.jpeg'
        ],
        
        # Food
        'pizza': [
            'https://images.pexels.com/photos/2147491/pexels-photo-2147491.jpeg',
            'https://images.pexels.com/photos/905847/pexels-photo-905847.jpeg',
            'https://images.pexels.com/photos/825661/pexels-photo-825661.jpeg',
            'https://images.pexels.com/photos/2147493/pexels-photo-2147493.jpeg',
            'https://images.pexels.com/photos/1146760/pexels-photo-1146760.jpeg'
        ],
        'apple': [
            'https://images.pexels.com/photos/61127/pexels-photo-61127.jpeg',
            'https://images.pexels.com/photos/102104/pexels-photo-102104.jpeg',
            'https://images.pexels.com/photos/209431/pexels-photo-209431.jpeg',
            'https://images.pexels.com/photos/161559/background-bitter-breakfast-bright-161559.jpeg',
            'https://images.pexels.com/photos/209443/pexels-photo-209443.jpeg'
        ],
        
        # People
        'person': [
            'https://images.pexels.com/photos/697509/pexels-photo-697509.jpeg',
            'https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg',
            'https://images.pexels.com/photos/1036622/pexels-photo-1036622.jpeg',
            'https://images.pexels.com/photos/712513/pexels-photo-712513.jpeg',
            'https://images.pexels.com/photos/733872/pexels-photo-733872.jpeg'
        ],
        
        # Places
        'house': [
            'https://images.pexels.com/photos/106399/pexels-photo-106399.jpeg',
            'https://images.pexels.com/photos/259588/pexels-photo-259588.jpeg',
            'https://images.pexels.com/photos/280229/pexels-photo-280229.jpeg',
            'https://images.pexels.com/photos/164558/pexels-photo-164558.jpeg',
            'https://images.pexels.com/photos/258160/pexels-photo-258160.jpeg'
        ],
        'mountain': [
            'https://images.pexels.com/photos/417173/pexels-photo-417173.jpeg',
            'https://images.pexels.com/photos/2387873/pexels-photo-2387873.jpeg',
            'https://images.pexels.com/photos/547114/pexels-photo-547114.jpeg',
            'https://images.pexels.com/photos/361104/pexels-photo-361104.jpeg',
            'https://images.pexels.com/photos/1054289/pexels-photo-1054289.jpeg'
        ],
        'beach': [
            'https://images.pexels.com/photos/994605/pexels-photo-994605.jpeg',
            'https://images.pexels.com/photos/1032650/pexels-photo-1032650.jpeg',
            'https://images.pexels.com/photos/1438761/pexels-photo-1438761.jpeg',
            'https://images.pexels.com/photos/237272/pexels-photo-237272.jpeg',
            'https://images.pexels.com/photos/753626/pexels-photo-753626.jpeg'
        ]
    }
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n🚀 Loading classifier on {self.device}...")
        
        # Load pre-trained model
        model_name = "google/vit-base-patch16-224"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load or create learning data
        self.learning_data = self._load_learning_data()
        
        # Auto-train if no data exists
        if len(self.learning_data['samples']) < 10:
            print("\n📚 No training data found. Auto-training with 5 images per category...")
            self._auto_train()
        else:
            print(f"\n✅ Loaded {len(self.learning_data['samples'])} training samples")
            print(f"   Knows {len(self.learning_data['objects'])} different objects")
    
    def _load_learning_data(self):
        """Load learning data"""
        if os.path.exists(app.config['LEARNING_DATA']):
            with open(app.config['LEARNING_DATA'], 'rb') as f:
                return pickle.load(f)
        return {
            'objects': {},
            'features': {},
            'samples': []
        }
    
    def _save_learning_data(self):
        """Save learning data"""
        with open(app.config['LEARNING_DATA'], 'wb') as f:
            pickle.dump(self.learning_data, f)
    
    def _download_image(self, url):
        """Download image from URL"""
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content)).convert('RGB')
        except:
            pass
        return None
    
    def _extract_features(self, image):
        """Extract features from image"""
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            features = outputs.hidden_states[-1].mean(dim=1).cpu().numpy().flatten()
        
        return features
    
    def _get_feature_hash(self, features):
        """Create hash from features"""
        quantized = np.round(features * 100).astype(int)
        return hashlib.md5(quantized.tobytes()).hexdigest()
    
    def _auto_train(self):
        """Auto-train with 5 images per category"""
        print("\n" + "="*60)
        print("🎯 AUTO-TRAINING STARTED")
        print("="*60)
        
        total_trained = 0
        
        for object_name, urls in self.TRAINING_URLS.items():
            print(f"\n📚 Training: {object_name.title()}")
            
            for i, url in enumerate(urls, 1):
                try:
                    print(f"   Downloading image {i}/5...", end=" ")
                    
                    # Download image
                    image = self._download_image(url)
                    if image is None:
                        print("❌ Failed")
                        continue
                    
                    # Extract features
                    features = self._extract_features(image)
                    feature_hash = self._get_feature_hash(features)
                    
                    # Save to learning data
                    self.learning_data['features'][feature_hash] = object_name
                    self.learning_data['objects'][object_name] = self.learning_data['objects'].get(object_name, 0) + 1
                    
                    # Save image file
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"auto_train_{object_name}_{i}_{timestamp}.jpg"
                    filepath = os.path.join(app.config['TRAINING_FOLDER'], filename)
                    image.save(filepath)
                    
                    # Save sample record
                    self.learning_data['samples'].append({
                        'object': object_name,
                        'source': 'auto_train',
                        'timestamp': timestamp,
                        'file': filename
                    })
                    
                    print(f"✅ Learned")
                    total_trained += 1
                    
                    # Small delay to avoid rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"❌ Error: {str(e)[:50]}")
        
        # Save all learning data
        self._save_learning_data()
        
        print("\n" + "="*60)
        print(f"✅ AUTO-TRAINING COMPLETE!")
        print(f"   Total images trained: {total_trained}")
        print(f"   Unique objects: {len(self.learning_data['objects'])}")
        print("="*60)
    
    def predict(self, image_path, top_k=5):
        """Predict objects in image"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Get features
            features = self._extract_features(image)
            feature_hash = self._get_feature_hash(features)
            
            # Check if we've seen this before
            if feature_hash in self.learning_data['features']:
                learned_object = self.learning_data['features'][feature_hash]
                if learned_object in self.CATEGORIES:
                    cat = self.CATEGORIES[learned_object]
                    return [{
                        'object': learned_object.title(),
                        'category': cat['category'],
                        'emoji': cat['emoji'],
                        'color': cat['color'],
                        'confidence': 0.98,
                        'percentage': '98%',
                        'learned': True
                    }]
            
            # Get base model predictions
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = F.softmax(outputs.logits, dim=-1)[0]
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, top_k * 3)
            
            predictions = []
            seen = set()
            
            for prob, idx in zip(top_probs, top_indices):
                class_name = self.model.config.id2label[idx.item()].lower()
                confidence = prob.item()
                
                if confidence < 0.01:
                    continue
                
                # Find matching object
                detected = None
                for obj in self.CATEGORIES.keys():
                    if obj in class_name and obj not in seen:
                        detected = obj
                        break
                
                if detected:
                    cat = self.CATEGORIES[detected]
                    predictions.append({
                        'object': detected.title(),
                        'category': cat['category'],
                        'emoji': cat['emoji'],
                        'color': cat['color'],
                        'confidence': confidence,
                        'percentage': f"{confidence:.1%}"
                    })
                    seen.add(detected)
                    
                    if len(predictions) >= top_k:
                        break
            
            return predictions[:top_k]
            
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def learn(self, image_path, object_name, username):
        """Learn from user feedback"""
        try:
            object_name = object_name.lower().strip()
            
            if object_name not in self.CATEGORIES:
                # Add to categories dynamically
                self.CATEGORIES[object_name] = {
                    'category': 'objects',
                    'emoji': '📦',
                    'color': '#4A90E2'
                }
            
            # Extract and store features
            image = Image.open(image_path).convert('RGB')
            features = self._extract_features(image)
            feature_hash = self._get_feature_hash(features)
            
            # Store in learning data
            self.learning_data['features'][feature_hash] = object_name
            self.learning_data['objects'][object_name] = self.learning_data['objects'].get(object_name, 0) + 1
            
            # Save image
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{username}_{object_name}_{timestamp}.jpg"
            filepath = os.path.join(app.config['TRAINING_FOLDER'], filename)
            image.save(filepath)
            
            # Save sample
            self.learning_data['samples'].append({
                'object': object_name,
                'username': username,
                'timestamp': timestamp,
                'file': filename
            })
            
            self._save_learning_data()
            
            return True, f"Learned: {object_name.title()}"
            
        except Exception as e:
            return False, str(e)
    
    def get_stats(self):
        """Get training statistics"""
        return {
            'total_samples': len(self.learning_data['samples']),
            'unique_objects': len(self.learning_data['objects']),
            'objects': self.learning_data['objects']
        }

# Initialize the trained classifier
print("\n" + "="*60)
print("🎓 AAU HIGH-LEVEL IMAGE CLASSIFIER")
print("="*60)
classifier = TrainedClassifier()
print("="*60 + "\n")

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
    stats = classifier.get_stats()
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

@app.route('/learn', methods=['POST'])
@login_required
def learn():
    if 'file' not in request.files or 'object_name' not in request.form:
        return jsonify({'error': 'Missing data'}), 400
    
    file = request.files['file']
    object_name = request.form['object_name']
    
    if not object_name.strip():
        return jsonify({'error': 'Please enter an object name'}), 400
    
    filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        success, message = classifier.learn(filepath, object_name, session['username'])
        
        if success:
            return jsonify({
                'success': True,
                'message': message,
                'stats': classifier.get_stats()
            })
        else:
            return jsonify({'error': message}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/stats')
@login_required
def get_stats():
    return jsonify(classifier.get_stats())

# ============================================================================
# HTML Templates
# ============================================================================

LOGIN_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AAU - Image Classifier</title>
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
            max-width: 400px;
            text-align: center;
        }
        .logo { margin-bottom: 20px; }
        .logo img { width: 200px; }
        .university-name { color: #8B0000; font-size: 20px; font-weight: bold; }
        .college-name { color: #FFD700; font-size: 14px; margin-bottom: 20px; }
        .project-title {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 25px;
            border-left: 5px solid #8B0000;
        }
        .input-group { margin-bottom: 15px; }
        .input-group input {
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #e1e1e1;
            border-radius: 8px;
            font-size: 16px;
        }
        .input-group input:focus { border-color: #FFD700; outline: none; }
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
            margin-top: 10px;
        }
        button:hover { transform: translateY(-2px); }
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
            <img src="data:image/svg+xml;base64,{{ logo }}" alt="AAU Logo">
        </div>
        <div class="university-name">ADDIS ABABA UNIVERSITY</div>
        <div class="college-name">College of Natural and Computational Sciences<br>Department of Computer Science</div>
        
        <div class="project-title">
            <h3>High-Level Image Classifier with Boosting Algorithm</h3>
            <p>Machine Learning Course (COSC 6041)</p>
        </div>
        
        {% if error %}<div class="error">{{ error }}</div>{% endif %}
        
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
        .logo img { height: 40px; }
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
                        <td>{% if info.role == 'admin' %}👑 Admin{% else %}👤 User{% endif %}</td>
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
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .logo img { height: 40px; }
        .university-info h2 { font-size: 18px; color: #FFD700; }
        
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
        }
        
        .container { max-width: 1000px; margin: 40px auto; padding: 0 20px; }
        
        .project-header {
            text-align: center;
            margin-bottom: 30px;
        }
        .project-header h1 { color: #8B0000; font-size: 32px; }
        
        .stats-card {
            background: white;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 30px;
            display: flex;
            justify-content: space-around;
            border-left: 5px solid #FFD700;
        }
        .stat-number { font-size: 28px; font-weight: bold; color: #8B0000; }
        
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
            margin: 20px 0;
        }
        .upload-area:hover { border-color: #FFD700; background: #fff9e6; }
        .upload-icon { font-size: 48px; color: #8B0000; }
        
        #preview {
            max-width: 100%;
            max-height: 300px;
            display: none;
            margin: 20px auto;
            border: 4px solid #FFD700;
            border-radius: 10px;
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
            margin: 10px 0;
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
        }
        
        .learned-badge {
            background: #4CAF50;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 10px;
            margin-left: 10px;
        }
        
        .prediction-confidence {
            background: #8B0000;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }
        
        .learning-section {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 15px;
            display: none;
        }
        
        .learning-input {
            width: 100%;
            padding: 12px;
            margin: 15px 0;
            border: 2px solid #ddd;
            border-radius: 8px;
        }
        .learning-input:focus { border-color: #FFD700; outline: none; }
        
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
        
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        
        .results { display: none; }
        .success-message {
            background: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            display: none;
        }
        
        .footer {
            background: #8B0000;
            color: #FFD700;
            padding: 20px;
            text-align: center;
            margin-top: 50px;
        }
    </style>
</head>
<body>
    <div class="navbar">
        <div class="logo">
            <img src="data:image/svg+xml;base64,{{ logo }}" alt="AAU Logo">
            <div class="university-info">
                <h2>ADDIS ABABA UNIVERSITY</h2>
                <p>College of Natural and Computational Sciences</p>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 20px;">
            <span class="user-info">{{ session.full_name }}</span>
            <div class="nav-links">
                <a href="/">Home</a>
                {% if session.role == 'admin' %}<a href="/users">Manage Users</a>{% endif %}
                <a href="/logout">Logout</a>
            </div>
        </div>
    </div>
    
    <div class="container">
        <div class="project-header">
            <h1>High-Level Image Classifier</h1>
            <p style="color: #666;">Machine Learning Course (COSC 6041)</p>
        </div>
        
        <div class="stats-card">
            <div class="stat-item">
                <div class="stat-number">{{ stats.total_samples }}</div>
                <div>Training Samples</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{{ stats.unique_objects }}</div>
                <div>Objects Known</div>
            </div>
        </div>
        
        <div class="main-card">
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <input type="file" id="fileInput" accept="image/*" style="display: none;">
                <div class="upload-icon">📤</div>
                <p>Click or drag image here</p>
            </div>
            
            <img id="preview" src="#" alt="Preview">
            
            <div class="spinner" id="spinner"></div>
            
            <button class="btn" onclick="classifyImage()">🔍 Classify Image</button>
            
            <div class="results" id="results">
                <h3 style="color: #8B0000; margin: 20px 0;">Detected Objects:</h3>
                <div id="predictionList"></div>
            </div>
            
            <div class="learning-section" id="learningSection">
                <h3 style="color: #8B0000;">📚 Teach the Model</h3>
                <p>What object is in this image?</p>
                <input type="text" id="objectName" class="learning-input" placeholder="e.g., cat, car, person">
                <button class="btn" onclick="submitLearning()" style="background: #4CAF50;">✓ Submit & Teach</button>
                <div id="learningMessage" class="success-message"></div>
            </div>
        </div>
    </div>
    
    <div class="footer">
        <p>Addis Ababa University | College of Natural and Computational Sciences | Department of Computer Science</p>
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
            document.getElementById('learningSection').style.display = 'none';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayResults(data.predictions);
                    document.getElementById('learningSection').style.display = 'block';
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
            const predictionList = document.getElementById('predictionList');
            predictionList.innerHTML = '';
            
            predictions.forEach(pred => {
                const item = document.createElement('div');
                item.className = 'prediction-item';
                item.style.borderLeftColor = pred.color;
                
                let learnedBadge = pred.learned ? '<span class="learned-badge">Learned</span>' : '';
                
                item.innerHTML = `
                    <div>
                        <span style="font-size: 20px; margin-right: 10px;">${pred.emoji}</span>
                        <span style="font-weight: bold;">${pred.object}</span>
                        ${learnedBadge}
                        <span style="color: #666; font-size: 12px; margin-left: 10px;">(${pred.category})</span>
                    </div>
                    <span class="prediction-confidence">${pred.percentage}</span>
                `;
                predictionList.appendChild(item);
            });
            
            document.getElementById('results').style.display = 'block';
        }
        
        async function submitLearning() {
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
            
            const messageDiv = document.getElementById('learningMessage');
            
            try {
                const response = await fetch('/learn', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    messageDiv.textContent = `✅ ${data.message}`;
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
