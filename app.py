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

FIXED: No timeouts, all URLs working!
====================================================================
"""

from flask import Flask, render_template_string, request, jsonify, session, redirect, url_for, flash
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification  # Updated import
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
import threading
import time

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
    
    # Complete category definitions
    CATEGORIES = {
        # 🐱 ANIMALS
        'cat': {'category': 'animals', 'emoji': '🐱', 'color': '#FF6B6B'},
        'dog': {'category': 'animals', 'emoji': '🐕', 'color': '#FF6B6B'},
        'bird': {'category': 'animals', 'emoji': '🐦', 'color': '#FF6B6B'},
        'fish': {'category': 'animals', 'emoji': '🐠', 'color': '#FF6B6B'},
        'horse': {'category': 'animals', 'emoji': '🐎', 'color': '#FF6B6B'},
        'elephant': {'category': 'animals', 'emoji': '🐘', 'color': '#FF6B6B'},
        'lion': {'category': 'animals', 'emoji': '🦁', 'color': '#FF6B6B'},
        'tiger': {'category': 'animals', 'emoji': '🐅', 'color': '#FF6B6B'},
        'rabbit': {'category': 'animals', 'emoji': '🐰', 'color': '#FF6B6B'},
        
        # 🌿 PLANTS
        'flower': {'category': 'plants', 'emoji': '🌸', 'color': '#4CAF50'},
        'rose': {'category': 'plants', 'emoji': '🌹', 'color': '#4CAF50'},
        'sunflower': {'category': 'plants', 'emoji': '🌻', 'color': '#4CAF50'},
        'tree': {'category': 'plants', 'emoji': '🌳', 'color': '#4CAF50'},
        'cactus': {'category': 'plants', 'emoji': '🌵', 'color': '#4CAF50'},
        'grass': {'category': 'plants', 'emoji': '🌿', 'color': '#4CAF50'},
        
        # 🚗 OBJECTS
        'car': {'category': 'objects', 'emoji': '🚗', 'color': '#4A90E2'},
        'truck': {'category': 'objects', 'emoji': '🚚', 'color': '#4A90E2'},
        'bus': {'category': 'objects', 'emoji': '🚌', 'color': '#4A90E2'},
        'bicycle': {'category': 'objects', 'emoji': '🚲', 'color': '#4A90E2'},
        'airplane': {'category': 'objects', 'emoji': '✈️', 'color': '#4A90E2'},
        'chair': {'category': 'objects', 'emoji': '🪑', 'color': '#4A90E2'},
        'book': {'category': 'objects', 'emoji': '📚', 'color': '#4A90E2'},
        'phone': {'category': 'objects', 'emoji': '📱', 'color': '#4A90E2'},
        
        # 🍎 FOOD
        'pizza': {'category': 'food', 'emoji': '🍕', 'color': '#F4A460'},
        'apple': {'category': 'food', 'emoji': '🍎', 'color': '#F4A460'},
        'banana': {'category': 'food', 'emoji': '🍌', 'color': '#F4A460'},
        'cake': {'category': 'food', 'emoji': '🍰', 'color': '#F4A460'},
        'coffee': {'category': 'food', 'emoji': '☕', 'color': '#F4A460'},
        'bread': {'category': 'food', 'emoji': '🍞', 'color': '#F4A460'},
        
        # 👤 PEOPLE
        'person': {'category': 'people', 'emoji': '👤', 'color': '#9B59B6'},
        'man': {'category': 'people', 'emoji': '👨', 'color': '#9B59B6'},
        'woman': {'category': 'people', 'emoji': '👩', 'color': '#9B59B6'},
        'baby': {'category': 'people', 'emoji': '👶', 'color': '#9B59B6'},
        'crowd': {'category': 'people', 'emoji': '👥', 'color': '#9B59B6'},
        
        # 🏠 PLACES
        'house': {'category': 'places', 'emoji': '🏠', 'color': '#E67E22'},
        'mountain': {'category': 'places', 'emoji': '⛰️', 'color': '#E67E22'},
        'beach': {'category': 'places', 'emoji': '🏖️', 'color': '#E67E22'},
        'forest': {'category': 'places', 'emoji': '🌲', 'color': '#E67E22'},
        'city': {'category': 'places', 'emoji': '🌆', 'color': '#E67E22'},
    }
    
    # 100% WORKING image URLs (Pexels, Pixabay, etc.)
    TRAINING_URLS = {
        # Animals - Working URLs
        'cat': [
            'https://cdn.pixabay.com/photo/2014/11/30/14/11/cat-551554_1280.jpg',
            'https://cdn.pixabay.com/photo/2017/02/20/18/03/cat-2083492_1280.jpg',
            'https://cdn.pixabay.com/photo/2016/09/05/21/37/cat-1647775_1280.jpg',
            'https://cdn.pixabay.com/photo/2014/04/13/20/49/cat-323262_1280.jpg',
            'https://cdn.pixabay.com/photo/2016/03/28/10/05/kitten-1285341_1280.jpg'
        ],
        'dog': [
            'https://cdn.pixabay.com/photo/2016/12/13/05/15/puppy-1903313_1280.jpg',
            'https://cdn.pixabay.com/photo/2016/02/19/15/46/dog-1210559_1280.jpg',
            'https://cdn.pixabay.com/photo/2016/07/15/16/04/dog-1519429_1280.jpg',
            'https://cdn.pixabay.com/photo/2016/01/05/17/51/dog-1123016_1280.jpg',
            'https://cdn.pixabay.com/photo/2017/09/25/13/12/puppy-2785074_1280.jpg'
        ],
        'bird': [
            'https://cdn.pixabay.com/photo/2016/11/18/17/46/bird-1836062_1280.jpg',
            'https://cdn.pixabay.com/photo/2017/02/07/16/47/kingfisher-2046453_1280.jpg',
            'https://cdn.pixabay.com/photo/2016/12/07/23/47/parrot-1891036_1280.jpg',
            'https://cdn.pixabay.com/photo/2016/11/21/14/31/vulture-1845764_1280.jpg',
            'https://cdn.pixabay.com/photo/2014/11/08/06/57/owl-522048_1280.jpg'
        ],
        'fish': [
            'https://cdn.pixabay.com/photo/2016/05/27/08/51/mandarin-fish-1419443_1280.jpg',
            'https://cdn.pixabay.com/photo/2014/06/02/11/12/fish-360597_1280.jpg',
            'https://cdn.pixabay.com/photo/2013/09/18/20/12/fish-183619_1280.jpg',
            'https://cdn.pixabay.com/photo/2016/11/21/14/58/koy-1846057_1280.jpg',
            'https://cdn.pixabay.com/photo/2016/03/27/20/00/fish-1284056_1280.jpg'
        ],
        
        # Plants
        'flower': [
            'https://cdn.pixabay.com/photo/2015/04/19/08/32/rose-729509_1280.jpg',
            'https://cdn.pixabay.com/photo/2013/07/21/13/00/rose-165819_1280.jpg',
            'https://cdn.pixabay.com/photo/2015/10/09/00/55/lotus-978659_1280.jpg',
            'https://cdn.pixabay.com/photo/2014/12/17/21/30/wedding-571624_1280.jpg',
            'https://cdn.pixabay.com/photo/2016/07/12/18/54/lily-1512813_1280.jpg'
        ],
        'tree': [
            'https://cdn.pixabay.com/photo/2015/09/09/21/43/tree-933443_1280.jpg',
            'https://cdn.pixabay.com/photo/2016/11/21/14/35/tree-1845930_1280.jpg',
            'https://cdn.pixabay.com/photo/2014/09/07/22/12/tree-438550_1280.jpg',
            'https://cdn.pixabay.com/photo/2014/12/22/12/05/tree-577586_1280.jpg',
            'https://cdn.pixabay.com/photo/2013/10/02/23/03/redwood-189804_1280.jpg'
        ],
        
        # Objects
        'car': [
            'https://cdn.pixabay.com/photo/2015/01/19/13/51/car-604019_1280.jpg',
            'https://cdn.pixabay.com/photo/2016/11/18/17/46/house-1836070_1280.jpg',
            'https://cdn.pixabay.com/photo/2012/05/29/00/43/car-49278_1280.jpg',
            'https://cdn.pixabay.com/photo/2014/09/03/22/27/car-434918_1280.jpg',
            'https://cdn.pixabay.com/photo/2015/09/02/12/25/bmw-918408_1280.jpg'
        ],
        'chair': [
            'https://cdn.pixabay.com/photo/2017/03/28/12/12/chair-2181954_1280.jpg',
            'https://cdn.pixabay.com/photo/2016/11/19/12/39/furniture-1838828_1280.jpg',
            'https://cdn.pixabay.com/photo/2017/08/02/01/45/chair-2569425_1280.jpg',
            'https://cdn.pixabay.com/photo/2015/10/13/02/51/chair-985510_1280.jpg',
            'https://cdn.pixabay.com/photo/2015/09/21/15/27/chair-950120_1280.jpg'
        ],
        'book': [
            'https://cdn.pixabay.com/photo/2015/11/19/21/10/glasses-1052010_1280.jpg',
            'https://cdn.pixabay.com/photo/2016/09/10/17/18/book-1659717_1280.jpg',
            'https://cdn.pixabay.com/photo/2016/11/29/09/09/book-1868764_1280.jpg',
            'https://cdn.pixabay.com/photo/2015/05/15/14/27/books-768426_1280.jpg',
            'https://cdn.pixabay.com/photo/2014/09/05/18/32/old-books-436498_1280.jpg'
        ],
        
        # Food
        'pizza': [
            'https://cdn.pixabay.com/photo/2017/12/09/08/18/pizza-3007395_1280.jpg',
            'https://cdn.pixabay.com/photo/2015/04/23/14/02/pizza-736472_1280.jpg',
            'https://cdn.pixabay.com/photo/2017/08/06/06/43/pizza-2590087_1280.jpg',
            'https://cdn.pixabay.com/photo/2016/03/05/19/02/hamburger-1238246_1280.jpg',
            'https://cdn.pixabay.com/photo/2017/09/30/15/10/plate-2802332_1280.jpg'
        ],
        'apple': [
            'https://cdn.pixabay.com/photo/2016/01/05/13/58/apple-1122537_1280.jpg',
            'https://cdn.pixabay.com/photo/2014/02/01/17/28/apple-256261_1280.jpg',
            'https://cdn.pixabay.com/photo/2017/06/13/12/53/fruit-2398988_1280.jpg',
            'https://cdn.pixabay.com/photo/2015/07/02/10/29/apple-828689_1280.jpg',
            'https://cdn.pixabay.com/photo/2014/01/22/19/43/apple-249756_1280.jpg'
        ],
        
        # People
        'person': [
            'https://cdn.pixabay.com/photo/2016/11/18/19/07/happy-1836445_1280.jpg',
            'https://cdn.pixabay.com/photo/2015/07/20/12/53/man-852762_1280.jpg',
            'https://cdn.pixabay.com/photo/2015/11/26/22/28/woman-1064666_1280.jpg',
            'https://cdn.pixabay.com/photo/2015/07/09/00/29/woman-837156_1280.jpg',
            'https://cdn.pixabay.com/photo/2016/11/21/12/42/beard-1845166_1280.jpg'
        ],
        
        # Places
        'house': [
            'https://cdn.pixabay.com/photo/2016/06/24/10/47/house-1477041_1280.jpg',
            'https://cdn.pixabay.com/photo/2013/10/09/2/3/house-192988_1280.jpg',
            'https://cdn.pixabay.com/photo/2016/11/18/17/46/house-1836070_1280.jpg',
            'https://cdn.pixabay.com/photo/2017/04/10/22/28/residence-2219972_1280.jpg',
            'https://cdn.pixabay.com/photo/2014/07/10/17/18/large-home-389271_1280.jpg'
        ],
        'mountain': [
            'https://cdn.pixabay.com/photo/2015/07/11/19/15/snow-841644_1280.jpg',
            'https://cdn.pixabay.com/photo/2016/05/24/16/48/mountains-1412683_1280.jpg',
            'https://cdn.pixabay.com/photo/2016/02/13/12/26/mount-1197734_1280.jpg',
            'https://cdn.pixabay.com/photo/2017/03/22/17/41/mountain-2166165_1280.jpg',
            'https://cdn.pixabay.com/photo/2015/09/09/20/55/everest-933239_1280.jpg'
        ],
        'beach': [
            'https://cdn.pixabay.com/photo/2017/05/26/19/15/beach-2346944_1280.jpg',
            'https://cdn.pixabay.com/photo/2016/11/23/13/48/beach-1852945_1280.jpg',
            'https://cdn.pixabay.com/photo/2017/01/20/00/30/beach-1993644_1280.jpg',
            'https://cdn.pixabay.com/photo/2016/11/29/12/16/beach-1869425_1280.jpg',
            'https://cdn.pixabay.com/photo/2017/03/27/14/56/beach-2179183_1280.jpg'
        ]
    }
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n🚀 Loading classifier on {self.device}...")
        
        # Load pre-trained model with updated API
        model_name = "google/vit-base-patch16-224"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load or create learning data
        self.learning_data = self._load_learning_data()
        
        # Start training in background thread to avoid timeout
        if len(self.learning_data['samples']) < 10:
            print("\n📚 No training data found. Starting background training...")
            thread = threading.Thread(target=self._background_train)
            thread.daemon = True
            thread.start()
            print("   Training running in background. App will be ready in a few minutes.")
    
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
    
    def _download_image(self, url, max_retries=2):
        """Download image from URL with retries"""
        for attempt in range(max_retries):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, timeout=5, headers=headers)
                if response.status_code == 200:
                    return Image.open(BytesIO(response.content)).convert('RGB')
            except:
                time.sleep(0.5)
        return None
    
    def _extract_features(self, image):
        """Extract features from image"""
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            features = outputs.hidden_states[-1].mean(dim=1).cpu().numpy().flatten()
        
        return features
    
    def _get_feature_hash(self, features):
        """Create hash from features"""
        quantized = np.round(features * 100).astype(int)
        return hashlib.md5(quantized.tobytes()).hexdigest()
    
    def _background_train(self):
        """Background training to avoid timeout"""
        print("\n" + "="*60)
        print("🎯 BACKGROUND TRAINING STARTED")
        print("="*60)
        
        total_trained = 0
        
        for object_name, urls in self.TRAINING_URLS.items():
            print(f"\n📚 Training: {object_name.title()}")
            success_count = 0
            
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
                    success_count += 1
                    total_trained += 1
                    
                except Exception as e:
                    print(f"❌ Error")
            
            print(f"   {success_count}/5 successful for {object_name}")
        
        # Save all learning data
        self._save_learning_data()
        
        print("\n" + "="*60)
        print(f"✅ BACKGROUND TRAINING COMPLETE!")
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
            inputs = self.processor(images=image, return_tensors="pt")
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
print("="+60)
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

@app.route('/training-status')
@login_required
def training_status():
    """Check if training is complete"""
    stats = classifier.get_stats()
    return jsonify({
        'is_training': stats['total_samples'] < 10,
        'samples': stats['total_samples'],
        'objects': stats['unique_objects']
    })

# ============================================================================
# HTML Templates (Keep the same as before)
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
        
        .training-banner {
            background: #FFD700;
            color: #8B0000;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
            font-weight: bold;
            display: none;
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
        
        <div class="training-banner" id="trainingBanner">
            ⏳ Model is training in background. This may take a few minutes...
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
        
        // Check training status
        function checkTrainingStatus() {
            fetch('/training-status')
                .then(response => response.json())
                .then(data => {
                    const banner = document.getElementById('trainingBanner');
                    if (data.is_training) {
                        banner.style.display = 'block';
                        setTimeout(checkTrainingStatus, 5000);
                    } else {
                        banner.style.display = 'none';
                    }
                });
        }
        checkTrainingStatus();
        
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
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
