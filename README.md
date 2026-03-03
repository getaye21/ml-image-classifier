---
title: ML Image Classifier
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# ML Image Classifier

A hybrid image classifier combining:
- **Deep Learning**: Hugging Face ViT for feature extraction
- **Boosting**: XGBoost for classification
- **Supervised Learning**: Trained on labeled images

## Live Demo
[https://huggingface.co/spaces/Getaye/ml-image-classifier](https://huggingface.co/spaces/Getaye/ml-image-classifier)

## Features
- Upload images for real-time classification
- Train custom models with your own labeled images
- No local setup required - runs in browser!

## How to Use
1. **Train**: Add images with labels and click "Start Training"
2. **Classify**: Upload any image to get predictions

## Technical Details
- **CNN**: Vision Transformer (ViT) from Hugging Face
- **Boosting**: XGBoost with 200 estimators
- **Framework**: Flask + PyTorch
