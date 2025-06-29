import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
import logging
from datetime import datetime
import shutil
import cv2
from explain import ExplainableAI  # Import your Grad-CAM explainer

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this for production

# Configuration
app.config.update({
    'UPLOAD_FOLDER': 'static/uploads',
    'RESULTS_FOLDER': 'static/results',
    'MODELS_FOLDER': 'models',
    'CLASSIFICATION_MODEL': 'classification_model.h5',
    'SEGMENTATION_MODEL': 'segmentation_model.h5',
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB
    'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg', 'bmp'},
    'CLASS_NAMES': ['Normal', 'Glioma', 'Meningioma', 'Pituitary'],
    'CLASSIFICATION_DIM': (168, 168),  # Must match classification model input size
    'SEGMENTATION_DIM': (160, 160),   # Must match segmentation model input size
    'N_CHANNELS': 1,                  # Grayscale
    'EXPLAIN_LAYER_NAME': 'block5_conv4',  # Last conv layer for Grad-CAM
    'EXPLAIN_IMG_SIZE': (168, 168)    # Size for explanation images
})

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('brain_tumor_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Model cache
classification_model = None
segmentation_model = None

def load_classification_model():
    """Load and cache the classification model"""
    global classification_model
    if classification_model is None:
        try:
            model_path = os.path.join(app.config['MODELS_FOLDER'], app.config['CLASSIFICATION_MODEL'])
            classification_model = tf.keras.models.load_model(model_path)
            logger.info("Classification model loaded successfully")
            
            if classification_model.input_shape[1:] != (*app.config['CLASSIFICATION_DIM'], app.config['N_CHANNELS']):
                logger.error(f"Model expects input shape {classification_model.input_shape[1:]}, but config is {app.config['CLASSIFICATION_DIM']} with {app.config['N_CHANNELS']} channel(s)")
                raise ValueError("Model input shape mismatch")
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    return classification_model

def load_segmentation_model():
    """Load and cache the segmentation model"""
    global segmentation_model
    if segmentation_model is None:
        try:
            model_path = os.path.join(app.config['MODELS_FOLDER'], app.config['SEGMENTATION_MODEL'])
            segmentation_model = tf.keras.models.load_model(model_path)
            logger.info("Segmentation model loaded successfully")
            
            if segmentation_model.input_shape[1:] != (*app.config['SEGMENTATION_DIM'], app.config['N_CHANNELS']):
                logger.error(f"Segmentation model expects input shape {segmentation_model.input_shape[1:]}, but config is {app.config['SEGMENTATION_DIM']} with {app.config['N_CHANNELS']} channel(s)")
                raise ValueError("Segmentation model input shape mismatch")
                
        except Exception as e:
            logger.error(f"Error loading segmentation model: {str(e)}")
            raise
    return segmentation_model

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_classification_image(image_path):
    """Preprocess image for classification model"""
    try:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize(app.config['CLASSIFICATION_DIM'])
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        return np.expand_dims(img_array, axis=0)        # Add batch dimension
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise

def preprocess_segmentation_image(image_path):
    """Preprocess image for segmentation model"""
    try:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize(app.config['SEGMENTATION_DIM'])
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        return np.expand_dims(img_array, axis=0)        # Add batch dimension
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise

def perform_segmentation(image_path):
    """Perform tumor segmentation on the image"""
    try:
        model = load_segmentation_model()
        img = Image.open(image_path).convert('L').resize(app.config['SEGMENTATION_DIM'])
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))
        
        predicted_mask = model.predict(img_array)[0, :, :, 0]
        binary_mask = (predicted_mask > 0.5).astype(np.uint8) * 255
        mask_image = Image.fromarray(binary_mask)
        
        original_filename = os.path.basename(image_path)
        mask_filename = f"mask_{original_filename}"
        mask_path = os.path.join(app.config['RESULTS_FOLDER'], mask_filename)
        mask_image.save(mask_path, format='PNG')
        
        return mask_filename
    except Exception as e:
        logger.error(f"Segmentation failed: {str(e)}")
        raise

def generate_explanation(image_path, model, class_id=None):
    """Generate Grad-CAM explanation for the prediction"""
    try:
        explainer = ExplainableAI(
            model=model,
            last_conv_layer_name=app.config['EXPLAIN_LAYER_NAME'],
            img_size=app.config['EXPLAIN_IMG_SIZE']
        )
        
        original_filename = os.path.basename(image_path)
        explanation_filename = f"explanation_{original_filename}"
        explanation_path = os.path.join(app.config['RESULTS_FOLDER'], explanation_filename)
        
        explainer.generate_explanation(
            img_path=image_path,
            class_id=class_id,
            output_path=explanation_path
        )
        
        return explanation_filename
    except Exception as e:
        logger.error(f"Explanation generation failed: {str(e)}")
        raise

def clear_old_files():
    """Remove files older than 1 hour from upload and results folders"""
    try:
        now = datetime.now()
        for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULTS_FOLDER']]:
            for filename in os.listdir(folder):
                filepath = os.path.join(folder, filename)
                file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                if (now - file_time).total_seconds() > 3600:  # 1 hour
                    try:
                        if os.path.isfile(filepath):
                            os.remove(filepath)
                            logger.info(f"Removed old file: {filepath}")
                    except Exception as e:
                        logger.error(f"Error removing file {filepath}: {str(e)}")
    except Exception as e:
        logger.error(f"Error clearing old files: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main route for file upload and processing"""
    clear_old_files()
    
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        file = request.files['image']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('index'))
            
        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload a PNG, JPG, or JPEG image.', 'error')
            return redirect(url_for('index'))

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_filename = secure_filename(file.filename)
            filename = f"{timestamp}_{original_filename}"
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            
            analysis_type = request.form.get('analysis_type', 'classification')
            results = {
                'uploaded_image': filename,
                'segmentation_image': None,
                'prediction': None,
                'explanation_image': None,
                'is_normal': False  # Add this flag
            }
            
            if analysis_type in ['classification', 'both']:
                model = load_classification_model()
                img_array = preprocess_classification_image(upload_path)
                predictions = model.predict(img_array)
                predicted_class = np.argmax(predictions[0])
                class_name = app.config['CLASS_NAMES'][predicted_class]
                
                results['prediction'] = {
                    'class_name': class_name,
                    'confidence': float(np.max(predictions[0])),
                    'probabilities': [
                        {'name': name, 'value': float(p), 'percentage': float(p)*100} 
                        for name, p in zip(app.config['CLASS_NAMES'], predictions[0])
                    ],
                    'class_id': predicted_class
                }
                
                # Check if normal
                if class_name == 'Normal':
                    results['is_normal'] = True
                    return render_template('results.html', **results)
                
                # Generate explanation only if not normal
                results['explanation_image'] = generate_explanation(
                    upload_path, 
                    model, 
                    predicted_class
                )
            
            # Skip segmentation if normal
            if analysis_type in ['segmentation', 'both'] and not results['is_normal']:
                results['segmentation_image'] = perform_segmentation(upload_path)
            
            return render_template('results.html', **results)

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            flash('Error processing image. Please try another.', 'error')
            return redirect(url_for('index'))

    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for AJAX requests"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
            
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        analysis_type = request.form.get('analysis_type', 'classification')
        include_explanation = request.form.get('include_explanation', 'false').lower() == 'true'
        results = {
            'is_normal': False
        }
        
        if analysis_type in ['classification', 'both']:
            model = load_classification_model()
            img_array = preprocess_classification_image(upload_path)
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            class_name = app.config['CLASS_NAMES'][predicted_class]
            
            results['classification'] = {
                'prediction': class_name,
                'confidence': float(np.max(predictions[0])),
                'probabilities': {
                    name: float(p) for name, p in zip(app.config['CLASS_NAMES'], predictions[0])
                }
            }
            
            # Check if normal
            if class_name == 'Normal':
                results['is_normal'] = True
                return jsonify(results)
            
            if include_explanation:
                explanation_filename = generate_explanation(upload_path, model, predicted_class)
                results['explanation'] = {
                    'image_url': url_for('static', filename=f"results/{explanation_filename}")
                }
        
        # Skip segmentation if normal
        if analysis_type in ['segmentation', 'both'] and not results['is_normal']:
            results['segmentation'] = {
                'mask_url': url_for('static', filename=f"results/{perform_segmentation(upload_path)}")
            }
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template('error.html', error="Internal server error"), 500

if __name__ == '__main__':
    try:
        load_classification_model()
        load_segmentation_model()
        
        if not os.path.exists('backups'):
            os.makedirs('backups')
        backup_dir = os.path.join('backups', datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(backup_dir)
        shutil.copytree(app.config['UPLOAD_FOLDER'], os.path.join(backup_dir, 'uploads'))
        shutil.copytree(app.config['RESULTS_FOLDER'], os.path.join(backup_dir, 'results'))
        
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}")