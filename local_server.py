#!/usr/bin/env python3
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import io
import os
import time
from PIL import Image, ImageFilter
from transparent_background import Remover
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global model variable
model = None

# Optimization settings
MAX_IMAGE_SIZE = 512  # Maximum width or height (reduced for faster processing)
JPEG_QUALITY = 85

# Alpha processing settings
DEFAULT_ALPHA_PROCESSING = 'blur'  # Options: 'soft', 'medium', 'hard', 'blur'
DEFAULT_THRESHOLD = 0.7

def resize_image(image, max_size=MAX_IMAGE_SIZE):
    """Resize image while maintaining aspect ratio"""
    width, height = image.size
    
    if width <= max_size and height <= max_size:
        return image
    
    # Calculate new size maintaining aspect ratio
    if width > height:
        new_width = max_size
        new_height = int((height * max_size) / width)
    else:
        new_height = max_size
        new_width = int((width * max_size) / height)
    
    print(f"📏 Resizing from {width}x{height} to {new_width}x{new_height} (reduction: {(width*height)/(new_width*new_height):.1f}x)")
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def post_process_alpha(image, processing_type=DEFAULT_ALPHA_PROCESSING, threshold=DEFAULT_THRESHOLD):
    """Post-process alpha channel to reduce semi-transparent pixels"""
    process_start = time.time()
    
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        if len(img_array.shape) != 3 or img_array.shape[2] != 4:
            print("⚠️ Image has no alpha channel")
            return image
        
        alpha = img_array[:, :, 3].astype(float) / 255.0
        original_alpha_stats = f"min: {alpha.min():.2f}, max: {alpha.max():.2f}, mean: {alpha.mean():.2f}"
        
        if processing_type == 'hard':
            # Hard threshold: 0 or 255
            alpha = (alpha > threshold).astype(float)
            print(f"🔥 Applied hard threshold: {threshold}")
            
        elif processing_type == 'medium':
            # Medium threshold with smoothing
            alpha_threshold = 0.3
            alpha = np.where(alpha > alpha_threshold, 
                            np.minimum(1.0, alpha * 1.5), 
                            0.0)
            print(f"⚖️ Applied medium alpha processing")
            
        elif processing_type == 'blur':
            # Blur with subsequent sharpening
            alpha_pil = Image.fromarray((alpha * 255).astype(np.uint8), mode='L')
            alpha_pil = alpha_pil.filter(ImageFilter.GaussianBlur(radius=1))
            alpha_pil = alpha_pil.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
            alpha = np.array(alpha_pil).astype(float) / 255.0
            print(f"🌀 Applied blur + sharpening to alpha")
        
        # Apply processed alpha channel
        img_array[:, :, 3] = (alpha * 255).astype(np.uint8)
        processed_alpha_stats = f"min: {alpha.min():.2f}, max: {alpha.max():.2f}, mean: {alpha.mean():.2f}"
        
        process_time = time.time() - process_start
        print(f"🎨 Alpha processing ({processing_type}): {process_time:.2f}s")
        print(f"   Before: {original_alpha_stats}")
        print(f"   After:  {processed_alpha_stats}")
        
        return Image.fromarray(img_array, 'RGBA')
        
    except Exception as e:
        print(f"❌ Error in alpha processing: {str(e)}")
        return image

def initialize_model():
    global model
    if model is not None:
        return model
    
    try:
        model_start = time.time()
        model_path = 'model/ckpt_base.pth'
        if os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            model = Remover(ckpt=model_path, device='cpu')
            model_time = time.time() - model_start
            print(f"✅ Model loaded successfully in {model_time:.2f} seconds!")
        else:
            print(f"Model file not found at {model_path}")
            raise FileNotFoundError("Model file not found")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e
    
    return model

@app.route('/remove-background', methods=['POST', 'OPTIONS'])
def remove_background():
    if request.method == 'OPTIONS':
        return '', 200
    
    total_start = time.time()
    
    try:
        # Get parameters
        alpha_processing = request.form.get('alpha_processing', DEFAULT_ALPHA_PROCESSING)
        threshold = float(request.form.get('threshold', DEFAULT_THRESHOLD))
        
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read and process image
        load_start = time.time()
        img = Image.open(file.stream).convert('RGB')
        original_size = img.size
        original_pixels = original_size[0] * original_size[1]
        load_time = time.time() - load_start
        print(f"📸 Original: {original_size} ({original_pixels:,} pixels, loaded in {load_time:.2f}s)")
        
        # Resize image for faster processing
        resize_start = time.time()
        img_resized = resize_image(img)
        processed_pixels = img_resized.size[0] * img_resized.size[1]
        resize_time = time.time() - resize_start
        if img_resized.size != original_size:
            reduction_factor = original_pixels / processed_pixels
            print(f"⚡ Resized in {resize_time:.2f}s (processing {processed_pixels:,} pixels, {reduction_factor:.1f}x reduction)")
        
        # Initialize model
        print("🔄 Initializing model...")
        init_start = time.time()
        remover = initialize_model()
        init_time = time.time() - init_start
        if init_time > 0.1:
            print(f"🤖 Model ready in {init_time:.2f}s")
        
        # Detailed AI processing timing
        print(f"🚀 Starting AI processing on {img_resized.size} image...")
        ai_start = time.time()
        
        # Process resized image
        result = remover.process(img_resized, type='rgba')
        
        ai_time = time.time() - ai_start
        pixels_per_second = processed_pixels / ai_time if ai_time > 0 else 0
        print(f"🎯 AI processing: {ai_time:.2f}s ({pixels_per_second:,.0f} pixels/sec)")
        
        # Apply alpha processing
        result = post_process_alpha(result, alpha_processing, threshold)
        
        # Upscale if needed
        upscale_start = time.time()
        if img_resized.size != original_size:
            result = result.resize(original_size, Image.Resampling.LANCZOS)
            upscale_time = time.time() - upscale_start
            print(f"📏 Upscaled result in {upscale_time:.2f}s")
        
        # Save result
        save_start = time.time()
        img_io = io.BytesIO()
        result.save(img_io, format='PNG', optimize=True)
        img_io.seek(0)
        save_time = time.time() - save_start
        
        total_time = time.time() - total_start
        
        print(f"✅ TOTAL: {total_time:.2f}s (Load: {load_time:.2f}s + AI: {ai_time:.2f}s + Alpha: included + Save: {save_time:.2f}s)")
        print(f"📊 Performance: {original_pixels/total_time:,.0f} original pixels/sec")
        
        return send_file(
            img_io,
            mimetype='image/png',
            as_attachment=True,
            download_name='background-removed.png'
        )
        
    except Exception as e:
        total_time = time.time() - total_start
        print(f"❌ Error after {total_time:.2f}s: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'message': 'Background removal server is running',
        'max_size': MAX_IMAGE_SIZE,
        'device': 'cpu',
        'model_loaded': model is not None,
        'alpha_processing': DEFAULT_ALPHA_PROCESSING,
        'threshold': DEFAULT_THRESHOLD
    })

if __name__ == '__main__':
    print("🚀 Starting optimized background removal server...")
    print(f"📏 Max image size: {MAX_IMAGE_SIZE}px")
    print(f"🎨 Default alpha processing: {DEFAULT_ALPHA_PROCESSING}")
    print("🌐 Access at: http://localhost:5002")
    app.run(host='0.0.0.0', port=5002, debug=True) 