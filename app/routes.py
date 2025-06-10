"""
Маршрути та вся логіка обробки зображень.
Взято з вашого `local_server.py`, але організовано у функцію `register_routes`.
"""
import io, os, time
from flask import request, jsonify, send_file
from PIL import Image, ImageFilter
import numpy as np
from transparent_background import Remover

# --- Налаштування ---
MAX_IMAGE_SIZE = 512
JPEG_QUALITY = 85
DEFAULT_ALPHA_PROCESSING = 'blur'
DEFAULT_THRESHOLD = 0.7

model = None   # глобальна змінна-кеш

def resize_image(image, max_size=MAX_IMAGE_SIZE):
    width, height = image.size
    if width <= max_size and height <= max_size:
        return image
    if width > height:
        new_width = max_size
        new_height = int((height * max_size) / width)
    else:
        new_height = max_size
        new_width = int((width * max_size) / height)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def post_process_alpha(image, processing_type=DEFAULT_ALPHA_PROCESSING, threshold=DEFAULT_THRESHOLD):
    # ... (ваш код без змін) ...
    try:
        img_array = np.array(image)
        if len(img_array.shape) != 3 or img_array.shape[2] != 4:
            return image
        alpha = img_array[:, :, 3].astype(float) / 255.0
        if processing_type == 'hard':
            alpha = (alpha > threshold).astype(float)
        elif processing_type == 'medium':
            alpha_threshold = 0.3
            alpha = np.where(alpha > alpha_threshold,
                             np.minimum(1.0, alpha * 1.5),
                             0.0)
        elif processing_type == 'blur':
            alpha_pil = Image.fromarray((alpha * 255).astype(np.uint8), mode='L')
            alpha_pil = alpha_pil.filter(ImageFilter.GaussianBlur(radius=1))
            alpha_pil = alpha_pil.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
            alpha = np.array(alpha_pil).astype(float) / 255.0
        img_array[:, :, 3] = (alpha * 255).astype(np.uint8)
        return Image.fromarray(img_array, 'RGBA')
    except Exception:
        return image

def initialize_model():
    global model
    if model is not None:
        return model
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'ckpt_base.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found")
    model = Remover(ckpt=model_path, device='cpu')
    return model
# ---------------------------------------------------

def register_routes(app):
    @app.route('/remove-background', methods=['POST', 'OPTIONS'])
    def remove_background():
        if request.method == 'OPTIONS':
            return '', 200
        start = time.time()

        alpha_processing = request.form.get('alpha_processing', DEFAULT_ALPHA_PROCESSING)
        threshold = float(request.form.get('threshold', DEFAULT_THRESHOLD))
        if 'image' not in request.files or request.files['image'].filename == '':
            return jsonify({'error': 'No image file provided'}), 400
        img = Image.open(request.files['image'].stream).convert('RGB')

        img_resized = resize_image(img)
        remover = initialize_model()
        result = remover.process(img_resized, type='rgba')
        result = post_process_alpha(result, alpha_processing, threshold)
        if img_resized.size != img.size:
            result = result.resize(img.size, Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        result.save(buf, format='PNG', optimize=True)
        buf.seek(0)
        return send_file(buf, mimetype='image/png',
                         as_attachment=True,
                         download_name='background-removed.png')

    @app.route('/health')
    def health():
        return jsonify({'status': 'healthy',
                        'model_loaded': model is not None,
                        'max_size': MAX_IMAGE_SIZE})