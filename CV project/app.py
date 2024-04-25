from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__, template_folder='templates')

def cartoonify_image(input_image):
    # Your existing cartoonify logic
    line_wdt = 9
    blur_value = 7
    totalColors = 15

    # Edge detection
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    grayblur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(grayblur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_wdt, blur_value)
    

    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges = cv2.resize(edges, (input_image.shape[1], input_image.shape[0]))

    # Color quantization
    data = np.float32(input_image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    ret, label, center = cv2.kmeans(data, totalColors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(input_image.shape)

    # Bilateral filtering
    smoothed_img = input_image
    for _ in range(5):
        smoothed_img = cv2.bilateralFilter(smoothed_img, 9, 75, 75)

    # Convert the images to 8-bit unsigned integer (CV_8U) data type
    result = cv2.convertScaleAbs(result)
    edges = cv2.convertScaleAbs(edges)

    # Ensure the mask (edges) is in grayscale and binary format
    edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
    _, edges = cv2.threshold(edges, 1, 255, cv2.THRESH_BINARY)

    # Perform bitwise AND operation
    cartoon_image = cv2.bitwise_and(smoothed_img, smoothed_img, mask=edges)

    return cartoon_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cartoonify', methods=['POST'])
def cartoonify():
    if 'image' in request.files:
        uploaded_image = request.files['image']
        image_np = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)

        # Perform cartoonify processing
        cartoon_image_np = cartoonify_image(image_np)

        # Convert cartoonified image to base64 for display in HTML
        _, cartoon_image_encoded = cv2.imencode('.jpg', cartoon_image_np)
        cartoon_image_base64 = base64.b64encode(cartoon_image_encoded).decode('utf-8')

        return jsonify({'image': cartoon_image_base64})
    return jsonify({'error': 'No image uploaded'})

if __name__ == '__main__':
    app.run(debug=True)
