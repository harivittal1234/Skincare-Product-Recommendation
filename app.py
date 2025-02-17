import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from flask import Flask, render_template, Response, request, jsonify
import cv2

# Initialize Flask app and camera
app = Flask(__name__)
camera = None

# Load the CSV files
result_csv_path = "result.csv"
result2_csv_path = "result2.csv"

df = pd.read_csv(result_csv_path)
makeup = pd.read_csv(result2_csv_path)

# Preprocess Skincare Data
df['concern 2'].fillna('', inplace=True)
df['concern 3'].fillna('', inplace=True)
df['concern'] = df['concern'] + ',' + df['concern 2'] + ',' + df['concern 3']
df.drop(columns=['concern 2', 'concern 3', 'spf', 'key ingredient', 'formulation'], inplace=True)

df2 = df[df['label'].isin(['face-moisturisers', 'mask-and-peel', 'cleanser', 'eye-cream'])]
df2 = df2[df2['skin type'].notna()]
df2['concern'] = df2['concern'].str.lower()
df2['brand'] = df2['brand'].str.lower()
df2['name'] = df2['name'].str.lower()
df2['skin type'] = df2['skin type'].str.lower()
df2['concern'] = df2['concern'].str.replace(' and ', ',').str.replace(' or ', ',')

top_concerns = {
    'face-moisturisers': 'general care',
    'mask-and-peel': 'daily use',
    'cleanser': 'general care',
    'eye-cream': 'fine lines,wrinkles,dark circles,eye bags'
}
for i in range(len(df2)):
    if pd.isnull(df2.iloc[i]['concern']):
        label = df2.iloc[i]['label']
        df2.at[i, 'concern'] = top_concerns[label]

# Define features and encode
features = ['normal', 'dry', 'oily', 'combination', 'acne', 'sensitive',
            'fine lines', 'wrinkles', 'redness', 'dull', 'pore', 'pigmentation',
            'blackheads', 'whiteheads', 'blemishes', 'dark circles', 'eye bags', 'dark spots']

entries = len(df2)
one_hot_encodings = np.zeros((entries, len(features)))

for i in range(entries):
    skin_type = df2.iloc[i]['skin type']
    if skin_type == 'all':
        one_hot_encodings[i][:5] = 1
    elif skin_type in features:
        one_hot_encodings[i][features.index(skin_type)] = 1

def search_concern(target, i):
    concern_value = str(df2.iloc[i]['concern']) if pd.notna(df2.iloc[i]['concern']) else ''
    return target in concern_value

for i in range(entries):
    for j in range(5, len(features)):
        if search_concern(features[j], i):
            one_hot_encodings[i][j] = 1

# Nearest Neighbors
nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(one_hot_encodings)
distances, indices = nbrs.kneighbors(one_hot_encodings)

def name2index(name):
    return df2[df2['name'] == name].index[0]

def index2prod(index):
    return df2.iloc[index]

def wrap(info_arr):
    return {
        'brand': info_arr[0],
        'name': info_arr[1],
        'price': info_arr[2],
        'url': info_arr[3],
        'skin type': info_arr[4],
        'concern': str(info_arr[5]).split(',')
    }

def recs_cs(vector=None, name=None, label=None, count=5):
    products = []
    if name:
        idx = name2index(name)
        fv = one_hot_encodings[idx]
    elif vector is not None:
        fv = vector
    else:
        return []

    cs_values = cosine_similarity(np.array([fv]), one_hot_encodings)[0]
    df2['cs'] = cs_values

    dff = df2[df2['label'] == label] if label else df2
    if name:
        dff = dff[dff['name'] != name]
    recommendations = dff.sort_values('cs', ascending=False).head(count)
    data = recommendations[['brand', 'name', 'price', 'url', 'skin type', 'concern']].to_dict('split')['data']
    for element in data:
        products.append(wrap(element))
    return products

def recs_essentials(vector=None, name=None):
    response = {}
    for label in df2['label'].unique():
        response[label] = recs_cs(vector=vector, name=name, label=label)
    return response

# Initialize Camera
def initialize_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)

def release_camera():
    global camera
    if camera and camera.isOpened():
        camera.release()

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    initialize_camera()
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Detect faces in the frame
            faces = detect_face(frame)
            
            # Draw green boxes around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Resize the frame for display
            frame = cv2.resize(frame, (500, 500))
            
            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Stream the frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Face detection function
def detect_face(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Skin condition analysis functions (detect skin tone, acne, dark spots, etc.)
def detect_skin(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin = cv2.bitwise_and(image, image, mask=mask)
    return skin, mask

def calculate_skin_tone(skin):
    hsv = cv2.cvtColor(skin, cv2.COLOR_BGR2HSV)
    non_zero_pixels = hsv[np.where((hsv != [0, 0, 0]).all(axis=2))]
    average_tone = np.mean(non_zero_pixels, axis=0) if len(non_zero_pixels) > 0 else [0, 0, 0]
    return average_tone

def classify_skin_type(average_tone):
    if average_tone[2] > 180:  # High brightness
        return "Normal"
    elif average_tone[1] > 70:  # High saturation
        return "Oily"
    elif average_tone[1] < 30:  # Low saturation
        return "Dry"
    else:
        return "Combination"

def analyze_skin_type(skin, mask):
    gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
    laplacian = cv2.Laplacian(masked_gray, cv2.CV_64F)
    variance = np.var(laplacian)
    return "Sensitive" if variance < 1700 else "Non-Sensitive"

def detect_acne(skin, mask):
    gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    acne_mask = cv2.bitwise_and(thresholded, thresholded, mask=mask)
    contours, _ = cv2.findContours(acne_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    acne_count = len(contours)
    return acne_count

def detect_dark_spots(skin, mask):
    gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    dark_spots_mask = cv2.bitwise_and(thresholded, thresholded, mask=mask)
    contours, _ = cv2.findContours(dark_spots_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dark_spot_count = len(contours)
    return dark_spot_count

def detect_fine_lines(skin, mask):
    gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    fine_lines = cv2.bitwise_and(edges, edges, mask=mask)
    fine_line_count = np.count_nonzero(fine_lines)
    return fine_line_count

@app.route('/predict', methods=['POST'])
def predict():
    try:
        initialize_camera()
        # Capture an image
        success, frame = camera.read()
        if not success:
            return render_template('results.html', results={"error": "Failed to capture image."})
        
        # Detect face before proceeding
        faces = detect_face(frame)
        if len(faces) == 0:
            return render_template('results.html', results={"error": "No face detected, please try again."})

        # Draw green box around the face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Process the image and analyze skin conditions
        skin, mask = detect_skin(frame)
        average_tone = calculate_skin_tone(skin)
        skin_type = classify_skin_type(average_tone)
        sensitivity = analyze_skin_type(skin, mask)
        acne_count = detect_acne(skin, mask)
        dark_spots_count = detect_dark_spots(skin, mask)
        fine_lines_count = detect_fine_lines(skin, mask)

        # Create vector based on the analysis
        analysis_vector = [0] * len(features)
        if skin_type.lower() in features:
            analysis_vector[features.index(skin_type.lower())] = 1
        if acne_count > 0:
            analysis_vector[features.index('acne')] = 1
        if dark_spots_count > 0:
            analysis_vector[features.index('dark spots')] = 1
        if fine_lines_count > 0:
            analysis_vector[features.index('fine lines')] = 1
        if sensitivity.lower() in features:
            analysis_vector[features.index('sensitive')] = 1

        # Get recommendations based on the analysis vector
        recommendations = recs_essentials(vector=analysis_vector)

        # Save the processed frame (with the green box)
        processed_image_path = "static/processed_frame.jpg"
        cv2.imwrite(processed_image_path, frame)

        # Prepare results dictionary
        results = {
            "skin_type": skin_type,
            "sensitivity": sensitivity,
            "acne_count": acne_count,
            "dark_spots_count": dark_spots_count,
            "fine_lines_count": fine_lines_count,
            "recommendations": recommendations if recommendations else {"error": "No recommendations available."}
        }

        return render_template('results.html', results=results, image_path=processed_image_path)

    except Exception as e:
        return render_template('results.html', results={"error": str(e)})

    finally:
        release_camera()




if __name__ == '__main__':
    app.run(debug=True)
