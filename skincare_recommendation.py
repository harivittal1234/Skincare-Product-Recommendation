import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Load Recommendation Data
result_csv_path = "result.csv"
result2_csv_path = "result2.csv"

df = pd.read_csv(result_csv_path)
makeup = pd.read_csv(result2_csv_path)

# Preprocessing Skincare Data
df['concern 2'].fillna('', inplace=True)
df['concern 3'].fillna('', inplace=True)
df['concern'] = df['concern'] + ',' + df['concern 2'] + ',' + df['concern 3']
df.drop(columns=['concern 2', 'concern 3', 'spf', 'key ingredient', 'formulation'], inplace=True)

df2 = df[df['label'].isin(['face-moisturisers', 'mask-and-peel', 'cleanser', 'eye-cream'])]
df2['concern'] = df2['concern'].str.lower()
df2['skin type'] = df2['skin type'].str.lower()

features = ['normal', 'dry', 'oily', 'combination', 'acne', 'sensitive',
            'fine lines', 'wrinkles', 'redness', 'dull', 'pore', 'pigmentation',
            'blackheads', 'whiteheads', 'blemishes', 'dark circles', 'eye bags', 'dark spots']

entries = len(df2)
one_hot_encodings = np.zeros((entries, len(features)))

# Encode skin types and concerns
for i in range(entries):
    skin_type = df2.iloc[i]['skin type']
    if skin_type == 'all':
        one_hot_encodings[i][:5] = 1
    elif skin_type in features:
        one_hot_encodings[i][features.index(skin_type)] = 1

def recs_cs(vector):
    products = []
    cs_values = cosine_similarity(np.array([vector]), one_hot_encodings)[0]
    df2['cs'] = cs_values
    recommendations = df2.sort_values('cs', ascending=False).head(5)
    for _, row in recommendations.iterrows():
        products.append({
            'brand': row['brand'],
            'name': row['name'],
            'price': row['price'],
            'url': row['url'],
            'skin type': row['skin type'],
            'concern': row['concern']
        })
    return products

# Skin Analysis Functions
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

def analyze_skin_type(skin, mask):
    gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
    laplacian = cv2.Laplacian(masked_gray, cv2.CV_64F)
    variance = np.var(laplacian)
    return "Smooth" if variance < 1700 else "Rough"

def detect_acne(skin, mask):
    gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    acne_mask = cv2.bitwise_and(thresholded, thresholded, mask=mask)
    contours, _ = cv2.findContours(acne_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

def detect_dark_spots(skin, mask):
    gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    dark_spots_mask = cv2.bitwise_and(thresholded, thresholded, mask=mask)
    contours, _ = cv2.findContours(dark_spots_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

# Camera Integration and Analysis
cap = cv2.VideoCapture(0)
print("Press 'SPACE' to capture an image and analyze.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to access the camera.")
        break
    cv2.imshow("Live Camera", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        print("Image captured! Starting analysis...")
        captured_image = frame.copy()
        resized = cv2.resize(captured_image, (640, 480))

        skin, mask = detect_skin(resized)
        skin_tone = calculate_skin_tone(skin)
        skin_type = analyze_skin_type(skin, mask)
        acne_count = detect_acne(skin, mask)
        dark_spot_count = detect_dark_spots(skin, mask)

        # Map detected values to feature vector
        input_vector = [0] * len(features)
        if skin_type == "Rough":
            input_vector[features.index('acne')] = 1
        if dark_spot_count > 50:
            input_vector[features.index('dark spots')] = 1

        # Get recommendations
        recommendations = recs_cs(input_vector)
        print("\nRecommended Products:")
        for rec in recommendations:
            print(f"Brand: {rec['brand']}, Name: {rec['name']}, Price: {rec['price']}, URL: {rec['url']}")

    if key == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
