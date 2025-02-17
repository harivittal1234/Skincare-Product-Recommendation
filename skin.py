import cv2
import numpy as np

def detect_skin(image):
    """
    Detect skin regions in an image using HSV color space.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin = cv2.bitwise_and(image, image, mask=mask)
    return skin, mask

def calculate_skin_tone(skin):
    """
    Calculate the average skin tone from the detected skin region.
    """
    hsv = cv2.cvtColor(skin, cv2.COLOR_BGR2HSV)
    non_zero_pixels = hsv[np.where((hsv != [0, 0, 0]).all(axis=2))]
    average_tone = np.mean(non_zero_pixels, axis=0) if len(non_zero_pixels) > 0 else [0, 0, 0]
    return average_tone

def classify_skin_type(average_tone):
    """
    Classify skin type based on HSV average tone.
    """
    if average_tone[2] > 180:  # High brightness
        return "Normal"
    elif average_tone[1] > 70:  # High saturation
        return "Oily"
    elif average_tone[1] < 30:  # Low saturation
        return "Dry"
    else:
        return "Combination"

def analyze_skin_type(skin, mask):
    """
    Analyze skin texture (smooth or rough) for sensitive skin detection.
    """
    gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    # Calculate Laplacian variance
    laplacian = cv2.Laplacian(masked_gray, cv2.CV_64F)
    variance = np.var(laplacian)

    # Adjust thresholds based on testing
    return "Sensitive" if variance < 1700 else "Non-Sensitive"

def detect_acne(skin, mask):
    """
    Detect acne spots on the skin.
    """
    gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    acne_mask = cv2.bitwise_and(thresholded, thresholded, mask=mask)
    contours, _ = cv2.findContours(acne_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    acne_count = len(contours)
    return acne_count, acne_mask

def detect_dark_spots(skin, mask):
    """
    Detect dark spots on the skin.
    """
    gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    dark_spots_mask = cv2.bitwise_and(thresholded, thresholded, mask=mask)
    contours, _ = cv2.findContours(dark_spots_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dark_spot_count = len(contours)
    return dark_spot_count, dark_spots_mask

def detect_fine_lines(skin, mask):
    """
    Detect fine lines or wrinkles on the skin.
    """
    gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)  # Edge detection for lines
    fine_lines = cv2.bitwise_and(edges, edges, mask=mask)
    fine_lines_count = np.sum(fine_lines > 0) // 1000  # Estimate count based on edge density
    return fine_lines_count, fine_lines

# Open camera
cap = cv2.VideoCapture(0)

print("Press 'SPACE' to capture an image and analyze.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to access the camera.")
        break

    # Show live camera feed
    cv2.imshow("Live Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    # Capture the image when space is pressed
    if key == ord(' '):
        print("\nImage captured! Starting analysis...")
        captured_image = frame.copy()

        # Resize for consistent processing
        resized = cv2.resize(captured_image, (640, 480))

        # Detect skin
        skin, mask = detect_skin(resized)

        # Analyze features
        average_tone = calculate_skin_tone(skin)
        skin_type = classify_skin_type(average_tone)
        sensitivity = analyze_skin_type(skin, mask)
        acne_count, acne_mask = detect_acne(skin, mask)
        dark_spot_count, dark_spots_mask = detect_dark_spots(skin, mask)
        fine_lines_count, fine_lines = detect_fine_lines(skin, mask)

        # Determine skin concerns
        skin_concerns = []
        if acne_count > 5:
            skin_concerns.append("Acne")
        if dark_spot_count > 5:
            skin_concerns.append("Dark Spots")
        if fine_lines_count > 2:
            skin_concerns.append("Fine Lines")
        if sensitivity == "Sensitive":
            skin_concerns.append("Sensitivity")

        # Print results
        print("\n--- Skin Analysis Results ---")
        print(f"Skin Tone (HSV): {average_tone}")
        print(f"Skin Type: {skin_type}")
        print(f"Acne Count: {acne_count}")
        print(f"Dark Spot Count: {dark_spot_count}")
        print(f"Fine Lines Count: {fine_lines_count}")
        print(f"Skin Concerns: {', '.join(skin_concerns) if skin_concerns else 'None'}")
        print("-----------------------------")

        # Display results
        cv2.imshow("Captured Image", resized)
        cv2.imshow("Skin Mask", skin)
        cv2.imshow("Acne Mask", acne_mask)
        cv2.imshow("Dark Spots Mask", dark_spots_mask)
        cv2.imshow("Fine Lines", fine_lines)

    # Quit on 'q'
    if key == ord('q'):
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
