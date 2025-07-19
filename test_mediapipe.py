# test_mediapipe.py
import cv2
import mediapipe as mp

# --- CONFIGURE YOUR IMAGE HERE ---
image_path = "thumb_up.jpg"
# ---------------------------------

print(f"Analyzing '{image_path}' for a thumbs-up gesture...")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

try:
    # Read the image file
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find hand landmarks
    results = hands.process(image_rgb)

    gesture_found = False
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # A simple logic for thumbs-up detection:
            # The tip of the thumb (landmark 4) should be above the middle knuckle of the thumb (landmark 3).
            # The tip of the index finger (landmark 8) should be below the middle knuckle of the index finger (landmark 6).
            # This ensures the thumb is up and other fingers are down.
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP] # Intermediate Phalanx
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP] # Proximal Interphalangeal

            if (thumb_tip.y < thumb_ip.y) and (index_finger_tip.y > index_finger_pip.y):
                gesture_found = True
                # Draw landmarks on the image to visualize
                annotated_image = image.copy()
                mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.imwrite("annotated_image.jpg", annotated_image)
                break # Stop after finding one thumbs-up

    print("\n--- MediaPipe Analysis Result ---")
    if gesture_found:
        print("✅ Result: Thumbs-up gesture DETECTED.")
        print("Check the file 'annotated_image.jpg' to see the detected hand landmarks.")
    else:
        print("❌ Result: Thumbs-up gesture NOT DETECTED.")

except Exception as e:
    print("\n--- ERROR ---")
    print(f"An error occurred. Details: {e}")

finally:
    # Release the hands object
    hands.close()