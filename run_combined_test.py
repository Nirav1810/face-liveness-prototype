# run_combined_test.py
import cv2
import mediapipe as mp
from deepface import DeepFace

# --- CONFIGURE YOUR IMAGES HERE ---
reference_image_path = "image6.jpg"
live_image_path = "image5.jpg"
# ------------------------------------

def verify_identity(img1_path, img2_path):
    """Uses DeepFace to verify if the faces in two images match."""
    print("\n--- Running Step 1: Identity Verification (DeepFace) ---")
    try:
        # The first time this is called, it will download and build the model.
        # *** FIX: Switched to Facenet512 model to avoid VGG-Face initialization bug. ***
        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name="Facenet512" 
        )
        if result["verified"]:
            print("✅ Identity Verified: Faces are a match.")
            return True
        else:
            print("❌ Identity Failed: Faces do not match.")
            return False
    except Exception as e:
        print(f"ERROR during face verification: {e}")
        return False

def verify_liveness_gesture(img_path):
    """Uses MediaPipe to verify if a thumbs-up gesture is present in an image."""
    print("\n--- Running Step 2: Liveness Gesture Verification (MediaPipe) ---")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    try:
        image = cv2.imread(img_path)
        if image is None:
            print(f"ERROR: Could not read liveness image at {img_path}")
            return False

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]

                # Simple logic to check for a thumbs-up
                if (thumb_tip.y < thumb_ip.y) and (index_finger_tip.y > index_finger_pip.y):
                    print("✅ Liveness Verified: Thumbs-up gesture detected.")
                    hands.close()
                    return True

        print("❌ Liveness Failed: Thumbs-up gesture NOT detected.")
        hands.close()
        return False

    except Exception as e:
        print(f"ERROR during gesture verification: {e}")
        hands.close()
        return False


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting verification process...")

    # Run both checks in sequence
    identity_verified = verify_identity(reference_image_path, live_image_path)

    liveness_verified = False
    if identity_verified:
        liveness_verified = verify_liveness_gesture(live_image_path)

    # Final decision
    print("\n--- FINAL RESULT ---")
    if identity_verified and liveness_verified:
        print("✅✅✅ SUCCESS: Identity and Liveness both confirmed. Attendance can be marked.")
    else:
        print("❌❌❌ FAILED: Attendance rejected. One or both checks failed.")
