# run_blink_test.py
import cv2
import mediapipe as mp
from deepface import DeepFace
from scipy.spatial import distance as dist

# --- CONFIGURE YOUR REFERENCE IMAGE HERE ---
reference_image_path = "image6.jpg"
# -----------------------------------------

def calculate_ear(eye):
    """Calculates the Eye Aspect Ratio (EAR) for a single eye."""
    # Vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Horizontal eye landmark
    C = dist.euclidean(eye[0], eye[3])
    # The EAR formula
    ear = (A + B) / (2.0 * C)
    return ear

def verify_identity(img1_path, img2_path):
    """Uses DeepFace to verify if the faces in two images match."""
    print("\n--- Running Step 2: Identity Verification (DeepFace) ---")
    try:
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

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting live blink detection...")
    print("Please look at the camera and blink. The window will close automatically.")

    # Constants for blink detection
    EAR_THRESHOLD = 0.20  # Threshold for closed eyes
    EAR_CONSEC_FRAMES = 2  # Number of consecutive frames the eye must be below the threshold

    # Initialize counters
    FRAME_COUNTER = 0
    BLINK_DETECTED = False

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Start webcam feed
    cap = cv2.VideoCapture(0)

    identity_verified = False

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        # and convert the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True

        # Convert the image color back so it can be displayed
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark

            # Extract landmarks for left and right eyes
            # See landmark map: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
            left_eye_landmarks = [(face_landmarks[i].x, face_landmarks[i].y) for i in [362, 385, 387, 263, 373, 380]]
            right_eye_landmarks = [(face_landmarks[i].x, face_landmarks[i].y) for i in [33, 160, 158, 133, 153, 144]]

            left_ear = calculate_ear(left_eye_landmarks)
            right_ear = calculate_ear(right_eye_landmarks)

            # Average the EAR for both eyes
            avg_ear = (left_ear + right_ear) / 2.0

            # Check if the eye aspect ratio is below the blink threshold
            if avg_ear < EAR_THRESHOLD:
                FRAME_COUNTER += 1
            else:
                # If the eyes were closed for a sufficient number of frames
                if FRAME_COUNTER >= EAR_CONSEC_FRAMES:
                    BLINK_DETECTED = True
                # Reset the frame counter
                FRAME_COUNTER = 0

        # Display instructions on the video feed
        cv2.putText(image, "Please look at the camera and BLINK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        if BLINK_DETECTED:
            cv2.putText(image, "BLINK DETECTED!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            # Save the frame where the blink was confirmed
            live_capture_path = "live_capture.jpg"
            cv2.imwrite(live_capture_path, image)
            print("\n--- Running Step 1: Liveness Verification (MediaPipe) ---")
            print("✅ Liveness Verified: Blink was detected.")

            # Now run the identity check
            identity_verified = verify_identity(reference_image_path, live_capture_path)
            break # Exit the loop

        # Show the video feed
        cv2.imshow('MediaPipe Face Mesh', image)

        # Press 'q' to quit early
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

    # Final decision
    print("\n--- FINAL RESULT ---")
    if BLINK_DETECTED and identity_verified:
        print("✅✅✅ SUCCESS: Identity and Liveness both confirmed. Attendance can be marked.")
    else:
        print("❌❌❌ FAILED: Attendance rejected. One or both checks failed.")
