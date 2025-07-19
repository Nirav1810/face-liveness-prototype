# run_test.py
from deepface import DeepFace

# --- CONFIGURE YOUR IMAGES HERE ---
img1_path = "image2.jpg"
img2_path = "image6.jpg"
# ------------------------------------

print(f"Comparing '{img1_path}' and '{img2_path}'...")

try:
    # This is the core function that does the comparison
    result = DeepFace.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name="VGG-Face" # A reliable model for comparison
    )

    # Print the results in a clear format
    print("\n--- DeepFace Analysis Result ---")
    if result["verified"]:
        print("✅ Result: The faces are a MATCH.")
    else:
        print("❌ Result: The faces do NOT match.")

    print(f"Similarity Distance: {result['distance']:.4f}")
    print(f"(A lower distance means a higher similarity)")

except Exception as e:
    print("\n--- ERROR ---")
    print("An error occurred. This might be because a face could not be detected in one of the images.")
    print(f"Details: {e}")