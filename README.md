Face Verification & Liveness Prototype
This project is a Python script that uses open-source AI models to perform a two-step security check using your webcam:

Liveness Detection: It verifies you are a live person by detecting an eye blink.

Identity Verification: It captures an image after the blink and compares it to a reference photo to confirm your identity.

Setup Instructions
Follow these steps to run the prototype on your own computer.

1. Clone the Repository
First, you need to download the project files.

Go to the main page of this GitHub repository.

Click the green < > Code button.

Make sure the HTTPS tab is selected, and click the copy icon to copy the repository URL.

Open your terminal, navigate to where you want to save the project, and run the git clone command, pasting the URL you just copied. The full command will look like this:

git clone https://github.com/YourUsername/Your-Repo-Name.git

Navigate into the newly created project folder:

cd Your-Repo-Name

2. Create a Python Virtual Environment
This creates a clean, isolated environment for the project.

# Create the environment
python -m venv venv

# Activate the environment (command differs by OS)
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

3. Install Required Libraries
This project has several dependencies. Run these commands one by one to install them. This may take several minutes.

pip install tensorflow==2.16.1
pip install deepface
pip install tf-keras
pip install mediapipe
pip install opencv-python

4. Prepare Your Reference Image
The script needs a photo of you to use as a baseline for identity verification.

Find a clear, well-lit photo of your face.

Place it in the project folder.

Rename it to exactly reference_face.jpg.

How to Run the Test
Make sure your virtual environment is still active (your terminal prompt should start with (venv)).

Run the script from your terminal:

python run_blink_test.py
