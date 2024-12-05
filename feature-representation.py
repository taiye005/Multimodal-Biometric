import cv2

# Load pre-trained face detector (Haar cascade)
face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/4.9.0_7/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')

# Load an image
image = cv2.imread('/Users/darl/captured-images/capture3a.jpeg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Store feature vectors in a list
feature_vectors = []

# Perform feature representation for each detected facez
for (x, y, w, h) in faces:
    face_roi = gray[y:y+h, x:x+w]
    resized_face = cv2.resize(face_roi, (100, 100))
    feature_vector = resized_face.flatten()
    feature_vectors.append(feature_vector)

# Display feature vectors in the console
for i, feature_vector in enumerate(feature_vectors):
    print(f"Feature vector for face {i + 3}: {feature_vector}")

# Save feature vectors to a file (optional)
with open('feature_vectors.txt', 'w') as f:
    for feature_vector in feature_vectors:
        f.write(f"{feature_vector}\n")



        