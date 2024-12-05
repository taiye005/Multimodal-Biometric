import cv2

# Load pre-trained face detector (Haar cascade)
face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/4.9.0_7/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')

# Load an image
image = cv2.imread('/Users/darl/captured-images/capture3a.jpeg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the image with detected faces
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()



