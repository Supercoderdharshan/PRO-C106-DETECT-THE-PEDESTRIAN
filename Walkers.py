import cv2


# Create our body classifier
Body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')


# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded

while True:
    
    # Read first frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    body = Body_cascade.detectMultiScale(gray, 1.1, 5)

    # Draw the rectangle around each face
    for (x, y, w, h) in body:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # Quit Window by Spacebar Key
    if cv2.waitKey(25) == 32:
        break
    #Convert Each Frame into Grayscale
    
    
    # Pass frame to our body classifier
    
        


cap.release()
cv2.destroyAllWindows()
