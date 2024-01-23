# import necessary packages
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r',encoding="utf-8")
classNames = f.read().split('\n')
f.close()
print(classNames)


# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    pil_frame = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_frame)


    # font_file = r'D:\\Project\\Latha.ttf'
    font_file = r'C:\\Users\\rosar\\Downloads\\nirmala-ui\\Nirmala.ttf'
    font = ImageFont.truetype(font_file, size=30)
    position = (10, 50)

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)
    
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            # print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]


    combined_frame = np.zeros_like(frame)
    # show the prediction on the frame
    draw.text(position, className, font=font, fill=(255, 255, 255, 0))
    frame_with_text = np.array(pil_frame)


    pil_frame_width, pil_frame_height = pil_frame.size
    combined_frame[:pil_frame_height, :pil_frame_width] = pil_frame

    # Show the final output
    cv2.imshow("Output", combined_frame) 

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
