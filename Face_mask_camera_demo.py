import tensorflow as tf
import cv2
import os
import numpy as np
new_model=tf.keras.models.load_model("trained_mdl.h5")
cascPath = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
model = tf.keras.models.load_model("trained_mdl.h5")

video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    print(faces)
    faces_list = []
    preds = []
    for (x, y, w, h) in faces:
        face_frame = frame[y:y + h, x:x + w]
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_frame = cv2.resize(face_frame, (224, 224))
        face_frame = tf.keras.preprocessing.image.img_to_array(face_frame)
        face_frame = np.expand_dims(face_frame, axis=0)
        face_frame = tf.keras.applications.mobilenet_v2.preprocess_input(face_frame)
        faces_list.append(face_frame)
        if len(faces_list) > 0:
          preds = model.predict(faces_list)
        for pred in preds:
            (mask, withoutMask) = pred
        if mask > withoutMask:
            label = "Mask"
        else:
            label= "No Mask"
        print(label)
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
