import streamlit as st
import cv2
import os
from PIL import Image
import  numpy as np
import tensorflow as tf
import math
caspath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
face_cascade=cv2.CascadeClassifier(caspath)
new_model=tf.keras.models.load_model("trained_mdl.h5")
#new_model=cv2.face.LBPHFaceRecognizer_create()
def detect_faces(image):
    FONT_SCALE = 2e-3  # Adjust for larger font size in all images
    THICKNESS_SCALE = 1e-3  # Adjust for larger thickness in all images

    height, width= image.size

    font_scale = min(width, height) * FONT_SCALE
    thickness = math.ceil(min(width, height) * THICKNESS_SCALE)
    img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),flags=cv2.CASCADE_SCALE_IMAGE)
    faces_list = []
    preds = []
    for (x, y, w, h) in faces:
     roi_gray = gray[y:y + h, x:x + w]
     roi_color = img[y:y + h, x:x + w]
     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
     facess = face_cascade.detectMultiScale(roi_gray)
     if (len(facess) == 0):
        print("face not detected")
     else:
        for (ex, ey, ew, eh) in facess:
            face_frame = img[y:y + h, x:x + w]
            face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
            face_frame = cv2.resize(face_frame, (224, 224))
            face_frame = tf.keras.preprocessing.image.img_to_array(face_frame)
            face_frame = np.expand_dims(face_frame, axis=0)
            face_frame = tf.keras.applications.mobilenet_v2.preprocess_input(face_frame)
            faces_list.append(face_frame)
            if len(faces_list) > 0:
                preds = new_model.predict(faces_list)
            for pred in preds:
                (mask, withoutMask) = pred
            if mask > withoutMask:
                label = "Mask"
            else:
                label = "No Mask"
            color = (0, 255, 0) if label == "Mask" else (255,0,0)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(img, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,thickness, 2)

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            for pred in preds:
                # mask contain probabily of wearing a mask and vice versa
                (mask, withoutMask) = pred
                print(mask, withoutMask)
            if mask > withoutMask:
                print('Wearing Mask')
            else:
                print('Not wearing Mask')
    return img
def main():
    st.title("Face Mask Detector")
    html_temp="""
    <body style="background-color:red;">
    <div style="background-color:teal;padding:10px">
    <h2 style="color:white;text-align:center;">Face Mask Detection app</h2>
    </div>
    </body>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    image_file=st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
    if image_file is not None:
        image=Image.open(image_file)
        st.text("original Image")
        st.image(image)
    if st.button("Recognise"):
        result_img=detect_faces(image)
        st.image(result_img)
if __name__== '__main__':
    main()