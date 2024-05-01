import numpy as np
import cv2
import os
import tensorflow as tf
from joblib import  load
from facenet_pytorch import MTCNN
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation
from tensorflow.keras.models import Model
from numpy import expand_dims
from cv2 import resize,INTER_CUBIC
import threading
from tensorflow.keras.preprocessing.image import  img_to_array
import streamlit as st
from PythonFiles.getClasses import classes
from PythonFiles.apiIdentified import addIdentifiedPerson
from PythonFiles.apiUnidentified import apiUnidentified
from PythonFiles.sendNotification import SendNotification
# from getClasses import classes
# from deliveryDetection import process_frame
from PythonFiles.deliveryDetection import process_frame
# import easyocr
# import cv2
# from roboflow import Roboflow


# rf = Roboflow(api_key="lNppjISoqkMNbiFnzSlI")
# project = rf.workspace('pranav-4yqfh').project("foodcareerbagsdetection")
# model = project.version("2").model
# reader = easyocr.Reader(['en'])

# # Function to perform object detection and OCR on frames
# def process_frame(frame):
#     # Perform object detection on the frame
#     try:
#         response = model.predict(frame).json()
#         # Check if delivery bag is detected
#         delivery_bag_detected = False
#         for bounding_box in response['predictions']:
#             if bounding_box['class'] == 'delivery_bags':
#                 delivery_bag_detected = True
#                 break

#         if delivery_bag_detected:
#             # Perform OCR on the frame
#             result = reader.readtext(frame)
            
#             # Print the OCR results
#             for detection in result:
#                 if(detection[1]=='zomato' or detection[1]=='swiggy'):
#                     print(f"{detection[1]} Delivery boys detected")
#                     break
#                 # print(detection[1])  # Text detected by Easy OCR
#             # else:
#             #     print("No bag detected")
#     except Exception as e:
#         print(f"error in process_frame : {e}")

def vgg_face():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    return model

model = vgg_face()

model.load_weights('PythonFiles/vgg_face_weights.h5')

model = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

scaler = load('PythonFiles/scaler.joblib')
pca = load('PythonFiles/pca_model.joblib')
#Change this to load different classifier
clf = load('PythonFiles/SVC.joblib')

def preprocess_image(img):
    img = img_to_array(img)
    img = img/255.0
    img = expand_dims(img, axis=0)
    return img

def Face_Recognition(roi,model,scaler,pca,clf):
    roi=resize(roi,dsize=(224,224),interpolation=INTER_CUBIC)
    roi=preprocess_image(roi)
    embedding_vector = model.predict(roi)[0]

    embedding_vector=scaler.transform(embedding_vector.reshape(1, -1))
    embedding_vector_pca = pca.transform(embedding_vector)
    result1 = clf.predict(embedding_vector_pca)[0]
    #print(result1)
    y_predict = clf.predict_proba(embedding_vector_pca)[0]
    print(y_predict)
    
    threshold = 0.6
    result = np.where(y_predict > threshold)[0]
    
    return result , y_predict


# Face Detection
mtcnn = MTCNN(image_size=160, margin=14, min_face_size=20, device='cpu', post_process=False)



csv_file = "PythonFiles/dataset.csv"
users = classes(csv_file=csv_file)

for user,id in users.items():
    print(f"user : {user} id : {id}")
print(f"{len(users)}")

def ImageClass(n):
    for x, y in users.items():
        y = int(y)
        if n == y:
            return x

size = (800, 600)

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255, 0, 0)
thickness = 2


# Excel file setup

def LiveFaceRecognition(
        placeholder
):
    placeholder.write("![Loading GIF](https://media2.giphy.com/media/3oEjI6SIIHBdRxXI40/200w.gif?cid=6c09b952yx63atue9btnehyn9cu2z2g2o7xcmky4a8i7bjb0&ep=v1_gifs_search&rid=200w.gif&ct=gf)")
    unknown_folder = 'unknown_faces'
    os.makedirs(unknown_folder,exist_ok=True)
    other = 0
    excel_file = 'PythonFiles/recognized_faces.xlsx'
    recognized_faces_df = pd.DataFrame(columns=['Date & Time', 'Recognized Face'])
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)

    # cap1 = cv2.VideoCapture(1)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Could not open Primary camera.")
        st.error("Primary Camera not open")
        cap.release()
        return 
    
    # if not cap1.isOpened():
    #     print("Error: Could not open camera.")
    #     st.error("Secondary Camera not open")
    #     cap1.release()
    #     return 

    # Define folder to save unknown faces
    unknown_folder = 'unknown_faces'
    os.makedirs(unknown_folder, exist_ok=True)
    recognized_faces = {}
    first_unknown_time = None
    for key,value in users.items():
        recognized_faces[value] = False
        
    for key,val in recognized_faces.items():
        print(f"Key = {key} , value = {val}")

    # Excel file setup
    excel_file = 'recognized_faces.xlsx'
    recognized_faces_df = pd.DataFrame(columns=['Date & Time', 'Recognized Face'])
    registered = []
    result_array = []
    while True:
        # frame1 = cap1.read()
        # process_frame(frame1)
        
        try:
            ret, frame = cap.read()
            if not ret:
                raise ValueError("Error reading frame")
            

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_CUBIC)
            frame = cv2.GaussianBlur(frame, ksize=(3, 3), sigmaX=0)
            frame_face = frame.copy()
            frame_face = cv2.resize(frame_face, (320, 320), interpolation=cv2.INTER_CUBIC)
            boxes, probs = mtcnn.detect(frame_face, landmarks=False)

            if not probs.all() == None and probs.all() > 0.98:
                for x1, y1, x2, y2 in boxes:
                    x1, x2, y1, y2 = int(x1 * 1600 / 640), int(x2 * 1600 / 640), int(y1 * 1200 / 640), int(y2 * 1200 / 640)
                    roi = frame[y1:y2, x1:x2]

                    # Ensure roi has valid dimensions before passing it to Face_Recognition
                    if roi.shape[0] > 0 and roi.shape[1] > 0:
                        # Recognize face
                        result, y_predict = Face_Recognition(roi, model, scaler, pca, clf)
                        if len(result) == 1:
                            current_face = str(ImageClass(result[0]))
                            result_array.append(result[0])
                            cv2.putText(frame, current_face, (x1 - 5, y1 - 5), font, fontScale, color, thickness,
                                        cv2.LINE_AA)
                            cv2.putText(frame, str(np.round(y_predict[result[0]], 2)), (x2, y1 - 10), font, fontScale, color,
                                        thickness, cv2.LINE_AA)
                        #Unknow condition 
                        else:
                            # recognized_faces = {}
                            # print("dataframe refreshed")
                            current_face = 'Unknown'
                            
                            if first_unknown_time is None:
                                try:
                                    delivery_person = threading.Thread(target=process_frame, args=(frame,))
                                    delivery_person.start()
                                except Exception as e:
                                    print(f"Unable to process for delivery persons : {e} ")
                                print("Saving the image of the first unknown face.")
                                # Save the image of the unknown face
                                unknown_img_path = os.path.join(unknown_folder, f'unknown_{datetime.now().strftime("%Y%m%d%H%M%S")}.png')
                                roi = cv2.resize(roi, (100, 100))
                                cv2.imwrite(unknown_img_path, cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                                try:
                                    apiUn = threading.Thread(target=apiUnidentified,args = (unknown_img_path,))
                                    apiNotfi = threading.Thread(target = SendNotification)
                                    apiUn.start()
                                    apiNotfi.start()
            
                                except:
                                    print("Unable upload unidentified form live recognition")
                                # Set the first_unknown_time variable
                                first_unknown_time = datetime.now()

                            elapsed_time = (datetime.now() - first_unknown_time).seconds
                            if elapsed_time > 20:
                                try:
                                    delivery_person = threading.Thread(target=process_frame, args=(frame,))
                                    delivery_person.start()
                                except Exception as e:
                                    print(f"Unable to process for delivery persons : {e} ")
                                print("Enough time has passed. Processing another unknown face.")
                                # Save the image of the unknown face
                                unknown_img_path = os.path.join(unknown_folder, f'unknown_{datetime.now().strftime("%Y%m%d%H%M%S")}.png')
                                roi = cv2.resize(roi, (100, 100))
                                cv2.imwrite(unknown_img_path, cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                                try:
                                    apiUn = threading.Thread(target=apiUnidentified,args = (unknown_img_path,))
                                    apiNotfi = threading.Thread(target = SendNotification)
                                    apiUn.start()
                                    apiNotfi.start()
                                except:
                                    print("Unable upload unidentified form live recognition")
                                # Reset the first_unknown_time variable
                                first_unknown_time = datetime.now()
                                #Upload the Unknown api here    
                            else:
                                print("Cooldown period. Skipping processing.")
                            
                            cv2.putText(frame, current_face, (x1 - 5, y1 - 5), font, fontScale, color, thickness,
                                             cv2.LINE_AA)


                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                for key,vald in recognized_faces.items():
                    if int(key) in result_array:
                        recognized_faces[key] = True
                    else:
                        recognized_faces[key] = False
                result_array = []
                # print("--------------------")
                # for key,vale in recognized_faces.items():
                #     print(f"Key = {key}, value = {vale}")
                # print("--------------------")

                for i,j in recognized_faces.items():
                    current_face = str(ImageClass(int(i)))
                    
                    if recognized_faces[i] and i not in registered:
                        registered.append(i)
                        print(f"Recognized and registered: {current_face}")
                        try:
                            apiIdentf = threading.Thread(target=addIdentifiedPerson, args=(str(result[0]),current_face,))
                            # apiNotfi = threading.Thread(target = SendNotification, args=("Identified Person",f"Notification Test : {current_face}",))
                            apiIdentf.start()
                            # apiNotfi.start()
                        except:
                            print("Unable to upload to database")
                        
                    elif recognized_faces[i]==False and i in registered:
                        registered.remove(i)
                        print(f"Remove person from frame: {current_face}")
                    # elif recognized_faces[i] == False:
                    #     print(f"{current_face} not in frame")
            
            else:
                registered = []

            # cv2.imshow('frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            placeholder.image(frame)

            # Exit when 'q' is pressed
            if cv2.waitKey(1) == ord('q'):
                break

        except Exception as e:
            print(f"Error: {e}")
            st.error(f"Error encounted : {e}")
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    LiveFaceRecognition()
