import cv2
from mtcnn import MTCNN
import streamlit as st
import os
import shutil

def generateUser(label,placeholder):
    # Create a folder to save the detected faces
    output_folder = f"PythonFiles/newUser/{label}"
    os.makedirs(output_folder, exist_ok=True)

    # Load the MTCNN model
    mtcnn = MTCNN()

    # Open a video capture object (you can change the argument to 0 for webcam or provide the path to a video file)
    video_capture = cv2.VideoCapture(0)

    # Set a counter to limit the number of frames processed
    frame_counter = 0
    max_frames = 100  # Adjust as needed

    while True:
        # Read a frame from the video
        ret, frame = video_capture.read()

        # Break the loop if the video is finished or the frame limit is reached
        if not ret :
            return Exception
        if frame_counter >= max_frames:
            break

        orgFrame = frame
        # Detect faces using MTCNN
        faces = mtcnn.detect_faces(frame)

        # Save the detected faces
        for i, face_info in enumerate(faces):
            x, y, w, h = face_info['box']
            face = frame[y:y+h, x:x+w]

            # Save the face in the output folder
            face_filename = os.path.join(output_folder, f"face_{frame_counter}_{i}.jpg")
            try:
                if(i==0):
                    cv2.imwrite(face_filename, face)
                else:
                    placeholder.error("More than one person detected restart the registration")
                    shutil.rmtree(f"PythonFiles/newUser/{label}")
                    raise ValueError("Restart the process")
            except Exception as e:
                placeholder.error(f"Error {e}")
                raise ValueError("Restart the process")
            
            # Draw a rectangle around the detected face on the original frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Increment the frame counter
        frame_counter += 1

        # Display the frame with rectangles around detected faces
        # cv2.imshow('Detected Faces', frame)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        placeholder.image(frame,caption = "Move your face to capture the face in all directions")

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    generateUser("0",placeholder=st.empty())
