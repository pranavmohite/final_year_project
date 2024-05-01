import easyocr
import cv2
import time
from roboflow import Roboflow
from PythonFiles.apiDelivery import apiDelivery
from PythonFiles.sendNotification import SendNotification
import datetime
import os
import threading
rf = Roboflow(api_key="UvSiOJXaFoCFgFBPlbex")
project = rf.workspace('pranav-4yqfh').project("foodcareerbagsdetection")
model = project.version("2").model



# Load the object detection model (assuming model is already defined)

# Initialize Easy OCR with English language
reader = easyocr.Reader(['en'])

# Function to perform object detection and OCR on frames
def process_frame(frame):
    # Perform object detection on the frame
    delivery_folder = "DeliveryPeople"
    
    response = model.predict(frame, confidence=40, overlap=30).json()

    # Check if delivery bag is detected
    delivery_bag_detected = False
    for bounding_box in response['predictions']:
        if bounding_box['class'] == 'delivery_bags':
            delivery_bag_detected = True
            break

    if delivery_bag_detected:
        # Perform OCR on the frame
        result = reader.readtext(frame)
        
        # Print the OCR results
        for detection in result:
            if(detection[1]=='zomato' or detection[1]=='swiggy'):
                print(f"{detection[1]} Delivery boys detected")
                frame = cv2.resize(frame, (100, 100))
                delivery_img_path = os.path.join(delivery_folder, f'delivery_{datetime.now().strftime("%Y%m%d%H%M%S")}.png')
                cv2.imwrite((delivery_img_path), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                try:
                    apiDel = threading.Thread(target = apiDelivery,args=(delivery_img_path,))
                    apiNotfi = threading.Thread(target = SendNotification,args = ("Delivery Person Detected", f"{detection[1]} Delivery boys detected",))
                    apiNotfi.start()
                    apiDel.start()
                except:
                    print("unable to send notification or update deliveryCollection")
                break
            # print(detection[1])  # Text detected by Easy OCR
        
    else:
        print("No bag detected")

# Open the video capture device (assuming 0 is the default camera)
def main():
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    # Continuously process frames from the video stream
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if the frame is read correctly
        if not ret:
            print("Error: Could not read frame.")
            break

        # Process the frame
        process_frame(frame)

        # Display the frame
        cv2.imshow('frame', frame)

        # Check for the 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # image = cv2.imread("PythonFiles/frame.jpg")
    # process_frame(image)
    # Release the video capture device and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()