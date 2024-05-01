import streamlit as st
from PythonFiles.sendNotification import SendNotification
from PythonFiles.databaseEntry import apiUpload
from PythonFiles.generateNewUser import generateUser
import uuid
import os
from contextlib import contextmanager
from PythonFiles.initFirebase import generate_random_id
from PythonFiles.newUser import count_users
import shutil

st.set_page_config(
    page_title="Register user"
)

csv_file = "PythonFiles/dataset.csv"

id = count_users(csv_file = csv_file)-1

@contextmanager
def acquire_camera():
    camera = st.camera_input("Profile")
    yield camera
    del camera


# @st.cache
def register():
    st.title("New society member Registration")
    placeholder = st.empty()
    container = st.container()
    placeForm = st.empty()

    with placeForm.form("userRegistration"):
        label_text = st.text_input(label="Name*",placeholder="your name",max_chars=18,)
        information = st.text_input(label="Information*",placeholder="address")   
        # Capture image from camera
        
        # profile_photo = st.camera_input("Please Capture profile photo:")

        # Display the captured image
        with acquire_camera() as profile_photo:
            # Display the captured image
            if profile_photo is not None:
                # st.image(profile_photo, caption="Captured profile photo", use_column_width=True)

                # Save the captured image for later use
                if not os.path.exists("profile_photos"):
                    os.makedirs("profile_photos")
                profile_photo_path = f"profile_photos/{id}.jpg"
                with open(profile_photo_path, "wb") as f:
                    f.write(profile_photo.getvalue())

                st.success("Profile photo captured successfully!")

        want_to_contiue = st.checkbox("Are you sure you want to register*")
        
        register = st.form_submit_button("Register now!")
        if register :
            if want_to_contiue and label_text!="" and information !="":
                try:
                    generateUser(label=f"{label_text}",placeholder=placeholder)
                    with placeholder.container():
                        st.success("Succesfully Registered user")
                except Exception as e:
                    with placeholder.container():
                        st.error(f"Unable to complete the registration : {e}")
                    return
                try:
                    # st.success(f"face_dataset/{label_text}/1.png")
                    with placeholder.container():
                        print("inside the placholder")    
                        print("image is taken")
                        # st.image(currImage)
                        apiUpload(
                            image_path=f'profile_photos/{id}.jpg',
                            name = label_text,
                            id = f"{id}",
                            information=information
                        )
                        print("image is uploaded")
                        SendNotification(given_title="new registration", given_content=f"{label_text} has registered on the system")
                        with placeholder.container():
                            container.success("successfully send data to backend ?")
                except Exception as e :
                    with placeholder.container():
                        container.error(f"unable to send data to backend : {e}")
                        shutil.rmtree(f"PythonFiles/newUser/{label_text}")
            else:
                st.error("Please check the compulsory fields")


if __name__ == "__main__":
    register()
