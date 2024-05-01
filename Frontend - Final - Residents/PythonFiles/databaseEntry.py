import firebase_admin
from firebase_admin import credentials, storage
from firebase_admin import firestore
import uuid
from PythonFiles.privateFiles import *
from PythonFiles.initFirebase import knownPeople_collection
import os

bucket = storage.bucket()

def apiUpload(
        information,
        image_path,
        id,
        name
        ):

    # Destination path in Firebase Storage
    destination_blob_name = 'KnowPeople/'+id

    # Upload the local file to Firebase Storage
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(image_path)

    print(f'File {image_path} uploaded to {destination_blob_name}.')

    # Get the download URL for the uploaded image
    download_url = blob.generate_signed_url(expiration=253402300799) # URL won't expire

    print('Download URL:', download_url)
    # id = generate_random_id(8)
    #Create document for uploading
    document = {
        "id" : id,
        "info" : get_information(information),
        "name" : name,
        "profilePhoto" : download_url
    }

    try:
        knownPeople_collection.document(id).set(document)
        print("Document for new user success")
    except Exception as e:
        print('Unable to add new user with unique id : '+f"{e}")
    

def get_information(information):
    return information

def get_name(image_name):
    image_name = image_name.replace(".png","")
    image_name = image_name.replace("face_dataset/","")
    image_name = image_name.replace("/1","")
    return image_name

if __name__ == "__main__":
    local_image_path = 'Dataset/Mrugendra Kulkarni/1.png'
    apiUpload(
        information = "He trained model for delivery recognition",
        image_path = local_image_path,
        id="234cvioub",
        name="Aryan"
        )

