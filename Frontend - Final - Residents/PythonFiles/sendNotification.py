import firebase_admin
import os
from firebase_admin import credentials
from firebase_admin import messaging
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
import sys
sys.path.append('network')
from PythonFiles.initFirebase import residents_collection
# Initialize the Firebase Admin SDK with the private key


def SendNotification(given_title = "Intruder ⚠️",given_content ="Intruder in the area"):
    print("Send Notification Api Started")
    tokens = residents_collection.stream()

    registration_tokens = {}

    for i,token in enumerate(tokens):
        ton = token.to_dict()
        id = ton['token']
        registration_tokens[id] = ton['username']

    for key,value in registration_tokens.items():
        if key == '':
            continue
        message = messaging.Message(
        
        notification=messaging.Notification(
            title=given_title,
            body=given_content
        ),
        token=key,
        
        )
        try:
            response = messaging.send(message)
            # print('Successfully sent message to : ',key)
        except Exception:
            print('failed to send message to : ',key,' ', value)
            query = residents_collection.where(filter=FieldFilter('token','==',key))
            docs = query.stream()
            for doc in docs:
                doc.reference.delete()
                # print('document with id : ',doc.id,'and name',key,'deleted successfully')
    print("Send Notification Success")
if __name__ == "__main__":
    SendNotification()
