import firebase_admin
import os
from firebase_admin import credentials
from firebase_admin import messaging
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from PythonFiles.privateFiles import Credentials
from PythonFiles.privateFiles import project_id
from PythonFiles.privateFiles import bucket_id

# from privateFiles import Credentials
# from privateFiles import project_id
# from privateFiles import bucket_id



cred = credentials.Certificate(Credentials)
if firebase_admin._apps:
    firebase_admin.delete_app(firebase_admin.get_app())

firebase_admin.initialize_app(cred, {
    'storageBucket': bucket_id 
    })
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= Credentials


db = firestore.Client(project=project_id)
# residents_collection = 'residents'
residents_collection = db.collection('residents')
knownPeople_collection = db.collection("KnownPeople")
identified_collection = db.collection("IdentifiedPeople")
unknownPeople_collection = db.collection("UnkownCollection")
delivery_collection = db.collection("DeliveryPeople")

def generate_random_id(length):
    Id = 0
    for item in sorted(os.listdir("Dataset")):
        item_path = os.path.join("Dataset", item)
        if os.path.isdir(item_path):
            # folder_names[item] = index
            Id += 1
    return Id