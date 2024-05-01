from PythonFiles.initFirebase import identified_collection
import datetime
import pytz
from pytz import timezone
import time

def addIdentifiedPerson(id,name):
    print("Api Identified Started")

    now_utc = datetime.datetime.now(tz=datetime.timezone.utc)
    now_ist = now_utc.astimezone(timezone('Asia/Kolkata'))
    current_time = now_ist.timestamp()
    data = {
        "id":id,
        "name":name,
        "timeStamp":now_ist
    }
    # print("get it here")
    try:
        # time.sleep(5)
        identified_collection.document(str(now_ist)).set(data)
        # print('doc added with id : '+ f'{doc_ref[1].id}')
        print('success to upload identified at backend')
    except Exception as e:
        print(f'unable to add the {name} : '+f'{e}')
    
if __name__ == "__main__":
    addIdentifiedPerson("0","Mrugendra")