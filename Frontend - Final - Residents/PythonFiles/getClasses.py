import os
# from initFirebase import knownPeople_collection 
import csv

def get_classes(directory):
    folder_names = {}
    index = 0
    for item in sorted(os.listdir(directory)):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folder_names[item] = index
            index += 1
    return folder_names

knownPeople = {}

def classes(csv_file):
    data_dict = {}
    with open(csv_file, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data_dict[row['name']] = row['id']
    return data_dict

def count_users(csv_file):
    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file)
        row_count = sum(1 for row in reader)
    return row_count

def addUser(csv_file,data):
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

# Example usage:
if __name__ == "__main__":
    directory_path = "PythonFiles/newUser"
    folders = get_classes(directory_path)
    classe = classes(csv_file = "PythonFiles\dataset.csv")
    print("Folder names and indices within", directory_path, "are:")
    
    for index,name in folders.items():
        print(f"name: {type(name)} and idex: { index}")

    for index,name in classe.items():
        print(f"Key: {type(index)} name : {name}")

    print(count_users("PythonFiles\dataset.csv")-1)






