from pymongo import MongoClient
import os


#mongo_uri = os.getenv("MONGO_URI")
mongo_uri = f"mongodb+srv://summerham22:FEm4DxNh3oxLK3rM@cluster0.5vlv3.mongodb.net/tut?retryWrites=true&w=majority&appName=Cluster0"

def connect_db():
    try:
        client = MongoClient(mongo_uri)
        db = client['tut']
        collection = db['ticket']
        print("MongoDB connected successfully!")

         # 'description' 필드를 가져오는 쿼리
        tickets = collection.find({}, {'description': 1})
        
        description_list = list()

        for ticket in tickets:
            description_list.append(ticket['description'])

    except Exception as e:
        print(f"MongoDB connection error: {e}")
    
    return description_list