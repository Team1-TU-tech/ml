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
        
        return collection

    except Exception as e:
        print(f"MongoDB connection error: {e}")
    
def get_all_performances():
    collection = connect_db()
    if collection is None:
        print("MongoDB 연결에 실패했습니다. 데이터를 가져올 수 없습니다.")
        return [] 
    
    performances = collection.find({}, {'_id': 1, 'title': 1, 'description': 1, 'start_date': 1, 'end_date': 1})
    return list(performances)

