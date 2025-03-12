
from pymongo.mongo_client import MongoClient
import dotenv
import os
dotenv.load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

uri = MONGO_DB_URL

# Create a new client and connect to the server
client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)