import os
import sys
import pymongo
import numpy as np
import pandas as pd
from typing import List
import pymongo.mongo_client
from sklearn.model_selection import train_test_split

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

## Configuration for the  Data Ingestion Config

from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self,data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    # def export_collection_as_dataframe(self):
    #     """
    #     Read the Data from the MongoDB
    #     """
    #     try:
    #         database_name = self.data_ingestion_config.database_name
    #         collection_name = self.data_ingestion_config.collection_name
    #         self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
    #         collection = self.mongo_client[database_name][collection_name]

    #         df = pd.DataFrame(list(collection.find()))

    #         print(f"Fetched {len(df)} records from MongoDB")  # Debugging
    #         logging.info(f"Fetched {len(df)} records from MongoDB")

    #         if df.empty:
    #             raise ValueError("MongoDB Collection is empty! Please check data ingestion.")

    #         if "_id" in df.columns.to_list():
    #             df.drop(columns=['_id'],axis=1)

    #         df.replace({"na":np.nan},inplace=True)

    #         return df
        
    #     except Exception as e:
    #         raise NetworkSecurityException(e,sys)

    # def export_collection_as_dataframe(self):
    #     try:
    #         database_name = self.data_ingestion_config.database_name
    #         collection_name = self.data_ingestion_config.collection_name
            
    #         # Add connection parameters to improve reliability
    #         client_options = {
    #             "connectTimeoutMS": 30000,  # Increase connection timeout to 30 seconds
    #             "socketTimeoutMS": 45000,   # Increase socket timeout to 45 seconds
    #             "serverSelectionTimeoutMS": 30000,  # Server selection timeout
    #             "retryWrites": True,
    #             "retryReads": True
    #         }
            
    #         # Parse the connection string and add options
    #         if "?" in MONGO_DB_URL:
    #             connection_string = f"{MONGO_DB_URL}&connectTimeoutMS={client_options['connectTimeoutMS']}&socketTimeoutMS={client_options['socketTimeoutMS']}&serverSelectionTimeoutMS={client_options['serverSelectionTimeoutMS']}"
    #         else:
    #             connection_string = f"{MONGO_DB_URL}?connectTimeoutMS={client_options['connectTimeoutMS']}&socketTimeoutMS={client_options['socketTimeoutMS']}&serverSelectionTimeoutMS={client_options['serverSelectionTimeoutMS']}"
            
    #         self.mongo_client = pymongo.MongoClient(connection_string)
    #         collection = self.mongo_client[database_name][collection_name]
            
    #         # Add logging to track connection progress
    #         logging.info(f"Attempting to connect to MongoDB database: {database_name}, collection: {collection_name}")
            
    #         # Test connection before proceeding
    #         self.mongo_client.admin.command('ping')
    #         logging.info("Successfully connected to MongoDB")
            
    #         # Use cursor with batch size to avoid loading everything at once
    #         cursor = collection.find({}, batch_size=1000)
    #         df = pd.DataFrame(list(cursor))
            
    #         print(f"Fetched {len(df)} records from MongoDB")
    #         logging.info(f"Fetched {len(df)} records from MongoDB")
            
    #         if df.empty:
    #             raise ValueError("MongoDB Collection is empty! Please check data ingestion.")
                
    #         if "_id" in df.columns.to_list():
    #             df = df.drop(columns=['_id'])  # Note: corrected to assign the result back to df
                
    #         df.replace({"na": np.nan}, inplace=True)
                
    #         return df
            
    #     except pymongo.errors.NetworkTimeout as e:
    #         logging.error(f"MongoDB connection timed out: {str(e)}")
    #         raise NetworkSecurityException(f"MongoDB connection timed out. Please check your network connection and MongoDB availability: {e}", sys)
    #     except Exception as e:
    #         logging.error(f"Error in export_collection_as_dataframe: {str(e)}")
    #         raise NetworkSecurityException(e, sys)
    #     finally:
    #         # Ensure connection is closed even if an error occurs
    #         if hasattr(self, 'mongo_client'):
    #             self.mongo_client.close()
    #             logging.info("MongoDB connection closed")

    def export_collection_as_dataframe(self):
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            
            # Connection string with parameters
            connection_string = MONGO_DB_URL
            
            self.mongo_client = pymongo.MongoClient(connection_string)
            collection = self.mongo_client[database_name][collection_name]
            
            logging.info(f"Connected to MongoDB. Retrieving data in chunks...")
            
            # Get total document count (use a lightweight operation)
            total_count = collection.count_documents({})
            logging.info(f"Total documents to retrieve: {total_count}")
            
            # Retrieve data in chunks
            chunk_size = 1000
            all_data = []
            
            # Use limit and skip for pagination
            for i in range(0, total_count, chunk_size):
                chunk = list(collection.find().skip(i).limit(chunk_size))
                all_data.extend(chunk)
                logging.info(f"Retrieved chunk {i//chunk_size + 1}, documents {i} to {min(i+chunk_size, total_count)}")
            
            df = pd.DataFrame(all_data)
            
            print(f"Fetched {len(df)} records from MongoDB")
            logging.info(f"Fetched {len(df)} records from MongoDB")
            
            if df.empty:
                raise ValueError("MongoDB Collection is empty! Please check data ingestion.")
                
            if "_id" in df.columns.to_list():
                df = df.drop(columns=['_id'])
                
            df.replace({"na": np.nan}, inplace=True)
                
            return df
                
        except Exception as e:
            logging.error(f"Error in export_collection_as_dataframe: {str(e)}")
            raise NetworkSecurityException(e, sys)
        finally:
            if hasattr(self, 'mongo_client'):
                self.mongo_client.close()
        
    def export_data_into_feature_store(self,dataframe: pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            #Creating folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        

    def split_data_as_train_test(self,dataframe: pd.DataFrame):
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size = self.data_ingestion_config.train_test_split_ratio
            ) 
            logging.info("Performed train test split on dataframe")

            logging.info(
                'Excited split_data_as_train_test method od Data_Ingestion class'
            )

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)

            os.makedirs(dir_path,exist_ok=True)

            logging.info("Exporting train and test file path")

            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False,header=True
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False,header=True
            )
            logging.info("Exporting train and test file path")


        except Exception as e:
            raise NetworkSecurityException(e,sys)

    # def initiate_data_ingestion(self):
    #     try:
    #         dataframe = self.export_collection_as_dataframe()
    #         dataframe = self.export_data_into_feature_store(dataframe)
    #         self.split_data_as_train_test(dataframe)
    #         dataingestionartifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
    #                                                       test_file_path=self.data_ingestion_config.testing_file_path)
    #         return dataingestionartifact
    #     except Exception as e:
    #         raise NetworkSecurityException(e,sys)

    def initiate_data_ingestion(self):
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                logging.info(f"Data ingestion attempt {attempt + 1} of {max_retries}")
                dataframe = self.export_collection_as_dataframe()
                dataframe = self.export_data_into_feature_store(dataframe)
                self.split_data_as_train_test(dataframe)
                
                dataingestionartifact = DataIngestionArtifact(
                    trained_file_path=self.data_ingestion_config.training_file_path,
                    test_file_path=self.data_ingestion_config.testing_file_path
                )
                
                logging.info("Data ingestion completed successfully")
                return dataingestionartifact
                
            except pymongo.errors.NetworkTimeout:
                if attempt < max_retries - 1:
                    logging.warning(f"Timeout occurred, retrying in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                else:
                    logging.error("Maximum retry attempts reached, giving up.")
                    raise
            except Exception as e:
                raise NetworkSecurityException(e, sys)