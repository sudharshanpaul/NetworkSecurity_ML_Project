import sys
import os
import numpy
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

from networksecurity.constant.training_pipeline import TARGET_COLUMN
from networksecurity.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.utils.main_utils.utils import save_numpy_array_data,save_object

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact = data_validation_artifact
            self.data_transformation_config: DataTransformationConfig=data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys) 
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)  
        
    # def get_data_transformer_object(cls)->Pipeline:
    #     """
    #     It initializes a KNN Imputer object with the parameters specified in the training_pipeline.py file and
    #     returns a Pipeline object with the KNN Imputer object as the first step.

    #     Args:
    #         cls: DataTransformation
        
    #     returns:
    #         A Pipeline object 
    #     """
    #     logging.info("Entered get_data_transformer_object method of transformation class")

    #     try:
    #         imputer: KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
    #         logging.info(f"Initialize KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}")
    #         processor:Pipeline = Pipeline([('imputer',imputer)])
    #         return processor
    #     except Exception as e:
    #         raise NetworkSecurityException(e, sys) 



    def get_data_transformer_object(self) -> Pipeline:
        """
        Returns a transformation pipeline that:
        - Applies StandardScaler to numerical features
        - Applies Label Encoding to categorical features
        """
        try:
            # numerical_columns = ['duration', 'src_bytes', 'dst_bytes']  
            categorical_columns = ['protocol_type', 'service', 'flag'] 
            numerical_columns = ['duration',
                 'src_bytes',
                 'dst_bytes',
                 'land',
                 'wrong_fragment',
                 'urgent',
                 'hot',
                 'num_failed_logins',
                 'logged_in',
                 'num_compromised',
                 'root_shell',
                 'su_attempted',
                 'num_root',
                 'num_file_creations',
                 'num_shells',
                 'num_access_files',
                 'num_outbound_cmds',
                 'is_host_login',
                 'is_guest_login',
                 'count',
                 'srv_count',
                 'serror_rate',
                 'srv_serror_rate',
                 'rerror_rate',
                 'srv_rerror_rate',
                 'same_srv_rate',
                 'diff_srv_rate',
                 'srv_diff_host_rate',
                 'dst_host_count',
                 'dst_host_srv_count',
                 'dst_host_same_srv_rate',
                 'dst_host_diff_srv_rate',
                 'dst_host_same_src_port_rate',
                 'dst_host_srv_diff_host_rate',
                 'dst_host_serror_rate',
                 'dst_host_srv_serror_rate',
                 'dst_host_rerror_rate',
                 'dst_host_srv_rerror_rate',
                 'level']  

            # Define the transformers
            numerical_transformer = Pipeline(steps=[
                ("imputer", KNNImputer(n_neighbors=5)),  # KNN Imputer for missing values
                ("scaler", StandardScaler())  # StandardScaler after imputing
            ])

            categorical_transformer = OneHotEncoder(handle_unknown="ignore")  # OneHotEncoder for categorical features

            # Create column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numerical_transformer, numerical_columns),
                    ("cat", categorical_transformer, categorical_columns),
                ]
            )

            # Create pipeline
            processor = Pipeline(steps=[("preprocessor", preprocessor)])
            return processor

        except Exception as e:
            raise NetworkSecurityException(e, sys)


        
    # def initiate_data_transformation(self)->DataTransformationArtifact:
    #     logging.info('Entered initiate_data_transformation method of DataTransformation class')
    #     try:
    #         logging.info('Starting Data Transformation')
    #         train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
    #         test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

    #         train_df.loc[train_df[TARGET_COLUMN] == "normal", "attack"] = 'normal'
    #         train_df.loc[train_df[TARGET_COLUMN] != 'normal', "attack"] = 'attack'

    #         test_df.loc[test_df[TARGET_COLUMN] == "normal", "attack"] = 'normal'
    #         test_df.loc[test_df[TARGET_COLUMN] != 'normal', "attack"] = 'attack'

    #         ## Training Dataframe
    #         input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN],axis=1)
    #         target_feature_train_df = train_df[TARGET_COLUMN]

    #         target_feature_train_df = target_feature_train_df.replace({'normal':0,'attack':1})

    #         ##Testing DataFrame
    #         input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN],axis=1)
    #         target_feature_test_df = test_df[TARGET_COLUMN]

    #         target_feature_test_df = target_feature_test_df.replace({'normal':0,'attack':1})

    #         preprocessor = self.get_data_transformer_object()
    #         preprocessor_object = preprocessor.fit(input_feature_train_df)
    #         transformed_input_train_feature = preprocessor.transform(input_feature_train_df)
    #         transformed_input_test_feature = preprocessor.transform(input_feature_test_df)
            
    #         train_arr = np.c_[transformed_input_train_feature,np.array(target_feature_train_df)]
    #         test_arr = np.c_[transformed_input_test_feature,np.array(target_feature_test_df)]

    #         ## save numpy array data
    #         save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,array=train_arr,)
    #         save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,array=test_arr,)
    #         save_object(self.data_transformation_config.transformed_object_file_path,preprocessor_object,)

    #         ## Preparing Artifacts

    #         data_transformation_artifact = DataTransformationArtifact(
    #             transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
    #             transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
    #             transformed_test_file_path= self.data_transformation_config.transformed_test_file_path
    #         )

    #         return data_transformation_artifact

    #     except Exception as e:
    #         raise NetworkSecurityException(e, sys) 

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info('Entered initiate_data_transformation method of DataTransformation class')
        try:
            logging.info('Starting Data Transformation')

            # Read training and testing data
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # Convert target column values
            train_df["attack"] = train_df[TARGET_COLUMN].apply(lambda x: 'normal' if x == 'normal' else 'attack')
            test_df["attack"] = test_df[TARGET_COLUMN].apply(lambda x: 'normal' if x == 'normal' else 'attack')

            # Separate input features and target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN].replace({'normal': 0, 'attack': 1})

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN].replace({'normal': 0, 'attack': 1})

            # Get the preprocessor
            preprocessor = self.get_data_transformer_object()

            # Fit and transform the training data, transform test data
            transformed_input_train_feature = preprocessor.fit_transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor.transform(input_feature_test_df)

            # Convert to NumPy arrays and add target column
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            # Save transformed data
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)

            # Prepare Artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

