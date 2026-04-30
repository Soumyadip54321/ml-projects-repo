'''
Script that chosen a set of input from UI is fed to ML models in the background to run predictions 
and display it back to UI.
'''

import pandas as pd
from joblib import load
import os
# run from project root - Healthcare_Premium. Python then treats it as root and App within it as a package.
# This path ensures model picks up the right transformer used for target encoding the train set
from encoders import oof_cv_target_encoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path_rest = os.path.join(BASE_DIR, "Artifacts", "bestmodel_fullpipeline_rest.joblib")
model_path_young = os.path.join(BASE_DIR, "Artifacts", "bestmodel_fullpipeline_young.joblib")

# import models with full preprocessing pipelines
model_rest = load(model_path_rest)
model_young = load(model_path_young)

def predict(input_dict):
    '''
    Function that predicts the healthcare premium amount. It checks for customer age such that
    the optimal performing model is used for inference.
    :param input_dict:
    :return: prediction
    '''

    # convert input to dataframe with attributes in the same order as they appeared in the model training set
    inference_df = pd.DataFrame(data=[input_dict.values()],columns=input_dict.keys())

    # make prediction using most optimal model basis customer Age
    if input_dict['Age']<25:
        prediction = model_young.predict(inference_df)
    else:
        prediction = model_rest.predict(inference_df)

    return prediction.item()
