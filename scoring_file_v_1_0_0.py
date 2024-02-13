# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Score text dataset from model produced by training run."""

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType
from azureml.automl.core import inference
from azureml.automl.dnn.nlp.common.constants import ScoringLiterals, OutputLiterals

import numpy as np
import pandas as pd
import json
import pickle
import os

data_sample = PandasParameterType(pd.DataFrame({"Question": pd.Series(["example_value"], dtype="object")}))
input_sample = StandardPythonParameterType({'data': data_sample})
method_sample = StandardPythonParameterType("predict")
sample_global_params = StandardPythonParameterType(1.0)

result_sample = NumpyParameterType(np.array(["example_value"]))
output_sample =  StandardPythonParameterType({'Results':result_sample})


def init():
    """This function is called during inferencing environment setup and initializes the model"""
    global model
    model_path = os.path.join(os.getenv(ScoringLiterals.AZUREML_MODEL_DIR_ENV), OutputLiterals.MODEL_FILE_NAME)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

@input_schema('GlobalParameters', sample_global_params, convert_to_provided_type=False)
@input_schema('Inputs', input_sample)
def run(Inputs, GlobalParameters=1.0) -> str:
    """ This is called every time the endpoint is invoked. It returns the prediction of the input data

    :param data: input data provided by the user
    :type data: pd.DataFrame
    :return: json string of the result
    :rtype: str
    """
    data = Inputs["data"]
    fin_outputs = model.predict_proba(data)
    classes = model.classes_
    concatenated_lists = [list(zip(classes, [str(ind) for ind in single_output])) for single_output in fin_outputs]
    return concatenated_lists