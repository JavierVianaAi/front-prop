import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import algo_bottom as algo_bottom

import tensorflow as tf
from tensorflow.keras.models import model_from_json

import torch

# ------------------------------------------------------------------------------
# Specs needed based on what we upload:

# Which framework we are using:
framework = "tensorflow" # "tensorflow" "pytorch"

# For tensorflow:
tensorflow_file_format = "json_h5"  # "h5"

# For Pytorch:
pytorch_file_format = "pt"  # "pth"

# ------------------------------------------------------------------------------
# Load the Model of the User

# If tensorflow:
if framework == "tensorflow":

  # If .json and .h5 format:
  if tensorflow_file_format == "json_h5":

    # Load json and create model
    with open('user_model_structure.json', 'r') as json_file:
        loaded_user_model_in_json = json_file.read()
    # Model from json:
    loaded_user_model = model_from_json(loaded_user_model_in_json)
    # Load weights into the new model
    loaded_user_model.load_weights("user_model_parameters.h5")

  # If .h5 format:
  elif tensorflow_file_format == "h5":
    pass # TODO


# If Pytorch:
elif framework == "pytorch":

  # If .pt format:
  if pytorch_file_format == "pt":
    # Read the model - It cannot be TorchScript saved model:
    # Hooks are not allowed on top (we need them to get interm. outs in the best way).
    # And conversion from TorchScript to nn.model is not supported by Pytorch.
    loaded_user_model = torch.load('user_model.pt')
    loaded_user_model.eval()

  # If .pth format:
  elif pytorch_file_format == "pth":
    pass # TODO


# ------------------------------------------------------------------------------
# Prepare the Model for the Explanation Flow 

# Get all the information of the model, we only do this once:
specs, arr_layers_info, ref_of_fake_layers, associated_true_layer_index = algo_bottom.study_model(loaded_user_model, framework)

# ------------------------------------------------------------------------------
# The User Submits a Request with New Input Data
df = pd.read_csv('input_data_for_the_request.csv', index_col=0, header=0)
# We extract the matrix X, this step might be a little different depending on how we transfer the data through the API:
X = df.values

# ------------------------------------------------------------------------------

# The names of the input dimensions:
param_names = ['Variable ' + str(i + 1) for i in range(X.shape[1])]

# ------------------------------------------------------------------------------

# For visibility:
print("Shape of the Inputs: ", X.shape)
print("Specs of the Model: ")
print("   Design Framework: ", specs.framework)
print("   Num. of Inputs: ", specs.n_ins)
print("   Num. of Outputs: ", specs.n_outs)
print("   Num. of True Layers: ", specs.true_n_lays)
print("   Num. of All Layers (fake and true): ", specs.all_n_lays)

# ------------------------------------------------------------------------------
# Prediction Flow
# Predict the outputs with the loaded_model for the training set:
y_pred_NN = algo_bottom.get_NN_output(X, loaded_user_model, specs)

# ------------------------------------------------------------------------------
# Explanation Flow
output_explanations = algo_bottom.run_explanations(X, y_pred_NN, loaded_user_model, specs, arr_layers_info,
                                                  ref_of_fake_layers, associated_true_layer_index, param_names) 

# ------------------------------------------------------------------------------
# Return the output explanations:
def return_output_explanations():
  return output_explanations

# ------------------------------------------------------------------------------
# Display the results for a given i_instance:
def display_explanation_of_instance(i_instance):
  print("Input Values: ", output_explanations[i_instance].input_values)
  print("Output Pred. by NN: ", output_explanations[i_instance].neural_network_output)
  print("Output Pred. by LF: ", output_explanations[i_instance].linear_function_output)
  for i in range(specs.n_outs):
    print("  For Output Dimension ", i+1)
    print("     Linear Func. Dep. Term: ", output_explanations[i_instance].input_dependencies[i])
    print("     Linear Func. Indep. Term: ", output_explanations[i_instance].independent_term[i])
    print("     Input Contributions: ", output_explanations[i_instance].input_contributions[i])
    print("     Text Explanations: ")
    print("         ", output_explanations[i_instance].text_explanation[i][0])
    print("         ", output_explanations[i_instance].text_explanation[i][1])
    print("         ", output_explanations[i_instance].text_explanation[i][2])
  print("Reference ID: ", output_explanations[i_instance].reference_id)

  return


# ------------------------------------------------------------------------------
# Visualization of the linear function prediction vs NN prediction:
# Function to compare the predictions of a linear function prediction vs NN prediction:
# Method to see how good the explanation is, ideally the scatterplot should be tangent to the 1-1 line.
def plot_explanation_performance(i_instance, M, proxim_thres, flag_normalization, which_out_dims=[0]):
  """ 
      i_instance: Index of the instance of the request that we want to evaluate.

      M: Number of points desired to plot in the vicinity of the instance.
      proxim_thres: Allowed maximum difference in every dimension to generate the sample.
      flag_normalization: Boolean to know if the input vector is normalized. If so, we make sure
                         that the random points generated are within the limits of 0 and 1.

      which_out_dims: To know which output dimesnions we want to plot. Could be "all" or
                      an array with the indexes of the out dims desired to plot. 
                      For example [0,3,5]. Indexes start at 0.     
  """

  # Get the information of this instance:
  inst = output_explanations[i_instance].input_values
  arr_LF_deps = output_explanations[i_instance].input_dependencies
  arr_LF_indep = output_explanations[i_instance].independent_term
  arr_out_pred_LF = output_explanations[i_instance].linear_function_output
  arr_out_pred_NN = output_explanations[i_instance].neural_network_output

  # Run the plots generator:
  algo_bottom.plot_LF_vs_NN_predictions(inst, arr_LF_deps, arr_LF_indep, arr_out_pred_LF, arr_out_pred_NN, 
                                        loaded_user_model, specs, M, proxim_thres, flag_normalization, which_out_dims)

  return










