from urllib.request import install_opener
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import copy

import tensorflow as tf
from tensorflow.keras.models import Model

import torch
import torch.nn as torchnn

# ------------------------------------------------------------------------------------------------------
# Function to just predict:
def get_NN_output(X, main_model, specs):

  # If tensorflow:
  if specs.framework == "tensorflow":
    # Predict:
    y_pred_NN = main_model.predict(X)

  # If Pytorch:
  elif specs.framework == "pytorch":
    # Conver to torch tensor:
    torch_X = torch.tensor(X, dtype=torch.float32)
    # Predict - Detach to remove grad so we can convert the tensor to numpy:
    y_pred_NN = main_model(torch_X).detach().numpy()

  return y_pred_NN

# ------------------------------------------------------------------------------------------------------
# The class for the specifications of the model:
class SpecsClass:
  """ 
      framework: string that can have the names: tensorflow or pytorch
      n_ins: Number of input dimensions
      n_outs: Number of output dimensions
      true_n_lays: Number of true layers 
      all_n_lays:  Number of all layers, includes fake and true layers (fakes are the afs considered as layers)
  """
  def __init__(self, framework, num_ins, num_outs, true_n_lays, all_n_lays):
    self.framework = framework
    self.n_ins = num_ins
    self.n_outs = num_outs
    self.true_n_lays = true_n_lays
    self.all_n_lays = all_n_lays


# ------------------------------------------------------------------------------------------------------

# The reference to convert the layer type names to our internal standard type names:

# For Tensorflow:
ref_tensorflow_lay_type_conversion = { "Dense": "dense", 
                                       "Dropout": "dropout",
                                       "BatchNormalization": "batch_norm",

                                       "elu": "elu",
                                       "exponential": "exponential",
                                       "gelu": "gelu",
                                       "hard_sigmoid": "hard_sigmoid",
                                       "linear": "linear",
                                       "relu": "relu",
                                       "selu": "selu",
                                       "sigmoid": "sigmoid",
                                       "softmax": "softmax",
                                       "softplus": "softplus",
                                       "softsign": "softsign",
                                       "swish": "swish",
                                       "tanh": "tanh"
                                     }

# For Pytorch:
ref_pytorch_lay_type_conversion = { "Linear": "dense", 
                                    "Dropout": "dropout",
                                    "BatchNorm1d": "batch_norm",

                                    "ELU": "elu",
                                    "GELU": "gelu",
                                    "Hardsigmoid": "hard_sigmoid",
                                    "ReLU": "relu",
                                    "SELU": "selu",
                                    "Sigmoid": "sigmoid",
                                    "Softmax": "softmax",
                                    "Softplus": "softplus",
                                    "Softsign": "softsign",
                                    "Tanh": "tanh"
                                  }

# TODO: You are missing Leaky Relu, and then some others in Pytorch that are very weird.

# ------------------------------------------------------------------------------------------------------

# Useful variable:
arr_all_tf_af_names = ['elu', 'exponential', 'gelu', 'hard_sigmoid', 'linear', 'relu',
                       'selu', 'sigmoid', 'softplus', 'softsign',  'swish', 'tanh']

# ------------------------------------------------------------------------------------------------------
# The classes for the different types of layers:

# Dense:
class DenseObj:
  """ 
      laytype: 'non_af'
      name: 'dense'
      ws: weights
      bs: biases
      use_bias: Boolean. "use_bias" in tensorflow
  """
  def __init__(self, ws, bs, use_bias):
    self.laytype = 'non_af'
    self.name = 'dense'
    self.ws = ws
    self.bs = bs
    self.use_bias = use_bias 

# Droput:
class DropoutObj:
  """ 
      laytype: 'non_af'
      name: 'dropout'
  """
  def __init__(self):
    self.laytype = 'non_af'
    self.name = 'dropout'

# Batch Normalization:
class BatchNormObj:
  """ 
      laytype: 'non_af'
      name: 'batch_norm'
      use_beta: Boolean. "center" in tensorflow
      use_gamma: Boolean. "scale" in tensorflow
  """
  def __init__(self, moving_mean, moving_var, beta, gamma, epsilon, use_beta, use_gamma):
    self.laytype = 'non_af'
    self.name = 'batch_norm'
    self.moving_mean = moving_mean
    self.moving_var = moving_var
    self.beta = beta
    self.gamma = gamma
    self.epsilon = epsilon
    self.use_beta = use_beta
    self.use_gamma = use_gamma
    

# Generic Activation Function:
# IDEALLY WE WOULD NOT HAVE A GENERIC AfObj BUT RATHER ONE PER EACH TYPE OF AF WITH THE PARAMETERES 
class AfObj:
  """ 
      laytype: 'af'
      name: the name of the af
  """
  def __init__(self, name):
    self.laytype = 'af'
    self.name = name

# ------------------------------------------------------------------------------------------------------
# Function to retrieve the information of the model:
def study_model(main_model, framework):
  """ 
      Loop through the layers of the base model, sequentially create new models, 
      get all the info of the activation funcs, get the arrays of weights and biases:
  """

  # If tensorflow ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if framework == "tensorflow":
    
    # The main model:
    main_model_layers = main_model.layers

    # Get the types of the layers
    layer_types = [layer.__class__.__name__ for layer in main_model_layers]

    # Convert these types names to our internal standard (std) names of types:
    std_layer_types = [ ref_tensorflow_lay_type_conversion[type_name] for type_name in layer_types ]

    # Array with the string names of the layer types or activation functions:
    arr_layers_info = {}
    # Counter for all the layers. Remember, in reality we have less layers, since all includes the fakes,
    # this is because we are going to consider activation functions as different layers:
    all_lays_count = 0
    # Reference to keep track of those layers that we created, the fake layers:
    ref_of_fake_layers = {}
    # Reference to keep track of the real indexes of the layers within fake that are true:
    associated_true_layer_index = {}

    # Loop through the layers:
    for i, layer in enumerate(main_model_layers):
      # Get the name of the layer:
      lay_name = std_layer_types[i]

      # If it is Dense:
      if lay_name == 'dense':
        # Get the weights and biases:
        ws_and_bs = layer.get_weights()
        # Extract the weights:
        lay_weights = ws_and_bs[0].transpose()
        # Extract the biases:
        if layer.use_bias:
          lay_biases = ws_and_bs[1]
        else:
          lay_biases = None
        # Now save the information of the dense layer:
        arr_layers_info[all_lays_count] = DenseObj(lay_weights, lay_biases, layer.use_bias)
        # Update the reference of fakes and the associated true layer index:
        ref_of_fake_layers[all_lays_count] = 'fake'
        associated_true_layer_index[all_lays_count] = None
        # Increase layer counter:
        all_lays_count += 1

        # Append as a new layer the activation function associated to the dense layer:
        #   TODO: The AF can have some parameters depending on the AF, get them and store in the object.
        arr_layers_info[all_lays_count] = AfObj(tf.keras.activations.serialize(layer.activation))
        # Update the reference of fakes and the associated true layer index:
        ref_of_fake_layers[all_lays_count] = 'non_fake'
        associated_true_layer_index[all_lays_count] = i
        # Increase layer counter:
        all_lays_count += 1


      # If it is Dropout:
      elif lay_name == "dropout":
        # Save the information of the dense layer:
        arr_layers_info[all_lays_count] = DropoutObj()
        # Update the reference of fakes and the associated true layer index:
        ref_of_fake_layers[all_lays_count] = 'non_fake'
        associated_true_layer_index[all_lays_count] = i
        # Increase layer counter:
        all_lays_count += 1
        # Update the current number of outputs:
        num_outs = layer.output_shape[1]


      # If it is Batch Normalization:
      elif lay_name == "batch_norm":
        # Get beta:
        if layer.center:
          lay_beta = layer.beta.numpy()
        else:
          lay_beta = None
        # Get gamma:
        if layer.scale:
          lay_gamma = layer.gamma.numpy()
        else:
          lay_gamma = None
        # Save the information of the dense layer:
        arr_layers_info[all_lays_count] = BatchNormObj(layer.moving_mean.numpy(), layer.moving_variance.numpy(),
                                                       lay_beta, lay_gamma, layer.epsilon,
                                                       layer.center, layer.scale)
        # Update the reference of fakes and the associated true layer index:
        ref_of_fake_layers[all_lays_count] = 'non_fake'
        associated_true_layer_index[all_lays_count] = i
        # Increase layer counter:
        all_lays_count += 1
        # Update the current number of outputs:
        num_outs = layer.output_shape[1]
        

      # If it is just an activation function:
      # TODO: You could increase runtime avoiding comparison with all names, and rather store info of af earlier...
      elif lay_name in arr_all_tf_af_names:
        # Append the activation function as a new layer:
        #   TODO: The AF can have some parameters depending on the AF, get them and store in the object.
        arr_layers_info[all_lays_count] = AfObj(lay_name)
        # Update the reference of fakes and the associated true layer index:
        ref_of_fake_layers[all_lays_count] = 'non_fake'
        associated_true_layer_index[all_lays_count] = i
        # Increase layer counter:
        all_lays_count += 1

    # Number of layers, number of inputs, number of outputs:
    num_ins = main_model_layers[0].input_shape[1]
    num_outs = main_model_layers[-1].output_shape[1]
    true_num_lays = len(main_model_layers)
    # Generate the specs object:
    specs = SpecsClass(framework, num_ins, num_outs, true_num_lays, all_lays_count)
    


  # If pytorch ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif framework == "pytorch":

    # Pytorch models have 4 important fields that you should know about for looping through layers:
    #    .modules()  - Info of the Sequential and the layers inside (ej. Linear, ReLu, etc.)
    #    .children() - Info of the inner layers (ej. Linear, ReLu, etc.)
    #    .named_modules()   -  Same as .modules() but also with the name. So when looping, you get name & module
    #    .named_children()  -  Same as .children() but also with the name. So when looping, you get name & child  

    
    # Get the types of the layers --------------------------------------------------------------------
    temp_layer_types = [layer_i._get_name() for layer_i in main_model.modules()]

    # If the we are working with a Sequential model:
    if temp_layer_types[0] == "Sequential":
      # Exclude the Sequential object and keep the rest:
      layer_types = temp_layer_types[1:]

    # If we are working with something else you would need to play with .modules and .children.
    else:
      # TODO: You will also need to support sub Sequential models inside the Sequential:
      pass


    # Convert these types names to our internal standard (std) names of types:
    std_layer_types = [ ref_pytorch_lay_type_conversion[type_name] for type_name in layer_types ]


    # Array with the string names of the layer types or activation functions:
    arr_layers_info = {}
    # Initialize all_lays_count, although in Pytorch yet we don't support fake layers yet,
    # but to be coherent with tensorflow. Also, in Pytorch is less frequent the need of fake layers.
    all_lays_count = 0
    # Reference to keep track of those layers that we created, the fake layers:
    ref_of_fake_layers = {}
    # Reference to keep track of the real indexes of the layers within fake that are true:
    associated_true_layer_index = {}


    # To get the number of outputs of the model is a little tricky since an AF layer in pytorch does not have
    # the .out_features or .in_features field. So we cannot look at the last layer, it might not have that info.
    # To solve it we just keep updating the num_outs variable every time we see a non-af layer

    # Get the weights and biases 
    for i, child in enumerate(main_model.children()):
      # Get the name of the layer:
      lay_name = std_layer_types[i]

      # If it is Dense:
      if lay_name == "dense":
        # Get the weights. Detach is used because you cannot convert to numpy a tensor that requires grad.
        lay_weights = child.weight.detach().numpy()
        # Get the biases:
        if child.bias is not None:
          lay_biases = child.bias.detach().numpy()
          lay_use_bias = True
        else:
          lay_biases = None
          lay_use_bias = False
        # Now save the information of the dense layer:
        arr_layers_info[all_lays_count] = DenseObj(lay_weights, lay_biases, lay_use_bias)
        # Update the reference of fakes and the associated true layer index:
        ref_of_fake_layers[all_lays_count] = 'non_fake'
        associated_true_layer_index[all_lays_count] = i
        # Increase layer counter:
        all_lays_count += 1
        # Update the current number of outputs:
        num_outs = child.out_features

      # If it is Dropout:
      elif lay_name == "dropout":
        # Save the information of the dense layer:
        arr_layers_info[all_lays_count] = DropoutObj()
        # Update the reference of fakes and the associated true layer index:
        ref_of_fake_layers[all_lays_count] = 'non_fake'
        associated_true_layer_index[all_lays_count] = i
        # Increase layer counter:
        all_lays_count += 1
        # TODO: I was not able to find the number of outputs in a dropout layer of pytorch.
        # I don't think however it would be done in the last layer of the net, but just in case correct it.
        # num_outs 

      # If it is Batch Normalization:
      elif lay_name == "batch_norm":
        # Save the information of the dense layer:
        arr_layers_info[all_lays_count] = BatchNormObj(child.running_mean.detach().numpy(), 
                            child.running_var.detach().numpy(), child.bias.detach().numpy(), 
                            child.weight.detach().numpy(), child.eps,
                            True, True) # Pytorch does not allow the option of not using beta or gamma, so True both.
        # Update the reference of fakes and the associated true layer index:
        ref_of_fake_layers[all_lays_count] = 'non_fake'
        associated_true_layer_index[all_lays_count] = i
        # Increase layer counter:
        all_lays_count += 1
        # Update the current number of outputs:
        num_outs = child.num_features
        

      # If it is just an activation function:
      # TODO: You could increase runtime avoiding comparison with all names, and rather store info of af earlier...
      elif lay_name in arr_all_tf_af_names:
        # Append the activation function as a new layer:
        #   TODO: The AF can have some parameters depending on the AF, get them and store in the object.
        #   I believe from child.parameters() but maybe there is a special field for each case.
        arr_layers_info[all_lays_count] = AfObj(lay_name)
        # Update the reference of fakes and the associated true layer index:
        ref_of_fake_layers[all_lays_count] = 'non_fake'
        associated_true_layer_index[all_lays_count] = i
        # Increase layer counter:
        all_lays_count += 1
        

    # Useful:
    main_model_layers = { i: child for i,child in enumerate(main_model.children()) }

    # Number of layers, number of inputs, number of outputs:
    num_ins = main_model_layers[0].in_features
    true_num_lays = len(main_model_layers.keys())
    # num_outs is already obtained in the looping.

    # Generate the specs object:
    specs = SpecsClass(framework, num_ins, num_outs, true_num_lays, all_lays_count)

    
  # TODO: Group ref_of_fake_layers & associated_true_layer_index in a class to look better
  return specs, arr_layers_info, ref_of_fake_layers, associated_true_layer_index


# ------------------------------------------------------------------------------------------------------
# Derivatives of Activation Functions, returns u (dep. term) and v (indep. term) of the derivative (except softmax):

# NOTE A A A A A A A:
# For all of our derivatives of the activations functions, both af_in and af_out must be numpy arrays
# But to run  the activation functions of tensorflow (not the derivative) some could be np arrays, others must be tensor.
# like in hardsigmoid

# Elu:
def af_elu(af_in, af_out, alpha=1):
  # Initialize:
  u = af_out + alpha
  v = af_out - u*af_in
  # Find the places where af_in is >0:
  arr_bool = af_in > 0
  # Replace:
  u[arr_bool] = 1
  v[arr_bool] = 0
  return u, v


# Exponential:
def af_exp(af_in, af_out):
  u = af_out
  v = af_out - u*af_in
  return u, v


# Gelu:
def af_gelu(af_in, af_out):
  u = af_out/af_in + (af_in/(np.sqrt(2*np.pi))) * np.exp( - (af_in**2)/2 )
  v = af_out - u*af_in
  return u, v


# Hard Sigmoid:
def af_hardsigmoid(af_in, af_out):
  # Initialize:
  u = 0.2 * np.ones(af_in.shape)
  v = 0.5 * np.ones(af_in.shape)
  # Where in between:
  arr_bool_low = af_in < -2.5
  arr_bool_hig = af_in >  2.5
  # Replace:
  u[arr_bool_low] = 0
  v[arr_bool_low] = 0
  u[arr_bool_hig] = 0
  v[arr_bool_hig] = 1
  return u, v


# Linear:
def af_linear(af_in, af_out):
  u = np.ones(af_in.shape)
  v = np.zeros(af_in.shape)
  return u, v


# Relu:
def af_relu(af_in, af_out, alpha=0, max_value=np.inf, threshold=0):
  # Initialize:
  u =  alpha * np.ones(af_in.shape)
  v = -alpha * threshold * np.ones(af_in.shape)
  # Where in between:
  arr_bool_thres = threshold < af_in
  arr_bool_maxvl = af_out >= max_value
  # Replace:
  u[arr_bool_thres] = 1
  v[arr_bool_thres] = 0
  u[arr_bool_maxvl] = 0
  v[arr_bool_maxvl] = max_value
  return u, v


# Selu:
def af_selu(af_in, af_out):
  # alpha and scale are pre-defined constants:
  alpha=1.67326324
  scale=1.05070098
  # Initialize:
  u =  scale*alpha*np.exp(af_in)
  v =  af_out - u*af_in
  # Where in between:
  arr_bool = af_in>=0
  # Replace:
  u[arr_bool] = scale
  v[arr_bool] = 0
  return u, v


# Sigmoid:
def af_sigmoid(af_in, af_out):
  u = af_out*(1-af_out)
  v = af_out - u*af_in
  return u, v


# Softmax (not normal af, uses info of all the activation functions in the layer):
def afs_softmax(mat_m_x_w, rowvec_afs_outs, rowvec_x):

  # Transform the outs of the activation functions to a column vector:
  colvec_afs_outs = rowvec_afs_outs.reshape((-1, 1))

  # Get the matrix of slopes for the outputs:
  sum_of_prod_1 = np.sum(np.multiply(mat_m_x_w, colvec_afs_outs), axis =0)
  diff_1 = np.subtract(mat_m_x_w, sum_of_prod_1)
  new_mat_slopes = np.multiply(diff_1, colvec_afs_outs)

  # Get the vector of independent terms for the outputs:
  sum_of_prod_2 = np.sum(np.multiply(new_mat_slopes, rowvec_x), axis=1).reshape((-1, 1))
  new_colvec_indeps = np.subtract(colvec_afs_outs, sum_of_prod_2)

  return new_mat_slopes, new_colvec_indeps


# Softplus:
def af_softplus(af_in, af_out):
  u = 1/(1+np.exp(-af_in))
  v = af_out - u*af_in
  return u, v


# Softsign:
def af_softsign(af_in, af_out):
  u = (1+np.abs(af_in))**(-2)
  v = af_out - u*af_in
  return u, v


# Swish:
def af_swish(af_in, af_out):
  u = (af_out/af_in) * (1+af_in-af_out)
  v = af_out - u*af_in
  return u, v


# Tanh:
def af_tanh(af_in, af_out):
  u = (np.cosh(af_in))**(-2)
  v = af_out - u*af_in
  return u, v


# ------------------------------------------------------------------------------------------------------
# Function to decide which activation function to use:
def get_uv(af_type, af_in, af_out):
      
  if af_type == 'elu':
    return af_elu(af_in, af_out) # , alpha)

  elif af_type=='exponential':
    return af_exp(af_in, af_out)

  elif af_type=='gelu':
    return af_gelu(af_in, af_out)

  elif af_type=='hard_sigmoid':
    return af_hardsigmoid(af_in, af_out)

  elif af_type=='linear':
    return af_linear(af_in, af_out)

  elif af_type=='relu':
    return af_relu(af_in, af_out) # , alpha, max_value, threshold)

  elif af_type=='selu':
    return af_selu(af_in, af_out)

  elif af_type=='sigmoid':
    return af_sigmoid(af_in, af_out)

  elif af_type=='softplus':
    return af_softplus(af_in, af_out)

  elif af_type=='softsign':
    return af_softsign(af_in, af_out)

  elif af_type=='swish':
    return af_swish(af_in, af_out)

  elif af_type=='tanh':
    return af_tanh(af_in, af_out)

  # Allow users to have their own activation functions if they provide the code and the differentiation?
  # For now its okay.
  else:
    print(f"The activation function {af_type} was not found")

# ------------------------------------------------------------------------------------------------------
# This is the fundamental function:
def core_algorithm(specs, arr_layers_info, outs_layers):
  """ 
      Bias:  what the nn model has in each neuron, trained variables.
      Independent term: what is carried from before, part of the explanation.

      specs: Contains the number of input dimensions, layers and output dimensions.
      arr_layers_info: Array where every entry has laytype (non_af or af) and name (dense, relu, softmax, etc.)

      outs_layers: The intermediate outputs of all the layers, including the fake layers.
                   The first entry contains the input.
  """

  # Create a ranger for the layers:
  ranger_layers = range(specs.all_n_lays)
  # Number of input dimensions:
  nn = specs.n_ins 
  # Number of instances:
  num_instances = outs_layers[-1].shape[0]

  # Initialize all the dicts:
  dict_mats_LF_dep_terms = {}
  dict_mats_LF_indep_term = {}
  dict_mats_input_contribs = {}
  dict_outputs_pred_LF = {}

  # Loop through the instances, i is the index of the ith instance:
  for ith in range(num_instances):

    # The input of the entire network:
    rowvec_x = outs_layers[-1][ith,:]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # SLOPES AND INDEPS 
    # Get the mat of arrays with the input contributions for each output:

    # Initialize the first entry:
    mat_slopes = np.array(np.matrix(np.eye(nn)))
    colvec_indep_terms = np.array([0 for _ in range(nn) ]).reshape(nn,1)

    # Go through the layers: 
    for layer_l in ranger_layers:

      # If we have a layer that is not an activation function:
      if arr_layers_info[layer_l].laytype == 'non_af':

        # If dense:
        if arr_layers_info[layer_l].name == 'dense':

          # kk: num of neurons previous layer
          # qq: num of neurons current layer
          kk = arr_layers_info[layer_l].ws.shape[1]
          qq = arr_layers_info[layer_l].ws.shape[0]

          # WEIGHTS:
          # Reformat the weights matrix for the dot product:
          rf_mat_weights = np.reshape( arr_layers_info[layer_l].ws, (qq, 1, kk) )

          # SLOPES:
          # Reformat the slopes matrix, so it has a 3d shape of (1, k, n).
          rsp_mat_slopes = np.reshape(mat_slopes, (1, kk, nn))
          # Dot product, avoid the 3D mat of the product, takes huge space.
          mat_slopes = np.dot(rf_mat_weights, rsp_mat_slopes).reshape((qq,nn))

          # INDEPS:
          # Reformat the colvec indeps, so it has a 3d shape of (1, k, 1).
          rsp_colvec_indeps = np.reshape(colvec_indep_terms, (1, kk, 1))
          # Initialize colvec_indep_terms, we consider later the addition of the bias.
          # Use the dot product, avoid the product and then summing, takes huge space:
          colvec_indep_terms = np.dot(rf_mat_weights, rsp_colvec_indeps).reshape(qq,1)

          # BIASES:
          # Check if using:
          if arr_layers_info[layer_l].use_bias:
            # Reformat the biases vector so its colvec, not needed to be 3d:
            colvec_biases = np.reshape( arr_layers_info[layer_l].bs, (qq, 1) )
            # Update the indep terms adding the biases:
            colvec_indep_terms += colvec_biases


        # If we have a dropout layer:
        elif arr_layers_info[layer_l].name == 'dropout':
          # During inference dropout should not modify the inputs of the layer:
          pass


        # If we have a batch normalization layer:
        elif arr_layers_info[layer_l].name == 'batch_norm':
          # Refactoring needed for broadcasting correctly:
          aux_epsilon = arr_layers_info[layer_l].epsilon
          aux_mov_mean = arr_layers_info[layer_l].moving_mean.reshape((qq, 1))
          aux_mov_var = arr_layers_info[layer_l].moving_var.reshape((qq, 1))
          # Remember, for the slopes only the derivative, do not use the mov_mean nor the beta.
          # We consider the use of gamma later, once we have the independent terms computed:
          # Use an aux factor to avoid repetition of code:
          aux_factor = 1 / np.sqrt(aux_mov_var + aux_epsilon)
          # Apply:
          mat_slopes *= aux_factor
          # The indep terms. We consider the use of gamma and beta later:
          colvec_indep_terms = (colvec_indep_terms - aux_mov_mean) * aux_factor
          # Consider the use of gamma:
          if arr_layers_info[layer_l].use_gamma:
            # Refactoring needed for broadcasting correctly:
            aux_gamma = arr_layers_info[layer_l].gamma.reshape((qq, 1))
            # Apply:
            mat_slopes *= aux_gamma
            colvec_indep_terms *= aux_gamma
          # Consider the use of beta:
          if arr_layers_info[layer_l].use_beta:
            # Refactoring needed for broadcasting correctly:
            colvec_indep_terms += arr_layers_info[layer_l].beta.reshape((qq, 1))


      # If we have an activation function:
      elif arr_layers_info[layer_l].laytype == 'af':
        
        # qq: num of neurons of current layer 
        # (which should be the same as the previous layer's current since we are now working with afs):
        qq = mat_slopes.shape[0]

        # TODO: I think we could do all the instnaces at once with np dots etc., maybe tricky for CNNs?
        # Ins and outs of this instance:
        rowvec_afs_ins = outs_layers[layer_l-1][ith,:]
        rowvec_afs_outs = outs_layers[layer_l][ith,:]

        # Reformat:
        colvec_afs_ins = np.reshape(rowvec_afs_ins, (qq, 1) )
        colvec_afs_outs =  np.reshape(rowvec_afs_outs, (qq, 1) )
        
        # If the activation function is not softmax:
        if arr_layers_info[layer_l].name != 'softmax':
          # First get the colvec arrays of u and v:
          colvec_u, colvec_v = get_uv(arr_layers_info[layer_l].name, colvec_afs_ins, colvec_afs_outs)
          # Multiply times the colvec u, update the next mat_slopes:
          mat_slopes = mat_slopes * colvec_u
          # Multiply times the colvec u and add v, update the next colvec_indep_terms:
          colvec_indep_terms = colvec_indep_terms * colvec_u + colvec_v

        # If the activation function is softmax:
        else:
          # Get the output slopes matrix and the output indep terms vec:
          mat_slopes, colvec_indep_terms = afs_softmax(mat_slopes, rowvec_afs_outs, rowvec_x)


    # NOTE: For now there is no need to return the entire array of intermediate slopes and indep terms.
    # Only the array with last layer's neurons (there might be more than one output).
    # In the future we would like to know which parts of the net are not useful to suggest simplifications.

    # Store the final slopes and indeps. 
    dict_mats_LF_dep_terms[ith] = mat_slopes
    dict_mats_LF_indep_term[ith] = colvec_indep_terms


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CONTRIBS
    # Get the mat of arrays with the input contributions for each output:

    # Create an array with each individual absolute value of slope*input:
    arr_abs_slopes_x_input = np.absolute(np.multiply(rowvec_x, mat_slopes))

    # Get the sum:
    aux_sum_of_arr = np.sum(arr_abs_slopes_x_input, axis=1).reshape(specs.n_outs,1)

    # Finally the matrix of contributions for all the outputs:
    mat_contribs = arr_abs_slopes_x_input/aux_sum_of_arr

    # Store the contributions:
    dict_mats_input_contribs[ith] = mat_contribs


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # LF PRED OUTS
    # Get the vector of outputs predicted with the LFs for each output dimension:

    # Get the output with all its dimensions. Reshape colvec mat_LF_indep_term_i so its a row vector.
    output_pred_LF = np.dot(mat_slopes,rowvec_x) + colvec_indep_terms.flatten()

    # Store the outputs:
    dict_outputs_pred_LF[ith] = output_pred_LF


  # Note about the entries inside the dicts:  mat_LF_dep_terms,  mat_LF_indep_term,  mat_input_contribs
  #   Their shape is different from the criterion we are following, for faster numpy ops and visibility.
  #   mat_LF_dep_terms is a 2D mat with rows that identify output dimensions, and cols the terms.
  #   The criterion is that the input is a row vector, and the output is also a row vector.
  #   However, when we save in the ExplClass, we will convert to lists, so we will respect criterion.

  return dict_mats_LF_dep_terms, dict_mats_LF_indep_term, dict_mats_input_contribs, dict_outputs_pred_LF 


# ------------------------------------------------------------------------------------------------------
# Function to retrieve the adjective within the ranges record using a value:
def get_adj_from_val(val, ranges):
  
  # A A A A this shouldn't be done, store it as a class so you have it already:
  locs = [ entry[0] for entry in ranges]
  names =[ entry[1] for entry in ranges]

  # Get where it fits the val:
  for ii in range(len(locs)-1):
      if locs[ii]<=val and val<locs[ii+1]:
          return names[ii]

  # If we reach this point is the last:
  return names[-1]


# ------------------------------------------------------------------------------------------------------
# Function to get the final text explanation:
def get_text_expl(inst, arr_expl_slopes, arr_expl_contribs, param_names):

  ranger = range(len(inst))

  # Our own baselines, rethink how they are saved A A A A A:
  norm_var_baseline = [[0, 'Very Low'], [0.2, 'Low'], [0.4, 'Average'], [0.6, 'High'], [0.8, 'Very High']]
  scld_var_baseline = [[-np.Inf, 'Very Low'], [-3, 'Low'], [-1, 'Average'], [1, 'High'], [3, 'Very High']]
  
  slopes_baseline = [[-np.Inf, 'Very Low'], [-3, 'Low'], [-1, 'Average'], [1, 'High'], [3, 'Very High']]
  contribs_baseline = [[0, 'Very Low'], [0.2, 'Low'], [0.4, 'Average'], [0.6, 'High'], [0.8, 'Very High']]
  
  # We construct for each input, using the information from the user (here we use the baselines):
  ranges_inputs = [ norm_var_baseline for _ in ranger ]
  ranges_slopes = [ slopes_baseline for _ in ranger ]
  ranges_contribs = [ contribs_baseline for _ in ranger ]
  # Also the names of the variables:
  names_inputs = [ param_names[i] + ' ' for i in ranger ]

  # The start of the 3 sentences:
  expl_txt_slopes = " For this instance, the output variable depends on: "
  expl_txt_contribs = " For this instance, the contributions of the variables are: "
  expl_txt_input = " Because the input has values of: "

  # Initialize the text_expl variable, each entry refers to one output dimension:
  # Inside each of these output dimensions, we have the 3 explanations.
  arr_text_expl = []

  for expl_slopes, expl_contribs in zip(arr_expl_slopes, arr_expl_contribs):

    # Go through the slopes:
    for val, ranges, name in zip(expl_slopes, ranges_slopes, names_inputs):
      txt_to_add = get_adj_from_val(val, ranges)
      expl_txt_slopes = expl_txt_slopes + name + txt_to_add + ','

    # Go through the contributions:
    for val, ranges, name in zip(inst, ranges_contribs, names_inputs):
      txt_to_add = get_adj_from_val(val, ranges)
      expl_txt_contribs = expl_txt_contribs + name + txt_to_add + ','

    # Go through the input:
    for val, ranges, name in zip(expl_contribs, ranges_inputs, names_inputs):
      txt_to_add = get_adj_from_val(val, ranges)
      expl_txt_input = expl_txt_input + name + txt_to_add + ','

    # Replace with a dot the last entry:
    expl_txt_input = expl_txt_input[:-1]+'.'
    expl_txt_contribs = expl_txt_contribs[:-1]+'.'
    expl_txt_input = expl_txt_input[:-1]+'.'

    # Append this entry:
    arr_text_expl.append([expl_txt_slopes, expl_txt_contribs, expl_txt_input])

  return arr_text_expl


# ------------------------------------------------------------------------------------------------------
# The class for the output explanations of the request:
class ExplClass:
  """
      The class for the output explanations of the request
  """
  def __init__(self, input_i, output_pred_NN_i, output_pred_LF_i, 
               mat_LF_dep_terms_i, mat_LF_indep_term_i, mat_input_contribs_i, arr_text_explanation_i, reference_id):

    # In the assignation we correct the shape of arr_LF_indep_term_i, mat_LF_indep_term_i, mat_input_contribs_i
    # to respect the criterion: rows instances, cols dimensions (both the input and outputs are row vecs) 
    self.input_values = input_i
    self.input_contributions = mat_input_contribs_i.tolist()
    self.input_dependencies = mat_LF_dep_terms_i.tolist()
    self.independent_term = mat_LF_indep_term_i.tolist()
    self.neural_network_output = output_pred_NN_i.tolist()
    self.linear_function_output = output_pred_LF_i.tolist()
    self.text_explanation = arr_text_explanation_i
    self.reference_id = reference_id


# ------------------------------------------------------------------------------------------------------
# Get the explanations function:
def run_explanations(X, y_pred_NN, user_model, specs, arr_layers_info, ref_of_fake_layers, associated_true_layer_index, param_names):

  # Initialize the variable that stores the explanations as a dictionary:
  generated_explanations = {}

  # If tensorflow:
  if specs.framework == "tensorflow":

    # The computationally most efficient approach:
    #    -  A single model that outputs all intermediate outputs at once.
    #    -  Calling this model only once for all the instances of the request.

    # Get the reference for the outputs of the true the layers:
    ref_outs_true_layers = [layer.output for layer in user_model.layers]
    # Construct the new model that returns the outputs of the true layers:
    model_of_outs_true_layers  = Model(inputs=user_model.input, outputs=ref_outs_true_layers)
    # Feed the model with all the instances of the request:
    # Is vital to process all instances together. Much less computational time than calling the model iteratively:
    outs_true_layers = model_of_outs_true_layers.predict(X)

    # The outs of all the layers, which includes the fake ones that we generated:
    outs_layers = {}
    # We add the input, as the first entry (key -1) of outs_layers:
    outs_layers[-1] = X
    # Loop through fake layers:
    for layer_l in range(specs.all_n_lays):
        # Check if fake layer:
        if ref_of_fake_layers[layer_l] == 'fake':
          outs_layers[layer_l] = (None)
        # If it is not fake:
        else:
          outs_layers[layer_l] = outs_true_layers[associated_true_layer_index[layer_l]]

    # Fill the None entries of outs_layers:
    for layer_l in range(specs.all_n_lays):
      # If it is fake we know that the entry in outs_layers is None, thus we need to fill it:
      if ref_of_fake_layers[0] == 'fake':
        # If the type of layer is dense:
        if arr_layers_info[layer_l].name == 'dense':
          # Remember that outs_layers has now the input before, so the index to replace in outs_ayers is +1
          outs_layers[layer_l] = np.dot(outs_layers[layer_l-1],arr_layers_info[layer_l].ws.T)
          # If we are using bias:
          if arr_layers_info[layer_l].use_bias:
            outs_layers[layer_l] += arr_layers_info[layer_l].bs.T 


  # If tensorflow:
  elif specs.framework == "pytorch":

    # Ideas obtained from:
    # https://kozodoi.me/blog/20210527/extracting-features
    # https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/

    # Dict to store the outputs of the intermediate layers:
    outs_layers = {}
    # Helper function to get the intermediate outputs:
    def getIntermediateOuts(given_name):
      # The hook signature:
      def hook(model, input, output):
        # Use detach to remove grad field. Now you can convert to numpy and manipulate:
        outs_layers[given_name] = output.detach().numpy()
      # Return the hook:
      return hook


    # Dictionary with hooks for each layer:
    dict_hooks = {}
    # Layers counter:
    layer_index = 0
    # Loop through the layers:
    for layer_i in user_model.children():
        # Now the module:
        dict_hooks[layer_index] = layer_i.register_forward_hook(getIntermediateOuts(layer_index))
        # Add the counter of the layers:
        layer_index += 1
   
    # Convert to torch tensor:
    torch_X = torch.tensor(X, dtype=torch.float32)

    # Predict - Forward pass - Getting the outputs
    outs_model = user_model(torch_X)

    # List of names for the keys only of the interm. lays (exclude the input, key:-1):
    keys_interm_outs = list(range(layer_index))

    # Remove the hooks after the prediction:
    # TODO: I am not sure if this is needed.
    if False:
      for lay_name in keys_interm_outs:
        dict_hooks[lay_name].remove()

    # We add the input, as the first entry (key -1) of outs_layers:
    outs_layers[-1] = X

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # CORE ALGO

  # Run the frontpropagation algorithm, mat_LF_dep_terms_i & mat_LF_indep_term_i are both arrays 
  # of arrays with the terms for all the outputs (one array of terms per output).
  dict_mats_LF_dep_terms, dict_mats_LF_indep_term, dict_mats_input_contribs, dict_outputs_pred_LF = core_algorithm(specs, arr_layers_info, outs_layers)
  
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # TEXT EXPLANATIONS

  # Get the dict of text explanations:
  dict_arr_text_explanation = { i: get_text_expl(input_i, dict_mats_LF_dep_terms[i], dict_mats_input_contribs[i], param_names) for i, input_i in enumerate(X) }
  
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # PACKAGING EXPLANATIONS

  # Loop through the instances:
  for i, input_i in enumerate(X):

    # Save the information of the instance in the ExplClass the structure:
    generated_explanations[i] = ExplClass(input_i, y_pred_NN[i,:], dict_outputs_pred_LF[i], 
                                          dict_mats_LF_dep_terms[i], dict_mats_LF_indep_term[i], 
                                          dict_mats_input_contribs[i], dict_arr_text_explanation[i], i)

  return generated_explanations



# ------------------------------------------------------------------------------
# Visualization of the linear function prediction vs NN prediction:
# Low level function to compare the predictions of a linear function prediction vs NN prediction:
def plot_LF_vs_NN_predictions(inst, arr_LF_deps, arr_LF_indep, arr_out_pred_LF, arr_out_pred_NN,
                              main_model, specs, M, proxim_thres, flag_normalization, which_out_dims):
  """ 
      inst: The instance of the request that we want to evaluate.
      
      arr_ : Each of the array entries represent a dimension.

      arr_LF_deps, arr_LF_indep: Dep and indep terms of the Linear Function approximation.

      arr_out_pred_LF, arr_out_pred_NN: Predicted output of the instance with the LF and the NN respectively.

      M: Number of points desired to plot in the vicinity of the instance.

      proxim_thres: Allowed maximum difference in every dimension to generate the sample.

      flag_normalization: Boolean to know if the input vector is normalized. If so, we make sure
                         that the random points generated are within the limits of 0 and 1.

      which_out_dims: To know which output dimesnions we want to plot. Could be "all" or
                      an array with the indexes of the out dims desired to plot. 
                      For example [0,3,5]. Indexes start at 0.     
  
  """

  # Convert to numpy and reshape for manipulation:
  inst = np.array(inst)
  arr_LF_deps = np.array(arr_LF_deps).reshape((specs.n_outs, 1, specs.n_ins))
  arr_LF_indep = np.array(arr_LF_indep).reshape((1, specs.n_outs)) 
  arr_out_pred_LF = np.array(arr_out_pred_LF)
  arr_out_pred_NN = np.array(arr_out_pred_NN)

  # Convert the variable which_out_dims to an array that serves as a flag:
  # If the user wants all:
  if which_out_dims =="all":
    arr_flags_plot_out_dims = [ True for a in range(specs.n_outs) ]
  # which_out_dims is an array with the dimensions:
  else: 
    # Initialize:
    arr_flags_plot_out_dims = [ False for a in range(specs.n_outs) ]
    # Loop:
    for ind in which_out_dims:
      # Chack that it belongs to the possible outputs:
      if 0<=ind<specs.n_outs:
        # Assing:
        arr_flags_plot_out_dims[ind] = True


  # Generation of the random points in the neighborhood of the instance:
    
  # Initialize the random number generator:
  rng = np.random.default_rng()

  # Get the random normal sample of M new points. Rows, points, columns dimensions:
  numbers = rng.normal(size=(M, specs.n_ins))

  # Normalize each column/dimension:
  numbers = numbers/np.max(np.abs(numbers), axis=0)

  # Get the differences for the dimension:
  diffs = numbers * proxim_thres

  # Check if we are using normalized inputs.
  # If that is the case, we need to make sure we are not exceeding the limits:
  if flag_normalization: 

    # Add instance values to max and min differences of each dimension (top and bottom values):
    tops = np.max(diffs, axis=0) + inst
    bots = np.min(diffs, axis=0) + inst

    # Get the two filters of being off-limits:
    filter_top = tops>1.0
    filter_bot = bots<0.0

    # See where they exceed the normalized values in either tops or bots:
    where_offlimits = set( np.where(filter_top)[0].tolist() + np.where(filter_bot)[0].tolist() ) 
    
    # Go through those columns where we are off limits:
    for col in where_offlimits:

      # Now get the positivies and the negatives:
      filter_positives = diffs[:,col]>0
      filter_negatives = ~ filter_positives
      positives = diffs[:,col][filter_positives]
      negatives = diffs[:,col][filter_negatives]

      # Check if you exceed the top:
      if filter_top[col]:
        # Denorm and renorm the positives so that the max is 1-instance[col]
        positives = (positives/np.max(positives))*(1-inst[col])

      if filter_bot[col]:
        # Denorm and renorm the negatives so that the min is -instance[col]
        negatives = (negatives/abs(np.min(negatives))) * inst[col]
      
      # Now join them, shuffle:
      new_diffs_of_this_col = np.concatenate((positives, negatives)) 
      np.random.shuffle( new_diffs_of_this_col )

      # Assign it to the column in diffs:
      diffs[:,col] = new_diffs_of_this_col


  # Get the euclidean distance directly from the differences:
  distances = np.mean( np.square(diffs), axis=1).reshape((M, 1))

  # Apply these differences to generate the random points in the neighborhood:
  random_points = (diffs + inst).reshape((M, specs.n_ins))

  # Get the predictions with the NN for the random points:
  Yrandom_pred_NN = get_NN_output(random_points, main_model, specs)

  # Get the predictions with the LF approximation of the instance for the random points:
  aux_1 = random_points*arr_LF_deps
  aux_2 = np.sum(aux_1, axis=2).transpose()
  Yrandom_pred_LF = aux_2 + arr_LF_indep

  # Loop to create the figures:
  for j in range(specs.n_outs):
      
    # Check if you want to plot this one:
    if arr_flags_plot_out_dims[j]:

      # Figure:
      fig = plt.figure()
      ax = fig.add_subplot(111)
      # Perform the scatterplot, should be a curve tangent to the 1-1 line:
      # Color indicates the distance from the instance:
      if specs.n_outs == 1:
        scat_im = ax.scatter(Yrandom_pred_NN, Yrandom_pred_LF, c = distances, s=5, cmap="jet", alpha=0.7) 
      else:
        scat_im = ax.scatter(Yrandom_pred_NN[:,j], Yrandom_pred_LF[:,j], c = distances, s=5, cmap="jet", alpha=0.7) 
      clb = plt.colorbar(scat_im)
      clb.ax.set_title('Eucl. Dist.', size=12)
      # Convert to white the ticks:
      clb.ax.tick_params(axis='y')
      # Get current limits of plot, and the ref lims:
      x_lims = plt.gca().get_xlim()
      y_lims = plt.gca().get_ylim()
      ref_lims = [np.min((x_lims[0], y_lims[0])), np.max((x_lims[1], y_lims[1])) ]
      # Show the 1-1 line:
      ax.plot( ref_lims, ref_lims, color='grey', alpha=0.5)
      # Plot two dashed lines to mark the location of the instance:
      ax.plot([ref_lims[0], arr_out_pred_NN[j]], [arr_out_pred_LF[j], arr_out_pred_LF[j]], linestyle='dashed', color='black')
      ax.plot([arr_out_pred_NN[j], arr_out_pred_NN[j]], [ref_lims[0], arr_out_pred_LF[j]], linestyle='dashed', color='black')
      # Show the instance:
      ax.scatter(arr_out_pred_NN[j], arr_out_pred_LF[j], s=25, c="black", marker="*")
      # Set the limits to the ref lims:
      ax.set_xlim(ref_lims)
      ax.set_ylim(ref_lims) 
      # Details:
      ax.set_xlabel(f"Output {j+1} Predicted with NN", size=12)
      ax.set_ylabel(f"Output {j+1} Predicted with LF Aproximation", size=12)
      ax.set_title(f"Rand. Points Near Instance - Out Dim. {j+1}")
      # Convert to white the sides and ticks:
      # for side in ["top", "bottom", "left", "right"]: 
      #   ax.spines[side].set_color('white')
      # for dim in ["x", "y"]: 
      #   ax.tick_params(axis=dim)
      # Black facecolor:
      # ax.set_facecolor('black')
      # Save the figure with transparent background:
      plt.savefig(f"figure_{j}.png", transparent=True, dpi=400)

  return




  