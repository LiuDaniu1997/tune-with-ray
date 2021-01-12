import random
import numpy as np
import pandas as pd
import os
import argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model, load_model

import ray
from ray import tune
from ray.tune import Trainable, run
from ray.tune.schedulers import PopulationBasedTraining

#############################################################################################################################################################

def collectData(folder,subfolders):
    dataInputs = pd.DataFrame() 
    dataTargets = pd.DataFrame() 
    namesInput = ['kappa','alpha','gamma','Fx','Fy','Fz','Mx','My','Mz','deflection','lodrad','effrad','rotvel'] 
    for subfolder in subfolders:
        filePathInput = os.path.join(folder,subfolder,"TrainingDataPredictors") 
        filePathTarget = os.path.join(folder,subfolder,"TrainingDataTargets") 
        dataInputs = dataInputs.append(pd.read_table(filePathInput, names = namesInput, sep = ', ', header = None, engine = 'python'), ignore_index=True) 
        dataTargets = dataTargets.append(pd.read_table(filePathTarget, sep = ', ', header = None, engine = 'python'), ignore_index=True) 
    
    dataInputs = dataInputs.drop(columns=['lodrad','effrad'])
        
    return dataInputs, dataTargets

def norm(x,stats): 
    a = (x) / stats['max'] 
    a[a.isnull()] = 0 
    return a

def getStats(dataInputs, dataTargets):
    
    statsInputs = dataInputs.describe() 
    statsInputs = statsInputs.transpose() 
    statsTargets = dataTargets.describe() 
    statsTargets = statsTargets.transpose() 

    return statsInputs, statsTargets

#############################################################################################################################################################

class MLPmodel(tune.Trainable): # Hier ist die Model for PBT
    def _build_model(self):
        model = Sequential()

        num_layers = int(self.config.get("num_layers")) # Choose the number of hidden layers
        af_output = self.config.get("af_output") # Choose the Outputfunction at the lst layer
        af_output = select_output_activaiton_function(af_output) # decode the Outputfunction

        for i in range(num_layers):
            units = self.config.get(f"units_{i}", 300) # Define the Hidden Untis of each hidden layer
            af = self.config.get(f"af_{i}") # Choose the activation function of hidden layer
            af = select_activaiton_function(af) # Decode the activation function
            if i == 0:
                model.add(layers.Dense(int(units), input_shape=(11,))) # if it is first layer, add a Input layer
                model.add(layers.BatchNormalization())
                model.add(layers.Activation(af))
            else:
                model.add(layers.Dense(int(units)))
                model.add(layers.BatchNormalization())
                model.add(layers.Activation(af))
        model.add(layers.Dense(100, activation=af_output)) # Define the output layer
        
        return model

    def setup(self, config):
        model = self._build_model()
        opt = keras.optimizers.Nadam(learning_rate=1e-3) 
        model.compile(loss=tf.keras.losses.LogCosh(), optimizer=opt, metrics=['mae','mse','msle']) 
        self.model = model
    
    def step(self):
        # early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=25) 
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-12) 

        self.model.fit(dataTrainingInputsNormed, dataTrainingTargetsNormed, epochs=2, 
                        batch_size=int(self.config.get("batch_size", 200)),
                        validation_split = 0.1,
                        shuffle=True,
                        verbose=1,
                        callbacks=[reduce_lr]) 
        mae, mse, mape, msle = self.model.evaluate(dataTestInputsNormed, dataTestTargetsNormed)
        return {"msle": msle}
    
    def save_checkpoint(self, checkpoint_dir):
        file_path = checkpoint_dir + "/model"
        self.model.save(file_path)
        return file_path

    def load_checkpoint(self, path):
        del self.model
        self.model = load_model(path)
    
    # def cleanup(self):
    #     saved_path = self.model.save(self.logdir)
    #     print("save model at: ", saved_path)
##############################################################################################


def select_activaiton_function(af): # decode for activation function
    if int(af) == 0:
        af = "sigmoid"
    elif int(af) == 1:
        af = "tanh"
    elif int(af) == 2:
        af = "relu"
    elif int(af) == 3:
        af = "softplus"

    return af

def select_output_activaiton_function(af_output): # decode for activation function of the output layer
    if int(af_output) == 0:
        af_output = "selu"
    elif int(af_output) == 1:
        af_output = "relu"
    elif int(af_output) == 2:
        af_output = "softplus"
    elif int(af_output) == 3:
        af_output = "exponential"
    
    return af_output


def add_units_afs(dic, max_layers): # add Searchspace(include number od hidden units and number for activation function) for PBT
    for i in range(0, max_layers):
        dic[f"units_{i}"] = tune.randint(11,512) # hier will number of hidden units between 11 and 512 chosen
        dic[f"af_{i}"] = tune.randint(0,3) # hier will activation relevat number chosen
    return dic

##############################################################################################


if __name__ == "__main__":
    folderIn = 'FTire_Uneven_A' 
    
    subfolders = next(os.walk(folderIn))[1] 
    numSamples = len(subfolders) 
    numTrain = int(numSamples*0.9) 

    # Collecting Data
    print('Collecting data...') 
    subfoldersTraining = subfolders[0:numTrain] 
    subfoldersTest = subfolders[numTrain:] 
    dataTrainingInputs, dataTrainingTargets = collectData(folderIn,subfoldersTraining) 
    dataTestInputs, dataTestTargets = collectData(folderIn,subfoldersTest) 

    statsTrainingInputs, statsTrainingTargets = getStats(dataTrainingInputs,dataTrainingTargets) 
   
    # Prepare the Data
    print('Prepare the data...')
    dataTrainingInputsNormed = norm(dataTrainingInputs,statsTrainingInputs) # normalize predictor data for training
    dataTrainingTargetsNormed = dataTrainingTargets 
    dataTestInputsNormed = norm(dataTestInputs,statsTrainingInputs)  # normalize predictor data for testing
    dataTestTargetsNormed = dataTestTargets


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    args, _ = parser.parse_known_args()
    ray.init(num_cpus=3, num_gpus=1)

    # units_list = build_units_list(8)
    # af_list = build_af_list(8)

    # Define the Search Space for PBT
    searchspace = {
        "num_layers": tune.randint(1,8),
 
        "af_output": tune.randint(0, 4),
        "batchsize": tune.randint(32, 1024),
    }

    searchspace = add_units_afs(searchspace, 8) # Hier will add a dic for die initialize of hidden units and activaiton functions

    # Define the Mutationspace
    mutationspace = {
            "num_layers": tune.randint(1,8),
            "batchsize": tune.randint(32,512),
            "af_output": tune.randint(0,4),
        }
    
    mutationspace = add_units_afs(mutationspace, 8)
    
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=2,
        hyperparam_mutations=mutationspace)

    results = tune.run(
        MLPmodel,
        name="pbt_test",
        local_dir=os.path.normpath('D:/probe/pbt_checkpoint/'),
        scheduler=pbt,
        metric="msle",
        mode="min",
        resources_per_trial={
            "cpu": 3,
            "gpu": 1
        },
        stop={"training_iteration": 4},
        num_samples=2,
        config=searchspace,
        keep_checkpoints_num = 3,
        checkpoint_freq=3,
        checkpoint_at_end=True,
        )
    
    print("Best hyperparameters found were: ", results.best_config)
    
