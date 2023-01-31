# PLSR Algorithm for Maltose Concentration Prediction
This project contains a python script to optimize partial least squares regression (PLSR) algorithm for Savitzky-Golay hyperparameters applied to mid-infrared spectroscopy data to predict maltose concentration. The data was collected from an in vitro human digestion experiment. Different types of starches (pregelatinized maize, potato, rice, gelose 50, gelose 80) were digested and the samples were collected during the experiment. The predictive model was built to determine the maltose concentration in these samples. 

This is different from the MIR_Maltose_prediction repo as this script analyses data from the two different batches of the in vitro digestion experiments. 

## Scripts
- functions.py - Helper functions to conduct PLSR
- optimize_mir.py - This script determines the optimal hyper parameters for Savitsky-Golay filter and factors for the model for the given set of wavenumber regions
- mir_plsr.py - This applies PLSR for the data and outputs the prediction statistics and the loadings plot. 


## Requirements
- Python 3.7 or higher
- Numpy
- Pandas
- Sklearn
- Matplotlib (optional, for plotting)

## Getting Started
- Clone the repository to your local machine: git clone git@github.com:vishnu123r/MIR_Maltose_prediction_diluted-Infogest.git
- Navigate to the project directory
- Run the script to optimize the hyperparameters and predict the maltose concentration: python optimize_mir.py


## Usage
The script can be modified to use your own mid-infrared spectroscopy data and desired range of hyperparameters. The optimized hyperparameters and predicted maltose concentration will be exported as a CSV file.

## Known Issues
- The code for the loadings plot has not been completed yet. 
- Modification are still required for the README file

## Authors
Ramanah Visnupriyan
