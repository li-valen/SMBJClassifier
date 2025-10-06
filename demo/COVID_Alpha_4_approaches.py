import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat
import os
import sys
from multiprocessing import Pool

# Change the working directory to where this file is located 
os.chdir(os.path.dirname(__file__))

# Import SMBJClassifier from upper directory
sys.path.append('..')
from SMBJClassifier import DPP, model

def load_data(filename, Expfile):
    """Load data and experiment parameters."""
    return DPP.readInfo(filename, Expfile)

def preprocess_data(data, Amp, Freq, Vbias):
    """Preprocess data to convert current traces into conductance traces."""
    DPP.createCondTrace(data, Amp, Freq, Vbias)

def run_classification(args):
    """Run the classifier and save the results."""
    a, data, RR, group, num_group, s, group_Label = args
    conf_mat = model.runClassifier(a, data, RR, group, num_group, s)
    filename = f'./Result/Alpha_A{a}_H{s}.mat'
    savemat(filename, {'conf_mat': conf_mat, 'group_Label': group_Label})
    return filename

def main():
    filename = 'COVID_Strand_Source.txt'
    Expfile = 'COVID_Exp_parameter.mat'
    data, Amp, Freq, RR, Vbias = load_data(filename, Expfile)

    preprocess_data(data, Amp, Freq, Vbias)

    num_group = 3
    group_Label = ['Alpha_MM1', 'Alpha_MM2', 'Alpha_PM']
    group = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
    approach = [2]
    sampleNum = [5, 10]

    # Create a pool of workers
    with Pool() as pool:
        # Prepare arguments for the function
        args = [(a, data, RR, group, num_group, s, group_Label) for a in approach for s in sampleNum]
        # Run classification in parallel
        results = pool.map(run_classification, args)

    print(f"Classification results saved to: {results}")

if __name__ == '__main__':
    main()
