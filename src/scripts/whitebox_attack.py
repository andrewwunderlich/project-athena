"""

@author: Andrew Wunderlich (andreww(at)email(dot)sc(dot)edu)

Script used to generate optimization-based whitebox attacks and test them 
against 

"""

import argparse
import numpy as np
import pandas as pd
import os
import time
from matplotlib import pyplot as plt
import scipy.io

from utils.model import load_pool, load_lenet
from models.athena import Ensemble, ENSEMBLE_STRATEGY
from utils.file import load_from_json
from utils.metrics import error_rate, get_corrections
from attacks.attack import generate

def generate_whitebox_ae(model, data, labels, attack_configs, save=False, output_dir=None):
    """
    Generate whitebox adversarial examples using the optimization approach.
    
    :param model: The targeted model. For a whitebox attack, this should be
        the full defended model.
    :param data: array. The benign samples to generate adversarial for.
    :param labels: array or list. The true labels.
    :param attack_configs: dictionary. Attacks and corresponding settings.
    :param save: boolean. True, if save the adversarial examples.
    :param output_dir: str or path. Location to save the adversarial examples.
        It cannot be None when save is True.
    :return:
    """
    
    img_rows, img_cols = data.shape[1], data.shape[2]
    num_attacks = attack_configs.get("num_attacks")
    data_loader = (data, labels)

    if len(labels.shape) > 1:
        labels = [np.argmax(p) for p in labels] #returns correct label
    
    
    for id in range(num_attacks): #outer loop steps through attacks
        key = "configs{}".format(id)
        data_adv = generate_whitebox(model=model,
                            data_loader=data_loader,
                            attack_args=attack_configs.get(key)
                            )
        # predict the adversarial examples
        predictions = model.predict(data_adv)
        predictions = [np.argmax(p) for p in predictions]
        
        err_rate = error_rate(np.asarray(predictions), np.asarray(labels));
        print('>>>Error Rate: ',err_rate)

        #dataTable[:,id+1] = predictions #insert predicted values into new column
        
        # plotting some examples
        num_plotting = min(data.shape[0], 2)
        for i in range(num_plotting):  #inner loop steps through images to plot
            img = data_adv[i].reshape((img_rows, img_cols))
            plt.imshow(img, cmap='gray')
            title = '{}: {}->{}'.format(attack_configs.get(key).get("description"),
                                        labels[i],
                                        predictions[i])
            plt.title(title)
            plt.show()
            plt.close()
    

        # save the adversarial example
        if save:
            if output_dir is None:
                raise ValueError("Cannot save images to a none path.")
            # save with a random name
            file = os.path.join(output_dir, "ae_{}.npy".format(attack_configs.get(key).get("description")))
            print("Saving the adversarial examples to file [{}].".format(file))
            np.save(file, data_adv)
    

if __name__ == '__main__':
    
    # probably need to edit the parser arguments here in order to change the 
    # targeted model 
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('-m', '--model-configs', required=False,
                        default='configs/experiment/model-mnist.json',
                        help='Folder where models are stored.')
    parser.add_argument('-t', '--trans-configs', required=False,
                        default='configs/demo/athena-mnist.json',
                        help='Configuration file for transformations.')
    parser.add_argument('-d', '--data-configs', required=False,
                        default='configs/experiment/data-mnist.json',
                        help='Folder where test data are stored.')
    parser.add_argument('-a', '--attack-configs', required=False,
                        default='configs/experiment/attack-zk-mnist.json',
                        help='Folder where attack data are stored.')
    parser.add_argument('-o', '--output-root', required=False,
                        default='results',
                        help='Folder for outputs.')
    parser.add_argument('--debug', required=False, default=True)

    args = parser.parse_args()

    print("------AUGMENT SUMMARY-------")
    print('TRANSFORMATION CONFIGS:', args.trans_configs)
    print("MODEL CONFIGS:", args.model_configs)
    print("DATA CONFIGS:", args.data_configs)
    print("ATTACK CONFIGS:", args.attack_configs)
    print("OUTPUT ROOT:", args.output_root)
    print("DEBUGGING MODE:", args.debug)
    print('----------------------------\n')

    # parse configurations (into a dictionary) from json file
    model_configs = load_from_json(args.model_configs)
    data_configs = load_from_json(args.data_configs)
    attack_configs = load_from_json(args.attack_configs)
    trans_configs = load_from_json(args.trans_configs)

    # load the targeted model
    model_file = os.path.join(model_configs.get("dir"), model_configs.get("um_file"))
    target = load_lenet(file=model_file, wrap=True)

    # load the benign samples
    data_file = os.path.join(data_configs.get('dir'), data_configs.get('bs_file'))
    data_bs = np.load(data_file)
    # load the corresponding true labels
    label_file = os.path.join(data_configs.get('dir'), data_configs.get('label_file'))
    labels = np.load(label_file)

    # generate adversarial examples 
    num_images = 15 #set to large number (maybe 1000) for final run, <50 while developing for speed
    data_bs = data_bs[:num_images]
    labels = labels[:num_images]

    generate_whitebox_ae(model=target, 
                data=data_bs, 
                labels=labels, 
                attack_configs=attack_configs,
                save=True, output_dir=('C:/Users/andre/CSCE585_local/'+
                                       'project-athena/Task 2/data'))
    '''
    evaluate(trans_configs=trans_configs,
             model_configs=model_configs,
             data_configs=data_configs,
             num_images = num_images,
             save=True,
             output_dir=('C:/Users/andre/CSCE585_local/'+
                         'project-athena/data'))
    '''