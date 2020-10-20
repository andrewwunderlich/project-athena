"""

@author: Andrew Wunderlich (andreww(at)email(dot)sc(dot)edu)

Modified version of src/tutorials/craft_adversarial_examples.py

"""


import argparse
import numpy as np
import pandas as pd
import os
import time
from matplotlib import pyplot as plt

from utils.model import load_lenet
from utils.file import load_from_json
from attacks.attack import generate


def generate_ae(model, data, labels, attack_configs, save=False, output_dir=None):
    """
    Generate adversarial examples
    :param model: WeakDefense. The targeted model.
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
    
    # initialize array for storing predicted values for each attack
    # each row corresponds to an image from the MNIST dataset
    # the first column contains the true values, and each subsequent column 
    # contains the predicted values for each attack.
    # The array is initialized with '-1' at all elements so that any values 
    # which are not overwritten with digits 0-9 are identifiable as erroneous
    dataTable = -np.ones((num_images, num_attacks+1), dtype = int)
    dataTable[:,0] = labels;
    
    # generate attacks one by one
    for id in range(num_attacks): #outer loop steps through attacks
        key = "configs{}".format(id)
        data_adv = generate(model=model,
                            data_loader=data_loader,
                            attack_args=attack_configs.get(key)
                            )
        # predict the adversarial examples
        predictions = model.predict(data_adv)
        predictions = [np.argmax(p) for p in predictions]
        

        dataTable[:,id+1] = predictions #insert predicted values into new column
        
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
            file = os.path.join(output_dir, "{}.npy".format(attack_configs.get(key).get("description")))
            print("Saving the adversarial examples to file [{}].".format(file))
            np.save(file, data_adv)
    if (dataTable.shape[0]<50): 
        #if <50 images run, print table to console for debug and analysis
        print("Less than 50 images run--Printing dataTable to Console")
        print(dataTable)
    else: 
        # if >50, save table to a file for analysis in Task 1 Jupyter notebook
        file = os.path.join(output_dir, "/data/dataTable_um.npy")
        print("Saving dataTable to "+file)
        np.save(file, dataTable)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('-m', '--model-configs', required=False,
                        default='configs/experiment/model-mnist.json',
                        help='Folder where models are stored.')
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
    num_images = 15 #set to full 10,000 for final run, <50 while developing for speed
    data_bs = data_bs[:num_images]
    labels = labels[:num_images]
    generate_ae(model=target, data=data_bs, labels=labels, attack_configs=attack_configs,
                save=False, output_dir=('C:/Users/andre/CSCE585_local/'+
                                       'project-athena/saved_attacks'))
