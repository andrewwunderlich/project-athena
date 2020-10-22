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
import scipy.io

from utils.model import load_pool, load_lenet
from models.athena import Ensemble, ENSEMBLE_STRATEGY
from utils.file import load_from_json
from utils.metrics import error_rate, get_corrections
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
        
        err_rate = error_rate(np.asarray(predictions), np.asarray(labels));
        print('>>>Error Rate: ',err_rate)

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
            file = os.path.join(output_dir, "ae_{}.npy".format(attack_configs.get(key).get("description")))
            print("Saving the adversarial examples to file [{}].".format(file))
            np.save(file, data_adv)
    if (dataTable.shape[0]<50): 
        #if <50 images run, print table to console for debug and analysis
        print("Less than 50 images run--Printing dataTable to Console")
        print(dataTable)
    else: 
        # if >=50, save table to a file for analysis in Task 1 Jupyter notebook
        file = os.path.join(output_dir, "dataTable.mat")
        print("Saving dataTable to "+file)
        #np.save(file, dataTable)
        scipy.io.savemat(file, {'dataTable':dataTable})
        
def evaluate(trans_configs, model_configs,
             data_configs, labels, save=False, output_dir=None):
    """
    Apply transformation(s) on images.
    :param trans_configs: dictionary. The collection of the parameterized transformations to test.
        in the form of
        { configsx: {
            param: value,
            }
        }
        The key of a configuration is 'configs'x, where 'x' is the id of corresponding weak defense.
    :param model_configs:  dictionary. Defines model related information.
        Such as, location, the undefended model, the file format, etc.
    :param data_configs: dictionary. Defines data related information.
        Such as, location, the file for the true labels, the file for the benign samples,
        the files for the adversarial examples, etc.
    :param labels: the correct labels for each image
    :param save: boolean. Save the transformed sample or not.
    :param output_dir: path or str. The location to store the transformed samples.
        It cannot be None when save is True.
    :return:
    """
    # Load the baseline defense (PGD-ADT model)
    pgd_adt = load_lenet(file=model_configs.get('pgd_trained'), trans_configs=None,
                                  use_logits=False, wrap=False)

    # get the undefended model (UM)
    file = os.path.join(model_configs.get('dir'), model_configs.get('um_file'))
    undefended = load_lenet(file=file,
                            trans_configs=trans_configs.get('configs0'),
                            wrap=True)
    print(">>> um:", type(undefended))

    # load weak defenses into a pool
    pool, _ = load_pool(trans_configs=trans_configs,
                        model_configs=model_configs,
                        active_list=True,
                        wrap=True)
    # create AVEP ensemble from the WD pool
    wds = list(pool.values())
    print(">>> wds:", type(wds), type(wds[0]))
    ensemble_AVEP = Ensemble(classifiers=wds, strategy=ENSEMBLE_STRATEGY.AVEP.value)


    # load the benign samples
    bs_file = os.path.join(data_configs.get('dir'), data_configs.get('bs_file'))
    x_bs = np.load(bs_file)
    img_rows, img_cols = x_bs.shape[1], x_bs.shape[2]

    ''' # this section replaced by passing in labels as a parameter
    # load the corresponding true labels
    label_file = os.path.join(data_configs.get('dir'), data_configs.get('label_file'))
    labels = np.load(label_file)
    if len(labels.shape) > 1:
        labels = [np.argmax(p) for p in labels] #returns correct label
    '''
    # get indices of benign samples that are correctly classified by the targeted model
    print(">>> Evaluating UM on [{}], it may take a while...".format(bs_file))
    pred_bs = undefended.predict(x_bs)
    corrections = get_corrections(y_pred=pred_bs, y_true=labels)

    # Evaluate AEs.
    results = {}
    ae_list = data_configs.get('ae_files')

    for ae_count in range(len(ae_list)): # step through ae's one by one
        ae_file = os.path.join(data_configs.get('dir'), ae_list[ae_count])
        x_adv = np.load(ae_file)

        # evaluate the undefended model on the AE
        print(">>> Evaluating UM on [{}], it may take a while...".format(ae_file))
        pred_adv_um = undefended.predict(x_adv) #num_images by 10 array
        pred_adv_um = [np.argmax(p) for p in pred_adv_um] # returns prediction
        #err_um = error_rate(y_pred=pred_adv_um, y_true=labels, correct_on_bs=corrections)
        # track the result
        #results['UM'] = err_um

        # evaluate the ensemble on the AE
        print(">>> Evaluating ensemble on [{}], it may take a while...".format(ae_file))
        pred_adv_AVEP = ensemble_AVEP.predict(x_adv)
        pred_adv_AVEP = [np.argmax(p) for p in pred_adv_AVEP]
        #err_AVEP = error_rate(y_pred=pred_adv_ens, y_true=labels, correct_on_bs=corrections)
        # track the result
        #results['Ensemble'] = err_AVEP

        # evaluate the baseline on the AE
        print(">>> Evaluating baseline model on [{}], it may take a while...".format(ae_file))
        pred_adv_pgd_adt = pgd_adt.predict(x_adv)
        pred_adv_pgd_adt = [np.argmax(p) for p in pred_adv_pgd_adt]
        #err_pgd_adt = error_rate(y_pred=pred_adv_pgd_adt, y_true=labels, correct_on_bs=corrections)
        # track the result
        #results['PGD-ADT'] = err_pgd_adt

    # TODO: collect and dump the evaluation results to file(s) such that you can analyze them later.
    print(">>> Evaluations on [{}]:\n{}".format(ae_file, results))


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
    '''generate_ae(model=target, 
                data=data_bs, 
                labels=labels, 
                attack_configs=attack_configs,
                save=False, output_dir=('C:/Users/andre/CSCE585_local/'+
                                       'project-athena/data'))
    '''
    evaluate(trans_configs=trans_configs,
             model_configs=model_configs,
             data_configs=data_configs,
             labels = labels
             save=False,
             output_dir=args.output_root)
