# Athena Tutorials
**This file will be updated constantly.**

**Questions regarding project, open [GitHub Issues](https://github.com/csce585-mlsystems/project-athena/issues).**

**Questions regarding programming techniques, Google, check it on StackOverflow, and open [GitHub Issues](https://github.com/csce585-mlsystems/project-athena/issues).**

**Before open a GitHub issue, check if your the same question has been asked.**

## configuration files
Configuration files are organized in the ``configs`` folder. ``experiment`` folder contains the configurations for the **Vanilla Athena**. Create your own configuration files for your experiments, such that you can evaluate your approaches on Vanilla Athena by providing corresponding json files we provided. ``demo`` folder contains the configurations (smaller scale) for tutorials.

1. ``athena-mnist.json`` defines the corresponding transformation settings for each individual weak defenses in the Vanilla Athena. ``num_transformations`` defines the total number of weak defenses, and ``configs<n>`` defines the settings for the corresponding transformation, where ``n`` is the ``id`` of the weak defense and ``0`` is for the _undefended model_ (the targeted model in the context of zero-knowledge threat model). You will need this file model and transformation related tasks, for example, load a model, transform an image, etc.
2. ``data-mnist.json`` describes the information related to test data. It consits of the folder to store the test data (``dir``), the benign sample (``bs_file``), the corresponding true labels (``label_file``), and the adversarial examples (``ae_files``). You will need it for almost all tasks like evaluate a model, generate adversarial example, etc. **You do NOT need it when you train a model.**
3. ``model-mnist.json`` defines the information related to trained models. The folder where the trained models locate (``dir``), the undefended model (``um_file``), the prefix and postfix of the model names (check models/cnn/readme.md for the naming patterns). You will need it when you load a trained model.
4. ``attack-zk-mnist.json`` define the attack configurations, in a similar structure of ``athena-mnist.json``. ``num_attacks`` defines the total number of attacks, ``configs<n>`` defines the adversarial settings for each attack.

## python tutorials
### craft_adversarial_examples.py
A sample to generate adversarial example in the context of zero-knowledge threat model, using the undefended model as the targeted model.

How to run
* click the run button in IDE.
* command line

``python craft_adversarial_examples.py --model-configs ../configs/demo/model-mnist.json --data-configs ../configs/demo/data-mnist.json --attack-configs ../configs/demo/attack-zk-mnist.json --output-root ../results``


### transformation.py
A sample to transform images.

How to run
* click the run button in IDE.
* command line

``python transformation.py --trans-configs ../configs/demo/athena-mnist.json --model-configs ../configs/demo/model-mnist.json --data-configs ../configs/demo/data-mnist.json --output-root ../results``

