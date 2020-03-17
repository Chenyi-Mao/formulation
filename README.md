<p align="left">
  <img src="https://github.com/Chenyi-Mao/formulation/blob/master/images/LOGO.png" width="900">
</p>

--------------

[![travis build](https://travis-ci.com/Chenyi-Mao/formulation.svg?branch=master)](https://travis-ci.com/Chenyi-Mao/formulation)
[![codecov](https://codecov.io/gh/Chenyi-Mao/formulation/branch/master/graph/badge.svg)](https://codecov.io/gh/Chenyi-Mao/formulation)
[![Issues](https://img.shields.io/github/issues/Chenyi-Mao/formulation)](https://github.com/Chenyi-Mao/formulation/issues)
[![License](https://img.shields.io/github/license/Chenyi-Mao/formulation)](https://github.com/Chenyi-Mao/formulation/blob/master/LICENSE)

We are presenting a model, where the formulation of a potential active pharmaceutical ingredient (API) is predicted by using random forest classifier. This model is built on 5 features coming from each API: number of hydrogen-bond acceptor, number of hydrogen-bond donor, polar surface area density, calculated log value of partition coefficient between octanol and water, and unchanged excretion percentage in urine. We believe this model would benefit the very first stage of drug screening, discovery and development.

## Overview

Bringing a new pharmaceutical drug to the market is not easy. An active pharmaceutical ingredient (API) faces not only a complicated drug development process involving from target validation to toxicity examination, from pre-clinical trial to phase 4 clinical studies, but tremendous costs and other resources. Therefore, there is an unmet need to partially or fully speed up the development process and/or to save costs and other resources. With concurrent advances in computing power, giant data availablility, and better algorithms, artificial intelligence (AI) is becoming more and more popular and has been used in various fields. Here, we are taking AI's advantages and applying on drug development. 

Although discovering APIs is crucial for the drug development, putting errors on deciding the form of the product out of an API is not negligible because a right formulation could directly improve the drug delivery accuracy and efficiency, leading to a higher likelihood of passing clinical studies and a less likelihood of wasting extra budgets and resources. Therefore, in this project we are focuing on predicing pharmaceutical formulation of any given API. 

To achieve this goal, we decide to use an AI-related algorithm, random forest classifier. Random forest classifier is a decision-tree-based learning method for data classification. 5 features mentioned from last section are initially used to train the algorithm so that an importance of each feature is reported. Then, based on the report, users subjectively decide how many features match with their goals and apply these features to train the algorithm one more time, leading to a consequent prediction and accuracy report. In addition, our module also provides raw data processing such as regression and feature cross validation prior to perform random forest classifier. 

## Get Started
The following instructions are prepared for random-forest-classifier-based algorithm, called `formulation`. This instruction contains five parts: prerequisites, installation, details, modules, and running tests. The goal of this instruction is to help users successfully get a copy of `formulation` running on their local machines, to perform a detailed tutorial to make sure every piece works properly, and to use it for desired purposes. 

### Prerequisites
Python is the primary software for the 'formulation'. We recommend python 3.7 or higher version here. Meanwhile, serval python packages are also required in advance. They are `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `math`, and can be easily installed by using *conda*, a package and environment control system. Finally, everything mentioned above should be performed by [Linux Operating System](https://www.linux.com/what-is-linux/) (Linux OS). 

### Installation
```

open Linux OS, jump to a directory or create a new one where you want to save formulation and then type the following command to get formulation:
git clone https://github.com/Chenyi-Mao/formulation.git
```

### Details
After the installation, users can find more details from here. The package contains three primary items: two folders called **formulation** and **paper**, **README.md**, and **examples.ipynb**. Formulation folder contains all modules needed to perform the random forest classifier and all tests developers wrote to test each module. Details can be found in the next two sections. Raw data used for the project comes from a literature article which is included in the folder, paper. 

#### Modules
The `formulation`has been dividied into 6 modules: *predict_missing_value.py*, *cross_validation.py*, *importance.py*, *classification*, and *predict*, . Each module perform one or two tasks and works mutually with others. 
1. *predict_missing_value* is used to selectively remove cells that have missing values. Once users get a sense of the raw data, they decide which features are important for the prediction and then data_dropna will help users clean empty cells corresponding to selected features and will return a cleaned dataframe for the further use. In addition, it is designed to predict the missing value for the feature of interest. Meanwhile, mean squared error will be given after the prediction so users can decide whether to include this newly filled feature into the training process or not. 

2. once features of interest are settled, *cross_validation* can be applied to find out the best n_estimator and max_depth combination leading to the best performance. 

4. *importance* reports the individual importance for each selected feature under above chosen n_estimator and max_depth values.

5. accoroding to the importance report, users can decide which features to eventually use in *classification* to build the classifer model.

6. lastly, *predict* wraps up everything and make a final prediction with respect to the unseen data. 

#### Running tests
Automated tests have been done by using [Travis CI](https://travis-ci.com/Chenyi-Mao/formulation), a integration service with Github. Two badges at the beginning indicates all test functions we have in the tests folder pass with _XXXXXXX_ coverage rate. 

## Contributing
As an open resource, contributing is always welcome. Authors recommend a discussion through opening an issue, writing an email, or any other approaches prior to making any changes. Then please follow the following procedures:
1. fork `formulation`
2. create your own feature branch
3. make changes and corresponding tests
4. commit changes
5. push to master branch
6. create a pull request

## Authors
We are grateful of having five developers, Chenyi Mao, Dawei Gu, Zichen Zhu, Jinge Xu, and Ling Zhang, working together in this project. Five developers major in diverse fields and have brought different insights into this project. Chenyi who has background in biochemistry initialized this project. Zichen and Jingge used their background in chemical engineering refined the goal. Dawei and Ling replying on their strength in computer science outlined modules. All authors contributed equally to make this project happen. 

## License
[MIT](https://en.wikipedia.org/wiki/MIT_License) - see [LICENSE](https://github.com/Chenyi-Mao/formulation/blob/master/LICENSE) for details.

## Acknowledgments
Authors want to thank David Beck and Ting Cao for offering two interesting and educational courses. Authors also want to thank David Beck, Ting Cao, Ted Cohen, Jimin Qian, Torin Stetina, and Caitlyn Wolf for any feedbacks. Lastly, C. Mao wants to thank his supervisor Josh Vaughan for allowing him take these two courses that are totally irrelevant to his research. 
