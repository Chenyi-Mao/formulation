<p align="left">
  <img src="https://github.com/Chenyi-Mao/formulation/blob/master/LOGO.png" width="900">
</p>

--------------

[![travis build](https://travis-ci.com/Chenyi-Mao/formulation.svg?branch=master)](https://travis-ci.com/Chenyi-Mao/formulation)
[![codecov](https://codecov.io/gh/Chenyi-Mao/formulation/branch/master/graph/badge.svg)](https://codecov.io/gh/Chenyi-Mao/formulation)
[![License](https://img.shields.io/github/license/Chenyi-Mao/formulation)](https://github.com/Chenyi-Mao/formulation/blob/master/LICENSE)

We are presenting a model, where the formulation of a potential active pharmaceutical ingredient (API) is predicted by using random forest classifier. This model is built on 5 features coming from each API: number of hydrogen-bond acceptor, number of hydrogen-bond doner, polar surface area density, calulated log value of partition coefficient between octanol and water, and unchanged excretion percentage in urine. We believe this model would benefit the very first stage of drug screening, discovery and development.

## Overview

Bringing a new pharmaceutical drug to the market is not easy. An active pharmaceutical ingradient (API) faces not only a complicated cycle involving from target validation to toxicity examination, from pre-clinical trial to phase 4 clinical studies, but tremendous costs and other resources. Therefore, there is an unmet need to partially or fully speed up the development cyle and/or to save costs and other resources. With concurrent advances in computing power, giant data availablilty, and better algorithms, artificial intelligence (AI) is becoming more and more popular and has been used in various areas. Here, we are taking AI's advantages and applying on draug development. 

Although discovering active pharmaceutical ingradient is crutial for the drug development, putting errors on deciding the form of the product out of an active pharmaceutical ingradient is not negligible because a right fomulation could directly improve the drug delivery accuracy and efficieny, leading to a higher likelihood of passing clinical studies and a less likelihood of using extra budgets and resources. Therefore, in this project we are focuing on predicing pharmaceutical formulation of any given API. 

To achieve this goal, we decide to choose an AI-related algorithm, random forest classifier. Random forest classifier is a decision-tree-based learning method for data classification. 5 features mentioned from last section are initally used to train the algorithm so that an importance of each feature is reported. Then, based on the report, users subjectively decide how many features match with their goals and apply these features to train the algorithm one more time, leading to a consequent prediction and accuracy report. In addition, our module also provides raw data processing such as regression and feature cross validation prior to perform random forest calssifier. 

## Get Started
The following instructions are prepared for random forest classifier based algorithm, called `formulation`. This instruction conatins five parts: prerequsites, installation, details, modules, and running tests. The goal of this instruction is to help you successfully get a copy of `formulation` running on your local machine, to perform a detailed tutorial to make sure every piece works properly, and to use it for desired purposes. 

### Prerequsites
Python is the primary software for the 'formulation'. We recommend python 3.7 or higher version here. Meanwhile, serval python packages are also required in advance. They are `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `math`, and can be easily installed by using conda, a package and environment control system. Finally, everything thing mentioned above should be performed by [Linux Operating System](https://www.linux.com/what-is-linux/) (Linux OS). 

### Installation
```

open Linux OS, jump to a directory where you want to save *formulation* and then type the following command to get **formulation**:
git clone https://github.com/Chenyi-Mao/formulation.git
```

### Details
After the installation, you can find more details from here. The packages contains three primary items: two folders called **formulation** and **paper** and **README.md**. Formulation folder contains all modules you will need to perform the random forest classifier and all tests developers wrote to test each modules. Details can be found in the next two sections. Raw data used for the porject comes from a literature article which is included in the folder, paper. 

#### Modules
The `formulation`has been dividied into 6 modules: *data_dropna*, *fill_missing_value*, *corss_validate_grid_search*, *cross_validate_n_predictors*, *importance*, *predict*, . Each module perform one task and works mutally with others. 
1. *data_dropna* is used to selectively remove cells that have missing values. Once users get a sense of the raw data, they decide which features are important for the prediction and then data_dropna will help users clean empty cells corresponding to selected features and will return a cleaned dataframe for a further use.  An screenshot is inculded to demosntrate the utility: 

XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

2. what if the feature of interest has a large number of value missing that could effect the efficieny of training process? No worries, *fill_missing_value* is here to help! Fill_missing_vlaue is designed to predict the missing value for the feature of interest. Meanwhile, mean squared error will be given after the prediction so users can decide to include this newly filled feature into the training process or not. 

3. once features of interest are settled, *cross_validate_grid_search* and *cross_validate_n_predictors* can be applied to find out the best n_estimator and max_depth combination leading to the best performance. The only difference between ____________________________

4. *importance* reports the individual importance for each selected feature under above chosen n_estimator and max_depth values.

5. accroding to the importance report, users can decide which features to include into the predcition step. *predict* outcomes the accuracy based on the overall test data. An screenshot is inculded to demosntrate the final outcome:
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

#### Running tests

## Contributing
As an open resource, contributing is always welcome. Authors recommend a discussion through openning an issue, writing an email, or any other approaches prior to making any changes. Then please follow the following procedures:
1. fork `formulation`
2. create your own feature branch
3. make changes and corresponding tests
4. commit changes
5. push to master branch
6. create a pull request

## Authors
We are greatful of having five developers, Chenyi Mao, Dawei Gu, Zichen Zhu, Jinge Xu, and Ling Zhang, working together in this project. Five developers major in diverse fields and have brought different insights into this project. Chenyi whose background in biochemistry initalized this project. Zichen and Jingge used their background in chemical engineering refined the goal. Dawei and Ling replying on their strength in computer science outlined modules. All authors contributed equally to make this project happen. 

## License
[MIT](https://en.wikipedia.org/wiki/MIT_License)

## Acknowledgments
Authors want to thank David Beck and Cao Ting for offering two interesting and eductational courses. Aythors also want to thank David Beck, Cao Ting, Ted Cohen, Jimin Qian, Torin Stetina, and Caitlyn Wolf for any feedbacks. Lastly, C.Mao wants to thank his supervisor Josh Vaughan for allowing him take these two courses that are totally irrelevant to his research. 
