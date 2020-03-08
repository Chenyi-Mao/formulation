<p align="center">
  <img src="https://github.com/Chenyi-Mao/formulation/blob/master/LOGO_for_DIRECT_w_title_calibri.png" width="800">
</p>

<p align="center">
  <img src="https://github.com/Chenyi-Mao/formulation/blob/master/LOGO_for_DIRECT_w_title.png" width="800">
</p>


# Random Forest Classifier, a novel tool on predicting pharmaceutical formulation

We are presenting a model, where the formulation of a potential active pharmaceutical ingredient (API) is predicted by using random forest classifier. This model is built on 5 features coming from each API: number of hydrogen-bond acceptor, number of hydrogen-bond doner, polar surface area density, calulated log value of partition coefficient between octanol and water, and unchanged excretion percentage in urine. We believe this model would benefit the very first stage of drug screening, discovery and development.

## Overview

Bringing a new pharmaceutical drug to the market is not easy. An active pharmaceutical ingradient (API) faces not only a complicated cycle involving from target validation to toxicity examination, from pre-clinical trial to phase 4 clinical studies, but tremendous costs and other resources. Therefore, there is an unmet need to partially or fully speed up the development cyle and/or to save costs and other resources. With concurrent advances in computing power, giant data availablilty, and better algorithms, artificial intelligence (AI) is becoming more and more popular and has been used in various areas. Here, we are taking AI's advantages and applying on draug development. 

Although discovering active pharmaceutical ingradient is crutial for the drug development, putting errors on deciding the form of the product out of an active pharmaceutical ingradient is not negligible because a right fomulation could directly improve the drug delivery accuracy and efficieny, leading to a higher likelihood of passing clinical studies and a less likelihood of using extra budgets and resources. Therefore, in this project we are focuing on predicing pharmaceutical formulation of any given API. 

To achieve this goal, we decide to choose an AI-related algorithm, random forest classifier. Random forest classifier is a decision-tree-based learning method for data classification. 5 features mentioned from last section are initally used to train the algorithm so that an importance of each feature is reported. Then, based on the report, users subjectively decide how many features match with their goals and apply these features to train the algorithm one more time, leading to a consequent prediction and accuracy report. In addition, our module also provides raw data processing such as regression and feature cross validation prior to perform random forest calssifier. 

## Get Started
The following instructions are prepared for random forest classifier based algorithm, called `formulation`. This instruction conatins five parts: prerequsites, installation, details, modules, and running tests. The goal of this instruction is to help you successfully get a copy of `formulation` running on your local machine, to perform a detailed tutorial to make sure every piece works properly, and to use it for desired purposes. 

### Prerequsites
### Installation
*don't know yet*
### Details
#### Modules
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
We are greatful of having five developers, Chenyi Mao, Dawei Gu, Zichen Zhu, Jinge Xu, and Ling Zhang, working together in this project. Five developers major in diverse fields and have brought different insights into this project. Chenyi whose background in biochemistry initalized this project. Zichen and Jingge used their background in chemical engineering refined the goal. Dawei and Ling replying on their strength in computer science outlined modules. 

## License
MIT
## Acknowledgments
