<p align="left">
  <img src="https://github.com/Chenyi-Mao/formulation/blob/master/LOGO.png" width="900">
</p>

--------------

## Usage Cases

* overview: we are aiming to build a model that can predict the formulation of a certain active pharmaceutical ingredient (API). We are going to reply on published features from about 800 drugs to build the model using random forest classifier. We believe the model can benefit the very first stage of drug screening, discovery and development. 
* potential users: any chemist or pharmaceutical company
* applied features for each API: number of hydrogen bond acceptor and donor, polar surface area density, partition coefficient between octane and water, and unexchanged excretion in urine. 
* data size 800 drugs constsiting of above features and respctive lables (tablets, capsules, and solution).
* resource: the data comes from a literature (doi: 10.1208/s12248-011-9290-9).
* primary uses: 1) predict the missing values for other important features; 2) predict the best formulation outcome that facilitates drug delivery and absorption; 3) extend the model to other supervised studies. 
