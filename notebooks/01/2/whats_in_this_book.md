What's In This Book?
====================

In this introductory chapter, we've barely scratched the surface of heliophysics and of machine learning. We specifically chose an example familiar to heliophysicists and for data scientists. In the following chapters, we'll cover examples from active research. Each chapter creates all the code necessary to reproduce a research paper published in a peer-reviewed scientific journal. These chapters cover a variety of topics, but they all employ machine learning methods to learn something about heliophysics, or the study of the Sun and its effects on our solar system -- the Earth, planets, minor objects, and all the space in between.

## Chapter 3 (This Chapter)
* Authors: James Paul Mason
* Objective: Fitting time series measurements of extreme ultraviolet irradiance emission line light curves
* ML method(s): 
    * Data cleaning via Imputing (`sklearn.preprocessing.Imputer`)
    * Training vs. prediction via shuffle splitting (`sklearn.model_selection.ShuffleSplit`)
    * Regression via Support Vector Machines (`sklearn.svm.SVR`)
    * Validation (`sklearn.model_selection.validation_curve`)
* Data source(s): Solar Dynamics Observatory (SDO) / Extreme Ultraviolet Variability Experiment (EVE)

## Chapter 4
* Authors: Monica Bobra
* Scientific Problem: The Sun's complex magnetic field drives a variety of eruptions. On occasion, the Sun emits a localized burst of energy called a solar flare. On others, the Sun throws magnetic flux and plasma into interplanetary space in an eruption called a coronal mass ejection. Both of these eruptions arise when the solar magnetic field contains a significant amount of free energy. This research uses properties of the solar magnetic field to predict whether the Sun will erupt in a flare or a coronal mass ejection.
* ML method(s): Support vector machine (from the python package [scikit-learn](https://scikit-learn.org/stable/)).
* Data sources(s): Images of the magnetic field on the solar surface, as well as tables of metadata with information about the timing and duration of solar flares and coronal mass ejections. 

## Chapter 5
* Authors: Carlos José Díaz Baso and Andrés Asensio Ramos
* Objective: Rapid and robust image resolution enhancement 
* ML method(s): 
    * Deep convolutional neural networks via the `keras` module
    * Applying penalties for optimization via `keras.regularizers.l2'`
* Data source(s): SDO/HMI

## Chapter 6
* Authors: Paul Wright, Mark Cheung, Rajat Thomas, Richard Galvez, Alexandre Szenicer, Meng Jin, Andrés Muñoz-Jaramillo, and David Fouhey
* Objective: Simulating data from a lost instrument (EVE) based on another of a totally different type (AIA); DEMs
* ML method(s): Deep convolutional neural networks
* Data source(s): SDO/AIA, SDO/EVE

Finally, we provide some other references in the next section. 
