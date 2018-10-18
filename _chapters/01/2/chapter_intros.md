---
title: 'Chapter Intros'
permalink: 'chapters/01/2/chapter_intros'
previouschapter:
  url: chapters/01/1/example_fitting_time_series
  title: 'Example Fitting Time Series'
nextchapter:
  url: chapters/01/3/other_references
  title: 'Other References'
redirect_from:
  - 'chapters/01/2/chapter-intros'
---
What You'll See in the Chapters to Come
====================

In this introductory chapter, we've barely scratched the surface of heliophysics and of machine learning. We specifically chose an example that would hopefully be a more familiar starting point for heliophysicists and for data scientists. In the following chapters, we'll dive a deeper into examples from real, active research being done, most of which has also been published in peer-review journals. The rabbit hole goes deeper still, and we provide some other references in the next section. 

# Available Now

## Chapter 3 (This Chapter)
* Authors: James Paul Mason
* Objective: Fitting time series data -- extreme ultraviolet irradiance emission line light curves
* ML method(s): 
    * Data cleaning via Imputing (`sklearn.preprocessing.Imputer`)
    * Training vs. prediction via shuffle splitting the available data (`sklearn.model_selection.ShuffleSplit`)
    * Regression via Support Vector Machines (`sklearn.svm.SVR`)
    * Validation (`sklearn.model_selection.validation_curve`)
* Data source(s): Solar Dynamics Observatory (SDO) / Extreme Ultraviolet Variability Experiment (EVE)

## Chapter 4
* Authors: Monica Bobra
* Objective: Predicting CMEs with from photospheric magnetic field
* ML method(s): 
    * Classification via Support Vector Machine (`sklearn.svm`)
    * Training vs. prediction via K-Fold method the available data (`sklearn.model_selection.StratifiedKFold`)
* Data source(s): 
    * Solar and Heliospheric Observatory (SOHO) Large Angle and Spectrometric Coronagraph Experiment (LASCO)
    * Solar Terrestrial Relations Observatory (STEREO) Sun Earth Connection Coronal and Heliospheric Investigation (SECCHI) Coronagraphs
    * Geostationary Operational Environmental Satellites (GOES) X-ray Sensor (XRS) flare catalogs
    * SDO Helioseismic and Magnetic Imager (HMI)

## Chapter 5
* Authors: Carlos José Díaz Baso and Andrés Asensio Ramos
* Objective: Rapid and robust image resolution enhancement 
* ML method(s): 
    * Deep convolutional neural networks via the `keras` module
    * Applying penalties for optimization via `keras.regularizers.l2'`
* Data source(s): SDO/HMI

# Upcoming

## Chapter 6
* Authors: Paul Wright
* Objective: Simulating data from a lost instrument (EVE) based on another of a totally different type (AIA); DEMs
* ML method(s): ? 
* Data source(s): SDO/AIA, SDO/EVE

## Chapter 7
* Authors: Sophie Murray
* Objective: Forecasting space weather
* ML method(s): ?
* Data source(s): ? 