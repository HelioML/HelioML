---
redirect_from:
  - "/table-of-contents"
title: 'Table of Contents'
prev_page:
  url: /acknowledgements.html
  title: 'Acknowledgements'
next_page:
  url: /01/whys_and_whats.html
  title: 'What Is This Book and Why Does It Exist?'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---
What's In This Book?
====================

In the following chapters, we'll cover examples from active research. Each chapter provides the motivation for the research and all of the code necessary to reproduce the results of a paper published in a peer-reviewed scientific journal. These chapters cover a variety of topics, but they all employ machine learning methods to heliophysics, which includes the study of the Sun and its effects on our solar system -- the Earth, planets, minor objects, and all of the space in between. 

Below is a short summary of each chapter. Each summary gives a brief overview of the machine learning methods and data types involved in solving a specific research problem. Each ">" symbol is designed to drill down from a general idea into a specific one. If some of these terms don't make sense, don't worry! Acronyms are defined at the bottom and the chapters explain each scientific and machine learning concept in detail.

## Chapter 1
* Author(s): James Paul Mason
* Objective: Fit time series measurements of solar ultraviolet light to contrast new and familiar concepts
* ML method(s) and concepts: 
	* Preprocessing > data cleaning > imputing ([sklearn.preprocessing.Imputer](https://sklearn.org/modules/generated/sklearn.preprocessing.Imputer.html))
	* Model selection > splitting data into training and validation sets > shuffle split ([sklearn.model\_selection.ShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html))
	* Regression > support vector machine > support vector regression ([sklearn.svm.SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html))
	* Model selection > determining best performing model > validation curve ([sklearn.model\_selection.validation\_curve](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html))
* Data source(s): 
	* Solar spectral irradiance > extreme ultraviolet light > extracted emission line time series > SDO/EVE
	
## Chapter 2
* Author(s): Monica Bobra
* Objective: Predict solar flares (outbursts of high energy light) and coronal mass ejections (outbursts of particles) based on measurements of the sun's surface magnetic field
* ML method(s) and concepts: 
	* Classification > support vector machine > support vector classifier ([sklearn.svm.svc](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html))
	* Model selection > splitting data into training and validation sets > stratified k-folds ([sklearn.model\_selection.StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html))
* Data sources(s): 
	* Solar surface magnetic field (AKA magnetograms) > SDO/HMI
	* Solar spectral irradiance > soft x-ray light > extracted flare peak intensity and time > GOES/XRS flare event database
	* Solar disk-blocked coronal images > visible light > extracted ejection occurrence and time > SOHO/LASCO and STEREO/SECCHI/COR coronal mass ejection database
* Published and refereed paper: [Bobra & Ilonidis, 2016, <i> Astrophysical Journal</i>, 821, 127](https://ui.adsabs.harvard.edu/#abs/2016ApJ...821..127B/abstract)

## Chapter 3
* Author(s): Carlos José Díaz Baso and Andrés Asensio Ramos
* Objective: Rapid and robust image resolution enhancement 
* ML method(s) and concepts: 
	* Classification > deconvolution > convolutional neural network ([keras](https://keras.io/))
	* Image processing > up-sampling > convolutional neural network ([keras](https://keras.io/))
	* Model selection > determining best performing model > regularization ([keras](https://keras.io/))
* Data source(s): 
	* Solar surface magnetic field (AKA magnetograms) > SDO/HMI
	* Solar surface images > visible light > SDO/HMI
	* Solar surface images > visible light > Hinode/SOT
* Published and refereed paper: [Díaz Baso & Asensio Ramos, 2018, <i> Astronomy & Astrophysics</i>, 614, A5](https://ui.adsabs.harvard.edu/#abs/2018A&A...614A...5D/abstract)

## Chapter 4
* Author(s): Paul Wright, Mark Cheung, Rajat Thomas, Richard Galvez, Alexandre Szenicer, Meng Jin, Andrés Muñoz-Jaramillo, and David Fouhey
* Objective: Simulating data from a lost instrument (EVE) based on another of a totally different type (AIA¹³)
* ML method(s) and concepts: 
	* Image transformation > mapping > convolutional neural networks ([pytorch](https://pytorch.org/))
	* Model selection > determining model performance > mean squared error loss ([torch.nn.MSELoss](https://pytorch.org/docs/0.3.1/nn.html#torch.nn.MSELoss))
* Data source(s): 
	* Solar spectral images > extreme ultraviolet light > SDO/AIA
	* Solar spectral irradiance > extreme ultraviolet light > extracted emission line time series > SDO/EVE
* Published and refereed paper: In progress

## Chapter 5 
* Author(s): Ryan M. McGranaghan, Anthony Mannucci, Brian Wilson, Chris Mattmann, Richard Chadwick
* Objective: Predicting high-latitude ionospheric scintillation
* ML method(s) and concepts:
	* Model selection > splitting data into training and validation sets > random ([sklearn.model\_selection.train\_test\_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html))
	* Classification > support vector machine > support vector classifier ([sklearn.svm.svc](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html))
	* Dimensionality reduction > feature selection > Fisher ranking score ([Gu et al. 2011](https://dslpitt.org/uai/papers/11/p266-gu.pdf))
* Data source(s): 
    * Solar wind > magnetic field strength and direction > ACE, Wind, IMP 8, and Geotail
    * Solar wind > velocity and pressure > ACE, Wind, IMP 8, and Geotail
    * Aurora > electrojets > various ground-based polar observatories complied by Kyoto WDCG
    * Geomagnetic field > symmetric disturbances > various ground-based polar observatories complied by Kyoto WDCG
    * Ionosphere > total electron content > GISTM
    * Ionosphere > radio spectrum > GISTM
    * Ionosphere > scintillation > GISTM
* Published and refereed paper: [McGranaghan et al., 2018, <i> Space Weather</i>, 16, 11](https://ui.adsabs.harvard.edu/#abs/2018SpWea..16.1817M/abstract)

## Chapter 6

* Author(s): Brandon Panos, Lucia Kleint, Cedric Huwyler, Säm Krucker, Martin Melchior, Denis Ullmann, Sviatoslav Voloshynovskiy
* Objective: Analyzing the behavior of a single spectral line (MgII) across many different flaring active regions
* ML method(s) and concepts: 
	* Clustering > K-means
* Data source(s): 
    * Solar spectral data > ultraviolet light > IRIS
* Published and refereed paper: [Panos et al., 2018, <i> Astrophysical Journal</i>, 861, 1](https://ui.adsabs.harvard.edu/#abs/2018ApJ...861...62P/abstract)   

## Chapter 7
* Author(s): Tobías Felipe and Andrés Asensio Ramos
* Objective: Detection of far-side active regions
* ML method(s) and concepts: 
	* Image transformation > mapping > convolutional neural networks ([pytorch](https://pytorch.org/))
	* Model selection > determining model performance > binary cross-entropy ([torch.nn.BCELoss](https://pytorch.org/docs/0.3.1/nn.html#torch.nn.BCELoss))
* Data source(s): 
	* Solar surface magnetic field (AKA magnetograms) > SDO/HMI
	* Solar far-side seismic maps > SDO/HMI	
* Published and refereed paper: [Felipe & Asensio Ramos, 2019, <i> Astronomy & Astrophysics</i>, 632, A82](https://ui.adsabs.harvard.edu/abs/2019A%26A...632A..82F/abstract)


## Future Chapters
* Contact us! Open an [issue on the GitHub repository](https://github.com/HelioML/HelioML/issues) with your idea. See our [guide for contributing here](https://github.com/HelioML/HelioML/blob/master/CONTRIBUTING.md). 

## Definitions
* ACE: Advanced Composition Explorer
* AIA: Atmospheric Imaging Assembly onboard SDO
* COR: Coronagraph onboard STEREO
* EVE: Extreme Ultraviolet Variability Experiment onboard SDO
* GISTM: Global Navigation Satellite System Ionospheric Scintillation and Total Electron Content Measurements Monitor
* GOES: Geostationary Operational Environmental Satellites
* HMI: Helioseismic Magnetic Imager onboard SDO'
* IRIS: Iterface Region Imaging Spectrograph
* Irradiance is the total output of light from the sun. Spectral irradiance is that intensity as a function of wavelength.
* LASCO: Large Angle and Spectrometric Coronagraph onboard SOHO
* SDO: Solar Dynamics Observatory
* SECCHI: Sun Earth Connection Coronal and Heliospheric Investigation suite of instruments onboard STEREO
* SOHO: Solar and Heliospheric Observatory
* SOT: Solar Optical Telescope onboard Hinode
* STEREO: Solar Terrestrial Relations Observatory
* WDCG: World Data Center for Geomagnetism
* XRS: X-Ray Sensor
