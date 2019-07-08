What's In This Book?
====================

In the following chapters, we'll cover examples from active research. Each chapter provides the motivation for the research and all of the code necessary to reproduce the results of a paper published in a peer-reviewed scientific journal. These chapters cover a variety of topics, but they all employ machine learning methods to heliophysics, which includes the study of the Sun and its effects on our solar system -- the Earth, planets, minor objects, and all of the space in between. 

Below is a short summary of each chapter. Each summary gives a brief overview of the machine learning methods and data types involved in solving a specific research problem. Each ">" symbol is designed to drill down from a general idea into a specific one. If some of these terms don't make sense, don't worry! The chapters explain each scientific and machine learning concept in detail.

## Chapter 1
* Author(s): James Paul Mason
* Objective: Fit time series measurements of solar ultraviolet light to contrast new and familiar concepts
* ML method(s) and concepts: 
	* Preprocessing > data cleaning > imputing ([sklearn.preprocessing.Imputer](https://sklearn.org/modules/generated/sklearn.preprocessing.Imputer.html))
	* Model selection > splitting data into training and validation sets > shuffle split ([sklearn.model\_selection.ShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html))
	* Regression > support vector machine > support vector regression ([sklearn.svm.SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html))
	* Model selection > determining best performing model > validation curve ([sklearn.model\_selection.validation\_curve](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html))
* Data source(s): 
	* Solar spectral irradiance¹ > extreme ultraviolet light > extracted emission line time series > SDO²/EVE³
	
## Chapter 2
* Author(s): Monica Bobra
* Objective: Predict solar flares (outbursts of high energy light) and coronal mass ejections (outbursts of particles) based on measurements of the sun's surface magnetic field
* ML method(s) and concepts: 
	* Classification > support vector machine > support vector classifier ([sklearn.svm.svc](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html))
	* Model selection > splitting data into training and validation sets > stratified k-folds ([sklearn.model\_selection.StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html))
* Data sources(s): 
	* Solar surface magnetic field (AKA magnetograms) > SDO/HMI⁴
	* Solar spectral irradiance > soft x-ray light > extracted flare peak intensity and time > GOES⁵/XRS⁶ flare event database
	* Solar disk-blocked coronal images > visible light > extracted ejection occurrence and time > SOHO⁷/LASCO⁸ and STEREO⁹/SECCHI¹⁰/COR¹¹ coronal mass ejection database
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
	* Solar surface images > visible light > Hinode/SOT¹²
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
    * Solar wind > 
    * geomagnetic activity, particle precipitation, and ionospheric Global Navigation Satellite System (GNSS) data
* Published and refereed paper: [McGranaghan et al., 2018, <i> Space Weather</i>, 16, 11](https://ui.adsabs.harvard.edu/#abs/2018SpWea..16.1817M/abstract)  

## Future Chapters
* Author(s): Brandon Panos, Lucia Kleint, Cedric Huwyler, Säm Krucker, Martin Melchior, Denis Ullmann, Sviatoslav Voloshynovskiy
* Objective: Analyzing the behavior of a single spectral line (MgII) across many different flaring active regions
* ML method(s) and concepts: 
	* Clustering > K-means
* Data source(s): 
    * Solar spectral data > ultraviolet light > IRIS
* Published and refereed paper: [Panos et al., 2018, <i> Astrophysical Journal</i>, 861, 1](https://ui.adsabs.harvard.edu/#abs/2018ApJ...861...62P/abstract)     

## Definitions
1. Irradiance is the total output of light from the sun. Spectral irradiance is that intensity as a function of wavelength.
2. SDO: Solar Dynamics Observatory
3. EVE: Extreme Ultraviolet Variability Experiment
4. HMI: Helioseismic Magnetic Imager
5. GOES: Geostationary Operational Environmental Satellites
6. XRS: X-Ray Sensor
7. SOHO: Solar and Heliospheric Observatory
8. LASCO: Large Angle and Spectrometric Coronagraph
9. STEREO: Solar Terrestrial Relations Observatory
10. SECCHI: Sun Earth Connection Coronal and Heliospheric Investigation
11. COR: Coronagraph
12. SOT: Solar Optical Telescope
13. AIA: Atmospheric Imaging Assembly
14. IRIS: Iterface Region Imaging Spectrograph