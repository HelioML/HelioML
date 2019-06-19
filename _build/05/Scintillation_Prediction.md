---
redirect_from:
  - "/05/scintillation-prediction"
title: 'Scintillation Prediction'
prev_page:
  url: /04/1/notebook
  title: 'Notebook'
next_page:
  url: /05/1/notebook
  title: 'Notebook'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---
New Capabilities for Prediction of High‐Latitude Ionospheric Scintillation: A Novel Approach With Machine Learning
============================================
*by Ryan M McGranaghan, Anthony J. Mannucci, Brian Wilson, Chris A Mattmann, and Richard Chadwick*

The purpose of this notebook is to detail the data science methodology used to produce classification (i.e., yes/no) predictions of high-latitude ionospheric phase scintillation. The reference for this work is:
McGranaghan, R.M., A.J. Mannucci, B.D Wilson, C.A. Mattmann, and R. Chadwick. (2018), New capabilities for prediction of high‐latitude ionospheric scintillation: A novel approach with machine learning, Space Weather, 16. https://doi.org/10.1029/2018SW002018.
and can be accessed here.
Abstract
As societal dependence on trans-ionospheric radio signals grows, understanding the impact of space weather on these signals is both increasingly important and remains unresolved. The challenge is particularly acute at high-latitudes where the effects of space weather are most direct and no reliable predictive capability exists. We take advantage of a large volume of data from Global Navigation Satellite Systems (GNSS) signals, increasingly sophisticated tools for data-driven discovery, and a machine learning algorithm known as the Support Vector Machine (SVM) to develop a novel predictive model for high-latitude ionospheric phase scintillation. This work, to our knowledge, represents the first time a SVM model has been created to predict high-latitude phase scintillation. We define a metric known as the Total Skill Score (TSS), which can be used to robustly compare between predictive methods, to evaluate the SVM model, thereby establishing a benchmark for high-latitude ionospheric phase scintillation. We find that the SVM model significantly outperforms persistence, doubling the predictive skill according to the TSS for a one hour predictive task. The increase in is even more pronounced as prediction time is increased. For a prediction time of three hours, persistence prediction is comparable to a random chance prediction, suggesting that the 'memory' of the ionosphere in terms of high-latitude plasma irregularities is on the order of or shorter than hours. The SVM model predictive skill only slightly decreases between the one and three hour predictive task, pointing to the potential of this method. Our findings can serve as a foundation on which to evaluate future predictive models, a critical development toward the resolution of space weather impact on trans-ionospheric radio signals.

This project was carried out during Ryan McGranaghan's Jack Eddy Living With a Star Fellowship at the NASA Jet Propulsion Laboratory. It was thereafter further explored and tailored for inclusion in the HelioML textbook. We gratefully thank Monica Bobra and James Mason for their efforts to bring this work into the wonderful HelioML book. The following acknowledgements were included in the original publication: 

This research was supported by the NASA Living With a Star Jack Eddy Postdoctoral Fellowship Program, administered by the University Corporation for Atmospheric Research and coordinated through the Cooperative Programs for the Advancement of Earth System Science (CPAESS). R. M. M. was also partially supported by the JPL Data Science Working Group. Portions of this research were carried out at the Jet Propulsion Laboratory, California Institute of Technology, under a contract with the National Aeronautics and Space Administration and funded by the Data Science Working Group Pilot Project Stretching Global Navigation Satellite Systems (GNSS) signals for Space Weather discovery. The authors gratefully acknowledge the input of P.T. Jayachandran, PI of the CHAIN network for guidance on data interpretation, details of phase scintillation/variation at high latitudes, and comments on the manuscript. CHAIN is supported by the Canadian Foundation for Innovation and the New Brunswick Innovation Foundation. CHAIN operation is conducted in collaboration with the Canadian Space Agency. Science funding is provided by the Natural Sciences and Engineering Research Council of Canada.
Data used in this work are
available from NASA’s Coordinated Data Analysis Web ([CDAWeb](https://cdaweb.sci.gsfc.nasa.gov/), the Oval Variation, Assessment, Tracking, Intensity, and Online Nowcasting Prime ([OVATION Prime] (http://sourceforge.net/ projects/ovation-prime/), and the [CHAIN](http://chain.physics.unb.ca/chain/). 
We openly and freely provide sample software to produce the results shown in this manuscript through a FigShare Project with
the same name as this paper and available at [McGranaghan et al., 2018a](https://doi.org/ 10.6084/m9.figshare.6813143).
Data from this work are also provided through the FigShare Project at  [McGranaghan et al., 2018b](https://doi.org/10.6084/ m9.figshare.6813131).
