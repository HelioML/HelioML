---
title: '2. Predicting Coronal Mass Ejections'
permalink: 'chapters/02/Predicting-Coronal-Mass-Ejections'
previouschapter:
  url: chapters/01/1/prototype_light_curve_fit
  title: '1.1 prototype light curve fit'
nextchapter:
  url: chapters/02/1/notebook
  title: '2.1 Notebook'
redirect_from:
  - 'chapters/02/predicting-coronal-mass-ejections'
---
Predicting Coronal Mass Ejections
=================================

A Coronal Mass Ejection (CME) throws magnetic flux and plasma from the Sun into interplanetary space. These eruptions are actually related to solar flares -- in fact, CMEs and solar flares are considered “a single magnetically driven event” ([Webb & Howard 2012](http://adsabs.harvard.edu/abs/2012LRSP....9....3W)), wherein a flare unassociated with a CME is called a confined or compact flare. <br>

In general, the more energetic a flare, the more likely it is to be associated with a CME ([Yashiro et al. 2005](http://adsabs.harvard.edu/abs/2005JGRA..11012S05Y)) -- but this is not, by any means, a rule. For example, [Sun et al. (2015)](http://adsabs.harvard.edu/abs/2015ApJ...804L..28S) found that the largest active region in the last 24 years, shown below, produced 6 X-class flares but not a single observed CME.<br>

In this notebook, we will be predicting whether or not a flaring active region will also emit a CME using a machine learning algorithm from the scikit-learn package called Support Vector Machine.

The analysis that follows is published in [Bobra & Ilonidis, 2016, <i> Astrophysical Journal</i>, 821, 127](http://adsabs.harvard.edu/abs/2016ApJ...821..127B).