*by C. J. Diaz Baso and A. Asensio Ramos*


In this chapter we will learn how to use and apply deep learning tecniques to improve the resolution of our images in a fast and robust way. We have developed a deep fully convolutional neural network which deconvolves and super-resolves continuum images and magnetograms observed with the Helioseismic and Magnetic Imager (HMI) satellite. This improvement allow us to analyze the smallest-scale events in the solar atmosphere.

We want to note that although almost all the examples/images are written in python, we have omitted some materials in their original format (usually large FITS files) to avoid increasing the size of this notebook.

The software resulting from this project, which was developed with the python library [keras](https://keras.io/), is hosted on [Github](https://github.com/cdiazbas/enhance) and published in [Díaz Baso & Asensio Ramos, 2018, <i> Astronomy & Astrophysics</i>, 614, A5](https://www.aanda.org/articles/aa/pdf/2018/06/aa31344-17.pdf).

![example](1/docs/imagen.gif)
Example of the software `Enhance` applied to real solar images.
