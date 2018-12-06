---
interact_link: notebooks/04/1/notebook.ipynb
title: 'Notebook'
permalink: 'chapters/04/1/notebook'
previouschapter:
  url: chapters/04/Differential_Emission_Measurements
  title: 'Differential Emission Measurements'
nextchapter:
  url: 
  title: ''
redirect_from:
  - 'chapters/04/1/notebook'
---

# DeepEM: A Deep Learning Approach for DEM Inversion
by Paul Wright$^{1}$, Mark Cheung$^{2,3}$, Rajat Thomas$^{4}$, Richard Galvez$^{5}$, Alexandre Szenicer$^{6}$, Meng Jin$^{2,7}$, Andrés Muñoz-Jaramillo$^{8}$, and David Fouhey$^{9}$


$^{1}$University of Glasgow, email: paul@pauljwright.co.uk;
$^{2}$Lockheed Martin Solar and Astrophysics Laboratory;
$^{3}$Stanford University;
$^{4}$University of Amsterdam;
$^{5}$New York University;
$^{6}$University of Oxford;
$^{7}$SETI Institute;
$^{8}$SouthWest Research Institute;
$^{9}$University of California, Berkeley 

---

The intensity observed through optically-thin <i>SDO</i>/AIA filters (94 Å, 131 Å, 171 Å, 193 Å, 211 Å, 335 Å) can be related to the temperature distribution of the solar corona (the differential emission measure; DEM) as

\begin{equation}
g_{i} = \int_{T} K_{i}(T) \xi(T) dT \, .
\end{equation}

In this equation, $g_{i}$ is the DN s$^{-1}$ px$^{-1}$ value in the $i$th SDO/AIA channel. This intensity corresponds to the $K_{i}(T)$ temperature response function, and the DEM, $\xi(T)$, is in units of cm$^{-5}$ K$^{-1}$. The matrix formulation of this integral equation can be represented in the form $\vec{g} = {\bf K}\vec{\xi}$, however this problem is an ill-posed inverse problem, and any attempt to directly recover $\vec{\xi}$ leads to significant noise amplification. 

There are numerous methods to tackle mathematical problems of this kind, and there are an increasing number of methods in the literature for recovering the differential emission measure from <i>SDO</i>/AIA observations, including methods based techniques such as Tikhonov Regularisation (<a href="https://doi.org/10.1051/0004-6361/201117576">Hannah & Kontar 2012</a>), on the concept of sparsity (<a href="https://doi.org/10.1088/0004-637X/807/2/143">Cheung <i>et al</i> 2015</a>).

In this notebook we will introduce a deep learning approach for DEM Inversion. <i>For this notebook</i>, DeepEM is a trained on one set of <i>SDO</i>/AIA observations (six optically thin channels; 6 x N x N) and DEM solutions (in 18 temperature bins from log$_{10}$T = 5.5 - 7.2, 18 x N x N; Cheung <i>et al</i> 2015) at a resolution of 512 x 512 (N = 512) using a 1x1 2D Convolutional Neural Network with a single hidden layer.

The DeepEM method presented here takes every DEM solution with no regards to the quality or existence of the solution. As will be demonstrated, when this method is trained with a single set images and DEM solutions, the DeepEM solutions have a similar fidelity to Basis Pursuit (with a significantly increased computation speed), and additionally, the DeepEM solutions find positive solutions at every pixel, and reduced noise in the DEM solutions.



{:.input_area}
```python
#This notebook has been written in PyTorch
import os
import json
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
```




{:.input_area}
```python
#cudaize determines if a gpu is available for training and testing
def cudaize(obj):
    return obj.cuda() if torch.cuda.is_available() else obj
```




{:.input_area}
```python
def em_scale(y):
    return np.sqrt(y/1e25)

def em_unscale(y):
    return 1e25*(y*y)

def img_scale(x):
    x2 = x
    bad = np.where(x2 <= 0.0)
    x2[bad] = 0.0
    return np.sqrt(x2)

def img_unscale(x):
    return x*x 
```


### Step 1: Obtain Data and Basis Pursuit Solutions for Training



We first load the <i>SDO</i>/AIA images and Basis Pursuit DEM maps.

N.B. While this simplified version of DeepEM has been trained on DEM maps from Basis Pursuit (Cheung <i>et al</i> 2015), we actively encourage the readers to try their favourite method for DEM inversion! 



{:.input_area}
```python
aia_files = ['AIA_DEM_2011-01-27','AIA_DEM_2011-02-22','AIA_DEM_2011-03-20']
em_cube_files = aia_files
status_files = aia_files
for k, (afile, emfile) in enumerate(zip(aia_files, em_cube_files)):
    afile_name = os.path.join('./DeepEM_Data/', afile + '.aia.npy')
    emfile_name = os.path.join('./DeepEM_Data/', emfile + '.emcube.npy')
    status_name = os.path.join('./DeepEM_Data/', emfile + '.status.npy')
    if k == 0:
        X = np.load(afile_name)
        y = np.load(emfile_name)
        status = np.load(status_name)
        
        X = np.zeros((len(aia_files), X.shape[0], X.shape[1], X.shape[2]))
        y = np.zeros((len(em_cube_files), y.shape[0], y.shape[1], y.shape[2]))
        status = np.zeros((len(status_files), status.shape[0], status.shape[1]))

        nlgT = y.shape[0]
        lgtaxis = np.arange(y.shape[1])*0.1 + 5.5
        
    X[k] = np.load(afile_name)
    y[k] = np.load(emfile_name) 
```


### Step 2: Define the Model

We first define the model as a 1x1 2D Convolutional Neural Network (CNN) with a kernel size of 1x1 and a single hidden layer. The model accepts a data cube of 6 x N x N (<i>SDO</i>/AIA data), and returns a data cube of 18 x N x N (DEM). When trained, this will transform the input (each pixel of the 6 <i>SDO</i>/AIA channels; 6 x 1 x 1) to the output (DEM at each pixel; 18 x 1 x 1).



{:.input_area}
```python
model = nn.Sequential(
    nn.Conv2d(6, 300, kernel_size=1),
    nn.LeakyReLU(), 
    nn.Conv2d(300, 300, kernel_size=1),
    nn.LeakyReLU(),
    nn.Conv2d(300, 18, kernel_size=1))

model = cudaize(model)
```


### Step 3: Train the Model

For training, we select one <i>SDO</i>/AIA data cube (6 x 512 x 512) and the corresponding Basis Pursuit DEM output (18 x 512 x 512). In the case presented here, we train the CNN on an image of the Sun obtained on the 27 Jan  2011; validate on an image of the Sun obtained one synodic rotation later (+26 days; 22-02-2011); and finally test on an image another 26 days later (20-03-2011).



{:.input_area}
```python
X = img_scale(X)
y = em_scale(y)

X_train = X[0:1] 
y_train = y[0:1] 

X_val = X[1:2] 
y_val = y[1:2] 

X_test = X[2:3] 
y_test = y[2:3]
```


#### Plotting SDO/AIA Observations ${\it vs.}$ Basis Pursuit DEM bins

For the test data set, the <i>SDO</i>/AIA images for 171 Å, 211 Å, and 94 Å, and the corresponding DEM bins near the peak sensitivity in these channels (log$_{10}$T = 5.9, 6.3, 7.0) are shown in Figure 1. Figure 1 shows a set of <i>SDO</i>/AIA images (171 Å, 211 Å, and 94 Å [top, left to right]) with the corresponding DEM maps (bottom) for temperature bins there are near the peak sensitivity of the <i>SDO</i>/AIA channel. Furthermore, it is clear from the DEM maps that a number of pixels that are $zero$. These pixels are primarily located off-disk, but there are a number of pixels on-disk that show this behaviour.



{:.input_area}
```python
fig,ax=plt.subplots(ncols=3,figsize=(9*2,9))

ax[0].imshow(X_test[0,2,:,:],vmin=0.01,vmax=30,cmap='Greys_r', origin='lower')
ax[0].text(5, 490, '${\it SDO}$/AIA 171 $\AA$', color="white", size='large')
ax[1].imshow(X_test[0,4,:,:],vmin=0.25,vmax=25,cmap='Greys_r', origin='lower')
ax[1].text(5, 490, '${\it SDO}$/AIA 211 $\AA$', color="white", size='large')
ax[2].imshow(X_test[0,0,:,:],vmin=0.01,vmax=3,cmap='Greys_r', origin='lower')
ax[2].text(5, 490, '${\it SDO}$/AIA 94 $\AA$', color="white", size='large')

for axes in ax:
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    
fig,ax=plt.subplots(ncols=3,figsize=(9*2,9))

ax[0].imshow(y_test[0,4,:,:],vmin=0.01,vmax=3,cmap='viridis', origin='lower')
ax[0].text(5, 490, 'Basis Pursuit DEM log$_{10}$T ~ 5.9', color="white", size='large')
ax[1].imshow(y_test[0,8,:,:],vmin=0.25,vmax=10,cmap='viridis', origin='lower')
ax[1].text(5, 490, 'Basis Pursuit DEM log$_{10}$T ~ 6.3', color="white", size='large')
ax[2].imshow(y_test[0,15,:,:],vmin=0.01,vmax=3,cmap='viridis', origin='lower')
ax[2].text(5, 490, 'Basis Pursuit DEM log$_{10}$T ~ 7.0', color="white", size='large')

for axes in ax:
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
```



![png](../../../images/chapters/04/1/notebook_16_0.png)



![png](../../../images/chapters/04/1/notebook_16_1.png)


<b>Figure 1:</b> Left to Right: <i>SDO</i>/AIA images in 171 Å, 211 Å, and 94 Å (top, left to right), with the corresponding DEM bins (chosen at the peak sensitivity of each of the <i>SDO</i>/AIA channels) shown below. In the DEM bins (bottom) it is clear that there are some pixels that have solutions of DEM = $zero$, as explicitly seen as dark regions/clusters of pixels on and off disk.

---

To implement training and testing of our model, we first define a DEMdata class, and define functions for training and validation/test: `train_model`, and `valtest_model`. 

N.B. It is not necessary to train the model, and if required, the trained model can be loaded to the cpu as follows:

```python
model = nn.Sequential(
    nn.Conv2d(6, 300, kernel_size=1),
    nn.LeakyReLU(),
    nn.Conv2d(300, 300, kernel_size=1),
    nn.LeakyReLU(),
    nn.Conv2d(300, 18, kernel_size=1))

dem_model_file = 'DeepEM_CNN_HelioML.pth'
model.load_state_dict(torch.load(dem_model_file))

model = cudaize(model)
```



{:.input_area}
```python
class DEMdata(nn.Module):
    def __init__(self, xtrain, ytrain, xtest, ytest, xval, yval, split='train'):
        
        if split == 'train':
            self.x = xtrain
            self.y = ytrain
        if split == 'val':
            self.x = xval
            self.y = yval
        if split == 'test':
            self.x = xtest
            self.y = ytest
            
    def __getitem__(self, index):
        return torch.from_numpy(self.x[index]).type(torch.FloatTensor), torch.from_numpy(self.y[index]).type(torch.FloatTensor)

    def __len__(self):
        return self.x.shape[0]
```




{:.input_area}
```python
def train_model(dem_loader, criterion, optimizer, epochs=500):
    model.train()
    train_loss_all_batches = []
    train_loss_epoch = []
    train_val = []
    for k in range(epochs):
        count_ = 0
        avg_loss = 0
        # =================== progress indicator ==============
        if k % ((epochs + 1) // 4) == 0:
            print('[{0}]: {1:.1f}% complete: '.format(k, k / epochs * 100))
        # =====================================================
        for img, dem in dem_loader:
            count_ += 1
            optimizer.zero_grad()
            # =================== forward =====================
            img = cudaize(img)
            dem = cudaize(dem)

            output = model(img) 
            loss = criterion(output, dem)

            loss.backward()
            optimizer.step()
            
            train_loss_all_batches.append(loss.item())
            avg_loss += loss.item()
        # =================== Validation ===================
        dem_data_val = DEMdata(X_train, y_train, X_test, y_test, X_val, y_val, split='val')
        dem_loader_val = DataLoader(dem_data_val, batch_size=1)
        val_loss, dummy, dem_pred_val, dem_in_test_val = valtest_model(dem_loader_val, criterion)
        
        train_loss_epoch.append(avg_loss/count_)
        train_val.append(val_loss)
        
        if k%10 == 0: 
            print('Epoch: ', k, 'trn_loss: ', avg_loss/count_, 'val_loss: ', train_val[k])
            
    torch.save(model.state_dict(), 'DeepEM_CNN_HelioML.pth')
    return train_loss_epoch, train_val

def valtest_model(dem_loader, criterion):

    model.eval()
    
    val_loss = 0
    count = 0
    test_loss = []
    for img, dem in dem_loader:
        count += 1
        # =================== forward =====================
        img = cudaize(img)
        dem = cudaize(dem)
        
        output = model(img)
        loss = criterion(output, dem)
        test_loss.append(loss.item())
        val_loss += loss.item()
        
    return val_loss/count, test_loss, output, dem
```


We chose the Adam optimiser with a learning rate of 1e-4, and weight_decay set to 1e-9. We use Mean Squared Error (MSE) between the Basis Pursuit DEM map and the DeepEM map as our loss function.



{:.input_area}
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-9); 
criterion = cudaize(nn.MSELoss())
```


Using the defined functions, `dem_data` will return the training data, and this will be loaded by the `DataLoader` with batch_size=1 (one 512 x 512 image per batch). For each epoch, `train_loss` and `valdn_loss` will be returned by `train_model`.



{:.input_area}
```python
dem_data = DEMdata(X_train, y_train, X_test, y_test, X_val, y_val, split='train')
dem_loader = DataLoader(dem_data, batch_size=1)

t0=time.time() #Timing how long it takes to predict the DEMs
train_loss, valdn_loss = train_model(dem_loader, criterion, optimizer, epochs=500)
ttime = "Training time = {0} seconds".format(time.time()-t0)
print(ttime)
```


{:.output_stream}
```
[0]: 0.0% complete: 
Epoch:  0 trn_loss:  3.328942060470581 val_loss:  3.5720224380493164
Epoch:  10 trn_loss:  1.6681122779846191 val_loss:  1.82981538772583
Epoch:  20 trn_loss:  0.8270796537399292 val_loss:  0.9576936364173889
Epoch:  30 trn_loss:  0.5821843147277832 val_loss:  0.7041575908660889
Epoch:  40 trn_loss:  0.5572682619094849 val_loss:  0.6688917279243469
Epoch:  50 trn_loss:  0.5128538012504578 val_loss:  0.62626713514328
Epoch:  60 trn_loss:  0.4758664667606354 val_loss:  0.5860872864723206
Epoch:  70 trn_loss:  0.4503079950809479 val_loss:  0.5547241568565369
Epoch:  80 trn_loss:  0.42846399545669556 val_loss:  0.526684045791626
Epoch:  90 trn_loss:  0.41002219915390015 val_loss:  0.504288911819458
Epoch:  100 trn_loss:  0.3946799635887146 val_loss:  0.4836472272872925
Epoch:  110 trn_loss:  0.3812086284160614 val_loss:  0.4666471481323242
Epoch:  120 trn_loss:  0.3687494397163391 val_loss:  0.4511677026748657
[125]: 25.0% complete: 
Epoch:  130 trn_loss:  0.3568281829357147 val_loss:  0.43708622455596924
Epoch:  140 trn_loss:  0.34481948614120483 val_loss:  0.42316001653671265
Epoch:  150 trn_loss:  0.3325572609901428 val_loss:  0.40893223881721497
Epoch:  160 trn_loss:  0.3206677734851837 val_loss:  0.3941940665245056
Epoch:  170 trn_loss:  0.3089371919631958 val_loss:  0.3809552490711212
Epoch:  180 trn_loss:  0.29662102460861206 val_loss:  0.3658013343811035
Epoch:  190 trn_loss:  0.2858353853225708 val_loss:  0.3530679941177368
Epoch:  200 trn_loss:  0.2759355306625366 val_loss:  0.3411480188369751
Epoch:  210 trn_loss:  0.2670588195323944 val_loss:  0.3305090665817261
Epoch:  220 trn_loss:  0.2591996192932129 val_loss:  0.3207198977470398
Epoch:  230 trn_loss:  0.2523084282875061 val_loss:  0.311989426612854
Epoch:  240 trn_loss:  0.246346578001976 val_loss:  0.3043992817401886
[250]: 50.0% complete: 
Epoch:  250 trn_loss:  0.241195946931839 val_loss:  0.2977723479270935
Epoch:  260 trn_loss:  0.2366849035024643 val_loss:  0.2920348644256592
Epoch:  270 trn_loss:  0.23266059160232544 val_loss:  0.28685781359672546
Epoch:  280 trn_loss:  0.22896899282932281 val_loss:  0.28227299451828003
Epoch:  290 trn_loss:  0.22538073360919952 val_loss:  0.27806150913238525
Epoch:  300 trn_loss:  0.2220344990491867 val_loss:  0.2743688225746155
Epoch:  310 trn_loss:  0.21884138882160187 val_loss:  0.2709724009037018
Epoch:  320 trn_loss:  0.2158205509185791 val_loss:  0.2678989768028259
Epoch:  330 trn_loss:  0.21288469433784485 val_loss:  0.26472389698028564
Epoch:  340 trn_loss:  0.21004413068294525 val_loss:  0.2616737484931946
Epoch:  350 trn_loss:  0.20724453032016754 val_loss:  0.25884389877319336
Epoch:  360 trn_loss:  0.2044813632965088 val_loss:  0.2559853792190552
Epoch:  370 trn_loss:  0.201777383685112 val_loss:  0.25315842032432556
[375]: 75.0% complete: 
Epoch:  380 trn_loss:  0.19900348782539368 val_loss:  0.2499287873506546
Epoch:  390 trn_loss:  0.19637078046798706 val_loss:  0.24727076292037964
Epoch:  400 trn_loss:  0.1936444491147995 val_loss:  0.24451886117458344
Epoch:  410 trn_loss:  0.1910942643880844 val_loss:  0.24197131395339966
Epoch:  420 trn_loss:  0.18861205875873566 val_loss:  0.23903696238994598
Epoch:  430 trn_loss:  0.18618348240852356 val_loss:  0.2365807741880417
Epoch:  440 trn_loss:  0.18382203578948975 val_loss:  0.23406410217285156
Epoch:  450 trn_loss:  0.18153594434261322 val_loss:  0.23166057467460632
Epoch:  460 trn_loss:  0.17931421101093292 val_loss:  0.2293083369731903
Epoch:  470 trn_loss:  0.1771312803030014 val_loss:  0.22700132429599762
Epoch:  480 trn_loss:  0.17498815059661865 val_loss:  0.22472193837165833
Epoch:  490 trn_loss:  0.17292694747447968 val_loss:  0.22251540422439575
Training time = 46.060962200164795 seconds

```

#### Plotting: MSE Loss for Training and Validation

In order to understand how well the model has trained we plot the training loss and validation loss as a function of Epoch in Figure 2. Figure 2 shows the MSE loss for training (blue) and validation (orange) as a function of epoch.



{:.input_area}
```python
plt.plot(np.arange(len(train_loss)), train_loss, color="blue")
plt.plot(np.arange(len(train_loss)), valdn_loss, color="orange")
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.show()
```



![png](../../../images/chapters/04/1/notebook_28_0.png)


<b>Figure 2:</b> Training and Validation MSE loss (blue, orange) as a function of Epoch.

---

### Step 4: Testing the Model

Now that the model has been trained, testing the model is a computationally cheap proceedure. As before, we choose the data using `DEMdata`, and load with `DataLoader`. Using `valtest_model`, the DeepEM map is created, and the MSE loss calculated as during training.



{:.input_area}
```python
dem_data_test = DEMdata(X_train, y_train, X_test, y_test, X_val, y_val, split='test')
dem_loader = DataLoader(dem_data_test, batch_size=1)
```




{:.input_area}
```python
t0=time.time() #Timing how long it takes to predict the DEMs
dummy, test_loss, dem_pred, dem_in_test = valtest_model(dem_loader, criterion)
performance = "Number of DEM solutions per second = {0}".format((y_test.shape[2]*y_test.shape[3])/(time.time()-t0))
```




{:.input_area}
```python
print(performance)
```


{:.output_stream}
```
Number of DEM solutions per second = 10867316.634142485

```

#### Plotting: AIA, Basis Pursuit, DeepEM

With the DeepEM map calculated, we can now compare the solutions obtained by Basis Pursuit and DeepEM. Figure 3 is similar to Figure 1 with an additional row corresponding to the solutions for DeepEM. Figure 3 shows <i>SDO</i>/AIA images in 171 Å, 211 Å, and 94 Å (left, top to bottom), with the corresponding DEM bins from Basis Pursuit (chosen at the peak sensitivity of each of the <i>SDO</i>/AIA channels) shown in the middle (top to bottom). The right-hand column row shows the DeepEM solutions that correspond to the same bins as the Basis Pursuit solutions. DeepEM provides solutions that are similar to Basis Pursuit, but importantly, provides DEM solutions for every pixel.



{:.input_area}
```python
fig,ax=plt.subplots(ncols=3,figsize=(9*2,9))

ax[0].imshow(X_test[0,2,:,:],vmin=0.01,vmax=30,cmap='Greys_r',origin='lower')
ax[0].text(5, 490, '${\it SDO}$/AIA 171 $\AA$', color="white", size='large')
ax[1].imshow(dem_in_test[0,4,:,:].cpu().detach().numpy(),vmin=0.01,vmax=3,cmap='viridis',origin='lower')
ax[1].text(5, 490, 'Basis Pursuit DEM log$_{10}$T ~ 5.9', color="white", size='large')
ax[2].imshow(dem_pred[0,4,:,:].cpu().detach().numpy(),vmin=0.01,vmax=3,cmap='viridis',origin='lower')
ax[2].text(5, 490, 'DeepEM log$_{10}$T ~ 5.9', color="white", size='large')

for axes in ax:
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    
fig,ax=plt.subplots(ncols=3,figsize=(9*2,9))

ax[0].imshow(X_test[0,4,:,:],vmin=0.25,vmax=25,cmap='Greys_r',origin='lower')
ax[0].text(5, 490, '${\it SDO}$/AIA 211 $\AA$', color="white", size='large')
ax[1].imshow(dem_in_test[0,8,:,:].cpu().detach().numpy(),vmin=0.25,vmax=10,cmap='viridis',origin='lower')
ax[1].text(5, 490, 'Basis Pursuit DEM log$_{10}$T ~ 6.3', color="white", size='large')
ax[2].imshow(dem_pred[0,8,:,:].cpu().detach().numpy(),vmin=0.25,vmax=10,cmap='viridis',origin='lower')
ax[2].text(5, 490, 'DeepEM log$_{10}$T ~ 6.3', color="white", size='large')

for axes in ax:
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    
fig,ax=plt.subplots(ncols=3,figsize=(9*2,9))

ax[0].imshow(X_test[0,0,:,:],vmin=0.01,vmax=3,cmap='Greys_r',origin='lower')
ax[0].text(5, 490, '${\it SDO}$/AIA 94 $\AA$', color="white", size='large')
ax[1].imshow(dem_in_test[0,15,:,:].cpu().detach().numpy(),vmin=0.01,vmax=3,cmap='viridis',origin='lower')
ax[1].text(5, 490, 'Basis Pursuit DEM log$_{10}$T ~ 7.0', color="white", size='large')
ax[2].imshow(dem_pred[0,15,:,:].cpu().detach().numpy(),vmin=0.01,vmax=3,cmap='viridis',origin='lower')
ax[2].text(5, 490, 'DeepEM log$_{10}$T ~ 7.0', color="white", size='large')

for axes in ax:
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
```



![png](../../../images/chapters/04/1/notebook_36_0.png)



![png](../../../images/chapters/04/1/notebook_36_1.png)



![png](../../../images/chapters/04/1/notebook_36_2.png)


<b>Figure 3</b>: Left to Right: <i>SDO</i>/AIA images in 171 Å, 211 Å, and 94 Å (left, top to bottom), with the corresponding DEM bins from Basis Pursuit (chosen at the peak sensitivity of each of the <i>SDO</i>/AIA channels) shown below (middle, top to bottom). The right-hand column shows the DeepEM solutions that correspond to the same bins as the Basis Pursuit solutions. DeepEM provides solutions that are similar to Basis Pursuit, but importantly, provides DEM solutions for every pixel.



---

Furthermore, as we have the original Basis Pursuit DEM solutions for the test set ("the ground truth"), we can compare the average DEM from Basis Pursuit to the average DEM from DeepEM, as they should be similar. Figure 4 shows the average Basis Pursuit DEM (black curve) and the DeepEM solution (dashed line).



{:.input_area}
```python
def PlotTotalEM(em_unscaled, em_pred_unscaled, lgtaxis, status):
    mask = np.zeros([status.shape[0],status.shape[1]])
    mask[np.where(status == 0.0)] = 1.0
    nmask = np.sum(mask)
    
    EM_tru_sum = np.zeros([lgtaxis.size])
    EM_inv_sum = np.zeros([lgtaxis.size])
    
    for i in range(lgtaxis.size):
        EM_tru_sum[i] = np.sum(em_unscaled[0,i,:,:]*mask)/nmask
        EM_inv_sum[i] = np.sum(em_pred_unscaled[0,i,:,:]*mask)/nmask
        
    fig = plt.figure   
    plt.plot(lgtaxis,EM_tru_sum, linewidth=3, color="black")
    plt.plot(lgtaxis,EM_inv_sum, linewidth=3, color="lightblue", linestyle='--')
    plt.tick_params(axis='both', which='major')#, labelsize=16)
    plt.tick_params(axis='both', which='minor')#, labelsize=16)
    
    dlogT = lgtaxis[1]-lgtaxis[0]
    
    plt.xlim(lgtaxis[0]-0.5*dlogT, lgtaxis.max()+0.5*dlogT)
    plt.xticks(np.arange(np.min(lgtaxis), np.max(lgtaxis),2*dlogT))
    plt.ylim(1e24,1e27)
    plt.yscale('log')
    plt.xlabel('log$_{10}$T [K]')
    plt.ylabel('Mean Emission Measure [cm$^{-5}$]')
    
    plt.show()
    return EM_inv_sum, EM_tru_sum
```




{:.input_area}
```python
em_unscaled = em_unscale(dem_in_test.detach().cpu().numpy())
em_pred_unscaled = em_unscale(dem_pred.detach().cpu().numpy())

# Status for the test data. While DeepEM was trained on all examples in the training set, 
# we only compare the DEMs where Basis Pursuit obtained a solution (status = 0)
status = status[2,:,:]

EMinv, EMTru = PlotTotalEM(em_unscaled,em_pred_unscaled,lgtaxis,status)
```



![png](../../../images/chapters/04/1/notebook_41_0.png)


<b>Figure 4</b>: Average Basis Pursuit DEM (plotted as mean emission measure, black line) and the Average DeepEM solution (dashed line). It is clear that this simple implementation of DeepEM provides, on average, DEMs that are similar to Basis Pursuit (Cheung <i>et al</i> 2015).

---

### Step 5: Synthesize <i>SDO</i>/AIA Observations

Finally, it is also of interest to reconstruct the <i>SDO</i>/AIA observations from both the Basis Pursuit, and DeepEM solutions. 

We are able to pose the problem of reconstructing the <i>SDO</i>/AIA observations from the DEM as a 1x1 2D Convolution. We first define the weights as the response functions of each channel, and set the biases to $zero$. By convolving the unscaled DEM at each pixel with the 6 filters (one for each <i>SDO</i>/AIA response function), we can recover the <i>SDO</i>/AIA observations.



{:.input_area}
```python
# We first load the AIA response functions:
cl = np.load('./DeepEM_Data/AIA_Resp.npy')
```




{:.input_area}
```python
# Used Conv2d to convolve?? every pixel (18x1x1) by the 6 response functions
# to return a set of observed fluxes in each channel (6x1x1)
dem2aia = cudaize(nn.Conv2d(18, 6, kernel_size=1))

chianti_lines_2 = cudaize(torch.zeros(6,18,1,1))
biases = cudaize(torch.zeros(6))

# set the weights to each of the SDO/AIA response functions and biases to zero
for i, p in enumerate(dem2aia.parameters()):
    if i == 0:
        p.data = Variable(cudaize(torch.from_numpy(cl).type(torch.FloatTensor)))
    else:
        p.data = biases 
```




{:.input_area}
```python
AIA_out = img_scale(dem2aia(Variable(em_unscale(dem_in_test))).detach().cpu().numpy())
AIA_out_DeepEM = img_scale(dem2aia(Variable(em_unscale(dem_pred))).detach().cpu().numpy())
```


#### Plotting SDO/AIA Observations and Synthetic Observations



{:.input_area}
```python
fig,ax=plt.subplots(ncols=3,figsize=(9*2,9))

ax[0].imshow(X_test[0,2,:,:],vmin=0.01,vmax=30,cmap='Greys_r',origin='lower')
ax[0].text(5, 490, '${\it SDO}$/AIA 171 $\AA$', color="white", size='large')
ax[1].imshow(AIA_out[0,2,:,:],vmin=0.01,vmax=30,cmap='Greys_r',origin='lower')
ax[1].text(5, 490, 'Basis Pursuit Synthesized 171 $\AA$', color="white", size='large')
ax[2].imshow(AIA_out_DeepEM[0,2,:,:],vmin=0.01,vmax=30,cmap='Greys_r',origin='lower')
ax[2].text(5, 490, 'DeepEM Synthesized 171 $\AA$', color="white", size='large')

for axes in ax:
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    
fig,ax=plt.subplots(ncols=3,figsize=(9*2,9))

ax[0].imshow(X_test[0,4,:,:],vmin=0.25,vmax=25,cmap='Greys_r',origin='lower')
ax[0].text(5, 490, '${\it SDO}$/AIA 211 $\AA$', color="white", size='large')
ax[1].imshow(AIA_out[0,4,:,:],vmin=0.25,vmax=25,cmap='Greys_r',origin='lower')
ax[1].text(5, 490, 'Basis Pursuit Synthesized 211 $\AA$', color="white", size='large')
ax[2].imshow(AIA_out_DeepEM[0,4,:,:],vmin=0.25,vmax=25,cmap='Greys_r',origin='lower')
ax[2].text(5, 490, 'DeepEM Synthesized 211 $\AA$', color="white", size='large')

for axes in ax:
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    
fig,ax=plt.subplots(ncols=3,figsize=(9*2,9))

ax[0].imshow(X_test[0,0,:,:],vmin=0.01,vmax=3,cmap='Greys_r',origin='lower')
ax[0].text(5, 490, '${\it SDO}$/AIA 94 $\AA$', color="white", size='large')
ax[1].imshow(AIA_out[0,0,:,:],vmin=0.01,vmax=3,cmap='Greys_r',origin='lower')
ax[1].text(5, 490, 'Basis Pursuit Synthesized 94 $\AA$', color="white", size='large')
ax[2].imshow(AIA_out_DeepEM[0,0,:,:],vmin=0.01,vmax=3,cmap='Greys_r',origin='lower')
ax[2].text(5, 490, 'DeepEM Synthesized 94 $\AA$', color="white", size='large')

for axes in ax:
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False) 
```



![png](../../../images/chapters/04/1/notebook_50_0.png)



![png](../../../images/chapters/04/1/notebook_50_1.png)



![png](../../../images/chapters/04/1/notebook_50_2.png)


<b>Figure 5:</b> Top to Bottom: <i>SDO</i>/AIA images in 171 Å, 211 Å, and 94 Å (left, top to bottom) with the corresponding synthesised observations from Basis Pursuit (middle, top to bottom) and DeepEM (right, top to bottom). DeepEM provides synthetic observations that are similar to Basis Pursuit, with the addition of being able to reconstruct <i>SDO</i>/AIA observations where the basis pursuit solution was $zero$.

---

### Doing this on your data

There are two way in which you can use this notebook:

1. Train your own model: Instead of Basis Pursuit solutions (as used here), you could use your favourite inversion technique to generate the training data and then feed that into the training.

2. Directly used the pre-trained model we provide and perform inference on your AIA images.

#### Formats:

* Input: 6 x N x N (where 6 is the number of input AIA/SDO channels) as .npy file
* Output: Nt x N x N (where Nt is the number of temperature bins, 18 in our case)

where N is the dimension of the image.

#### CAUTION:
Training your model on a CPU might long!
Inference on the other hand is fast.

### Discussion

This chapter has provided a simple example of how a 1x1 2D Convolutional Neural Network can be used to improve computational cost for DEM inversion. Future development of DeepEM is on-going, and this notebook can be improved in a few ways:

1. By using both the original, and synthesised data from the DEM, the ability of the DEM to recover the original or supplementary data (such as spectroscopic EUV data) can be used as an additional term in the loss function. 

2. This implementation of DeepEM has been trained on a <i>single</i> set of observations. While there are 512$^{2}$ DEMs in one set of observations, it would be advisable to train the model to further images of the Sun in various states of activity including times of solar flaring.

3. For simplicity this implementation of DeepEM has been trained on every single pixel in the training set with no with no regards to the quality or existence of the solution. If trained for enough Epochs, DeepEM will start to remember which combinations of <i>SDO</i>AIA values lead to DEMs equal to $zero$ in the original training set. By utilising the status files included in this notebook it would be advisable to only train DeepEM on pixels where the solutions exist.

---

### Appendix A: What has the CNN learned about our training set?

If we say that our training set is now our test set, we can see how much the CNN has learned about the training data.



{:.input_area}
```python
X_test = X_train 
y_test = y_train
```




{:.input_area}
```python
dem_data_test = DEMdata(X_train, y_train, X_test, y_test, X_val, y_val, split='test')
dem_loader = DataLoader(dem_data_test, batch_size=1)

dummy, test_loss, dem_pred_trn, dem_in_test_trn = valtest_model(dem_loader, criterion)
```




{:.input_area}
```python
AIA_out = img_scale(dem2aia(Variable(em_unscale(dem_in_test_trn))).detach().cpu().numpy())
AIA_out_DeepEM = img_scale(dem2aia(Variable(em_unscale(dem_pred_trn))).detach().cpu().numpy())
```




{:.input_area}
```python
fig,ax=plt.subplots(ncols=3,figsize=(9*2,9))


ax[0].imshow(X_test[0,2,:,:],vmin=0.01,vmax=30,cmap='Greys_r',origin='lower')
ax[0].text(5, 490, '${\it SDO}$/AIA 171 $\AA$', color="white", size='large')
ax[1].imshow(dem_in_test_trn[0,4,:,:].cpu().detach().numpy(),vmin=0.01,vmax=3,cmap='viridis',origin='lower')
ax[1].text(5, 490, 'Basis Pursuit DEM log$_{10}$T ~ 5.9', color="white", size='large')
ax[2].imshow(dem_pred_trn[0,4,:,:].cpu().detach().numpy(),vmin=0.01,vmax=3,cmap='viridis',origin='lower')
ax[2].text(5, 490, 'DeepEM log$_{10}$T ~ 5.9', color="white", size='large')

for axes in ax:
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    
fig,ax=plt.subplots(ncols=3,figsize=(9*2,9))

ax[0].imshow(X_test[0,4,:,:],vmin=0.25,vmax=25,cmap='Greys_r',origin='lower')
ax[0].text(5, 490, '${\it SDO}$/AIA 211 $\AA$', color="white", size='large')
ax[1].imshow(dem_in_test_trn[0,8,:,:].cpu().detach().numpy(),vmin=0.25,vmax=10,cmap='viridis',origin='lower')
ax[1].text(5, 490, 'Basis Pursuit DEM log$_{10}$T ~ 6.3', color="white", size='large')
ax[2].imshow(dem_pred_trn[0,8,:,:].cpu().detach().numpy(),vmin=0.25,vmax=10,cmap='viridis',origin='lower')
ax[2].text(5, 490, 'DeepEM log$_{10}$T ~ 6.3', color="white", size='large')

for axes in ax:
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    
fig,ax=plt.subplots(ncols=3,figsize=(9*2,9))

ax[0].imshow(X_test[0,0,:,:],vmin=0.01,vmax=3,cmap='Greys_r',origin='lower')
ax[0].text(5, 490, '${\it SDO}$/AIA 94 $\AA$', color="white", size='large')
ax[1].imshow(dem_in_test_trn[0,15,:,:].cpu().detach().numpy(),vmin=0.01,vmax=3,cmap='viridis',origin='lower')
ax[1].text(5, 490, 'Basis Pursuit DEM log$_{10}$T ~ 7.0', color="white", size='large')
ax[2].imshow(dem_pred_trn[0,15,:,:].cpu().detach().numpy(),vmin=0.01,vmax=3,cmap='viridis',origin='lower')
ax[2].text(5, 490, 'DeepEM log$_{10}$T ~ 7.0', color="white", size='large')

for axes in ax:
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
```



![png](../../../images/chapters/04/1/notebook_62_0.png)



![png](../../../images/chapters/04/1/notebook_62_1.png)



![png](../../../images/chapters/04/1/notebook_62_2.png)


<b>Figure A1</b>: Left to Right: <i>SDO</i>/AIA images in 171 Å, 211 Å, and 94 Å (left, top to bottom) for the training set, with the corresponding DEM bins from Basis Pursuit (chosen at the peak sensitivity of each of the <i>SDO</i>/AIA channels) shown below (middle, top to bottom). The right-hand column shows the DeepEM solutions that correspond to the same bins as the Basis Pursuit solutions. DeepEM provides solutions that are similar to Basis Pursuit, but importantly, provides DEM solutions for every pixel.



What this shows is that even in training the model has not learned the exact mapping from specific <i>SDO</i>/AIA observations to DEMs, and there is sufficient generalisation that the $zero$ DEMs are not learned by the model. 

Finally, we can synthesise the <i>SDO</i>/AIA observations, as previously.



{:.input_area}
```python
fig,ax=plt.subplots(ncols=3,figsize=(9*2,9))

ax[0].imshow(X_test[0,2,:,:],vmin=0.01,vmax=30,cmap='Greys_r',origin='lower')
ax[0].text(5, 490, '${\it SDO}$/AIA 171 $\AA$', color="white", size='large')
ax[1].imshow(AIA_out[0,2,:,:],vmin=0.01,vmax=30,cmap='Greys_r',origin='lower')
ax[1].text(5, 490, 'Basis Pursuit Synthesized 171 $\AA$', color="white", size='large')
ax[2].imshow(AIA_out_DeepEM[0,2,:,:],vmin=0.01,vmax=30,cmap='Greys_r',origin='lower')
ax[2].text(5, 490, 'DeepEM Synthesized 171 $\AA$', color="white", size='large')

for axes in ax:
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    
fig,ax=plt.subplots(ncols=3,figsize=(9*2,9))

ax[0].imshow(X_test[0,4,:,:],vmin=0.25,vmax=25,cmap='Greys_r',origin='lower')
ax[0].text(5, 490, '${\it SDO}$/AIA 211 $\AA$', color="white", size='large')
ax[1].imshow(AIA_out[0,4,:,:],vmin=0.25,vmax=25,cmap='Greys_r',origin='lower')
ax[1].text(5, 490, 'Basis Pursuit Synthesized 211 $\AA$', color="white", size='large')
ax[2].imshow(AIA_out_DeepEM[0,4,:,:],vmin=0.25,vmax=25,cmap='Greys_r',origin='lower')
ax[2].text(5, 490, 'DeepEM Synthesized 211 $\AA$', color="white", size='large')

for axes in ax:
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    
fig,ax=plt.subplots(ncols=3,figsize=(9*2,9))

ax[0].imshow(X_test[0,0,:,:],vmin=0.01,vmax=3,cmap='Greys_r',origin='lower')
ax[0].text(5, 490, '${\it SDO}$/AIA 94 $\AA$', color="white", size='large')
ax[1].imshow(AIA_out[0,0,:,:],vmin=0.01,vmax=3,cmap='Greys_r',origin='lower')
ax[1].text(5, 490, 'Basis Pursuit Synthesized 94 $\AA$', color="white", size='large')
ax[2].imshow(AIA_out_DeepEM[0,0,:,:],vmin=0.01,vmax=3,cmap='Greys_r',origin='lower')
ax[2].text(5, 490, 'DeepEM Synthesized 94 $\AA$', color="white", size='large')

for axes in ax:
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
```



![png](../../../images/chapters/04/1/notebook_65_0.png)



![png](../../../images/chapters/04/1/notebook_65_1.png)



![png](../../../images/chapters/04/1/notebook_65_2.png)


<b>Figure A2:</b> Top to Bottom: <i>SDO</i>/AIA images in 171 Å, 211 Å, and 94 Å (left, top to bottom) with the corresponding synthesised observations from Basis Pursuit (middle, top to bottom) and DeepEM (right, top to bottom). DeepEM provides synthetic observations that are similar to Basis Pursuit, with the addition of solutions where the basis pursuit solution was $zero$.

---

 

This project was initiated during the 2018 NASA Frontier Development Lab (FDL) program, a partnership between NASA, SETI, NVIDIA Corporation, Lockheed Martin, and Kx. We gratefully thank our mentors for guidance and useful discussion, as well as the SETI Institute for their hospitality.
