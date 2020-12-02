Automated Detection of Magnetopause Crossings
=============================================

*by Colin Small, Matthew R. Argall, and Marek Petrik*

Global-scale energy flow throughout Earth’s magnetosphere is catalyzed by processes that occur at Earth’s magnetopause (MP) in the electron diffusion region (EDR) of magnetic reconnection. Until the launch of the Magnetospheric Multiscale (MMS) mission, only rare, fortuitous circumstances permitted a glimpse of the electron dynamics that break magnetic field lines and energize plasma. MMS employs automated burst triggers onboard the spacecraft and a  Scientist-in-the-Loop (SITL) on the ground to select intervals likely to contain diffusion regions. Only low-resolution survey data is available to the SITL, which is insufficient to resolve electron dynamics. A strategy for the SITL, then, is to select all MP crossings. This has resulted in over 35 potential MP EDR encounters but is labor- and resource-intensive; after manual reclassification, just ∼ 0.7% of MP crossings, or 0.0001% of the mission lifetime during MMS’s first two years contained an EDR.

In this notebook, we develop a Long-Short Term Memory (LSTM) neural network to detect magnetopause crossings and automate the SITL classification process. An LSTM developed with this notebook has been implemented in the MMS data stream to provide automated predictions to the SITL.

This work is published in [Argall, Small, et al., 2020, <i> Front. Astron. Space Sci.</i>, 7](https://www.frontiersin.org/articles/10.3389/fspas.2020.00054)
