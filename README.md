# Adversarial Robustness of Deep Learning using a V1 Front-End
## KCL Computer Science Final Project

This is where the code used for my King's College London Computer Science BSc final project will go once I am allowed to publicly share the code.

Finding techniques to bolster the adversarial robustness of convolutional neural networks is a cutting-edge area of study in a rapidly growing industry of machine learning technology. As larger or more delicate systems begin to increasingly rely on computer vision, it is imperative that they are as resistant as possible to any anomalies such as malevolent actors or even weather effects on image-capturing devices. My meager contribution to this research was to attempt to find a link between simulating a V1 (primary visual cortex) before any CNN and an increased robustness to both white-box and black-box attack. This was done by replicating the results of this paper: https://www.biorxiv.org/content/10.1101/2020.06.16.154542v2, which uses a linear-nonlinear-Poisson cascade model (LNP) to simulate the V1; and comparing those to a CNN front-ended by a Predictive Coding/Biased Competition model (PC/BC) of the V1, outlined in this paper: https://www.jneurosci.org/content/30/9/3531. The objective was to assess whether there was something specific to the LNP model that could explain the large improvement to robustness or if any V1 simulation front-ending a CNN could do the same.

The software was written in Python using Pytorch, foolbox, and other libraries. 
