---
layout: page
title: FAQ
permalink: /faq/
group: navigation
---

{% include JB/setup %}

- I'm having trouble with __DeepCell__. Can I get help.
	- Yes. If you think you have discovered a bug that needs to be fixed please file a report on the GitHub page. 

- What kind of hardware do I need to run __DeepCell__?
	- You will need a CUDA/cuDNN capable Nvidia GPU. We have had good success with the Nvidia GTX 980, Titan X, and GTX 1080 graphics cards.

- Does __DeepCell__ work with TensorFlow?
	- Unfortunately no. We have found that TensorFlow is unable to use numpy like indexing to address tensors. This makes it significantly harder to implement d-regularly pooling kernels - but we're working on it.

- Does __DeepCell__ track cells from frame to frame?
	- Unfortunately no. Right now, this software package focuses solely on the image segmentation problem for live-cell experiments. However, we are aware that cell tracking is an issue for a number of labs and we're actively working on deep learning approaches to this problem.

- What cells can __DeepCell__ segment?
	- So far we have trained convolutional neural networks to segment fluorescently labeled nuclei, as well as phase images of E. coli, MCF10A cells, NIH-3T3 cells, HeLa-S3 cells, RAW264.7 cells, and bone marrow derived macrophages.

- Do I need a nuclear label to segment the cytoplasm of mammalian cells?
	- For our approach, yes. The nuclear labels are necessary to refine the segmentation prediction. 

- Should I train my own network?
	- We recommend it. There are laboratory-to-laboratory differences (lighting, microscope, camera, pixel size, etc.) that do matter.

- Where should I go if I want to learn more about deep learning?
	- Two great websites are [deeplearning.net](http://www.deeplearning.net) and [Stanford's cs231n course notes](http://cs231n.stanford.edu/).
