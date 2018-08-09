---
title: "Neural Network -- an introduction to function approximation"
layout: post
date: 2017-07-22
image: /assets/images/nnTitle.jpg
headerImage: true
tag:
- Machine Learning
- Neural Network
- Function Approximation
- Supervised Learning
category: blog
author: Alan Yu
description: A demystification to neural network and introduction to its application in function approximation

---

## What is neural network?

### Background 

As computer sciecne students, we often heard these fancy stuffs in life: image classification, pattern recognition, convolutional neural network, machine learning, etc. Sometimes, we are so overwhelmed by the jargons in the field that we do not wanna explore the field ourselves.

However, we are all human beings living in the 21st century -- <a href="https://www.tandfonline.com/doi/full/10.1080/20964471.2017.1397411"> the era of big data </a>, making techniques such as machine learning useful and meaningful for data analysis.

<a href="https://en.wikipedia.org/wiki/Machine_learning">Machine learning</a> is a century-old field that recently became tremendously popular due to the demand for convenient tools facilitates data analysis. It has two intersecting subfields based on techniques, namely statistical machine lerning and neural network. Whereas statistical machine learning involves a heavy load of statistics, neural networks stress more on designing and parameter-tuning. 

Based on the data we have, we can also divide machine learning into <a href="https://en.wikipedia.org/wiki/Supervised_learning">supervised</a> and <a href="https://en.wikipedia.org/wiki/Unsupervised_learning">unsupervised learning</a>. While supervised learning is to figure out the pattern with the guidance of a given standard, unsupervised learning is to recognize the pattern without any additional information.

Neural network is one group of algorithms used for machine learning that models the data using graphs of artificial neurons. It tries to mimic how the neurons in human brains work. 

In this post, we will focus majorly on applying neural network to function approximation. Since we will be informed of what our outputs should look like, we will be discussing about supervised neural network techniques only. Moreover, for the sake of illustration we only focus on Artificial Neural Network architecture. 

For those who are interested in Long-short Term Memory Neural Network, Convolutional Neural Network and Generative Adversarial Network, please check out the links below:

<ul> 
	<li> Long-short Term Memory: 
	<a href="http://colah.github.io/posts/2015-08-Understanding-LSTMs/">LSTM Blog</a>
	<a href="http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture10.pdf">Stanford CS231n</a>
	</li>
	<li> Convolutional: 
	<a href="https://www.google.com/search?q=convolutional+neural+network&rlz=1C5CHFA_enUS708US708&oq=convolutional&aqs=chrome.0.0j69i57j69i65j69i60j0l2.1802j1j9&sourceid=chrome&ie=UTF-8"> CNN Blog </a>
	<a href = "http://cs231n.github.io/convolutional-networks/">Stanford CS231n</a>
	</li>
	<li> Generative Adversarial: 
	<a href ="https://medium.com/@jonathan_hui/gan-why-it-is-so-hard-to-train-generative-advisory-networks-819a86b3750b"> GAN Blog</a>
	<a href="https://arxiv.org/abs/1406.2661">Original Paper</a>
	</li>
</ul>


### Technical detail

#### Components

**Image** on the left and **Text** on the right:


<div class="side-by-side">
	<div class="toleft">
		<img class="image" src="{{ page.insert1 }}}" alt="Alt Text">
		<figcaption class="caption">Illustration for one layer</figcaption>
	</div>
	<div class="toright">
		<p> A neuron network is a layer-by-layer structure. At each layer, it consists of processing elements (referred as PEs afterwards) and transfer functions. </p>
		<p> Usually, the first layer of a network is called input layer, the last layer is called output layer and the layers in between are hidden layers. </p>
		<p> The architecture of a neuron network is composed of the way that these layers combine together. For example, a network with 3-3-1 is a neural network where the first and second layer consist of 3 PEs and the output layer is of 1 PE. Note that the neural network on the left is of the topology N-1 with only two layers, one input layer and one output layer. </p>
	</div>
</div>

Useful video explaning neural networks in more details: 
<iframe width="699" height="393" src="https://www.youtube.com/embed/aircAruvnKk" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

------- 

#### Forward propagation

<div class="side-by-side">
	<div class="toleft">
		<img class="image" src="{{ page.insert2 }}" alt="Alt Text">
		<figcaption class="caption">what does weight look like</figcaption>
	</div>
	<div class="toright">
		<p> Weight is a mathematical representation of how important a factor is in the neural network. Assuming the identity transfer function, the higher the value of weight is, the more effect of that weight will be taken into account when calculating the output of the current layer. </p>
		<p> The weight in the form of the left is the weight of the first layer from jth PE to the ith PE in the next layer.</p>
	</div>
</div>

<div class="side-by-side">
	<div class="toleft">
		<img class="image" src="{{ page.insert3 }}" alt="Alt Text">
		<figcaption class="caption">transfer function</figcaption>
	</div>
	<div class="toright">
		<p> An transfer function a node defines the output of that node given an input or set of inputs</p>
		<p> Simplest transfer function is identity function which gives an identity map from the output of the previous layer to the input of the next layer. Researchers and scientists normally choose sigmoid function or tanh function as transfer functions. However, the choice of the transfer function is open to discuss and their relative advantages are discussed <a href="https://papers.nips.cc/paper/874-how-to-choose-an-transfer-function.pdf">here</a>. </p>
	</div>
</div>

In forward propagation, we will first aggregate the results calculated from the previous layer, applying transfer function as indicated above. We will do this layer by layer toward the output layer.


<div class="side-by-side">
	<div class="toleft">
		<img class="image" src="{{ page.insert4 }}" alt="Alt Text">
		<figcaption class="caption">Mathematical formula for forward propagation</figcaption>
	</div>
	<div class="toright">
		<p> The mathematic formula for forward propagation </p>
	</div>
</div>

Useful video for further illustration:
<iframe width="699" height="393" src="https://www.youtube.com/embed/UJwK6jAStmg" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

#### Backward propogation

When we reach the output layer in our neural network, we want to see how good our predicted result is compared to the desired output. So we will designate the network with an error function (usually Mean Square Error or Edit Distance) to evaluate how well we are doing with our current weights values and structures.

<div class="side-by-side">
	<div class="toleft">
		<img class="image" src="{{ page.insert5 }}" alt="Alt Text">
		<figcaption class="caption">Mathematical formula for forward propagation</figcaption>
	</div>
	<div class="toright">
		<p> We start from the output layer, comparing the desired results with predicted results then tracing back one layer at a time. </p>
		<p> The adjustments are made to our weights through various methods where gradient descent is the most popular one. </p>
		<p> Once we are done adjusting weights and reach the input layer, we will redo the forward propagation again. </p>
	</div>
</div>

Useful video for further illustration:
<iframe width="640" height="360" src="https://www.youtube.com/embed/GlcnxUlrtek" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

#### Stopping criteria

Before we start training our neural network, we will pre-define a stopping criteria for our network. For example, we can say if the error is within 10e-5 then we will stop the training or if we have gone 10e5 iterations then we stop training further.

Stopping criteria is crucial since we need to set a goal for our network to reach. There are various ways to determine where to stop. Check the links below:

<ul>
	<li><a href="https://www.ibm.com/support/knowledgecenter/en/SS3RA7_15.0.0/com.ibm.spss.modeler.help/idh_neuralnet_stopping_rules.htm">IBM blog</a></li>
	<li><a href="http://image.diku.dk/shark/sphinx_pages/build/html/rest_sources/tutorials/first_steps/when_to_stop.html">Shark developer post</a></li>
	<li><a href="https://www.researchgate.net/post/How_do_I_know_when_to_stop_training_a_neural_network">Research Gate Discussion Forum</a></li>
</ul>



#### Implementation

I have a ready-to-go 3-layer neural network implemented in Matlab for you. It can be easily translated to Python using TensorFlow or pyTorch. You can also build your own neural networks. 

<ul> 
	<li>A tutorial for TensorFlow users: <a href="http://web.stanford.edu/class/cs20si/"> Stanford CS 20SI</a></li>
	<li>Implement an ANN using pyTorch: <a href="https://github.com/AlanFermat/Blogs/tree/master/ANNPytorch"> PyTorch Implementation</a></li>
</ul>

## What is function approximation?

In general, a function approximation problem asks us to select a function among a well-defined class[clarification needed] that closely matches ("approximates") a target function in a task-specific way. 

By <a href="http://math.uchicago.edu/~may/REU2016/REUPapers/Gaddy.pdf">Stone–Weierstrass theorem </a>, every continuous function defined on a closed interval can be uniformly approximated as closely as desired by a polynomial function. We know from above that, if we choose linear function as our transfer function, then at each iteration of neural network, we are doing a matrix multiplication at each step. 

If we expand matrix multiplication, it is easy to see that the whole process of forward propogation is equivalent to using a polynomial to approximate the target function. By Stone-Weierstrass theorem, we have that neural network has the ability to approximate functions satisfying specific requirements (some constraints including continuity, domain, etc.).

There is a formal theorem supporting this! The <a href="https://en.wikipedia.org/wiki/Universal_approximation_theorem">universal approximation theorem </a> states that a feed-forward network with a single hidden layer containing a finite number of neurons can approximate continuous functions on compact subsets of n-dimensional Euclidean space, under mild assumptions on the transfer function. If you are interested in this theorem, you probably wanna read the following:





## Other applications

Use 3-layer neural network to teach a robotic arm how to draw certain pictures. 

Check this out: 
<a href = "https://github.com/AlanFermat/2R-robotic-arm-with-neural-network"> ANN application in Robotics </a>

## Reference
https://www.tandfonline.com/doi/full/10.1080/20964471.2017.1397411

http://math.uchicago.edu/~may/REU2016/REUPapers/Gaddy.pdf




