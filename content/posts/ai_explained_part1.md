---
title: "From Linear Models to Neural Networks: When Straight Lines Aren’t Enough"
date: 2025-07-03T12:00:00-04:00
draft: false
author: "Massil Ait Abdeslam"
tags: [ "AI"]
description: "This is the first in a three-part series where we build neural networks from the ground up. Today: understanding the problem they were designed to solve."
cover:
  image: "images/ai_explained_part1/cover.jpg"
  alt: "A cover picture with a brain"
---

# Introduction

One of the biggest topics today is AI. You hear about AI everywhere, every company tries to include “AI” in their products as a marketing strategy, because it’s what sells right now. But what exactly is this AI thing, and how does it actually work? That’s what we’re going to explore in this three-part series.

The first question is: what is AI? AI stands for artificial intelligence, but its definition is not very precise. Broadly, we define AI as the ability of machines to mimic human intelligence. For example, according to this definition, you could argue that a simple calculator is a form of AI, because it mimics the human ability to perform calculations. But of course, most of us don’t think of a calculator as AI, because it simply follows strict rules to perform computations faster than a human without any complex reasoning. This is part of what makes defining AI so challenging, the boundary between what counts as AI and what doesn’t is fuzzy.

AI is not at all a new field, it has been a topic of study since the early days of computing in the 1950s. While the history of AI is fascinating, this post will focus more on some key techniques and how they work. If you’re interested in the story of how AI evolved, you can read this other [post](https://www.tableau.com/data-insights/ai/history).

For the rest of this serie, we’ll focus on one topic that has dominated the AI world in recent years: neural networks.

# Traditional AI vs Machine Learning 

## Traditional AI 

Without diving too deeply into details, one of the biggest milestones in the history of AI was when IBM’s computer Deep Blue defeated the world chess champion Garry Kasparov in 1997. In this section, I want to explore the fundamental difference between how Deep Blue worked compared to modern AI.

Deep Blue was a computer program explicitly designed to play chess. At the time, chess programs were much weaker than top human players, even good amateurs could often beat them. Although computers are extremely fast at computation, chess has so many possible positions that it’s impossible for a computer to explore every possible outcome of the game from a given position. To play chess well, you need intuition about which positions are good and which are bad, something very hard to program, because computers only follow strict instructions and don’t “think” on their own.

Here’s how Deep Blue tackled this problem. At any given position, Deep Blue would explore all possible sequences of moves to a certain depth. For example, it might consider moving a bishop, then anticipate all possible responses by the opponent, then its own responses, and so on building a tree of possible moves. The challenge is that you can’t explore all the way to the end of the game, knowing if a sequence of move result in a win or a loss. For each branch, the tree grows too quickly. The figure below shows how you build a small tree for a given position to a depth of 2. 

<img src="/blog/images/ai_explained_part1/chess_minimax.png" alt="Alt text" width="80%" style="display: block; margin: auto; padding-bottom: 20px">


So you need a way to evaluate whether a position at a certain point in the tree is good or bad. This is exactly what Deep Blue did. Professional chess players and programmers worked together to craft a function that evaluated how good a position was. For example, one component of that evaluation function was the number of pieces: the more pieces you have compared to your opponent, the better. Other components accounted for the strength of a piece’s position on the board, some squares are strategically powerful, others are weak, etc.
In summary, Deep Blue explored millions of possible positions up to a certain depth and chose the move that led to the most advantageous position according to its evaluation function. The algorithm it used is called minimax, which has additional refinements that I won’t cover here, but this is the core idea: the programmers did the reasoning in advance, translating chess knowledge into rules and a scoring function. This is the hard part, being able to translate complex reasoning into a list of clear instructions. 

What made Deep Blue powerful was not that it “understood” chess better than Kasparov, in fact, its ability to evaluate positions was worse. Its strength came from being able to calculate and evaluate millions of positions per second, something no human could match. While Kasparov could evaluate a position at a depth of 3-4 moves, Deep Blue could reach 8 moves. 

## Machine Learning

You may have already guessed that explicitly translating the reasoning process of a complex task into a clear list of instructions that a computer can understand is extremely challenging and it greatly limits the number of applications where AI can be used.

Take a seemingly simple task: determining whether an image contains a dog. As a human, you can recognize a dog by its eyes, ears, tail, fur, and overall shape. You can extract these features from an image and associate them with the concept of a dog. But in real life, dogs come in many different shapes, sizes, and colors. In an image, they can appear at different positions, angles, lighting conditions, and even partially hidden. This makes it very difficult to translate our pattern recognition strategy into a clear and deterministic set of rules that a computer can follow.
This is where machine learning becomes useful. A funny example is determining whether an image contains a chihuahua or a muffin, a task that can get really challenging for a computer.

<img src="/blog/images/ai_explained_part1/dog.jpeg" alt="Alt text" width="80%" style="display: block; margin: auto; padding-bottom: 20px">


The idea behind machine learning is conceptually simple. Instead of explicitly telling the machine how to perform a task, what if the machine could learn the reasoning itself from a large amount of data? In this approach, the programmer doesn’t need to encode an explicit strategy,  which is highly task-specific, time-consuming, and not easily generalizable. Instead, we just provide the machine with enough data, and it figures out how to perform the task on its own.

All we need is an algorithm that is capable of learning a task from data. Crucially, this algorithm is general-purpose, it doesn’t need to be designed specifically for the task at hand. The task is learned from the data itself.

For example, instead of telling the machine what features to look for in an image to determine if it contains a dog, we simply give it a large dataset of images, labeled as “dog” or “no dog,” and let it learn on its own what patterns in the images correspond to the presence of a dog. We can also train this model to learn if a cat is in an image, or a car, just by giving it different data, our model is really generalizable. 
Of course, this is easier said than done! In the rest of this post, we’ll explore how this is actually achieved. One of the most powerful approaches today is through neural networks, which we’ll build up to step by step.

> **💡 Note — Supervised vs Unsupervised Learning :**
>
>There are two main types of machine learning algorithms: supervised and unsupervised. Supervised models use data that is labeled, for example, in the dog recognition task, the images fed to the model are labeled as “dog” or “not dog.” Unsupervised learning, on the other hand, refers to models that work with raw, unlabeled data and try to discover patterns or structure within it. In the rest of this post, we’ll focus only on supervised learning. 
>

# Functions

You may recall from your math classes what a function is: something that takes an input and produces a unique output. Usually, when we think of functions, we limit ourselves to mathematical ones like:

$$f(x)=2x^2$$

But in fact, functions describe much more than math, they can model many aspects of the world. Almost anything can be thought of as a function, as long as it takes an input and produces a unique output.

For example, in a certain sense, you can be modeled as a function. Your inputs are your senses, which react to stimuli, your outputs are your movements and actions. If you accept that humans are deterministic, meaning that for the same input you would always produce the same output, then humans can be described as extremely complex functions.

We can also define a function that associates a car with its brand. In this case, the function takes a car as input and outputs the brand that made it. So you can actually describe almost anything as some kind of function.
More formally, a function is an association from one set of elements to another, as long as each input element has exactly one associated output element. The image below illustrates this idea.

<img src="/blog/images/ai_explained_part1/function.jpg" alt="Alt text" width="50%" style="display: block; margin: auto; padding-bottom: 20px">

For example, mathematical functions often associate a real number with another real number. We call these functions $f:\mathbb{R}→\mathbb{R}$, meaning they map from the set of real numbers to itself.

Bringing this back to our earlier example of dog recognition, what we want to learn is essentially a function from the set of all possible images to the set {dog, no dog}. The input set consists of all possible pixel combinations that form an image, and the output set contains just two elements. The function takes an image as input and outputs whether or not there is a dog in it.

We know this function exists because we, as humans, can perform this task ourselves. The problem is that this function is extremely complex, and we’re not able to formalize it explicitly.

The aim of machine learning is to approximate such a function, given enough pairs of input and output data. We define a data point as a pair $(X,Y)$, where $X$ is the input and $Y$ is the output.

In the next section, we’ll explore our first machine learning model: the linear function.

# Linear Approximation

We said earlier that we want to learn an arbitrary complex function, so why are we starting with the simplest possible function, the linear function:

$$f(x)=w \cdot x+b$$

That’s a good question. The reason is that this simple example helps us build the tools and intuition we need, which can then be scaled up to more powerful and flexible function approximators, like neural networks.

The idea behind linear approximation is as follows: given a problem, we assume that the output is linearly correlated to the input, meaning it’s just a matter of scaling and shifting the input to get the output. Our task is to find the scaling factor $w$ and the shifting term $b$ that best describe the relationship between $X$ and $Y$.

Of course, this is a strong assumption that often doesn’t hold in practice. For example, in our dog recognition problem, this would mean assuming that if we sum each pixel in the image (weighted) and shift the result by a certain amount, we’d get a number directly correlated with whether or not the image contains a dog. More concretely: if this number is larger than a certain threshold, the image contains a dog; otherwise, it doesn’t. This is a classification problem, where the goal isn’t to predict a real number but to decide if an input belongs to a particular class. To achieve this, we apply a mathematical trick: we set a threshold to divide the output space into two halves, one for each class. This approach is formalized in a method called [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression).
We can express this idea as:

$$f(p_{0,0}, p_{0,1}, ... p_{n,n})= \sum_{j=0}^{n}\sum_{i=0}^{n}w_{i,j}\cdot p_{i,j}+b$$

With: 
$$f(p_{0,0}, p_{0,1}, ... p_{n,n}) > k \Rightarrow \text{dog}$$
$$f(p_{0,0}, p_{0,1}, ... p_{n,n}) \leq k \Rightarrow \text{no dog}$$

Where:
- $p_{i,j}$ : the pixel value at position $(i,j)$ in the image.  
- $w_{i,j}$ : the scaling (weight) factor applied to pixel $(i,j)$.  
- $b$ : the bias term (shifting parameter).  
- $k$ : the threshold (usually set to $0$) used to decide between the two classes.

From now on, we’ll refer to $w$ as the weights and $b$ as the bias.

Of course, the linear assumption is clearly too simplistic for a complex task like dog recognition but it provides a useful foundation. Let’s now look at a case where linear approximation is reasonable. 

## Model 

I’ll take the same example as Andrew Ng in his [machine learning course](https://www.deeplearning.ai/courses/machine-learning-specialization/) (which is an excellent course): predicting the price of a house.

In this case, let’s suppose that the price of a house is linearly correlated with its size in square meters, which seems reasonable. So we assume that the price of the house, $f(x)$, is proportional to $x$, the size of the house. That gives us the relationship:

$$f(x)=w \cdot x+b$$

Our task is to find the values of $w$ and $b$ that allow our model to predict the price of a house as accurately as possible. The first step is to collect data in the form of $(X,Y)$, where $X$ is the size of a house and $Y$ is the price it sold for. We use this historical data to train our model, with the ultimate goal of using $f(x)$ to predict the price of a house that hasn’t been sold yet.

I chose some fictitious data points. We can plot them on a graph and clearly see a linear correlation : 

<img src="/blog/images/ai_explained_part1/house_size_vs_price.png" alt="Alt text" width="80%" style="display: block; margin: auto; padding-bottom: 20px">

Now, our goal is to find an algorithm that can determine the best w and b to fit the data and make good predictions.

## The algorithm

There are multiple algorithms that can find the best $w$ and $b$ for a set of data points in the case of linear approximation, but we’ll focus on one that can also scale to more complex models: backpropagation.

The main idea behind backpropagation is to frame the problem as an optimization problem: Given the data points, what are the $w$ and $b$ that best fit them?

But how do we define “best fit”? We need a way to measure numerically how well a given curve fits a set of points. This is where the concept of a loss function comes in.

### Loss function

We can choose arbitrary $w$ and $b$ and plot $f(x)$ :

<img src="/blog/images/ai_explained_part1/example_curves.png" alt="Alt text" width="80%" style="display: block; margin: auto; padding-bottom: 20px">

Intuitively, it seems like the first curve fits the data best right ? But how do we actually measure this?

We compute predicted values $Y_{pred}=f(X)$ for all data points, then compare them to the true $Y$ values to see how far off they are. The figure below shows the distances between $Y_{pred}$ and $Y$ :  

<img src="/blog/images/ai_explained_part1/example_distances.png" alt="Alt text" width="80%" style="display: block; margin: auto; padding-bottom: 20px">


We can then compute the error, or how poorly the curve fits the data, by summing up all these distances. To normalize this (and make it independent of the number of points), we can average it:

$$\frac{1}{N}\sum_{i=1}^{N}\text{distance}_i$$

Where $N$ is the number of data points. 

This gives us a quantitative measure of how well the curve fits the data: the smaller the error, the better the fit. But how exactly should we define this “distance”?

A first idea might be to take $Y−Y_{pred}$ , but since differences can be positive or negative, they might cancel each other out. We could take the absolute value to keep only the magnitude of the differences, but an even better choice is to square the differences: $(Y−Y_{pred})^2$. There’s actually a way to derive this squared error from first principles, based on a probabilistic interpretation of machine learning, but that will be the focus of another post.

For now, we’ve arrived at a way to express, numerically, how well a curve fits the data. This is called a loss function. For our model, we’ll use the following loss function called Mean squared error :

$$\frac{1}{N}\sum_{i=1}^{N}(Y_i-\hat{Y_{i}})^2$$

where $Y_i$ represents the $i$-th true data point and $\hat{Y}_i$ represents our $i$-th predicted data point.

### Gradient Descent

Now we come to the most important idea in all of machine learning: how can we find the best $w$ and $b$, given the loss function?

This is exactly what gradient descent does, and that’s what we’ll explore in the rest of this section. The first question is: how should $w$ and $b$ relate to the loss? We said earlier that the better the fit, the smaller the loss. So for a perfect set of $w$ and $b$ (perfectly fitting the data), the loss should be zero. For bad choices of $w$ and $b$, the loss will be large.

Here is the key idea: we want to find the pair of $w$ and $b$ that minimizes the loss. For the next part, I’ll assume you know a bit of calculus, but even if you don’t, you can still follow the intuition.

We can write the loss function explicitly as a function of $w$ and $b$, since $X$ and $Y$ are the given data (constants) :

$$L(w,b)=\frac{1}{N}\sum_{i=1}^{N}(Y_i-\hat{Y_{i}})^2$$

Replacing $\hat{Y_i}=f(X_i) = w \cdot X_i +b$ :

$$L(w,b)=\frac{1}{N}\sum_{i=1}^{N}(Y_i-(w \cdot X_i +b))^2$$

If this is your first time seeing a multivariable function, don’t worry, it’s not much more complicated than a single-variable one. In a single-variable function, the input is a line, and you can plot the function in 2D (input vs output). In a two-variable function like $L(w, b)$, the inputs ($w$ and $b$) live in a plane. The output (the loss) gives a height above each point in the $(w, b)$ plane, forming a 3D surface (2D plane for the input and 1D line for the output).  We can plot the loss function in 3D (the log of the loss for a stipper curve):

<img src="/blog/images/ai_explained_part1/loss_landscape.png" alt="Alt text" width="80%" style="display: block; margin: auto; padding-bottom: 20px">

On this surface, we can clearly see that there is a minimum region, the lowest point of that lowest region corresponding to the best choice of $w$ and $b$. Now the question becomes: starting from any arbitrary point $(w, b)$, how can we move toward this minimum? This is where gradient descent comes in, an algorithm for iteratively moving toward the lowest point on the surface.

This is where calculus comes in. For people who aren’t familiar with calculus, you can simply think of it like this: From any point on a function, you can compute the direction in which you need to move each input to get closer to a maximum or minimum of the function. So in our case, we have a way to compute how to update $w$ and $b$ to get closer and closer to the minimum. If you accept this idea, you can skip to the next section.

For those who know some calculus, let’s go a bit deeper. In a single-input function, when we compute the derivative at a point, it gives us the slope of the function at that point. In a sense, it tells us which direction leads to a maximum or minimum:

- If the derivative is positive, the function is increasing, increasing the input makes the output bigger, while decreasing the input makes the output smaller.

- If the derivative is negative, it’s the opposite.

- The closer you are to a maximum or minimum, the smaller the derivative becomes indicating that you’re close.

- At a maximum or a minimum, the derivative is 0.

We can generalize this idea to functions of multiple inputs using something called the gradient. Here, we compute the derivative of the function with respect to each input, called the partial derivative, which tells us how much to adjust each input to reach the maximum or minimum. For example, consider this function:

$$f(x,y)=x^2+2y$$

Its partial derivatives (we use $\partial$ for partial derivatives instead of d) are:

$$\frac{\partial f(x,y)}{\partial x} = 2x$$
$$\frac{\partial f(x,y)}{\partial y} = 2$$

When computing the partial derivative with respect to one variable, we treat the others as constants. When we collect all the partial derivatives, we get the gradient, which points in the direction of steepest increase of the function. To reach the minimum, we simply move in the opposite direction of the gradient. So the gradient gives us the direction in wich to move the inputs to reach a maximum/minimum. For a two-input function, the gradient looks like this:

$$\nabla f(x,y) = \left[\frac{\partial f(x,y)}{\partial x} \text{  } \frac{\partial f(x,y)}{\partial y}\right]^T$$

So with the gradient, we have a way to get to the minimum of the loss function. We can compute the gradient of the loss (the derivative of a sum is the sum of the derivative) :

$$\frac{\partial L(w,b)}{\partial w}=\frac{1}{N}\sum_{i=1}^{N}-2(Y_i-(w \cdot X_i +b))X_i= -\frac{2}{N}\sum_{i=1}^{N}(Y_i-\hat{Y_i})X_i$$

$$\frac{\partial L(w,b)}{\partial w}=\frac{1}{N}\sum_{i=1}^{N}-2(Y_i-(w \cdot X_i +b))= -\frac{2}{N}\sum_{i=1}^{N}(Y_i-\hat{Y_i})$$

These gradients tell us how to adjust $w$ and $b$ step by step to move toward the minimum of the loss function.

### Backpropagation 

Now that we’ve covered the theoretical part, we can finally get to our goal, and the most important algorithm in machine learning: Backpropagation.

One of the key components of the algorithm is its ability to compute the gradient efficiently using something called the chain rule, but that will be the topic of another post.

For now, we can still develop a good understanding of the key ideas behind how backpropagation works.
It is composed of three steps:
- Forward pass: compute the predictions $\hat{Y_i}$.
- Backward pass: compute the gradients.
- Update step: update the weights and bias to get closer to the minimum.

We repeat these three steps until the loss reaches a minimum.

Let’s continue with our house price prediction example.

**0. Initialization**

We set the weight $w$ and bias $b$ to random initial values.

$$w = 496.71$$ 
$$b = -138.26$$

**1. Forward Pass**

We compute $\hat{Y_i}$ for each $X_i$ using the current $w$ and $b$ :

<img src="/blog/images/ai_explained_part1/initial_prediction.png" alt="Alt text" width="90%" style="display: block; margin: auto; padding-bottom: 20px">

**2. Backward Pass**

We compute the gradient using our formulas :

$$\frac{\partial L(w,b)}{\partial w}= -\frac{2}{N}\sum_{i=1}^{N}(Y_i-\hat{Y_i})X_i = -231362061$$

$$\frac{\partial L(w,b)}{\partial w}= -\frac{2}{N}\sum_{i=1}^{N}(Y_i-\hat{Y_i}) = -1097369$$

**3. Update step**

Notice that the gradients are quite large. Instead of moving fully in the gradient’s direction, we take a small step $\eta$, known as the learning rate, in the opposite direction. In this example, we chose $\eta=10^{-5}$.  

We update the parameters as follows:

$$w=w-\eta \frac{\partial L(w,b)}{\partial w} = 2801.3$$
$$w=b-\eta \frac{\partial L(w,b)}{\partial b} = -127.3$$

We now get this curve :

<img src="/blog/images/ai_explained_part1/linear_regression_1.png" alt="Alt text" width="90%" style="display: block; margin: auto; padding-bottom: 20px">

We then repeat these three steps, forward pass, backward pass, and update, until we reach a minimum (where the gradient becomes almost zero).

At the end of this process (after 10 repeat), we have a curve that fits our data quite well : 

<img src="/blog/images/ai_explained_part1/linear_regression_fit.png" alt="Alt text" width="90%" style="display: block; margin: auto; padding-bottom: 20px">

So, in summary: without explicitly telling the computer anything about the task of predicting the price of a house based on its size, we developed an algorithm that can automatically learn the correct relationship between the input and output data. We’ve built a general algorithm that allows a computer to discover patterns in the data on its own, quite impressive!

### Limitation 

We saw that our model performed quite well on the dataset, but we made two really big assumptions that won’t hold for more complex tasks:
1.	We assumed a single input function, where the price depended only on the size of the house.
2.	We assumed a linear correlation between the input and the output.

#### Number of inputs 

The first assumption was made mainly to simplify the visualization and explanation, but we can easily generalize what we did to multiple-input linear functions by writing:

$$f(x_0,x_1,...x_n)=\sum_{i=1}^nw_ix_i+b$$

All the arguments we made before still apply, but now our loss function depends on $w_0,w_1,…,w_n$ and $b$, so the search space becomes much larger. Still, we can compute the partial derivative for each weight and the bias, and update all of them just like before.

For example, we could include additional inputs such as:
- the year of construction
- the number of owners
- the number of rooms

Hopefully, this additional data will make our model better. That’s why data is crucial in machine learning, the more high-quality data you have, the better your model can learn. More data gives the model a finer understanding of the world and allows it to capture more nuanced patterns.

#### Linearity

This is the biggest assumption we made. Even though many phenomena are approximately linear, many others are not. For example, our dog detection problem is clearly not linear, the presence or absence of a dog in an image is not linearly dependent on the pixel values. The relationship is far more subtle and complex.

We could try defining polynomial models, which are more expressive than purely linear relationships. For example, for two inputs:

$$f(x_0,x_1)=w_{x_0}x_0+w_{x_1}x_1+w_{x_{1,2}}x_0x_1+w_{x_{0,0}}x_0^2+w_{x_{1,1}}x_1^2+b$$

Here the relationship is assumed quadratic, which is more flexible than linear. But the problem remains: The programmer needs to know in advance what kind of relationship exists between the inputs and outputs. In our dog detection example, we know the relationship isn’t linear, but we still don’t know what it actually looks like.

And this is exactly where neural networks come into play.

# Neural Networks

Unfortunately, in this post I won’t be able to give you a full understanding of how neural networks are used in such a wide variety of seemingly unrelated tasks. In this section, I’ll focus on giving you an idea of how neural networks solve the problem we previously identified: even with backpropagation, the programmer still needs to make assumptions about the relationship between inputs and outputs.

The key feature of a neural network is that it is a general function approximator, exactly what we were looking for! The same neural network, depending on its weights and biases, can learn an arbitrarily complex function. As a programmer, you don’t need to assume the shape of the function; you only need to decide which inputs are relevant to the output and feed them into the neural network. Even better, neural networks are resilient to noise. If one of your inputs actually doesn’t impact the output (i.e., it’s just noise), the network can often learn to ignore it and focus only on the important inputs.

Feed-forward neural networks (the simplest architecture) are, as the name suggests, a network of interconnected units called neurons. A neuron takes multiple inputs and produces an output. Neural networks are made up of layers of these neurons. The image below illustrates what a neural network looks like.

<img src="/blog/images/ai_explained_part1/neural_net.jpg" alt="Alt text" width="80%" style="display: block; margin: auto; padding-bottom: 20px">

Each neuron acts like a simple linear function, just like the ones we studied earlier:

$$z(x_0,x_1,...x_n)=\sum_{i=1}^nw_ix_i+b$$

We usually write this more compactly using vector notation:

$$z(\vec{x})=\vec{w}\cdot \vec{x}+b$$

But here is the key difference that makes a neural network a general function approximator, we add a non-linear activation function $\sigma()$ at the output of each neuron:

$$a(\vec{x})=\sigma(z(\vec{x}))=\sigma(\vec{w}\cdot \vec{x}+b)$$

With enough neurons and layers, and with the right weights and biases, a neural network can approximate any arbitrarily complex function. That’s what makes them so powerful. And, just like with the linear model, we use the backpropagation algorithm to find the optimal weights and biases for each neuron to fit the data as well as possible.

And as we saw earlier, almost anything can be framed as a function of some sort. So a neural network can approximate any function given enough neurons and layers. We also saw an algorithm that makes it possible to train a neural network on any function, provided we have enough data to properly tune the weights and biases. With all that in mind, the possibilities of what a neural network can do become enormous and you can start to see why they’re so powerful!

If you’re interested in a great visual and intuitive explanation of feed-forward neural networks, I highly recommend 3Blue1Brown’s excellent [introduction](https://www.3blue1brown.com/topics/neural-networks) on the subject. It’s a perfect follow-up while you wait for the next parts of this series!


# What’s Next?
I’ll wrap it up here for today, otherwise this post would get too long! I know it feels like we’re stopping at an exciting point, we’ve just uncovered the incredible potential of neural networks, but there are still plenty of important questions to answer:
- What exactly are these activation functions $\sigma()$.
- How much data, how many neurons, and how many layers do we need in a neural network?
- How can we efficiently compute the gradients of a large neural network?
- How can we feed different types of data like images, text, or audio into a neural network?
- How can a neural network perform creative tasks, like generating images or writing text?

I’ll tackle these topics in two or three upcoming posts:

In the first, I’ll dive deeper into feed-forward neural networks, explain how the chain rule makes backpropagation efficient, explore which activation functions work best and why, and discuss why training a neural network can sometimes be challenging.

In the second (and possibly a third), I’ll introduce you to other neural network architectures such as convolutional neural networks, diffusion models, transformers, and more and show how they excel at solving specific kinds of problems.

I hope you found this introduction clear and insightful. Thank you for taking the time to read my post, the next parts are coming soon!

Massil Ait Abdeslam





