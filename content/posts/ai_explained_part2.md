---
title: "Where Do Loss Functions Come From? A Probabilistic Perspective on Machine Learning"
date: 2025-07-09T12:00:00-04:00
draft: false
author: "Massil Ait Abdeslam"
tags: [ "AI"]
description: "This is the second part of our series on neural networks. Today: A probabilistic perspective on machine learning."
cover:
  image: "images/ai_explained_part2/cover.jpg"
  alt: "A cover picture with dices"
---

# Introduction

Previously, we saw that machine learning models are designed to approximate functions. For example, the function might detect whether an object is present in an image, predict whether a stock price will rise or fall, and so on. In short, we learned that many tasks can be framed as functions  and with enough data, we can train a machine learning model to approximate such a function.

This interpretation of machine learning, while useful as an introduction, misses a subtle but important point: we can’t fully model complex phenomena. Let’s revisit the example from the previous post, predicting the price of a house. At first, we assumed the price was entirely determined by its size. That already gave reasonable results, but we saw that we could improve the model by adding more parameters: the number of rooms, year of construction, number of previous owners, etc. Even with all this additional data, the price still depends on countless smaller factors, nuances that are hard, if not impossible, to fully capture. The timing of the sale, the number of interested buyers, neighborhood dynamics, these are difficult to measure yet still influence the final price.

And here lies the key idea we’ll explore today: our models have inherent uncertainty. The assumption that we’re learning a perfectly deterministic function is useful, but incomplete. Instead of learning deterministic functions, our models actually approximate probability distributions, functions that explicitly account for uncertainty and randomness. We will see how this change in perspective allows us to derive our usual loss functions. We’ll see how this probabilistic perspective naturally gives rise to the loss functions we use in machine learning.

# Probability theory

Before diving further, let’s review some basic probability concepts and formulas that we’ll use throughout the post.

## Probability distributions

A probability distribution is a special kind of mathematical function that models random phenomena. For example, consider rolling a die. The outcome of the roll is what we call a random variable, because we don’t know in advance what value it will take. The probability distribution describes the likelihood of each possible outcome of this random variable. We write it as:

$$P(X=x)$$

where $X$ is the random variable (the die roll) and $x$ is a particular outcome ($1,2,3,..,6$). For a fair six-sided die, the probability of rolling any number is $\frac{1}{6}$. For example, the probability of rolling a 5 is written as:

$$P(X=5)=\frac{1}{6}$$

This formalism may feel excessive for something as simple as a die roll, but it becomes essential when modeling more complex phenomena like predicting weather patterns, estimating disease risks, or forecasting stock market movements.

In short, a probability distribution gives the probability of each possible outcome of a random variable.

## Conditional probability
Often, events are not independent, some phenomena depend on others. For example, your probability of passing an exam depends on whether or not you studied.Let’s formalize this.We define 2 random variables :
- $X$: whether you studied ($X=1$) or not ($X=0$)
- $Y$: whether you pass ($Y=1$) or fail ($Y=0$) the exam

Those two random variables are binary, there are four possible scenarios:
- studied & passed
- studied & failed
- didn’t study & passed
- didn’t study & failed

We express the probability of $Y$ given $X$, called a **conditional probability** as:

$$P(Y=y|X=x)$$

This reads: *the probability that $Y=y$ (pass/fail) given that $X=x$ (studied or not)*.

We can summarize the conditional probabilities in a table (with hypothetical numbers):

<table style="
  margin-left: auto !important;
  margin-right: auto !important;
  display: block !important;
  width: fit-content !important;
  border-collapse: collapse;
  text-align: center;
">
  <thead>
    <tr>
      <th></th>
      <th style="padding: 8px;">X=0</th>
      <th style="padding: 8px;">X=1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style="padding: 8px;">Y=0</th>
      <td style="padding: 8px;">0.9</td>
      <td style="padding: 8px;">0.2</td>
    </tr>
    <tr>
      <th style="padding: 8px;">Y=1</th>
      <td style="padding: 8px;">0.1</td>
      <td style="padding: 8px;">0.8</td>
    </tr>
  </tbody>
</table>

Notice that for each row (i.e., fixed $X$), the probabilities add up to 1, since you either pass or fail. Also, it is more likely to fail than pass if you didn’t study (first column), and more likely to pass if you did study (second column).

That’s all the probability theory we need for now. These concepts, probability distributions and conditional probabilities, will underpin the rest of our discussion.

# Model

In the functional approach, we defined our model as:

$$Y_{pred}=f(X)$$ 

Where $f(\cdot)$ is the function learned by the model. In this notation, we implicitly assume that $Y_{pred}$ is certain, we don’t assign any likelihood or uncertainty to a particular output. For example, suppose we want to determine whether an image contains a dog. In the deterministic view, the model simply outputs either “dog” or “no dog.” However, it’s often more informative if the model instead outputs a confidence score, a number that reflects how sure it is about its prediction. For instance:
- If the model outputs $1$, it’s $100%$ confident that there is a dog.
- If it outputs $0$, it’s $0%$ confident (i.e. sure there is no dog).
- Values in between, like $0.85$, indicate high confidence but with some uncertainty.

If we still need to make a binary decision, we can set a threshold (e.g., if the output is $\geq0.5$, predict “dog”; otherwise, predict “no dog”). This probabilistic behavior is closely related to the idea of a conditional probability distribution. Rather than learning a deterministic function $Y_{pred}=f(X)$, the model is actually approximating:

$$P(Y=y|X=x)$$

That is, the probability that $Y=y$ (the image contains a dog) given that we observed $X=x$ (the image itself).

> **💡 Note - Bayesian vs frequentist interpretation :**
>
>*Here we adopt the Bayesian interpretation of probability, where probability reflects our degree of belief or confidence in an observation. This contrasts with the frequentist interpretation, where probability is defined as the long-run frequency of an event occurring over many trials.The Bayesian interpretation provides better intuition for notions like a 50% probability of rain tomorrow. We can’t repeat tomorrow many times, like rerolling a die, so in this case the probability represents our degree of belief in the event.*
>

# Three types of distributions

In machine learning, we typically encounter two kinds of tasks:
1.	**Regression tasks**: where we predict a continuous value. Examples: predicting the price of a house, the temperature tomorrow, or the time a runner will finish a race.
2.	Classification tasks: where we classify something into categories.
    - **Binary classification**: predicting one of two outcomes. Example : whether an image contains a dog or not.
    - **Multiclass classification**: predicting one of several categories. Example: whether an image contains a car, a dog, a house, etc.

In this section, we’ll see how these tasks correspond to well-known probability distributions.

## Regression tasks

In regression, $Y_{pred}$ can take on a continuum of values for a given $X$. This is modeled by a continuous probability distribution,we most commonly use the normal distribution (also called the Gaussian or bell curve). 

Many natural phenomena follow a normal distribution, such as human heights, test scores on an exam, etc. The prevalence of the normal distribution in nature can be explained by the central limit theorem, which essentially states that the sum of many small, independent effects tends to be normally distributed.(For a great explanation, check out 3Blue1Brown’s [video](https://www.youtube.com/watch?v=zeJD6dqJ5lo&ab_channel=3Blue1Brown) on the central limit theorem.) 

In this case, our model approximates the following conditional probability distribution :

$$P(Y=y|X=x)=\mathcal{N}(\mu_x, \sigma^2_x)=\frac{1}{\sqrt{2\pi}\sigma_x}exp\left(-\frac{(y-\mu_x)^2}{2\sigma_x^2}\right)$$

where:
- $\mu_x$ is the mean (most likely output given $x$)
- $\sigma_x^2$ is the variance (how spread out the possible outcomes are around $\mu_x$)

The following figure shows the shape of the normal distribution. We can see that it is centered at $\mu_x$, and the size of the bell depends on $\sigma_x$. 

<img src="/blog/images/ai_explained_part2/normal.png" alt="Alt text" width="70%" style="display: block; margin: auto; padding-bottom: 20px">


It’s important to note that for each input $x$, the output $Y$ follows a **different** normal distribution with its own $\mu_x$ and $\sigma_x^2$. However, we’ll see later that we often assume $\sigma_x^2$ constant (equal to 1) and train the model to predict only $\mu_x$, since the most likely outcome is given by the mean.

> **💡 Note - Probability mass function vs Probability density function**
>
>*Since $Y$ is continuous, the normal distribution gives us a probability density function (PDF) rather than the exact probability (probability mass function) of a specific value, which is technically zero. Instead, we talk about the probability of $Y$ falling within an interval $[a,b]$. This is a technicality but still important to mention.*
>

##  Binary classification

In binary classification, $Y_{pred}$ can only take two values: 0 or 1. Here, we use the Bernoulli distribution, which gives the probability of observing 1 (success) or 0 (failure):

$$P(Y=y|X=x)=p_x^y(1-p_x)^{1-y}$$

Where :
- $p_x$ is the probability of $Y=1$ given $X=x$.
- $y$: the outcome (0 or 1)

So the model actually learns a diffrent $p_{x}$ for each input $x$. Also, note that the probability of $Y=0$ for a given $X=x$ is simply $1−p_x$.

## Multiclass classification

In multiclass classification, $Y_{pred}$ can take one of $C$ possible outcomes: $y_1,y_2,…,y_C$. We model this with the categorical distribution, which generalizes the Bernoulli distribution to more than two outcomes. The conditional probability distribution is :

$$P(Y=y∣X=x)=\prod_{i=1}^C p_{x,i}^{y_i}$$

Where:
- $p_{x,i}$ probability of class $y_i$ given $x$
- $y_i$  the outcome for $y_i$ (1 if $y_i$ is the outcome , else 0). We can only have one outcome $y_i$

Also, by definition of probability we need to respect:
$$\sum_{i=1}^C p_{x,i}=1$$

For each input $x$, the model learns the full set of probabilities $p_{x,1}, p_{x,2},..., p_{x,C}$, one for each class.
In words, the probability of observing $y_i$  given $x$ is $p_{x,i}$, since all other terms are raised to 0.

For example, if the we want the probaility of the outcome {y_2} for a given $x$, then :

$$P(Y=y_2∣X=x)=\prod_{i=1}^C p_{x,i}^{y_i}=p_{x,1}^0\cdot p_{x,2}^1\cdot...\cdot p_{x,C}^0=p_{x,2}$$

This framework, using normal, Bernoulli, or categorical distributions, allows us to view machine learning models as learning to approximate these distributions, depending on the task.

# Maximum Likelihood

Here’s where things start to get really interesting: how does our model actually learn the best possible distribution, given enough data? That’s what we’ll cover in this section.

## Definition 

In machine learning, our goal is to find the best approximation function so that we can perform well not just on the training data, but also on unseen data. We’ve seen that this amounts to learning the best possible conditional probability distribution:

$$P(Y=y∣X=x)$$ 

This distribution depends on some parameters, which we’ll denote as $\theta$.
For example:
- In a normal distribution, $\theta=(\mu,\sigma^2)$, the mean and variance.
- In a Bernoulli distribution, $\theta=p$, the probability of success.
- In a categorical distribution, $\theta=(p_1,p_2,…,p_C)$, the probabilities of each class.

More formally, we write the conditional probability as:
$$P(Y=y∣X=x ; θ)$$

The probability of observing outcome $y$, given input $x$ and parameters $\theta$. So our task becomes: Find the parameters $\theta$ that best explain the data we observed.

### Negative Log Likelihood 

Suppose we collect a dataset of $N$ observations:

$$(y_0,x_0),(y_1,x_1),…,(y_N,x_N) $$

For a single sample $(y_i,x_i)$ our model assigns the likelihood of observing it as:

$$P(Y=y_i∣X=x_i;θ)$$

We hope this likelihood is high, because if our model represents reality well, then the data we actually observe should appear likely under the model. If the likelihood is low for a data point, there are two possible explanations:
1.	The data point is genuinely rare, an unlikely event that we happened to observe by chance.
2.	The model parameters $\theta$ are wrong and don’t accurately reflect the underlying distribution of the data.

To evaluate how well the model explains the entire dataset, we assume the data points are independent and compute the likelihood of observing the dataset under the model as the product of the individual probabilities:
$$L(\theta)=\prod_{i=1}^N P(Y=y_i∣X=x_i;θ)$$

This is called the likelihood function, it measures how likely the observed data are under parameters $\theta$. Again, if this overall likelihood is low, it probably means our parameters $\theta$ are poorly chosen, because the chance of observing many rare events at once is extremely small. This leads us to the principle of maximum likelihood estimation (MLE): Find the parameters $\theta$ that make the observed data as likely as possible. Formally:

$$\hat{\theta}=\arg\max_{\theta}\ L(\theta)= \arg\max_{\theta}\ \prod_{i=1}^N P(Y=y_i∣X=x_i;θ)$$

In words: We choose $\hat{\theta}$ to maximize the likelihood of the data under the model. In the next section, we’ll see how this MLE principle naturally leads us to the standard loss functions we use in machine learning depending on the task (regression, binary classification, multiclass classification).

Before we derive the loss function formulas, there are two additional steps we usually take to make the math easier and more stable. First, since probabilities are always $\leq 1$, the product of many of them quickly becomes extremely small, which can lead to numerical underflow and make computations tricky. To make the numbers more manageable, we take the logarithm of the likelihood. This also has the bonus property that products turn into sums since $\log(ab)=\log(a)+\log(b)$:

$$\log\left(L(\theta)\right)=\sum_{i=1}^N \log\left(P(Y=y_i∣X=x_i;θ)\right)$$

Second, in machine learning, it’s customary to phrase optimization as a minimization problem rather than a maximization one, we typically minimize some notion of “error” or “cost.” To turn our maximization of the log-likelihood into a minimization, we simply take its negative:

$$NLL(\theta)=-\log\left(L(\theta)\right)= -\sum_{i=1}^N \log\left(P(Y=y_i∣X=x_i;θ)\right)$$

This is called the negative log-likelihood (NLL), and minimizing it is equivalent to maximizing the likelihood. It also aligns conceptually with minimizing the error, just like we do with standard loss functions in supervised learning.

# Loss derivation

We finally arrive at the point where we can derive the appropriate loss functions for our three types of tasks! From here, you’ll see that the derivations are quite straightforward. Before diving in, let’s quickly recall how the parameters of the distributions are modeled. We saw earlier that the parameters θ of each distribution are:
- In a normal distribution, $\theta=(\mu,\sigma^2)$, the mean and variance.
- In a Bernoulli distribution, $\theta=p$, the probability of success.
- In a categorical distribution, $\theta=(p_1,p_2,…,p_C)$, the probabilities of each class.

We actually use function approximators, like a linear model or a neural network to learn these parameters! For example, in the regression case, we can write $\mu_x=f(x)$, where $f(x)$ is the function (e.g., a neural network) that outputs the mean given input $x$. So, our model still learns a function $f(x)$, but instead of predicting the output $Y_{pred}$ directly, it predicts the parameters of the distribution $P(Y∣X)$. This subtle distinction is exactly what motivated the whole probabilistic view of machine learning we’ve been building up to. Now, let’s derive the loss functions for each task.

## Binary Classification

In binary classification, we model $P(Y∣X)$ using the Bernoulli distribution:

$$P(Y=y|X=x)=p_x^y(1-p_x)^{1-y}$$

Where $p_x=f(x)$ is the probability of $y=1$ given $x$, predicted by our model.

The negative log-likelihood (NLL) is:

$$NLL(\theta)=-\sum_{i=1}^N \log\left(P(Y=y_i∣X=x_i;θ)\right)$$
$$\Rightarrow NLL(\theta)=-\sum_{i=1}^N \log\left(f(x_i)^{y_i}(1-f(x_i))^{1-y_i}\right)$$
$$ \Rightarrow NLL(\theta)=-\sum_{i=1}^N {y_i}\log\left(f(x_i)\right)+(1-y_i)\log\left(1-f(x_i)\right)$$

This is exactly the binary cross-entropy loss.

## Multiclass classification

In multiclass classification, we model $P(Y∣X)$ using the categorical distribution:
$$P(Y=y∣X=x)=\prod_{j=1}^C p_{x,j}^{y_j}$$

The parameters of our distribution are the $C$ probabilities $p_j$, one for each class $j$, representing the probability of observing class $y_j=1$ given the input $X$. Since we need to predict one probability per class, our model must produce a multi-output function, outputting $C$ values at once, one for each class .We can write this conveniently in vector notation: 
$$\vec{p_x}=\vec{f(x)} $$

Where $\vec{f(x)}$  is a vector-valued function approximator (e.g., a neural network), and its j-th component $f_j(x)$ gives the predicted probability $p_{x,j}$ of class $j$. Here, $\vec{f(x)}$ can still be any function approximator, linear model, neural network, etc. but it must produce a vector of $C$ outputs instead of a single scalar.

The NLL becomes:

$$NLL(\theta)=-\sum_{i=1}^N \log\left(P(Y=y_i∣X=x_i;θ)\right)$$
$$\Rightarrow NLL(\theta)=-\sum_{i=1}^N \log\left(\prod_{j=1}^C f_j(x_i)^{y_{i,j}}\right)$$
$$ \Rightarrow NLL(\theta)=-\sum_{i=1}^N \sum_{j=1}^C y_{i,j}\log\left(f_j(x_i)\right)$$

In some resources, the second sum is sometimes omitted because all the $y_{i,j}$ are zero except for the observed class. When we sample, we only observe one class; so across all $C$ classes, only the outcome for the observed class is one. Thus, we can also write more compactly the NLL as:

$$NLL(\theta)=-\sum_{i=1}^N \log\left(f_{y_i}(x_i)\right)$$

In both notation, this is the standard categorical cross-entropy loss.

## Regression task

For regression, we model $P(Y∣X)$ using the normal distribution:

$$P(Y=y|X=x)=\frac{1}{\sqrt{2\pi}\sigma_x}exp\left(-\frac{(y-\mu_x)^2}{2\sigma_x^2}\right)$$

Here, the parameters of the distribution are $\mu_x$ and $\sigma_x^2$, the mean and variance. In most applications, however, we don’t really care about the variance, because we’re usually interested in predicting the most likely outcome and for a normal distribution, this is simply the mean $\mu_x$ So, in practice, we often assume that the variance $\sigma_x^2$ is constant (equal to 1) and focus only on learning the correct $\mu_x$. We model this as: 

$$\mu_x=f(x)$$

Where $f(x)$ is our function approximator (e.g., a neural network) that predicts the mean of the distribution for a given $x$. We can now derive the loss function by taking the negative log-likelihood (NLL):

$$NLL(\theta)=-\sum_{i=1}^N \log\left(P(Y=y_i∣X=x_i;θ)\right)$$
$$\Rightarrow NLL(\theta)=-\sum_{i=1}^N \log\left(\frac{1}{\sqrt{2\pi}}exp\left(-\frac{(y_i-f(x_i))^2}{2}\right)\right)$$
$$ \Rightarrow NLL(\theta)=-\sum_{i=1}^N \log\left(\frac{1}{\sqrt{2\pi}}\right)+ \frac{1}{2}\sum_{i=1}^N(y_i-f(x_i))^2$$

We can see that this expression contains some constant terms that don’t depend on the model parameters and therefore don’t influence the optimization. We can safely drop these constants since they don’t affect where the minimum occurs. What remains is :

$$ Loss(\theta)=\sum_{i=1}^N(y_i-f(x_i))^2$$

This is known as the mean squared error (MSE) loss.

> **💡 Note - Averaged Losses**
>
>*You’ll often see these losses averaged over the dataset (dividing by N).
While this isn’t strictly necessary (since the minimum stays the same), it makes the loss scale-independent and helps optimization behave more smoothly.*
>

# Conclusion

With all this, you can see why we often say that probability theory is at the heart of machine learning. We derived the most common loss functions, binary cross-entropy, categorical cross-entropy, and mean squared error,  directly from probabilistic principles. What we’ve covered here is just the beginning. More advanced models and architectures build on these ideas and use even richer probabilistic concepts to tackle more complex tasks. Generative models especially rely on more advanced probabilistic modeling. I didn’t cover some important concepts used in generative models, such as KL-divergence or ELBO, in order to keep this post accessible, these might be the subject of another post in the future.

I hope you enjoyed the read, and thanks again for taking the time to go through my post!

Massil Ait Abdeslam
