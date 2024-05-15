---
layout: post
title: CasCast
date: 2024-05-14 00:32:13
description: A blog Describing Skillful High Resolution Precipitation Nowcasting via CasCaded Modeling
tags: deep-learning weather-forecasting
categories: blog-posts
tabs: true
---

This is a blog post that conducts an analysis of the [CasCast](https://arxiv.org/abs/2402.04290) model.

## Introduction

The CasCast paper represents an advancement in the field of meteorological forecasting, particularly in the accurate prediction of precipitation using high-resolution radar data. The paper specifically addresses the challenges faced in nowcasting, which is the prediction of weather conditions for a short period, usually up to two hours ahead. Accurate Weather Forecast in the immediate future are of critical importance for the fields of disaster management and in various social sectors. This paper aims to provide a robust solution to improve prediction accuracy, especially for extreme weather events.

## Motivation

Around the world extreme weather events result in significant damages every year. One of the most destructive consequences that can be caused by these events is flooding. Flooding is due to high amounts of precipitation that occur.
Precipitation events involve multiple scales of atmospheric systems, making accurate predictions challenging.
Furthermore most current methods struggle with Short-term forecasting (NowCasting), and these predictions are essential for emergency management and disaster mitigation.

## Deep Learning Approach

##### Mapping The Problem

In order to approach this problem, its structure must be well-defined. Using multiple inputs, experts are attempting to predict the weather conditions using deep Learning Models.
The inputs used are generally Radar Data (High Resolution Radar Echo Images) as well as a variety of Atmospheric Variables; these can include temperature, humidity or wind patterns for example.

This is an example of a High Resolution Radar Echo Image that can be used as input.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/radar_echo_input.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

The desired outputs are accurate precipitation maps of the affected areas, below is an example of a desired precipitation map.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/output_sample_1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

##### Loss Function

Multiple Loss Function can be used to train the models.
A couple of loss functions that can be used are:
* **Mean Squared Error (MSE)**: Can be used for training deterministic models. It measures the average squared difference between the estimated values and what is observed. This loss helps in minimizing the forecast error in terms of the general precipitation distribution.
* **Noise Prediction Loss**: Can be used in probabilistic models where a diffusion process is involved. This loss function helps in refining the generation of local weather phenomena, focusing on the specifics that the deterministic model might miss.
* **Hybrid Loss**: Different loss function can be combined to train complex, multi-part models. 

## CasCast Model

CasCast is a deep learning model designed for high-resolution precipitation nowcasting, which tackles the challenge of accurately predicting precipitation in the short term using radar data. This is a novel model that incorporates a cascaded architecture.

#### Cascaded Architecture

It is structured into two main components; a deterministic model and a probabilistic model, that work in tandem. This cascaded approach allows the model to effectively handle the complexities of precipitation systems that operate at different scales which was a challenge to previous models.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/cascaded_model1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>


#### Deterministic Model

This first part of the model can incorporate conventional neural network architectures such as CNNs, RNNs, or Transformers, that are trained to minimize mean squared error (MSE) loss. This part of the model is responsible for predicting the mesoscale aspects of precipitation (larger, more predictable patterns). This allows the model to capture the broad movements in weather patterns. The output of this models provides a solid foundation for the second type of models used in this architecture - the probabilistic model.
Seen here is the deterministic part of the model architecture.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/deterministic.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>


#### Probabilistic Model

Using the output of the deterministic model, the probabilistic model generates the fine-grained details and local variations within the precipitation pattern. It aims to model the stochasticity inherent in meteorological systems, this is particularly useful for capturing the nuances of extreme weather events.
In CasCast the probabilistic model is a frame-wise-guided diffusion transformer. This is a generative model that simulates the process of adding and removing noise to generate detailed predictions. This part is crucial for enhancing the resolution and accuracy of predictions at a localized scale.
Seen below is the probabilistic part of the model architecture.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/probabilistic.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>


## Results

## Analysis

<!-- 
{% raw %}

```liquid
{% tabs group-name %}

{% tab group-name tab-name-1 %}

Content 1

{% endtab %}

{% tab group-name tab-name-2 %}

Content 2

{% endtab %}

{% endtabs %}
```

{% endraw %}

With this you can generate visualizations like:

{% tabs log %}

{% tab log php %}

```php
var_dump('hello');
```

{% endtab %}

{% tab log js %}

```javascript
console.log("hello");
```

{% endtab %}

{% tab log ruby %}

```javascript
pputs 'hello'
```

{% endtab %}

{% endtabs %}

## Another example

{% tabs data-struct %}

{% tab data-struct yaml %}

```yaml
hello:
  - "whatsup"
  - "hi"
```

{% endtab %}

{% tab data-struct json %}

```json
{
  "hello": ["whatsup", "hi"]
}
```

{% endtab %}

{% endtabs %}

## Tabs for something else

{% tabs something-else %}

{% tab something-else text %}

Regular text

{% endtab %}

{% tab something-else quote %}

> A quote

{% endtab %}

{% tab something-else list %}

Hipster list

- brunch
- fixie
- raybans
- messenger bag

{% endtab %}

{% endtabs %} -->
