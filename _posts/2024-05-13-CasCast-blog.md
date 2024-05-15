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

## **Introduction**

The CasCast paper represents an advancement in the field of meteorological forecasting, particularly in the accurate prediction of precipitation using high-resolution radar data. The paper specifically addresses the challenges faced in nowcasting, which is the prediction of weather conditions for a short period, usually up to two hours ahead. Accurate Weather Forecast in the immediate future are of critical importance for the fields of disaster management and in various social sectors. This paper aims to provide a robust solution to improve prediction accuracy, especially for extreme weather events.

## **Motivation**

Around the world extreme weather events result in significant damages every year. One of the most destructive consequences that can be caused by these events is flooding. Flooding is due to high amounts of precipitation that occur.
Precipitation events involve multiple scales of atmospheric systems, making accurate predictions challenging.
Furthermore most current methods struggle with Short-term forecasting (NowCasting), and these predictions are essential for emergency management and disaster mitigation.

## **Deep Learning Approach**

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

## **CasCast Model**

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

#### Overall Architecture

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/overall_architecture.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

## **Results**

The results obtained from the CasCast model shows that it has powerful capabilities in high-resolution precipitation nowcasting, particularly in handling complex weather events.

#### Datasets Used

The CasCast model was trained and tested on three radar precipitation datasets to evaluate its performance and robustness in different geographic and climatic conditions. These datasets were:
* The **SEVIR** Dataset comprising weather radar observations mostly from the United States. It features a spatial resolution of 1km and covers a large geographic area.
* The **HKO-7** Dataset that comes from the Hong Kong Observatory and is primarily used for studying the regional weather conditions around Hong Kong. It contains radar CAPPI (Constant Altitude Plan Position Indicator) reflectivity images, which are critical for analyzing rainfall and storm patterns at an altitude of 2km. The dataset has a resolution of 480x480 pixels, covering a 512km x 512km area around Hong Kong.
* The **MeteoNet** Dataset from Meteo France. This dataset includes comprehensive weather data from different regions of France. The data is recorded with a spatial resolution of approximately 0.01 degrees. For practical applications, a portion of 400x400 pixels from the top left corner is often used to ensure quality and consistency.


#### Comparative Results

CasCast has demonstrated substantial improvements over existing models, especially in predicting extreme weather events. The model surpasses baseline models by up to 91.8% in regional extreme-precipitation nowcasting. This leap in performance showcases the efficacy of the cascaded modeling approach and the strength of CasCast.

Using metrics like the Critical Success Index (**CSI**), Heidke Skill Score (**HSS**), and Continuous Ranked Probability Score (**CRPS**) the model shows high performance.

The model had the highest **CSI** scores. Improvements in **CSI**, particularly at finer scales, indicate that the model can more accurately detect where and how intense precipitation events will occur, matching the actual observed data more closely than previous models.

Furthermore CasCast had the highest **HSS** score when compared to other models. Improved **HSS** scores suggest that CasCast is better at distinguishing between events and non-events, which is crucial for preventing false alarms—a common issue in weather prediction.

Finally CasCast have the lowest **CRPS** scores among the models tested, lower **CRPS** score indicate that the probabilistic forecasts of CasCast closely resemble the actual outcomes, suggesting high predictive reliability and better uncertainty estimation in forecasts.

The results of these comparative tests can be seen below.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/cascast_results_1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

#### Visual Comparisons

Comparisons with outputs from other models highlight CasCast's superior performance in capturing both macro and micro-precipitation dynamics without issues like blurring or oversimplification that can be seen in other forecasting models.
The results demonstrate that the model can distinguish between different precipitation intensities and spatial distributions with high fidelity.

Some output image comparisons can be seen below.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/cascast_results_2.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/cascast_results_3.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

#### Computational Efficiency

Despite the high-resolution outputs, CasCast efficiently manages computational resources, making it suitable for real-time operational use. This efficiency is very important for practical deployment in meteorological stations and emergency management systems where time is essential when producing predictions.


## **Analysis**

The CasCast model presents several strengths and limitations. Below is an overview of some of its pros based on its described capabilities and performance:

#### Pros

* **Enhanced Prediction Accuracy for Extreme Events:** CasCast excels in predicting extreme precipitation events. The model surpasses baseline models in accuracy for regional extreme-precipitation nowcasting, it can be a valuable tool for sectors that rely heavily on accurate weather predictions, such as agriculture, transportation, and public safety as well as disaster management and emergency services.

* **Efficient Computational Performance:** CasCast also uses computational resources efficiently. This efficiency makes it feasible for real-time applications.

#### Cons

* **Complexity in Training and Implementation:** The dual-model structure of CasCast adds complexity in training and model tuning. This complexity could pose challenges in terms of the time and resources needed for model training and optimization.

* **Generalizability Across Different Regions:** However one diffeciency is that CasCast needs to be retrained for different geographic regions and conditions to maintain its accuracy and effectiveness. This requirement limits its applicability in global settings without retraining.

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
