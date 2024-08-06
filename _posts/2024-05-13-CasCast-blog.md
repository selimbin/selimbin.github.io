---
layout: post
title: CasCast
date: 2024-08-05 12:30:13
description: A blog Describing Skillful High Resolution Precipitation Nowcasting via CasCaded Modeling
tags: deep-learning weather-forecasting
categories: blog-posts
tabs: true
toc:
  sidebar: right
---

Welcome to my blog post summarizing and presenting the paper "[CasCast](https://arxiv.org/abs/2402.04290): Skillful High-resolution Precipitation Nowcasting via Cascaded Modelling". The paper will be presented in this blog post as it showcases a novel approach with great promise.

## **Introduction**

The CasCast paper represents an advancement in the field of meteorological forecasting, particularly in the accurate prediction of precipitation using high-resolution radar data. The paper specifically addresses the challenges faced in nowcasting, which is the prediction of weather conditions for a short period, usually up to two hours ahead. Accurate Weather Forecast in the immediate future are of critical importance for the fields of disaster management and in various social sectors. This paper aims to provide a robust solution to improve prediction accuracy, especially for extreme weather events.

## **Motivation**

Around the world extreme weather events result in significant damages every year. One of the most destructive consequences that can be caused by these events is flooding. Flooding is due to high amounts of precipitation that occur.
Weather Forecasting is essential to handle disaster, plan for them and has impacts on many sectors (transportation, event planning, etc..).
Precipitation events involve multiple scales of atmospheric systems, making accurate predictions challenging.
Furthermore most current methods struggle with Short-term forecasting (NowCasting). Nowcasting is defined here as forecasting events that will occur within the next 2 hours.  These predictions are essential for emergency management and disaster mitigation.
This precipitation data can be used to give Real-Time warning to impacted communities.

## **Some information to know beforehand**

Previous Research in this field has faced multiple problems.
First of all, precipitation events involve multiple scales of atmospheric systems, making accurate predictions challenging. These systems are influenced by Mesoscale precipitation as well as small scale systems.

Previous research also faced challenges in predicting extreme precipitation events, which are seen in small scales. Which is important because over the past 50 years, extreme-precipitation events have caused more than 1 million related deaths, and economic losses beyond US$ 2.8 trillion 

### Precipitation Systems

Mesoscale precipitation systems evolve over spatial ranges of tens to hundreds of kilometers and time scales of several hours, driven and constrained by relatively stable large-scale circulation.

Small-scale systems evolve within a range of a few kilometers and operating on time scales of minutes, is influenced by local processes such as heating, surface features, and other physical factors, which introduce stochasticity and unpredictability into the systems behavior.

### Models Used

Another problem is that short term forecasting has its limitations. Each type of model whether deterministic or probabilistic has its limitations. Deterministic models are unable to capture the fine-grained detail of precipitation patterns. On the other hand, Probabilistic models are unable to capture large scale movements.

Deterministic models aim to predict the overall motion of mid-scale precipitation systems with a single-value forecast, but they often lack detail and appear blurry because they average out the randomness of small-scale systems. 

Probabilistic models, on the other hand,  sample from various latent variables to represent the randomness of future weather, capturing small-scale phenomena better. 
However, they struggle with accurately forecasting the large-scale, predictable distribution of precipitation. 

In summary, current models still face challenges in simultaneously predicting both mesoscale and small-scale systems.

## **Deep Learning Approach**

### Mapping The Problem

In order to approach this problem, its structure must be well-defined. Using multiple inputs, experts are attempting to predict the weather conditions using deep Learning Models.

The inputs used are generally Radar Data (High Resolution Radar Echo Images) as well as a variety of Atmospheric Variables; these can include temperature, humidity or wind patterns for example. This is the data from the past, from time 0 to T, where T is the current time step (present). The data covers T time steps

This is an example of a High Resolution Radar Echo Image that can be used as input.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/radar_echo_input.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Example of a High Resolution Radar Echo Image
</div>

The desired outputs are accurate precipitation maps of the affected areas, below is an example of a desired precipitation map. 

In this image areas of precipitation are colored from lightest precipitation to highest precipitation, this progression is as follows: green, yellow, orange, red and pink. The darker the shade in each color the higher the precipitation.

It is also important to note that the pink areas are the areas of extreme precipitation.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/output_sample_1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

<div class="caption">
    Output Precipitation Map from CasCast
</div>

### Loss Function

Multiple Loss Function can be used to train the models.
A couple of loss functions that can be used are:
* **Mean Squared Error (MSE)**: Can be used for training deterministic models. It measures the average squared difference between the estimated values and what is observed. This loss helps in minimizing the forecast error in terms of the general precipitation distribution.
* **Noise Prediction Loss**: Can be used in probabilistic models where a diffusion process is involved. This loss function helps in refining the generation of local weather phenomena, focusing on the specifics that the deterministic model might miss.
* **Hybrid Loss**: Different loss function can be combined to train complex, multi-part models. 

## **What loss functions and scoring functions are used ?**
### Loss Functions
#### Mean Squared Error Loss

In deterministic component the mean squared error loss is used.

$$
L_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} \left( y_i - \hat{y}_i \right)^2
$$

where $$y_i$$ is the observed value and $$\hat{y}_i$$ is the predicted value.
MSE is used to minimize the average squared differences between the predicted and observed values, this ensures that the model captures the general precipitation patterns accurately.

#### Noise Prediction Loss

One of the loss functions used is the Noise Prediction Loss, which was presented by [Ho et al](https://arxiv.org/pdf/2006.11239) in 2020:

$$
 L_{\theta_p} = \mathbb{E}_{\epsilon, k} \left[ \| \epsilon - \epsilon_{\theta_p}(z_k, k, z_{\text{cond}}) \|_2^2 \right]
$$

where:
- $$\epsilon$$ is the true noise added to the data during the forward diffusion process.
- $$k$$ is the current time step in the diffusion process.
- $$z_k$$ is the latent variable at time step $$k$$.
- $$\epsilon_{\theta_p}(z_k, k, z_{\text{cond}})$$ is the predicted noise by the model.
- $$z_{\text{cond}}$$ represents the conditional information, including the latent representations of the initial radar observations and deterministic model outputs.

The noise prediction loss is utilized in the probabilistic component, particularly within the diffusion model framework.
The objective of the noise prediction loss is to train the model to accurately predict the noise ϵ added at each step of the diffusion process. 
By minimizing this loss, the model learns to reverse the noise addition, effectively denoising the data to retrieve the original high-resolution precipitation patterns.

#### Hybrid Loss

By integrating both losses, the hybrid loss ensures that the model captures both broad precipitation patterns (mesoscale) and fine-grained details (small-scale), balancing deterministic accuracy and probabilistic realism.

### Scoring Formulas

In the paper 4 different scoring methods are used to compare the results from different models. These scoring methods are : Critical Success Index (CSI), Heidke Skill Score (HSS), Continuous Ranked Probability Score (CRPS) and Continuous Ranked Probability Score (CRPS).

Some of these scoring methods can be viewed in detail below.

<!-- #### Critical Success Index (CSI) -->

{% details Critical Success Index (CSI) %}
<br/>
It measures the accuracy of binary event forecasts, particularly useful for precipitation where the event is the occurrence of rainfall above a certain threshold.

$$
\text{CSI} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}
$$

CSI Measures the accuracy of binary event forecasts, which is useful for precipitation where the event is the occurrence of rainfall above a certain threshold (precipitation above a particular intensity).

{% enddetails %}


<br/>
<!-- #### Heidke Skill Score (HSS) -->
{% details Heidke Skill Score (HSS) %}
<br/>
Evaluates the accuracy of forecasts relative to random chance, considering the number of correct predictions.

$$
\text{HSS} = \frac{2(\text{TP} \cdot \text{TN} - \text{FP} \cdot \text{FN})}{(\text{TP} + \text{FN})(\text{FN} + \text{TN}) + (\text{TP} + \text{FP})(\text{FP} + \text{TN})}
$$

HSS is used to evaluate the overall skill of the model compared to random chance, providing a balanced measure of accuracy.
{% enddetails %}
<br/>
<!-- #### Continuous Ranked Probability Score (CRPS) -->
{% details Continuous Ranked Probability Score (CRPS) %}
<br/>
Measures the accuracy of probabilistic forecasts by comparing the predicted cumulative distribution function (CDF) to the observed outcome.

$$
\text{CRPS}(F, y) = \int_{-\infty}^{\infty} \left[ F(x) - \mathbf{1}(x \geq y) \right]^2 dx
$$

where $$F(x)$$ is the CDF of the forecasted distribution at value $$x$$, and $$\mathbf{1}(x \geq y)$$ is the indicator function that equals 1 if $$x \geq y$$ and 0 otherwise.

CRPS evaluates the probabilistic predictions, ensuring that the predicted distributions align well with the actual observed values.
{% enddetails %}

<!-- #### Structural Similarity Index Measure (SSIM) -->
<br/>
{% details Structural Similarity Index Measure (SSIM) %}
<br/>
Measures the perceived quality of the predictions compared to the ground truth, considering luminance, contrast, and structure.

SSIM is calculated using a sliding window approach over the images, typically involving:

$$
\text{SSIM}(x, y) = \frac{(2 \mu_x \mu_y + c_1)(2 \sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}
$$

where $$\mu_x$$ and $$\mu_y$$ are the means of the two images $$x$$ and $$y$$, $$\sigma_x^2$$ and $$\sigma_y^2$$ are the variances, $$\sigma_{xy}$$ is the covariance, and $$c_1$$ and $$c_2$$ are constants to stabilize the division.

SSIM is used to assess the visual similarity between predicted precipitation patterns and the actual observations, providing a measure of structural accuracy. 
{% enddetails %}
<br/>

## **CasCast Model**

CasCast is a deep learning model designed for high-resolution precipitation nowcasting, which tackles the challenge of accurately predicting precipitation in the short term using radar data. This is a novel model that incorporates a cascaded architecture.

### Cascaded Architecture

It is structured into two main components; a deterministic model and a probabilistic model, that work in tandem. This cascaded approach allows the model to effectively handle the complexities of precipitation systems that operate at different scales which was a challenge to previous models.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/cascaded_model1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

<div class="caption">
    CasCast Model Design
</div>

### Deterministic Model

This first part of the model can incorporate conventional neural network architectures such as CNNs, RNNs, or Transformers, that are trained to minimize mean squared error (MSE) loss. 

This part of the model is responsible for predicting the mesoscale aspects of precipitation (larger, more predictable patterns). 
 
This allows the model to capture the broad movements in weather patterns. The output of this models provides a solid foundation for the second type of models used in this architecture - the probabilistic model.

Seen here is the deterministic part of the model architecture.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/deterministic.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    CasCast Deterministic Component Architecture
</div>


### Probabilistic Model

Using the output of the deterministic model, the probabilistic model generates the fine-grained details and local variations within the precipitation pattern. It aims to model the stochasticity inherent in meteorological systems, this is particularly useful for capturing the nuances of extreme weather events.

In CasCast the probabilistic model is a frame-wise-guided diffusion transformer. This is a generative model that simulates the process of adding and removing noise to generate detailed predictions. This part is crucial for enhancing the resolution and accuracy of predictions at a localized scale.

Seen below is the probabilistic part of the model architecture.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/probabilistic.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    CasCast Probabilistic Component Architecture
</div>

### Overall Architecture

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/overall_architecture.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    CasCast Architecture
</div>

As you can see here the model has 2 parts a deterministic and probabilistic part (shown on the left side).

The deterministic model takes the input and generates mesoscale precipitation.
All this data is then combined and given to the Casformer.

The diffusion denoise process generates the fine-grained details using the input data and denoising the image.

Diffusion models learn the reverse process of gradually noising data $$x_0$$ into Gaussian noise.

CasFormer has a frame-wise encoding stage and a sequence-wise decoding stage. The frame-wise encoding provides better-matched conditions for each frame-wise latent vector, reducing the complexity of the denoising conditioned by a sequence of blurry predictions.

Sequence-wise decoding utilizes the sequence features from the sequence aggregator to ensure the spatiotemporal consistency of precipitation nowcasting. 

This frame-wise guidance in diffusion transformer ensures a frame-to-frame correspondence between blurry predictions and latent vectors, resulting in better optimization for the generation of small-scale patterns.

## **Results**

The results obtained from the CasCast model shows that it has powerful capabilities in high-resolution precipitation nowcasting, particularly in handling complex weather events.

### Datasets Used

The CasCast model was trained and tested on three radar precipitation datasets to evaluate its performance and robustness in different geographic and climatic conditions. These datasets were:
* The **SEVIR** Dataset comprising weather radar observations mostly from the United States. It features a spatial resolution of 1km and covers a large geographic area.
* The **HKO-7** Dataset that comes from the Hong Kong Observatory and is primarily used for studying the regional weather conditions around Hong Kong. It contains radar CAPPI (Constant Altitude Plan Position Indicator) reflectivity images, which are critical for analyzing rainfall and storm patterns at an altitude of 2km. The dataset has a resolution of 480x480 pixels, covering a 512km x 512km area around Hong Kong.
* The **MeteoNet** Dataset from Meteo France. This dataset includes comprehensive weather data from different regions of France. The data is recorded with a spatial resolution of approximately 0.01 degrees. For practical applications, a portion of 400x400 pixels from the top left corner is often used to ensure quality and consistency.


### Comparative Results

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
<div class="caption">
    Comparative Results between multiple models using multiple datasets
</div>

### Visual Comparisons

Comparisons with outputs from other models highlight CasCast's superior performance in capturing both macro and micro-precipitation dynamics without issues like blurring or oversimplification that can be seen in other forecasting models.
The results demonstrate that the model can distinguish between different precipitation intensities and spatial distributions with high fidelity.

Some output image comparisons can be seen below.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/cascast_results_2.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Sample Model Output Image
</div>

The animation below shows an important distinction between CasCast and other models. As can be seen below CasCast is the model most able to predict the ground truth. 

The clear advantage of CasCast is that it is the only model that it able to detect the areas of extreme precipitation (pink areas).

<div class="col-sm mt-3 mt-md-0">
    {% include video.liquid path="assets/img/high_precip.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=true %}
</div>
<div class="caption">
    Video Comparing Resutls using Different Models
</div>
### Computational Efficiency

Despite the high-resolution outputs, CasCast efficiently manages computational resources, making it suitable for real-time operational use. This efficiency is very important for practical deployment in meteorological stations and emergency management systems where time is essential when producing predictions.


## **Analysis**

The CasCast model presents several strengths and limitations. Below is an overview of some of its pros based on its described capabilities and performance:

### Pros

* **Enhanced Prediction Accuracy for Extreme Events:** CasCast excels in predicting extreme precipitation events. The model surpasses baseline models in accuracy for regional extreme-precipitation nowcasting, it can be a valuable tool for sectors that rely heavily on accurate weather predictions, such as agriculture, transportation, and public safety as well as disaster management and emergency services.

* **Efficient Computational Performance:** CasCast also uses computational resources efficiently. This efficiency makes it feasible for real-time applications.

### Cons

* **Complexity in Training and Implementation:** The dual-model structure of CasCast adds complexity in training and model tuning. This complexity could pose challenges in terms of the time and resources needed for model training and optimization.

* **Generalizability Across Different Regions:** However one diffeciency is that CasCast needs to be retrained for different geographic regions and conditions to maintain its accuracy and effectiveness. This requirement limits its applicability in global settings without retraining.

## **Conclusion**

### Future Work

The authors of this paper propose to further explore incorporating additional data sources like wind and temperature data to train the model.

Another aspect that needs to be further researched is multi region training.
To train the model to work on any region or design a unified model that can work on multiple datasets.

### What should you remember?

So, what should you remember from this paper? 

* First It proposes a dual model approach: CasCast combines deterministic and Probabilistic models to achieve high resolutions.
* CasCast can enhance disaster preparedness, by providing more detailed and reliable forecasts.
* Finally, Cascast is more efficient. It manages computational resources efficiently, making it suitable for real-time applications.

## **References**

* [CasCast: Skillful High-resolution Precipitation Nowcasting via Cascaded Modelling](https://arxiv.org/abs/2402.04290)
* [Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.](https://arxiv.org/pdf/2006.11239)
* Chen, & Zhang, Chunze & Liu, Jingtian & Zeng,. (2019). Generative Adversarial Networks Capabilities for Super-Resolution Reconstruction of Weather Radar Echo Images. Atmosphere. 10. 555. [10.3390/atmos10090555](https://www.researchgate.net/publication/335862244_Generative_Adversarial_Networks_Capabilities_for_Super-Resolution_Reconstruction_of_Weather_Radar_Echo_Images)
* All images, tables and figures in this post can be found in the references above.


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
