---
layout: post
title: AudioLDM & MusicGen for Voice-to-Music Generation
date: 2025-07-05 14:00:00
description: A deep learning blog post on extending AudioLDM and MusicGen for voice-guided multi-modal music generation
tags: deep-learning music-generation diffusion-models transformers
categories: blog-posts
tabs: true
toc:
  sidebar: right
---

Welcome to this blog post on **generating instrumental music from vocal input**, a novel and intuitive interface for musical expression. This post presents and summarizes work done on extending two prominent models—**AudioLDM** and **MusicGen (AudioCraft)**—to perform **voice-to-music generation** with optional multi-modal conditioning.

## **Introduction**

Audio synthesis has recently seen breakthroughs via **latent diffusion models** and **autoregressive transformers**. Among these, **AudioLDM** and **MusicGen** are state-of-the-art frameworks in the realm of controllable sound generation.

However, despite these advances, using the **human voice** as a control modality remains underexplored. Voice—through singing or humming—provides a natural way for non-musicians to guide music generation.

In this project:
- We extend **AudioLDM** to support **audio-based conditioning** (voice input).
- We fine-tune **MusicGen** to generate music from **melody and genre prompts**.
- We build a custom **vocal-instrumental dataset** with 416 paired tracks across 42 artists and multiple languages.
- We evaluate both models and show that **multi-modal conditioning** leads to better generation quality.

## **Background**

### AudioLDM

AudioLDM is a **latent diffusion model** designed for text-to-audio generation. It works in Mel spectrogram space and includes:
- A **VAE** for encoding and decoding spectrograms
- The **CLAP** encoder for extracting semantic information from text/audio
- A **UNet**-based diffusion model to generate latent audio

#### AudioLDM Pipeline

1. Encode text using CLAP → embedding
2. Encode input audio into latent Mel via VAE
3. Generate latent audio with diffusion model (conditioned on embedding)
4. Decode Mel → waveform using HiFi-GAN

### MusicGen (AudioCraft)

MusicGen, from Meta’s **AudioCraft**, is an **autoregressive Transformer** for high-fidelity music generation using:
- Discrete audio tokens from **EnCodec**
- Melody conditioning using **chroma pitch features**
- Optional **text conditioning** for genre/style control

#### MusicGen Pipeline

1. Extract chroma features from melody (e.g., vocal input)
2. Encode text prompt (T5 encoder)
3. Transformer generates EnCodec tokens
4. Decode tokens into waveform (32kHz audio)

#### Applications

- **Voice-to-instrument** synthesis
- **Prompt-based music composition**
- **Genre transfer**
- **Melody harmonization**

## **Dataset**

To train both models, we created a dataset of:

- **416 tracks** across **42 artists**
- Paired **vocals** and **instrumentals**
- Metadata: genre, BPM, key, mood, artist name
- **Languages**: English (69%), Arabic (19%), French (12%)

Tracks were segmented for training:
- 30s chunks for **MusicGen**
- 20s chunks for **AudioLDM**

## **Methodology**

### Extending AudioLDM

Original AudioLDM only supported **text conditioning**. We extended it to support **multi-modal conditioning** by:

- Using both **CLAP text and audio encoders**
- Concatenating embeddings for unified context
- Injecting context via FiLM layers or direct input to UNet

This allows generation from:
- Voice only
- Voice + text prompt (e.g., genre)
- Text only

### Fine-tuning MusicGen

We fine-tuned the **1.5B parameter MusicGen-Melody model** in two stages:

1. **Voice-to-music**: using isolated vocals to guide melody
2. **Voice + text**: adding genre-specific text prompts (e.g., “Disco music for input vocals”)

## **Experimental Setup**

Training was done on an **NVIDIA A40 (48 GB VRAM)** GPU.

### MusicGen Fine-Tuning

- Model: MusicGen-Melody (1.5B)
- Batch size: 2
- Epochs: 48
- Duration: ~48 hours

### AudioLDM Fine-Tuning

- Model: AudioLDM-s (330M)
- Trained for 10k steps / 12 hrs per variant
- Variants:
  - Voice only
  - Voice + text
  - Refined text prompts (to match MusicGen prompt format)

## **Results**

### MusicGen Output

MusicGen showed:
- Strong alignment to both **melody and genre prompts**
- Stylistic consistency across genres
- Clearer, more coherent outputs than the pretrained baseline

### AudioLDM Output

- **Voice-only**: poor output, often noisy or incoherent
- **Voice + text**: major improvement
- **Refined prompts**: better genre alignment and clarity

While improved, AudioLDM still underperforms compared to MusicGen, especially in musical coherence and fidelity.



### 🎧 Voice-to-Music Generation Results

{% tabs test-cases %}

{% tab test-cases Test 1: Pop %}

**Input Vocal**  
<audio controls src="/assets/audio/test1_input_2.wav"></audio>  

**Ground Truth Instrumental**  
<audio controls src="/assets/audio/test1_gt_2.wav"></audio>  

**Genre Prompt:** `Pop`  

**MusicGen Pretrained Output**  
<audio controls src="/assets/audio/test1_pretrained_2.wav"></audio>  

**MusicGen Fine-tuned Output**  
<audio controls src="/assets/audio/test1_musicgen_ft_2.wav"></audio>  

**AudioLDM Fine-tuned Output**  
<audio controls src="/assets/audio/test1_audioldm_ft_2.wav"></audio>  

{% endtab %}

{% tab test-cases Test 2a: Pop Prompt %}

**Input Vocal**  
<audio controls src="/assets/audio/test2_input.wav"></audio>  

**Ground Truth Instrumental**  
<audio controls src="/assets/audio/test2_gt.wav"></audio>  

**Genre Prompt:** `Pop`  

**MusicGen Fine-tuned Output**  
<audio controls src="/assets/audio/test2_musicgen_pop.wav"></audio>  

**AudioLDM Fine-tuned Output**  
<audio controls src="/assets/audio/test2_audioldm_pop.wav"></audio>  

{% endtab %}

{% tab test-cases Test 2b: Disco Prompt %}

**Input Vocal**  
<audio controls src="/assets/audio/test2_input.wav"></audio>  

**Ground Truth Instrumental**  
<audio controls src="/assets/audio/test2_gt.wav"></audio>  

**Genre Prompt:** `Disco`  

**MusicGen Fine-tuned Output**  
<audio controls src="/assets/audio/test2_musicgen_disco.wav"></audio>  

**AudioLDM Fine-tuned Output**  
<audio controls src="/assets/audio/test2_audioldm_disco.wav"></audio>  

{% endtab %}

{% tab test-cases Test 3: Disco %}

**Input Vocal**  
<audio controls src="/assets/audio/test3_input.wav"></audio>  

**Ground Truth Instrumental**  
<audio controls src="/assets/audio/test3_gt.wav"></audio>  

**Genre Prompt:** `Disco`  

**MusicGen Pretrained Output**  
<audio controls src="/assets/audio/test3_pretrained.wav"></audio>  

**MusicGen Fine-tuned Output**  
<audio controls src="/assets/audio/test3_musicgen_ft.wav"></audio>  

**AudioLDM Fine-tuned Output**  
<audio controls src="/assets/audio/test3_audioldm_ft.wav"></audio>  

{% endtab %}

{% tab test-cases Test 4: Rock %}

**Input Vocal**  
<audio controls src="/assets/audio/test4_input.wav"></audio>  

**Ground Truth Instrumental**  
<audio controls src="/assets/audio/test4_gt.wav"></audio>  

**Genre Prompt:** `Rock`  

**MusicGen Pretrained Output**  
<audio controls src="/assets/audio/test4_pretrained.wav"></audio>  

**MusicGen Fine-tuned Output**  
<audio controls src="/assets/audio/test4_musicgen_ft.wav"></audio>  

**AudioLDM Fine-tuned Output**  
<audio controls src="/assets/audio/test4_audioldm_ft.wav"></audio>  

{% endtab %}

{% endtabs %}

### 🌍 Cultural Generalizability: Unclean Vocal Inputs

To evaluate how well the models generalize across **languages**, **accents**, and **noisy vocal inputs**, we tested on vocal tracks from different cultural backgrounds using known songs in **Arabic**, **French**, **Egyptian Arabic**, and **English**.

Each case uses the original unclean vocals and generates instrumental output via the fine-tuned models.

{% tabs cultural-generalizability %}

{% tab cultural-generalizability Arabic — كفّك إنتَ %}

**Culture:** Arabic  
**Song:** *Kefak enta (كفّك إنتَ)*  

**Input Vocal:**  
<audio controls style="width: 100%;">
  <source src="/assets/audio/culture_arabic_input.wav" type="audio/wav">
</audio>

**Generated Instrumental Output:**  
<audio controls style="width: 100%;">
  <source src="/assets/audio/culture_arabic_output.wav" type="audio/wav">
</audio>

{% endtab %}

{% tab cultural-generalizability French — La Vie en Rose %}

**Culture:** French  
**Song:** *La Vie en Rose — Édith Piaf*  

**Input Vocal:**  
<audio controls style="width: 100%;">
  <source src="/assets/audio/culture_french_input.wav" type="audio/wav">
</audio>

**Generated Instrumental Output:**  
<audio controls style="width: 100%;">
  <source src="/assets/audio/culture_french_output.wav" type="audio/wav">
</audio>

{% endtab %}

{% tab cultural-generalizability Egyptian — CairoKee %}

**Culture:** Egyptian Arabic  
**Song:** *Cairokee – James Dean*  

**Input Vocal:**  
<audio controls style="width: 100%;">
  <source src="/assets/audio/culture_egypt_input.wav" type="audio/wav">
</audio>

**Generated Instrumental Output:**  
<audio controls style="width: 100%;">
  <source src="/assets/audio/culture_egypt_output.wav" type="audio/wav">
</audio>

{% endtab %}

{% tab cultural-generalizability English — Sweet Caroline %}

**Culture:** English  
**Song:** *Sweet Caroline*  

**Input Vocal:**  
<audio controls style="width: 100%;">
  <source src="/assets/audio/culture_english_input.wav" type="audio/wav">
</audio>

**Generated Instrumental Output:**  
<audio controls style="width: 100%;">
  <source src="/assets/audio/culture_english_output.wav" type="audio/wav">
</audio>

{% endtab %}

{% endtabs %}

## 🔍 Evaluation Results

We conducted a comprehensive evaluation of our models using both **qualitative** and **quantitative** methods. The goal was to compare the performance of our fine-tuned models—**MusicGen Fine-tuned** and **AudioLDM Fine-tuned**—against the **pretrained MusicGen** baseline in generating instrumental music from vocal input.

---

### Qualitative Evaluation: User Listening Survey

To assess perceptual quality and alignment, we conducted a user study with **10 participants**, each comparing outputs from all three models across **4 different songs**. Participants answered **12 questions** covering:

* Vocal alignment
* Genre fit
* Overall audio quality


#### Win Rate Comparison (Per Question Basis)

| Model               | Win Rate (%) |
| ------------------- | ------------ |
| MusicGen Fine-tuned | **58.57%**   |
| MusicGen Pretrained | 43.33%       |
| AudioLDM Fine-tuned | 40.00%       |

> ℹ️ While MusicGen Fine-tuned led in preference, the relatively close win rates highlight the complexity of modeling user expectations and stylistic alignment in music generation.

#### 📊 Genre Preference Bar Chart (Simplified)

Across four test tracks, participants selected preferred outputs based on genre fit:

```
Track   | MusicGen Fine-tuned | AudioLDM Fine-tuned
--------|----------------------|---------------------
Test 1  | 7 votes              | 3 votes
Test 2  | 5 votes              | 5 votes
Test 3  | 5 votes              | 5 votes
Test 4  | 5 votes              | 5 votes
```

🎵 Interpretation: While MusicGen was generally preferred for melodic coherence, both models showed similar strength in capturing genre cues.

---

### Quantitative Evaluation

We used two established metrics to evaluate realism and prompt consistency:

#### CLAP Score (Text-Audio Alignment)

CLAP (Contrastive Language-Audio Pretraining) measures similarity between the generated audio and the text prompt.

| Model               | CLAP Score ↑ |
| ------------------- | ------------ |
| MusicGen Fine-tuned | **0.180**    |
| MusicGen Pretrained | **0.180**    |
| AudioLDM Fine-tuned | 0.117        |

**Higher is better.** MusicGen clearly excels at maintaining alignment with semantic prompts, while AudioLDM showed weaker consistency, despite its improvements from multi-modal fine-tuning.

#### Fréchet Audio Distance (FAD)

FAD assesses the realism of generated audio by comparing the statistical distribution of embeddings against real instrumentals.

| Model               | FAD Score ↓ |
| ------------------- | ----------- |
| MusicGen Pretrained | **10.64**   |
| MusicGen Fine-tuned | 10.70       |
| AudioLDM Fine-tuned | **9.48**    |

**Lower is better.** Interestingly, AudioLDM Fine-tuned achieved the lowest FAD score, indicating that it generates more acoustically realistic audio—even if semantically weaker. This suggests it captures low-level audio features well.

---

### Summary of Findings

* **MusicGen Fine-tuned** was preferred in qualitative tests and matched baselines in CLAP score.
* **AudioLDM Fine-tuned** produced more realistic audio per FAD but lacked in semantic alignment.
* Combining **voice + text conditioning** yields stronger results than using audio-only inputs.
* While MusicGen appears better suited for structured, genre-aware music generation, AudioLDM benefits from its latent-domain realism and could be enhanced further with architectural tuning.



## **Conclusion**

This work proposes a **voice-guided music generation** framework by extending two powerful audio generation models. Key contributions:

- A **custom dataset** with paired vocals/instrumentals and rich metadata
- **Multi-modal conditioning** for both AudioLDM and MusicGen
- **Transformer-based generation** (MusicGen) outperforms diffusion-based generation (AudioLDM) in quality

### Takeaways

- **Voice + text prompts** offer the best control and realism
- **MusicGen** is better suited for voice-to-instrument tasks today
- **AudioLDM** can improve with further architecture tuning

### Future Work

- Larger datasets with studio-quality separation
- Conditioning on **chord progressions**, **lyrics**, or **emotions**
- Real-time applications in music apps, or web tools

## **References**

- [AudioLDM](https://arxiv.org/abs/2301.12503)
- [MusicGen (AudioCraft)](https://arxiv.org/abs/2306.05284)
- [CLAP: Contrastive Language-Audio Pretraining](https://arxiv.org/abs/2301.12661)
- [EnCodec](https://arxiv.org/abs/2210.13438)
- [HTSAT-CLAP Implementation](https://github.com/LAION-AI/CLAP)

---

