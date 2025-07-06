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
<audio controls src="/assets/audio/test1_input.wav"></audio>  

**Ground Truth Instrumental**  
<audio controls src="/assets/audio/test1_gt.wav"></audio>  

**Genre Prompt:** `Pop`  

**MusicGen Pretrained Output**  
<audio controls src="/assets/audio/test1_pretrained.wav"></audio>  

**MusicGen Fine-tuned Output**  
<audio controls src="/assets/audio/test1_musicgen_ft.wav"></audio>  

**AudioLDM Fine-tuned Output**  
<audio controls src="/assets/audio/test1_audioldm_ft.wav"></audio>  

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
- Real-time applications in **DAWs**, music apps, or web tools

## **References**

- [AudioLDM](https://arxiv.org/abs/2301.12503)
- [MusicGen (AudioCraft)](https://arxiv.org/abs/2306.05284)
- [CLAP: Contrastive Language-Audio Pretraining](https://arxiv.org/abs/2301.12661)
- [EnCodec](https://arxiv.org/abs/2210.13438)
- [HTSAT-CLAP Implementation](https://github.com/LAION-AI/CLAP)

---

