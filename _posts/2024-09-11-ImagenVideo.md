---
title: "[Paper Reivew] Imagen Video: High Definition Video Generation with Diffusion Models (Imagen Video)"
description: Imagen Video는 텍스트 조건에 따라 고해상도 비디오를 생성하는 시스템으로,  Cascaded video diffusion model을 기반으로 합니다. 
toc: true
comments: true
# layout: default
math: true
date: 2024-09-11 17:10:00 +09:00
categories: [Deep Learning, Generative Model]
tags: [diffusion model, generative model, google, t2v]     # TAG names should always be lowercase
image: /posts/20240910_ImagenVideo/thumbnail.webp
alt : Thumbnail
author: Daemin
---

> arXiv 2022. [[Paper](https://arxiv.org/abs/2210.02303)] [[Demo](https://imagen.research.google/video/)]

> Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P. Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, Tim Salimans

>Google Research, Brain Team

>5 Oct 2022

  
  
  

# TL;DR

**Imagen** 이라는 Text-to-Image 모델을 비디오까지 확장한 모델(Video Diffusion)입니다.

  

**Imagen Video**는 텍스트 조건에 따라 **고해상도 비디오를 생성하는 시스템**으로, **Cascaded video diffusion model**을 기반으로 합니다.

  

텍스트 프롬프트를 입력받아 기본 비디오 생성 모델과, **spatial 및 temporal super-resolution model**을 순차적으로 결합하여 고해상도 비디오를 생성합니다.

  

> [Demo Page](https://imagen.research.google/video/)를 보고 오시죠!

  

## 1. Introduction

저자들의 목표는 텍스트로부터 비디오를 생성하는 것이라고 합니다. 이전의 비디오 생성 모델은 주로 Autoregressive (AR) 기반의 모델이지만, 저자들의 이전 연구에서 diffusion 기반의 모델도 비디오 가능성을 보여주었습니다.

  

이번 연구에서 저자들은 **Imagen Video**라는 Video Diffusion을 기반으로 한 Text-to-Video 시스템을 제안합니다. **Imagen Video**는 고해상도의 비디오를 생성할 수 있으며, 높은 프레임 fidelity, 강한 시간적 일관성, 그리고 깊은 언어 이해력을 갖추고 있습니다. 이전 연구는 주로 낮은 해상도의 짧은 비디오 생성에 중점을 두었지만, **Imagen Video**는 128프레임의 **1280×768 해상도의 비디오**를 초당 24프레임으로 생성할 수 있도록 확장되었습니다.

  

**Imagen Video**는 **간단한 아키텍처**로 구성됩니다. 이 모델은 **T5 텍스트 인코더**, 기본 비디오 확산 모델, 그리고 공간 및 시간적 **초해상도 확산 모델**로 구성됩니다. 이 논문의 주요 기여는 다음과 같습니다:

  

1. 고해상도 비디오 생성을 위한 **cascaded diffusion video model**의 단순성과 효과를 입증합니다.

2.  **텍스트-이미지 설정**에서 발견된 최근 연구 결과들이 **비디오 생성**에도 적용될 수 있음을 확인합니다. (예를 들어, **frozen encoder text conditioning**과 **CFG**의 효과 등)

3. Video Diffusion model에 대한 새로운 발견을 제시하며, 이는 확산 모델 전반에 영향을 미칠 수 있습니다. 예를 들어, **v-prediction 파라미터화**가 샘플 품질에 미치는 영향과 **progressive distillation**의 효과를 보여줍니다.

4. Imagen Video는 **3D 객체 이해**와 텍스트 애니메이션 생성, 다양한 예술적 스타일의 비디오 생성과 같은 **qualitative controllability** 기능을 입증합니다.

  
  
  

## 2. Imagen Video

**Imagen Video**는 비디오 확산 모델을 기반으로 한 계단식 시스템으로, 텍스트 기반의 비디오 생성, 공간 초해상도, 시간 초해상도를 수행하는 7개의 하위 모델로 구성되어 있습니다. 이 시스템을 통해 **1280×768 해상도**에서 **초당 24프레임**의 **128프레임(약 5.3초)** 비디오를 생성하며, 약 **1억 2600만 픽셀**에 해당합니다.

  

![fig6](/posts/20240910_ImagenVideo/fig6.png){: width="800" height="300"}

  

### 2.1. Diffusion Models

Imagen Video는 continuous time에서 정의된 diffusion model입니다. $$ x \sim p(x) $$에서 시작하는 forward process $$ q(z \vert x) $$를 따르는 latent variable $$ z = \{ z_t \vert t \in [0, 1] \} $$ model입니다.

  

- Forward Process :

$$

\begin{equation}

q(z_t \vert x) = \mathcal{N}(z_t; \alpha_t x, \sigma_t^2 I), \quad q(z_t \vert z_s) = \mathcal{N}(z_t; (\alpha_t / \alpha_s) z_s, \sigma_{t \vert s}^2 I) \\

\end{equation}

$$

  

$$
\textrm{where} \; \quad  0  \le s < t \le  1, \quad  \sigma_{t \vert s}^2 = (1 - e^{\lambda_t - \lambda_s}) \sigma_t^2, \quad  \lambda_t = \log [\alpha_t^2 / \sigma_t^2]
$$

  

- Reverse Process :

위 forward process의 역과정을 학습하기 위해 $$  \mathbf z_t \sim q(\mathbf {z}_t|\mathbf x) $$로부터 노이즈를 점차적으로 제거해 $$ \hat {\mathbf x}_\theta(\mathbf z_t, \lambda_t) \approx \mathbf x $$를 예측하도록 한다. objective function은 다음과 같다.

  

$$
\begin{align}
\mathcal L(x) = \mathbb{E}_{\epsilon  \sim  \mathcal{N}(0, I), t \sim U(0,1)} [\left\| \hat{\epsilon}_{\theta}(z_t, \lambda_t) - \epsilon  \right\|^2_2]
\end{align}
$$

  

이때, $$  \mathbf z_t = \alpha_t  \mathbf x + \sigma  \mathbf  \epsilon  $$, $$  \hat{\epsilon}(z, \lambda) = \sigma^{-1}_t \left( z_t - \alpha_t  \hat{x_\theta}(z_t, \lambda_t) \right) $$이다.

  
  

**조건부 생성 모델링**에서는 텍스트와 이전 단계의 저해상도 비디오 등 조건 정보가 모델에 제공되며, 이 조건들을 사용하여 모델이 공간 및 시간 superresolution을 처리합니다. 텍스트 임베딩을 모든 초해상도 모델에 적용하는 것이 중요하며, 이는 더 높은 이미지 품질을 보장하는 데 중요한 요소로 작용합니다.

  

저자들은 샘플링 과정에서는 **discrete time ancestral sampler**를 사용합니다.

$$
\begin{equation}
q(z_s \vert z_t, x) = \mathcal{N} (z_s; \tilde{\mu}_{s \vert t} (z_t, x), \tilde{\sigma}_{s \vert t}^2 I) \\
\end{equation}
$$

$$  \textrm{where} \quad  \tilde{\mu}_{s \vert t} (z_t, x) = e^{\lambda_t - \lambda_s} (\alpha_s / \alpha_t) z_t + (1 - e^{\lambda_t - \lambda_s} \alpha_s) x \\
\textrm{ , } \quad  \tilde{\sigma}_{s \vert t}^2 = (1 - e^{\lambda_t - \lambda_s}) \sigma_s^2  
$$
  

$$ z_1  \sim  \mathcal{N}(0, 1) $$에서부터 시작해, ancestral sampler는 다음의 규칙을 따릅니다.

  

$$
\begin{equation}
z_s = \tilde{\mu}_{s \vert t} (z_t, \hat{x}_\theta (z_t)) + \sqrt{(\tilde{\sigma}_{s \vert t}^2)^{1 - \gamma}(\tilde{\sigma}_{t \vert s}^2)^\gamma} \epsilon
\end{equation}
$$

  

또한, **DDIM sampler**를 사용할 수 있는데, 샘플링을 가속화하고, 더 빠른 생성 과정을 위해 **Progressive Distillation(PD)** 을 적용하면 효과적이였다고 합니다.

  
  
  

### 2.2. Cascaded Diffusion models and text conditioning

저자들은 **Imagen**과 같이 base diffusion model에 super-resolution model을 연결하여 각 sub-model을 단순하게 유지하면서 high-dimensional 이미지(비디오)를 만들 수 있었다고 합니다.

Imagen Video의 전체적인 파이프라인은 다음과 같습니다.

- Frozen pretrained text encoder(T5-XXL)

- Base video diffusion model

- 3개의 Spatial Super-Resolution model(SSR)

- 3개의 Temporal Super-Resolution model(TSR)

**Cascaded Models**의 장점 중 하나는 각 확산 모델을 독립적으로 훈련할 수 있어 7개의 모델을 병렬로 훈련할 수 있다는 점이며, 인력 텍스트 프롬프트의 conditioning는 고정된 T5-XXL 텍스트 인코더의 임베딩을 활용합니다.

### 2.3. Video Diffusion Architecture

일반적인 Diffusion model은 2D U-Net구조를 이용하는데, Imagen Video는 각 해상도에서 spatial attention과 convolution이 합쳐진 형태로 구성함으로써 비디오 프레임간 의존성을 높일 수 있었다고 합니다.


저자들의 초기 연구에서 Video-U-Net를 연구했는데, 2D diffusion model을 3D space-time 구조로 확장한 것입니다. 각 노이즈 제거 모델이 여러 비디오 프레임을 동시에 처리하여 전체 비디오 프레임을 한 번에 생성합니다.


![fig7](/posts/20240910_ImagenVideo/fig6.png){: width="800" height="300"}

SSR, TSR 모델은 입력 비디오에 대해, 노이즈 데이터 $$  \mathbf z_t$$를 입력 채널별로 연결하여 conditioning합니다.

Base diffusion 모델은 낮은 프레임 수, 낮은 해상도의 데이터를 생성하며, Temporal attention을 사용하지만, SSR, TSR 모델은 temporal convolution을 사용해 메모리, 계산 cost를 줄였다고 합니다. (첫 두개의 super-resolution 모델에서만 Temporal attention도 사용)

  