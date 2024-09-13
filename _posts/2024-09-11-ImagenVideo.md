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




### 2.4. v-prediction

  

**v-parameterization**을 사용하여 모든 모델을 파라미터화합니다. 여기서 $$ v_t \equiv  \alpha_t  \epsilon - \sigma_t $$입니다. 이 접근 방식은 diffusion process에 수치적 안정성을 제공하여 모델의 progressive distillation을 가능하게 합니다.

  

또한 고해상도 모델에서 v-parameterization을 사용하면 색상 이동 아티팩트를 피할 수 있으며, 샘플 품질 메트릭의 수렴 속도가 빨라지는 장점도 있었다고 합니다.

  

### 2.5. Conditioning Augmentation

저자들은 SSR과 TSR에서 noise conditioning augmentation을 사용했다고 합니다. 이는 cascaded diffuison model에서 class-conditional 생성시 매우 중요하다고 합니다. 특히 cascaded된 각 모델들의 병렬 학습을 가능하게 하며, 각 stage의 domain gap을 줄여주는 역할을 합니다.

  

### 2.6. Video-Image Joint Training

저자의 이전연구를 따라, 이미지와 비디오를 함께 이용해 Imagen Video를 학습했다고 합니다. 학습시 개별 이미지를 video frame의 한 장면으로 간주하여 독립적인 이미지를 동일한 길이의 비디오로 묶어서 처리하는데, temporal convolution은 computation path에따라 masking된다고 합니다. 이러한 전략을 통해 video-text dataset에비해 훨씬 많은 image-text dataset을 사용할 수 있으며, 비디오 샘플의 품질을 크게 향상시킬 수 있었다고 합니다.

  

#### 2.6.1. Classifier Free Guidance

> [CFG, 이전포스트](https://daemini.github.io/posts/Classifier-Free-Diffusion-Guidance/)

  

Conditional generation 세팅에서, data $$  \mathbf x $$는 signal $$ \mathbf c $$(텍스트 프롬프트의 embedding)에 의해 conditioning되어 생성된다. Diffusion 모델은 $$ \mathbf c $$를 denosining의 추가 입력으로 사용해 학습시킬 수 있다. $$ \hat{\mathbf x}_\theta (z_t, c) $$ 학습이 완료되면 guidance scale을 적용해, 다음과 같이 표현할 수 있다.

  

$$

\begin{equation}

\tilde{x}_\theta (z_t, c) = (1 + w) \hat{x}_\theta (z_t, c) - w \hat{x}_\theta (z_t)

\end{equation}

$$

  

> 이 식은 $$\mathbf v$$-space와 $$\mathbf  \epsilon$$-space에서도 똑같이 활용 가능하다.

  

guidance weight $$ w > 0 $$의 경우 conditioning을 과하게 강조하는 효과가 있으며, 다양성은 낮지만 높은 품질의 샘플을 생성하는 경향이 있다.

  

#### 2.6.2. Large Guidance Weights

너무 큰 guidance weight를 사용하면 train-test mismatch가 발생하는 문제가 있습니다. 이를 위해 Imagen과 마찬가지로 dynamic clipping을 사용합니다.

> e.g. $$  \text{np.clip}(x, -s, s) / s $$

  

하지만 dynamic clipping만으로는 over-staturation 문제가 여전히 발생해, 저자들은 guidance weight를 각 sampling 단계마다 high->low로 바꾸는 *oscillating guidance* 방법을 적용해 이를 해결했습니다.

  

(1) 샘플링 처음 시작 시 constant high guidance weight -> 텍스트를 강조

(2) 이후 high guidance weight($$ w= 15  $$)는 강한 텍스트 정렬을 유지

(3) 그 다음, low guidance weight($$ w= 1  $$)는 sturation artifact 줄이는 데 도움

(2), (3) 번갈아서 weight 바꾸기.

  

하지만 **80×48 이상의 해상도**에서 진동 가이드를 적용했을 때는 샘플 품질 개선 없이 더 많은 시각적 아티팩트가 발생했습니다. 따라서 저자들은 이 진동 가이드를 기본 모델과 초기 두 개의 **SR(Super-Resolution) 모델**에만 적용했다고 합니다.



### 2.7. Progressive Distillation with Gudiance and Stochastic Samplers

  

Diffusion model의 빠른 sampling을 위한 방법으로 *progressive distillation*가 있습니다. 이는 이미 학습된 DDIM의 sampler를 증류하여 샘플링 step을 줄이면서도 perceptual quality를 유지하는 방법입니다.

  

이 방법의 확장으로 guidance를 추가한 새로운 stochastic sampler가 있는데, 저자들은 이 방법이 video diffusion에도 효과적이였다고 합니다. 저자들은 DDIM sampler에 두 단계의 distillation 접근법을 적용합니다.

  

1. First stage에서 conditional, unconditional 모두 하나의 diffusion을 이용해 학습합니다.

2. Second stage에서 더 적은 step으로 샘플링 할 수 있도록 progressive distillation을 적용합니다.

  

Distillation이 끝나면, $$ N $$-step stochastic sampler를 사용합니다.

1. 먼저 deterministic DDIM update를 step size의 두 배 만큼으로 1번 진행합니다.

2. 이후 stochastic step을 거꾸로 원래 step size만큼 1 번 진행합니다.

  

이를 통해 모든 비디오 확산 모델을 8단계로 샘플링할 수 있으며, 인식 품질의 손실 없이 빠른 샘플링이 가능했다고 합니다.

  

## 3. Experiments

**데이터 및 평가**: 모델은 **내부 데이터셋**(14M 비디오-텍스트 쌍과 60M 이미지-텍스트 쌍)과 **LAION-400M** 이미지-텍스트 데이터셋을 기반으로 학습되었습니다.

  

모델 성능은 FID(개별 프레임 평가), FVD(시간적 일관성 평가), CLIP 점수(비디오-텍스트 정렬 평가)로 측정되었습니다.

  
  

### 3.1. Unique Video Generation Capabilties

  

Imagen Video는 **고해상도 비디오**를 생성할 수 있으며, 전통적인 unstructured 생성 모델에서는 찾아보기 힘든 몇 가지 독특한 기능을 가지고 있습니다.

  

![fig8](/posts/20240910_ImagenVideo/fig8.png){: width="800" height="300"}

모델이 이미지 정보를 학습하여 **반 고흐 스타일**이나 **수채화 스타일**의 비디오를 생성할 수 있음을 보여줍니다.

  
  

![fig9](/posts/20240910_ImagenVideo/fig9.png){: width="800" height="300"}

객체가 회전하는 동안 구조를 대체로 유지하며, 3D 구조를 이해하는 능력이 있음을 보여줍니다. 비록 회전 중 3D 일관성이 완벽하지는 않지만, Imagen Video는 3D 일관성을 강제하는 방법의 사전 모델로서 효과적일 수 있음을 시사합니다.

  

![fig10](/posts/20240910_ImagenVideo/fig10.png){: width="800" height="300"}

**Fig. 10**에서는 다양한 애니메이션 스타일로 텍스트를 신뢰성 있게 생성할 수 있음을 보여줍니다. 이러한 결과들은 Imagen Video와 같은 **범용 생성 모델**이 **고품질 콘텐츠 생성**의 난이도를 크게 낮출 수 있음을 시사합니다.

  
  

### 3.2. Scaling

![fig11](/posts/20240910_ImagenVideo/fig11.png){: width="800" height="300"}

  

**Fig. 11**에서는 **영상 U-Net**의 파라미터 수를 확장했을 때 모델의 성능이 크게 향상된다는 것을 보여줍니다. 우리는 네트워크의 기본 채널 수와 깊이를 증가시켜 이 확장을 수행했습니다. 기존 text-image 모델과는 반대되는 결과지만, 저자들은 text-video가 더 어려운 task이기때문에, 현재 모델 크기에서 saturation되지 않았을 것이라 합니다.

  

### 3.3. Comparing Prediction Parameterizations

![fig12](/posts/20240910_ImagenVideo/fig11.png){: width="800" height="300"}

  

저자들은 초기 실험에서 **$$  \epsilon  $$-예측 모델**이 **$$  \mathbf v$$-예측**모델보다 성능이 떨어진다는 것을 발견했습니다, 특히 높은 해상도에서. 고해상도 SSR 모델의 경우, $$  \epsilon  $$-예측은 샘플 품질 메트릭에서 상대적으로 느리게 수렴하며, 생성된 비디오에서 색상 이동과 색상 불일치 문제가 발생했습니다.

  

![fig13](/posts/20240910_ImagenVideo/fig13.png){: width="800" height="300"}

  
  

### 3.4. Perceptual Quality and Distillation

![tab1](/posts/20240910_ImagenVideo/tab1.png){: width="800" height="300"}

모델 샘플과 그 증류 버전의 지각 품질 metric (CLIP 점수 및 CLIP R-정확도)비교 입니다.

  

Distillation은 샘플링 시간과 지각 품질 간의 매우 유리한 trade-offf를 제공합니다. distilled cascade약 18배 빠르며, 원래 모델의 샘플과 유사한 품질의 비디오를 생성합니다.

  

FLOPs 측면에서도, distilled 모델은 약 36배 더 효율적이기도 합니다. 원래는 각 모델을 두 번 평가하여 CFG를 적용하지만, distilled 모델은 가이드를 단일 모델로 증류했기 때문에 두 번 평가할 필요가 없다는 장점도 있습니다.

  

![fig14](/posts/20240910_ImagenVideo/fig14.png){: width="800" height="300"}

  

## 4. Limitations

  

**Imagen Video**는 텍스트 기반 이미지 생성 모델의 발전을 이어 받아 텍스트 기반 비디오 생성 기능을 향상시킨 모델입니다. 이 모델은 창의성을 증대시키는 데 긍정적인 영향을 미칠 수 있지만, **허위 정보, 유해한 콘텐츠**를 생성하는 등 악용될 가능성도 있습니다. 이러한 문제를 완화하기 위해 **입력 텍스트 필터링** 및 **출력 비디오 콘텐츠 필터링**을 적용했으나, 여전히 훈련 데이터에 내재된 **편향**을 완전히 해결하지는 못했습니다. **Imagen Video**와 **T5-XXL 텍스트 인코더**는 사회적 편향과 고정관념이 포함된 문제적 데이터로 훈련되었으며, 이를 필터링하는 데 한계가 존재합니다.

  

> 이러한 윤리적 문제로 인해 모델과 소스 코드는 이러한 문제가 해결될 때까지 **공개되지 않기로 결정**되었다고 합니다.

  

## 5. Conclusion

저자들의 주요 기여는 다음과 같습니다.

  

1.  **Imagen Video**, 즉 **텍스트 기반 비디오 생성 시스템**을 제안하여,**Imagen**의 텍스트-이미지 생성 모델을 **시간 축**으로 확장하고, 이미지와 비디오 데이터를 함께 학습하여 **높은 품질의 비디오**를 생성할 수 있습니다.

  

2. 이미지 생성에서 유용했던 **v-prediction**, **conditioning augmentation**, **CFG**등의 기법을 비디오 생성에도 성공적으로 적용했습니다.

  

비디오 모델링은 여전히 많은 계산 자원을 필요로 하지만, **Progressive distillation**을 통해 속도를 크게 개선할 수 있었습니다.

  
  

## Other Postings...

1. [Imagen](https://daemini.github.io/posts/Imagen/)

2. [Classifier-free guidance(CFG)](https://daemini.github.io/posts/Classifier-Free-Diffusion-Guidance/)

3. Progressive Distillation(PD)-> 업로드 예정입니다 :)

  
  

## **Reference**

[JiYeop Kim's blog](https://kimjy99.github.io/)를 참고하여 작성하였습니다.