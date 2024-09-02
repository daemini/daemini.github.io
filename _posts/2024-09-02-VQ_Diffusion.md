---
title: "[Paper Reivew] Vector Quantized Diffusion Model for Text-to-Image Synthesis (VQ-Diffusion)"
description: VQ-VAE의 latent space를 diffusion model을 이용해 모델링하여, 기존 AR 방식 모델들이 갖는 문제점을 해결한 연구입니다.
toc: true
comments: true
# layout: default
math: true
date: 2024-09-02 16:21:00 +09:00
categories: [Deep Learning, Generative Model]
tags: [diffusion model, generative model, vector quantized, cvpr, t2i, vae, microsoft]     # TAG names should always be lowercase
image: /posts/20240828_VQ_Diffusion/Thumbnail.jpeg
alt : Thumbnail
---



>CVPR 2022 (Oral). [[Paper](https://arxiv.org/abs/2111.14822)] [[Github](https://github.com/microsoft/VQ-Diffusion)]  
>Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, Baining Guo  
>University of Science and Technology of China | Microsoft Research  
>29 Nov 2021



# TL;DR
기존 AR 기반 Text-to-Image 모델은 뛰어난 성능을 보이지만, (1) Unidirectional Bias, (2) Accumulated Prediction Errors의 문제점을 갖고 있다. 이에 저자들은 **V**ector **Q**uantized **Diffusion**(VQ-Diffusion) model을 제안한다.

VQ-Diffusion 모델은 VQ-VAE를 기반으로, latent space를 conditional variant DDPM으로 모델링한다. **Latent-space** 방법론은 (1) Unidirectional Bias을 제거할 뿐만 아니라, (2) Accumulated Prediction Errors를 제거하기 위한 **mask-and-replace diffusion** 전략을 사용할 수 있게 한다. 

또한 noise-free image를 예측하도록 **reparameterization**을 적용해 기존 AR보다 15배나 빠르면서 더 좋은 품질의 이미지를 생성할 수 있게 되었다.

## 1. Introduction

NLP 분야 Transformer 성공은 Computer Vision 분야의 적용까지 확장되고 있다. **Autoregressive(AR)** 을 기반으로 DALL-E 같은 모델은 Text-to-Image(T2I) 모델에서 놀라운 성능을 보여주고 있긴 하지만 몇 가지 문제점을 갖고 있다.

**1. Unidirectional Bias**
기존 T2I 모델에서 이미지는 토큰을 읽는 고정된 순서로 예측을 진행하는데, 중요한 맥락적 정보가 임의의 위치에서 올 수 있음에도 불구하고 **특정 방향으로만 정보를 얻어** 편향된 결과를 얻을 수 있다는 문제가 있다.

**2. Accumulated Prediction Errors**
학습 단계에서는 _Teacher-forcing_ 방법론으로 ground truth를 제공하지만, Inference stage에서는 **이전 예측을 기반으로 다음 예측**을 만들기 때문에 오차가 축적될 수 있다. 

저자들은 위와 같은 문제점을 해결하기 위해 **VQ-Diffusion Model**을 제안한다. VQ-VAE를 기반으로 하여 이 모델의 parameter는 Conditional DDPM을 이용해 latent space를 학습한다. 저자들은 이런 latent-space model이 T2I 모델에 적합함을 확인하였다고 한다.

### Eliminates the unidirectional bias
VQ-Diffuison 모델은 독립적인 **text encoder**와 **diffusion image decoder**를 사용한다. 
- Inference의 시작에서, 이미지 토큰은 **마스킹되거나 무작위**로 설정된다. 

- Denosing step에서는 input text에 맞게 이미지 토큰의 확률 밀도를 예측하게 된다.
- 각 step에서, diffusion image decoder는 이전 **예측된 모든 토큰의 맥락정보**를 활용해 **새로운 확률 밀도**를 추정하고, 이를 이용해 **현재 step의 토큰**을 예측한다.
이런 **양방향 주의(bidirectional attention)** 은 global context를 제공하고, unidirectional bias를 제거한다.

### Avoids the accumulation of errors
VQ-Diffusion 모델은 **mask-and-replace diffusion** 전략을 사용해 accumulation of errors를 피할 수 있다.

-   Training stage에서 "teacher-forcing"을 사용하지 않고, **마스킹된 토큰**과 **무작위 토큰**을 도입하여 네트워크가 마스킹된 토큰을 예측하고 잘못된 토큰을 수정하도록 한다.

-   추론 단계에서는 모든 토큰의 밀도 분포를 업데이트하고 **새로운 분포에** 따라 모든 토큰을 **재샘플링**하여 오류를 수정하고 오류 누적을 방지한다.

-   전통적인 replace-only diffusion 전략과 비교하여, 마스킹된 토큰은 네트워크의 attention을 마스킹된 영역으로 유도하여 토큰 조합 수를 크게 줄일 수 있으며, 네트워크의 수렴 속도는 크게 빨라진다.


### Performance Assessment

VQ-Diffusion 모델은 다양한 데이터셋(CUB-200, Oxford-102, MSCOCO)에서 실험을 통해 AR 모델과 비교해 훨씬 나은 결과를 보였다. 이미지 품질 지표와 시각적 검토에서 더 우수한 성능을 나타내며, GAN 기반 방법들과 비교해 복잡한 장면을 다루고 이미지 품질이 향상되었다고 한다.

DALL-E나 CogView와 같은 대형 모델들과 비교해도 특정 이미지 유형(*e.g.* VQ-Diffusion이 학습 단계에서 본 유형의 이미지)에서는 유사하거나 더 나은 성능을 보였으며, VQ-Diffusion은 FFHQ, ImageNet dataset으로 확인한 결과 unconditional, conditional 둘 다에서 일반적이고 강력한 성능을 보였다.

VQ-Diffusion 모델은 또한 이미지 해상도에 무관하게 각 토큰 예측의 global context를 제공하며, 이를 통해 이미지 생성 속도와 이미지 품질간 효과적인 trade-off를 가능하게 한다. 

이는 간단한 **reparameterization**을 통해 가능한데, diffusion image decoder가 noise-reduced image 대신 **noise-free image**를 예측하도록 하는 것이다. 실험 결과 VQ-Diffusion (with reparameterization)은 기존 AR 방법보다 15배 빠른 추론 속도, 뛰어난 이미지 품질을 보였다. 

## Related Work

### GAN-based Text-to-image generation

**GAN-INT-CLS**은 T2I에서 **conditional GAN**을 처음으로 사용한 연구이다. 이 연구를 기반으로한 여러 방법들은 이미지 생성의 품질을 개선했지만, convnet의 inductive bias로 모델들은 단일 도메인 데이터셋(*e.g.* 새, 꽃)에서는 높은 품질의 이미지를 생성하지만, **복잡한 장면**(*e.g.* MS-COCO dataset)**에서는 잘 동작하지 않는다.**


### Autoregressive Model
**AR 모델**은 density estimation에서 강력한 성능을 보여주었고, 최근에도 이미지 생성에 활용된다. **PixelRNN, Image Transformer, ImageGPT**는 이미지의 확률 밀도를 픽셀 단위로 팩터화(factorize)하여 큰 해상도에서는 너무 많은 계산량 때문에 **저해상도 이미지($$ 64 \times 64 $$)** 에서 동작한다.

### VQ-VAE, VQGAN 및 ImageBART

**VQ-VAE, VQGAN, ImageBART** 는 **인코더를 사용해** 이미지를 저차원의 **discrete latent space으로 압축**하고, 숨겨진 변수의 밀도를 fitting한다. 이 접근 방식은 이미지 생성 성능을 크게 향상시켰다.

### DALL-E, CogView, M6

**DALL-E, CogView, M6**은 **AR기반의 T2I 생성 framework**이다. 이 모델들은 텍스트와 이미지 토큰의 joint distribution을 모델링하며, large scale의 transformer구조나, 많은 양의 text-image pair를 이용하여 이미지 생성 품질을 많이 향상시켰다. 하지만, AR 모델의 한계로 unidirectional bias, accumulated prediction errors의 문제점을 갖는다.

### Diffusion Model

Diffusion 기반 생성 모델은 최근 이미지 생성 및 이미지 super-resolution에서 강력한 성과를 거두고 있다. 하지만 대부분의 연구는 **연속적인 확산 모델에만 초점을** 맞췄으며, 이산적 확산 모델은 저해상도 이미지($$ 32 \times 32 $$) 생성에만 적용되고 있다.

## 3. Background: Learning Discrete Latent Space of Images Via VQ-VAE

### Transformer architecture
**Transformer architecture**는 뛰어난 표현력으로 이미지 생성 분야에서 높은 잠재성을 갖고 있다. 하지만 computational cost는 sequnce lenghth에 quadratic하기 때문에 pixel 단위로 이를 적용시키기는 어렵다. 이에 저자들은 이미지를 image-token으로 나누어 효율적으로 학습시키는 방법을 택했다고 한다.

### VQ-VAE


 VQ-VAE는 Encoder $$ E $$, Decoder $$ D $$, 유한개의 embedding vector를 포함하는 codebook $$ \mathcal{Z} = \{z_k\}_{k=1}^K \in \mathbb{R}^{K \times d} $$로 구성되어 있다. 이때 $$ K $$는 codebook의 크기, $$ d $$는 code의 차원을 의미한다. 주어진 이미지 $$ x \in \mathbb{R}^{H \times W \times 3} $$ 에 대해, 인코더는 이미지 토큰 $$ z = E(x) \in \mathbb{R}^{h \times w \times d} $$를 spatial-wise quantizer $$ Q(\cdot) $$에 넣어 codebook entry $$ z_k $$와 가장 가까운 $$ z_{ij}$$에 mapping한다.

$$
\begin{equation}
z_q = Q (z) = \bigg( \underset{z_k \in \mathcal{Z} }{\arg \min} \| z_{ij} - z_k \|_2^2 \in \mathbb{R}^{h \times w \times d} \bigg)
\end{equation}
$$


여기서 $$ h \times w $$는 인코딩된 시퀀스 길이를 나타내며, 이는 일반적으로 원본 이미지 $$ H \times W $$보다 훨씬 작다. 그 후 디코더 $$ G $$ 를 통해 이미지를 재구성할 수 있다.

$$
\tilde{x} = G(z_q)
$$

따라서 이미지 합성은 latent distribution으로부터 이미지 토큰을 샘플링 하는 것으로 이해할 수 있고, 이 이미지 토큰들은 이산적인 값을 갖는 quantized latent variable이라는 것이다. 다음 loss function을 이용해 end-to-end Training이 가능하다.

$$
\begin{equation}
\mathcal{L}_{\textrm{VQVAE}} = \|x - \tilde{x} \|_1 + \| \textrm{sg}[E(x)] - z_q \|_2^2 + \beta \| \textrm{sg}[z_q] - E(x) \|_2^2
\end{equation}
$$

> 실용적인 측면에서, 두번째 항을 EMA로 바꾸어 codebook 전체를 update하는 방식을 사용해 성능을 높인다.


## 4. Vector Quantized Diffusion Model

주어진 text-image 쌍에 대해, pre-trained VQ-VAE를 통해 discrete image token $$ x \in \mathbb{Z}^N $$를 얻는다. 이때 $$ N =hw $$이다. VQ-VAE codebook size가 $$ K $$라 하면, $$ i $$번째 위치의 이미지 토큰은 $$ x^i $$ codebook의 특정 인덱스를 의미한다.  $$ x_i \in \{1, 2, \cdots, K\} $$

반면 Text 토큰 $$ y \in \mathbb{Z}^M $$은 BEP-encdoing을 통해 얻을 수 있다. 전반적인 T2I framework는 text 토큰이 주어졌을 때, 이미지 토큰의 conditional transition distribution $$ q(x \vert y) $$을 최대화하는 것으로 볼 수 있다.

**DALL-E**나 **CogView** 같은 기존의 AR 모델들은, 텍스트 토큰과 이전에 예측된 이미지 토큰을 기반으로 각 이미지 토큰을 **순차적으로 예측**한다. 이는 주어진 텍스트에 따라 이미지의 각 부분을 예측하는 방식으로, 높은 품질의 T2I 생성 결과를 달성했다.
$$
\begin{equation}
q(x|y) = \prod_{i=1}^N q(x^i | x^1, \cdots, x^{i-1}, y)
\end{equation}
$$
하지만 이런 AR 기반 모델은 다음과 같은 문제점을 갖는다.

1.  **unidirectional ordering**: 기존 모델들은 일반적으로 고정된 순서(_e.g._ 좌상단에서 우하단으로)로 이미지를 예측하며, 이는 2D 데이터의 구조를 무시하고 이미지 모델링의 표현력을 제한 할 수 있다.

2. **Error accumulation**: 훈련 중에는 정답이 제공("teacher forcing")되지만, 추론 단계에서는 이전에 예측된 토큰에 의존해야 하므로 오차가 누적되는 문제가 발생한다.

3.   **비효율성**: 각 토큰을 예측하기 위해 네트워크의 순방향 패스를 사용해야 한다. 저해상도 이미지에서도 샘플링에 너무 많은 시간이 필요하다.

이런 문제를 해결하기 위해 저자들은 non-AR 방식으로 VQ-VAE의 latent space를 모델링하는 방식이다. 이번 연구에서 저자들은 Diffusion model과 같이 $$ q(x \vert y) $$을 최대화하는 **VQ-Diffusion 방법론**과 T2I 생성을 위한 **conditional variant discrete diffusion process**를 제안한다.

### 4.1. Discrete diffusion process

구체적으로 살펴보면, $$ x_{t-1} $$을 $$ x_t $$로 전이할 확률 행렬($$ [Q_t]_{mn} = q(x_t = m \vert x_{t−1} = n) \in \mathbb{R}^{K \times K} $$)을 정의할 수 있다.

forward Markov diffusion process는 다음과 같이 표현할 수 있다.

$$
\begin{equation}
q(x_t | x_{t-1}) = v^\top (x_t) Q_t v(x_{t-1})
\end{equation}
$$

$$ x_t $$에 대한 categorical 분포는 $$ Q_t v(x_{t−1}) $$로 주어진다. Markov chain의 특성으로, $$ x_0 $$로부터 중간 과정 없이 임의의 time step $$ x_t $$를 유도할 수 있다. 

$$
\begin{equation}
q_t(x_t | x_0) = v^\top (x_t) \overline{Q}_t v(x_0), \quad \overline{Q}_t = Q_t \cdots Q_1
\end{equation}
$$

또 다른 주목할만한 특징은, $$ z_0 $$로 conditioning함으로써, posterior가 tractable해진다는 것이다. 

$$
\begin{equation}
q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}, x_0) q(x_{t-1} | x_0)}{q(x_t | x_0)} = \frac{(v^\top (x_t) Q_t v(x_{t-1})) (v^\top (x_{t-1}) \overline{Q}_{t-1} v(x_0))}{v^\top (x_t) \overline{Q}_t v(x_0)}
\end{equation}
$$
따라서 Transition matrix $$ Q $$가 discrete diffusion model에서 매우 중요하며, reverse process에서 잘 동작하기 위해 세심한 디자인이 필요하다.

이전 연구에서는 작은 크기의 일정한 노이즈를 가하는 방법이 제안되었다.  

$$
\begin{equation}
Q_t = \begin{bmatrix}
    \alpha_t + \beta_t & \beta_t & \cdots & \beta_t \\
    \beta_t & \alpha_t + \beta_t & \cdots & \beta_t \\
    \vdots & \vdots & \ddots & \vdots \\
    \beta_t & \beta_t & \cdots & \alpha_t + \beta_t \end{bmatrix}
\end{equation}
$$

이때, $$ \alpha_t \in [0,1], \beta_t = (1-\alpha_t) / K$$이다. 각 토큰은 $$ (\alpha_t + \beta_t) $$확률로 이전 값을 유지하고, $$ K\beta_t $$의 확률로 $$ K $$ 카테고리 중 하나로 resample 된다.

하지만 unifrom diffusion 방식으로 데이터를 손상시키는 것은 reverse estimation에 문제가 될 수 있는 다소 공격적인 process이다. 

- unifrom diffusion은 이미지 토큰을 무작위로 다른 카테고리로 대체하는데, 이 과정에서 토큰간 의미적 충돌이 발생할 수 있다.
- 그 결과 네트워크는 이런 대체된 토큰을 감지하고 수정하기 위해 추가적인 작업을 수행해야 하는데, reverse estimation에서 딜레마에 빠지게 된다.

#### Mask-and-replace diffusion strategy
위와 같은 문제를 해결하기 위해 저자들은 mask language modeling에서 영감을 받아 확률적으로 일부 토큰을 마스킹함으로써, reverse network가 명시적으로 알아차릴 수 있는 전략을 취했다. 구체적으로 [$$ MASK $$]라는 **special 토큰**을 도입하여, 각 토큰이 $$ (K+1) $$개의 상태를 갖는다. 

1. 각 일반 토큰은 $$ \gamma_t $$의 확률 -> [$$ MASK $$] 토큰으로 대체
2. $$ K\beta_t $$의 확률 -> uniformly diffusion
3.  o.w. $$ \alpha_t = 1 - K \beta_t - \gamma_t $$ 확률 -> 유지.
4. [$$ MASK $$] 토큰은 항상 자체 상태를 유지한다.

따라서 transition matrix $$ Q_t \in \mathbb{R}^{(K+1) \times (K+1)} $$는 다음과 같이 표현할 수 있다.

$$
\begin{equation}
Q_t = \begin{bmatrix}
    \alpha_t + \beta_t & \beta_t & \beta_t & \cdots & 0 \\
    \beta_t & \alpha_t + \beta_t & \beta_t & \cdots & 0 \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    \gamma_t & \gamma_t & \gamma_t & \cdots & 1 \end{bmatrix}
\end{equation}
$$

이 mask-and-replace transition은 다음과 같은 장점을 갖는다.

1. corrupted token은 network가 구분 가능하다. 이는 reverse process를 쉽게한다.
2. mask-only 방법론과 비교했을 때, 이론적으로 token masking외에도 적은 noise를 포함하는 것이 필요하다는 것을 증명할 수 있다.
3. random token replacement는 $$ [MASK] $$만 보는 것이 아니라, network가 context를 이해하도록 강제한다.
4. cumulative transition matrix $$ \overline{Q}_t $$는 다음과 같은 colsed form 으로 쉽게 계산할 수 있다.

$$ 
\begin{equation}
\overline{Q}_t v(x_0) = \overline{\alpha}_t v(x_0) + (\overline{\gamma}_t - \overline{\beta}_t) v(K+1) + \overline{\beta}_t \\
(\overline{\alpha}_t = \prod_{i=1}^t \alpha_i, \overline{\gamma}_t = 1- \prod_{i=1}^t (1 - \gamma_i), \overline{\beta}_t = \frac{1 - \overline{\alpha}_t - \overline{\gamma}_t}{K})
\end{equation}
$$

여기서 $$ \overline{\gamma}_t $$는 미리 계산해서 저장할 수 있으므로, $$ q (x_t \vert x_0) $$의 계산 비용이 $$ O(tK^2) $$에서 $$ O(K) $$로 줄어든다.

### 4.2. Learning the reverse process
Reverse process를 위해 저자들은 denoising network $$ p_\theta (x_{t−1} \vert x_t, y) $$를 학습시켜 $$ q(x_{t−1} \vert x_t, x_0) $$를 예측하도록 했다. network는 다음과 같은 VLB를 이용해 학습한다.

$$
\begin{aligned}
\mathcal{L}_{\textrm{vlb}} &= \mathcal{L}_{0} + \mathcal{L}_{1} + \cdots + \mathcal{L}_{T-1} + \mathcal{L}_{T} \\
\mathcal{L}_{0} &= -\log p_\theta (x_0 | x_1, y) \\
\mathcal{L}_{t-1} &= D_{KL} (q(x_{t-1} | x_t, x_0) \; \| \; p_\theta (x_{t-1} | x_t, y)) \\
\mathcal{L}_{T} &= D_{KL} (q(x_T | x_0) \; \| \; p(x_T))
\end{aligned}
$$

$$ p(x_T) $$는 timestep $$ T $$에서 prior dist. 이며, 저자들이 제안한 mask-and-replace 기법에서는 다음과 같다.

$$ 
\begin{equation}
p(x_T) = [ \overline{\beta}_T, \overline{\beta}_T, \cdots, \overline{\beta}_T, \overline{\gamma}_T ]^\top
\end{equation}
$$

#### **Reparameterization trick on discrete stage.**

네트워크의 **parameterization**은 이미지 합성 결과에 큰 영향을 미친다. 최신 연구에 따르면 직접 $$ q(x_{t-1} \vert x_t, x_0) $$를 예측하는 것보다,  **노이즈가 없는 target data** $$ q(x_0) $$를 예측하는 것이 더 나은 품질을 제공한다고 한다.

따라서 저자들은 discrete setting에서 네트워크가 $$ p_\theta (\tilde{x}_0 \vert x_t, y) $$를 예측하도록 한다. 

$$
\begin{equation}
p_\theta (x_{t-1} | x_t, y) = \sum_{\tilde{x}_0 = 1}^K q(x_{t-1} | x_t, \tilde{x}_0) p_\theta (\tilde{x}_0 | x_t, y)
\end{equation}
$$

reparameterization trick과 함께, noiseless token $$ x_0 $$를 예측하도록 다음과 같은 **auxiliary denoising objective**를 제안한다.

$$
\begin{equation}
\mathcal{L}_{x_0} = -\log p_\theta (x_0 | x_t, y)
\end{equation}
$$ 

이를 $$ \mathcal{L}_{\textrm{vlb}} $$와 함께 사용해 이미지 생성 품질을 올릴  수 있었다고 한다.

#### **Model architecture.**

![fig1](/posts/20240828_VQ_Diffusion/fig1.png){: width="600" height="300"}

저자들의 framework는 크게 두가지로 되어 있다.
1. Text encoder
**Text encoder**는 text token $$ y $$를 입력으로 받아 conditional feature sequence를 생성한다.

2. Diffusion image decoder 
**Diffusion image decoder** 는 image token $$ x_t $$, Time step $$ t $$를 입력으로 받아 noiseless token dist. $$ p_\theta (\tilde{x}_0 \vert x_t, y) $$를 생성한다. Decoder는 여러 transformer, softmax layer로 구성되어 있으며, 각 transformer block은 full-attention, cross attention, feed forward 네트워크 블록으로 구성되어 있다. timestep $$ t $$는 Adaptive Layer Normalization(AdaLN)으로 네트워크에 제공된다.

$$
\begin{equation}
\textrm{AdaLN}(h, t) = a_t \textrm{LayerNorm}(h) + b_t
\end{equation}
$$

$$ h $$는 중간 activation이며, $$ a_t, b_t$$는 timestep embedding을 linear projection하여 얻는다. 

#### **Fast inference strategy**
Reparameterization trick을 이용해 inference stage에서 몇 step을 스킵하여 더욱 빠른 이미지 생성이 가능하다고 한다. **Time stride**를 $$ \Delta_t $$라 한다면,  $$ x_T, x_{T-1}, x_{T-2}, \cdots, x_0 $$ 로 생성하는 대신, $$ x_T, x_{T-\Delta_t}, x_{T-2\Delta_t}, \cdots, x_0 $$과 같이 이미지를 생성할 수 있다.

$$
\begin{equation}
p_\theta (x_{t-\Delta_t} | x_t, y) = \sum_{\tilde{x}_0 = 1}^K q(x_{t-\Delta_t} | x_t, \tilde{x}_0) p_\theta (\tilde{x}_0 | x_t, y)
\end{equation}
$$

작은 생성 품질을 야기하지만, 샘플링 과정을 훨씬 효율적으로 만들 수 있다고 한다.


<center>
  <img src='{{"/posts/20240828_VQ_Diffusion/al1.png" | relative_url}}' width="48%">
  &nbsp;
  <img src='{{"/posts/20240828_VQ_Diffusion/al2.png" | relative_url}}' width="48%">
</center>


## 5. Experiments
이 섹션에서는 다양한 실험 결과를 통해 저자들이 제안한 Text-to-image 합성 방법론의 우수성을 입증한다. VQ-Diffusion은 unconditional, conditional 이미지 합성에서 뛰어난 성능을 보였다고 한다.

#### **Datasets.**
-   **CUB-200**: 200종의 새를 포함한 8855개의 훈련 이미지와 2933개의 테스트 이미지를 포함한다. 각 이미지에는 10개의 텍스트 설명이 있다.
-   **Oxford-102**: 102개의 꽃 카테고리에 속하는 8189개의 이미지를 포함하며, 각 이미지에도 10개의 텍스트 설명이 있다.
-   **MSCOCO**: 훈련용 82,000개의 이미지와 테스트용 40,000개의 이미지를 포함한다. 각 이미지에는 5개의 텍스트 설명이 있다.

이외에도 저자들은 대규모 데이터셋에서 모델의 확장성을 입증하기 위해 **Conceptual Captions** (CC3M 및 CC12M 포함)과 **LAION-400M** 데이터셋에서도 실험을 진행했다. **Conceptual Captions** 데이터셋은 1,500만 개의 이미지를 포함하고, 텍스트 및 이미지 분포를 균형 있게 맞추기 위해 700만 개의 서브셋으로 필터링했다고 한다. **LAION-400M** 데이터셋은 4억 개의 이미지-텍스트 쌍을 포함하며, 여기서 3개의 서브셋(만화, 아이콘, 인간)을 필터링하여 사용했다.

#### **Traning Details.**

-   **VQ-VAE**: 인코더와 디코더는 **VQGAN** 설정을 따르며, GAN 손실을 활용해 더 현실적인 이미지를 생성한다. T2I 이미지 생성 실험을 위해 Open-Images 데이터셋에서 훈련된 공개 VQGAN 모델을 사용하며, 256×256 이미지를 32×32 토큰으로 변환하고, 코드북 크기 K=2886K = 2886K=2886으로, 불필요한 코드를 제거한 후 사용했다고 한다.
-   **텍스트 인코더**: CLIP 모델의 공개된 토크나이저를 텍스트 인코더로 채택하여 길이가 77인 조건부 시퀀스를 생성한다. 훈련 중에는 이미지 및 텍스트 인코더를 고정한다.

다른 모델과 공정한 비교를 위해 저자들의 VQ-Diffusion은 두 가지 다른 setting을 사용한다고 한다.

-   **VQ-Diffusion-S (Small)**: 18개의 트랜스포머 블록과 192 차원의 모델로 구성되며, 3400만 개(*34M*)의 파라미터로 구성되어 있다.
-   **VQ-Diffusion-B (Base)**: 19개의 트랜스포머 블록과 1024 차원의 모델로 구성되며, 3억 7000만 개(*370M*)의 파라미터로 구성되어 있다.

이후 Conceptual caption database에서 base model을 학습한 뒤, 각 database에서 finetuning을 진행했다고 한다. 이 모델은 **VQ-Diffusion-F** 라고 한다.

-   **시간 단계**:  $$ T = 100 $$ 
-   **손실 가중치**: $$ \lambda = 0.0005 $$ 
-   **전이 행렬**: $$ \gamma_t $$ 와 $$ \beta_t $$​를 각각 0에서 0.9, 0.1로 선형적으로 증가시킵니다.
-   **최적화**: AdamW 옵티마이저를 사용하며, $$ \beta_1 = 0.9, \beta_2 = 0.96 $$으로 설정합니다. 학습률은 5000 iteration warm-up 후 0.00045로 setting한다.


### 5.1. Comparison with state-of-the-art methods
저자들이 제안한 VQ-Diffusion 모델을 GAN 기반 방법들, DALL-E, CogView와 비교는 MSCOCO, CUB-200, Oxford-102 데이터셋에서 비교한 결과를 나타낸 것이다. 


![tab1](/posts/20240828_VQ_Diffusion/tab1.png){: width="500" height="300"}
_FID comparison of different text-to-image synthesis method on MSCOCO, CUB-200, and Oxford-102 datasets_

-   **VQ-Diffusion-S** 모델(소형 모델)은 이전의 GAN 기반 모델들과 유사한 파라미터 수를 가지면서도 CUB-200과 Oxford-102 데이터셋에서 강력한 성능을 보였다.
-   **VQ-Diffusion-B** 모델(기본 모델)은 더 나은 성능을 보여주었으며, **VQ-Diffusion-F** 모델은 모든 이전 방법을 크게 능가하는 SOTA를 달성했다.
-   특히 MSCOCO 데이터셋에서는 DALL-E와 CogView를 초과하는 성능을 보였으며, 이는 두 모델이 VQ-Diffusion보다 10배 더 많은 파라미터를 가지고 있음에도 불구하고 성능이 더 우수했다.

![fig2](/posts/20240828_VQ_Diffusion/fig2.png){: width="800" height="400"}

### 5.2. In the wild text-to-image synthesis

저자들은 모델의 실제 이미지 생성 능력을 입증하기 위해, LAION-400M 데이터셋의 세 가지 서브셋(만화, 아이콘, 인간)을 사용해 모델을 학습했다. VQ-Diffusion은 DALL-E 및 CogView와 같은 기존의 대규모 모델보다 훨씬 작은 모델임에도 불구하고 강력한 성능을 보여주었다.

![fig3](/posts/20240828_VQ_Diffusion/fig3.png){: width="500" height="300"}

AR 방식(from top-left to down-right)과 비교해, 저자들의 모델은 이미지를 global manner로 생성하기 때문에, mask inpainting과 같은 여러 vision task에 적용할 수 있다고 한다. 이때 모델을 다시 학습시키지 않고, 단순히 불규칙한 영역을 $$ [MASK] $$ Token으로 대체하면 된다고 한다. 

![fig5](/posts/20240828_VQ_Diffusion/fig5.png){: width="600" height="200"}

### 5.3. Ablations
#### Number of timesteps.
저자들은 CUB-200 데이터셋에서 training, inference 단계에서 timestep을 바꿔가며 실험을 진행한 결과 10~100 까지는 성능이 증가했지만, 200으로 증가할 때는 성능이 saturation 되는 것을 확인했다고 한다. Inference stage의 3/4를 줄여도 여전히 성능이 좋았다고 한다.

#### Mask-and-replace diffusion strategy.
Oxford-102 데이터셋에서 **mask-and-replace** 전략의 효과를 실험하였다. 저자들은 다른 mask rate $$ \overline{\gamma}_T $$(0일 때, replace only, 1일 때, mask only)를 바꿔가며 FID score를 측정한 결과, mask rate $$ M = 0.9 $$ 에서 가장 성능이 좋았다고 한다. $$  M > 0.9 $$일 때는 error accumulation, $$  M < 0.9 $$일 때는 모델이 어느 부분에 집중해야 되는지 어려워했다고 한다.

![fig4](/posts/20240828_VQ_Diffusion/fig4.png){: width="700" height="300"}

#### Truncation.
**Truncation sampling** 전략은 discrete diffusion base 방법에서 매우 중요한 역할을 하는데, 이 전략은 네트워크가 낮은 확률 토큰을 샘플링하는 것을 방지할 수 있다고 한다. 추론 단계에서 $$ p_\theta(\tilde{x}_0 \vert x_t, y) $$의 상위 $$ r $$ 토큰만 유지하도록 설정했으며, CUB-200 데이터셋에서 다양한 truncation rate $$ r $$을 평가한 결과 $$ r = 0.86$$ 에서 최고의 성능을 얻을 수 있었다고 한다.

#### VQ-Diffusion vs VQ-AR.
VQ-Diffusion의 이미지 decoder를 AR decoder로 대체한 후, 동일한 네트워크 구조와 설정을 유지하여 비교 실험을 진행하였다. 이 모델은 VQ-AR-S 및 VQ-AR-B로 명명되며, 각각 VQ-Diffusion-S 및 VQ-Diffusion-B에 대응된다. 실험은 CUB-200 dataet에서 진행되었다.

-  VQ-Diffusion 모델이 두 가지 설정(-S 및 -B) 모두에서 VQ-AR 모델을 큰 차이로 능가했다.
-  V100 GPU에서 배치 크기 32로 두 방법의 처리량을 평가한 결과, fast inference 전략을 적용한 VQ-Diffusion은 VQ-AR 모델보다 15배 빠르면서도 더 나은 FID 점수를 기록했습니다.

![tab3](/posts/20240828_VQ_Diffusion/tab3.png){: width="500" height="300"}

### 5.4. Unified generation model
저자들이 제안한 방법은 T2I 이미지 생성뿐만 아니라 다른 이미지 합성 작업에도 적용할 수 있는 General한 방법론이라고 한다. 예를 들어, unconditional, conditional 이미지 생성에서도 사용 가능하다. 

![tab4](/posts/20240828_VQ_Diffusion/tab4.png){: width="500" height="300"}

일부 작업에 특화된 GAN 모델들이 더 나은 FID 점수를 보고하기도 했지만, 저자들이 제안한 접근법은 다양한 작업에서 뛰어난 성능을 발휘하는 통합된 모델을 제공한다는 점에서 더 좋다고 한다.

## 6. Conclusion
저자들은 본 논문에서 **VQ-Diffusion**이라는 새로운 T2I 구조를 제안한다. VQ-VAE의 latent space를 non-AR 방식으로 모델링하는 것이다. 저자들이 제안한 **mask-and-replace** 전략은 기존 AR 기반 모델이 갖는 한계점을 극복하도록 설계되었으며 GAN 기반 T2I 모델의 결과를 뛰어넘는 성능을 보였다. 또한 저자들의 모델은 **범용적**으로 사용할 수 있으며, conditional, unconditional 모두에서 좋은 성능을 발휘한다고 한다.


## **Reference**
[JiYeop Kim's blog](https://kimjy99.github.io/)를 참고하여 작성하였습니다.


