---
title: "[Paper Reivew] Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model (Transfusion)"
description: 텍스트와 이미지 생성을 위한 단일 모델로, 두 가지 다른 최적화 목표를 결합해 효율적인 mult-modality 생성을 구현한 방법입니다.
toc: true
comments: true
# layout: default
math: true
date: 2024-09-05 14:45:00 +09:00
categories: [Deep Learning, Generative Model]
tags: [diffusion model, generative model, vector quantized, meta, multimodal, t2i]     # TAG names should always be lowercase
image: /posts/20240904_Transfusion/thumbnail.webp
alt : Thumbnail
---


> arXiv 2024. [[Paper](https://www.arxiv.org/abs/2408.11039)] [[Github](https://github.com/lucidrains/transfusion-pytorch)]  
> Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, Omer Levy  
> Meta | Waymo | University of Southern California  
> 20 Aug 2024


나온지 2주 정도된 따끈따끈한 논문입니다.
굉장히 놀라운 결과네요...

이 논문이 Baseline으로 비교하는 Chameleon 포스팅이 궁금하다면 -> [Chameleon, 이전 포스팅](https://daemini.github.io/posts/Chameleon/)

# TL;DR

**Transfusion**은 텍스트와 이미지 생성을 위한 단일 모델로, 두 가지 다른 최적화 목표를 결합해 효율적인 **mult-modality** 생성을 구현한 방법입니다.

![fig1](/posts/20240904_Transfusion/fig1.png){: width="800" height="300"}

## 1. Introduction
**Multi-modal** 생성 모델은 discrete, continuous 두 가지 종류의 정보를 인식(perceive), 처리(process), 생성(produce)할 수 있어야 합니다. **Language Model**은 next token prediction을 통해 discrete modality에 지배적인 반면, **Diffusion**은 continuous modality에서 지배적입니다. Language 모델이 Diffusion 모델을 사용하거나, pretrained diffusion 모델을 language 모델에 결합하는 방식으로 두 가지 접근법을 통합하려는 많은 연구가 있었습니다. 혹은, 다른 접근법으로 continuous modality를 quantization하여 discrete하게 훈련하는 방법도 있지만, 정보 손실을 야기한다는 문제점이 있다고 합니다. 


저자들은 이번 연구에서 정보 손실 없이 두 가지 modality, discrete text token과 diffuse continuous image를 모두 예측할 수 있는 **하나의 모델, Transfusion**을 제안합니다.  **Transfusion**은 텍스트와 이미지를 각각 50%씩 사용하여 transformer를 pretrain합니다. 이때 텍스트는 다음 토큰을 예측, 이미지는 diffusion을 목표로 학습합니다. 

1. **Text token** : Standard embedding layer를 통해 벡터로 변환. causal attention 적용
2. **Image** :  patchification layer를 통해 patch 벡터의 시퀀스로 변환. bidirectional attention 적용

Inference에서는 decoding 알고리즘을 적용해, language model로 부터 text generation을, diffusion model로부터 Image generation을 수행합니다.

Transfusion은 **Chameleon**의 discretization 접근법과 비교 실험에서 모든 modality 조합에서 더 나은 scale을 보였습니다.

-   **text-to-image generation**: Transfusion은 Chameleon보다 3분의 1 이하의 계산량으로도 더 우수한 FID와 CLIP 점수를 기록했습니다.
-   **image-to-text generation**: Transfusion은 FLOPs(계산 복잡도)가 21.8%에 불과할 때에도 Chameleon과 유사한 성능을 보였습니다.
-   **text-to-text prediction**: Transfusion은 Chameleon보다 50-60%의 FLOPs로 유사한 perplexity를 달성했습니다.

Ablation 실험을 통해 intra-image bidirectional attention이 중요하며, U-Net up/down block을 통해 encoding/decoding 함으로써, 더 큰 이미지 patch를 압축할 수 있었다고 합니다.

Transfusion은 이미지 생성에서 DALL-E 2, SDXL과 같은 모델보다 뛰어난 성능을 보였으며, 텍스트 생성에서도 Llama 1과 유사한 수준에 도달했다고 합니다.

## 2. Background
Transfusion은 하나의 모델을 두 개의 objective를 가지고 학습합니다. 각각의 Objective는 discrete, continuous data modeling의 SOTA 방식을 따랐으며 이에 대해 간단히 설명합니다.

### 2.1. Language Modeling
Discrete token의 시퀀스 $$ y = y_1, ..., y_n $$가 주어졌을 때, Language 모델은 $$ P(y) $$를 예측합니다. 일반적인 Language 모델은 $$ P(y) $$를 conditional probability $$ \prod_{i=1}^n P_\theta(y_i|y_{<i}) $$로 decompose합니다.

각각의 토큰 $$ y_i $$는 이전 시퀀스 $$ y_{<i} $$에 의해 결정되는 autoregressive classification task로 $$ \theta $$로 모델링되는 $$ P_\theta $$에 의해 예측됩니다.

모델은 주어진 데이터의 empirical data distribution과 $$ P_\theta $$간의 cross-entropy를 최소화 하는 방향으로 최적화 되며, 이를 $$ \textit{LM loss}$$라 합니다.

$$
\begin{equation}
\mathcal{L}_{\text{LM}} = \mathbb{E}_{y_i} \big[ - \log P_\theta (y_i | y_{<i}) \big]
\end{equation}
$$

모델이 학습되면, temperature, top-p truncation을 통해, model distribution $$ P_\theta $$로부터 토큰을 하나씩 샘플링하여, 텍스트를 생성할 수 있습니다.


### 2.2. Diffusion
> [이전 포스트](https://daemini.github.io/posts/Diffusion-Basic/) 혹은 [What are diffusion models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)를 참고하면 더욱 좋습니다. (아마도...)

**DDPM** 혹은 **diffusion model**이라 불리는 Denoising diffusion probabilistic models은 점진적인 노이즈 추가 과정을 통해 이미지를 noisy하게 바꾸고, 이를 역으로 제거하는 방법을 학습하는 이미지 생성 모델입니다. Language model과는 다르게 연속적인 벡터를 다루기 때문에, 이미지 같은 연속적인 데이터를 생성하는데 적합합니다. Diffusion model은 크게 두 가지 process로 구성됩니다.

#### **Forward process** 
수학적인 측면에서 forward process는 데이터에 노이즈를 얼마나 더할지 정의하는 과정입니다. 주어진 이미지 $$ \mathbf{x}_0 $$에 Markov chain을 통해 Gaussian noise를 $$ T $$ step동안 점차적으로 더하게 됩니다. 각 step은 다음과 같이 정의할 수 있습니다. $$ q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I}) $$

하지만 수학적으로 열심히 계산해서(?) $$ \mathbf{x}_0 $$에서 임의의 time step $$ t $$를 바로 계산할 수 있습니다. 

$$
\begin{equation}
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}
\end{equation}
$$

이때 $$ \bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s) $$이며, $$ \beta_t $$는 미리 정해진 noise schedule에 따라 점차적으로 증가하는 값입니다.

#### **Reverse Process**
Diffusion model은 reverse process $$ p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) $$를 수행하도록 학습됩니다. 노이즈가 제거된 이미지를 직접 예측할수도 있지만, 저자들은 Ho et al. [2020]의 접근법을 따라 noise $$ \boldsymbol{\epsilon} $$를 예측하는 방법을 택했다고 합니다. 또한, 이미지 외에도 추가적인 contextual information $$ c $$가 포함된 경우 다음의 MSE를 최소화하는 방향으로 최적화 됩니다.

$$
\begin{equation}
\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{\mathbf{x}_0, t, \boldsymbol{\epsilon}}\big[||\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{\theta}(\mathbf{x}_t, t, c)||^2\big]
\end{equation}
$$

#### **Noise Schedule**
Forward process를 따라 noisy한 이미지 $$ \mathbf{x}_t $$를 생성할 때, $$ \bar{\alpha}_t $$가 Gaussian noise의 variance를 결정하게 됩니다. 저자들의 연구에서는 주로 사용되는 Nichol and Dhariwal [2021, IDDPM]의 방법을 조금 변형하여 사용했다고 합니다. $$ \sqrt{\bar{\alpha}_t} \approx \cos(\frac{t}{T}\cdot\frac{\pi}{2}) $$


#### **Inference**
Pure Gaussian noise $$ \mathbf{x}_T $$로부터, 모델 $$ \boldsymbol{\epsilon}_{\theta}(\mathbf{x}_t, t, c) $$은 각 time step에서의 더해진 noise를 예측하게 됩니다. 이 노이즈는 위에서 정한 noise scheduling에 따라 scaling되어 $$ \mathbf{x}_t $$에서 제거되며, $$ \mathbf{x}_{t-1} $$을 생성합니다. 

CFG([이전포스팅](https://daemini.github.io/posts/Classifier-Free-Diffusion-Guidance/)) 방식은 두 배의 계산량이 필요하긴 하지만 추가적인 학습 없이 conditioned context $$ c $$와 unconditional prediction을 대조하여 이미지 생성 품질을 높이기 위해 주로 사용됩니다.

### 2.3. Latent Image Representation
초기 Diffusion model은 Pixel space에서 denoising과정을 학습했지만, computational cost가 매우 크다는 문제점이 있었습니다. 이에 VAE를 활용하여 이미지를 더 낮은 latent space로 encoding하여 계산량을 줄일 수 있었습니다. (VAE는 depp CNN으로 구성되어 있으며, downstream에서 활용될 수 있습니다.)

이런 pretrained VAE를 활용해 **[LDM, 이전포스팅](https://daemini.github.io/posts/Latent-Diffsuion-Model/)** 8×8 픽셀 패치를 8차원 벡터로 표현하는 등 이미지 patch embedding을  효과적으로 처리할 수 있습니다.

이와 같은 접근법은 **Autoregressive(AR) language model**에서도 활용되며, 이미지가 discrete한 형태로 변환되어야 합니다. Discrete autoencoder 일종인 **[VQ-VAE, 이전포스팅](https://daemini.github.io/posts/VQ_Diffusion/)** 는 quantization layer와 관련된 정규화 손실을 도입하여 continuous latent embeddings을 discrete tokens으로 매핑함으로써 이를 실현합니다.


## 3. Transfusion
Transfusion은 하나의 모델로 discrete, continuous modality를 이해하고, 생성하는 방법론입니다. 저자들의 주요 관심사는 같은 데이터, 모델 파라미터에 대해 각 Modality에따라 다른 loss를 사용할 수 있는지 확인하는 것이였다고 합니다,
### Data Representation

-Discrete text : integer로 표현되는 고정된 vocabulary로부터 discrete 토큰의 sequence로 변환 **(integers representing text tokens)**

-Continuous images : VAE를 이용해, latent patch로 변환되며, 각 patch는 continuous vector로 표현. BOI(*Beginning of Image*), EOI(*End of Image*)로 감싸집니다. **(vectors representing image patches)**

최종적으로 discrete element와 continuous elements를 **하나의 sequence**로 표현할 수 있습니다.

### Model Architecture
모델 파라미터의 대부분은 modality에 상관없이 모든 sequence를 처리하는 하나의 transformer가 차지합니다. Transformer는 고차원 벡터 $$ \mathbb{R}^d $$를 입력으로 받아 비슷한 벡터를 출력합니다. 원래 데이터를 vector space로 보내기 위해 lightweight modlity specific component를 사용합니다. (파라미터 공유 x)

- Fot text :  embedding matrix를 이용해 input integer를 vector space로 변환. 
- For image : $$ k \times k $$ 이미지 patch를 (1) Linear layer, (2) Up/Down U-Net block을 이용해 변환하는 실험을 진행했다고 합니다.

![fig3](/posts/20240904_Transfusion/fig3.png){: width="400" height="200"}


### Transfusion Attention

- Fot text : 일반적으로 sequential한 특성이 있으므로, causal attention
- For image : contextual 정보가 여러 곳에 분포 되어 있으므로, bidirectional attention

Transfusion은 두 가지 masking을 각각 적용하면서, 이미지는 모든 patch에 attend 할 수 있으며, 텍스트는 이전 sequence에만 attend할 수 있다.

![fig4](/posts/20240904_Transfusion/fig4.png){: width="400" height="200"}


### Training Objective
- Fot text : 토큰 단위로 $$ \mathcal{L}_\text{LM} $$
- For image : 이미지 patch 단위로 $$ \mathcal{L}_\text{DDPM} $$

따라서 Transfusion의 training objective는 다음과 같다.

$$
\begin{equation}
\mathcal{L}_\text{Transfusion} = \mathcal{L}_\text{LM} + \lambda  \cdot \mathcal{L}_\text{DDPM}
\end{equation}
$$

저자들은 위 loss가 discrete distribution loss와 continuous distribution loss를 합쳐 같은 모델을 최적화하는 방법론이 될 것이라 합니다.

### Inference
- Fot text : *LM mode*로, 예측한 distribution으로부터 토큰별로 샘플링하게 된다.
- For image : BOI 토큰이 샘플링되면, *Diffusion mode*로 바뀌게 된다. Pure noise $$ \mathbf{x}_T$$로부터, denosing step을 $$ T $$번 반복하며 이미지를 생성하고 , EOI 토큰이 샘플링되면 다시 *LM mode*모드로 전환된다.

## 4. Experiments

### 4.1. Setup

#### **Evaluation**

-   **text-to-text**: Wikipedia와 C4 데이터셋에서 2천만 개(20M)의 토큰을 대상으로 perplexity를 측정하며, Llama 2의 사전 학습 평가 스위트에서 정확도를 평가합니다.

-   **text-to-image**: MS-COCO 벤치마크에서 무작위로 선택된 3만 개(30K)의 프롬프트에 대해 이미지를 생성하고, 이를 평가하기 위해 zero-shot **Frechet Inception Distance (FID)**와 **CLIP score**를 사용합니다. 또한, 이미지 캡션 생성 능력을 평가하기 위해 **CIDEr score**를 측정했습니다.
-   **GenEval** 벤치마크에서 Transfusion의 성능을 평가하여, 모델이 프롬프트에 맞는 이미지를 정확하게 생성할 수 있는 능력을 측정하여 최신 Diffusion 모델과 성능을 비교 합니다.

#### Baseline
-   **Chameleon** 모델과 비교 실험을 진행합니다. Chameleon은 이미지 데이터를 discrete 토큰으로 quantization하여 표준 언어 모델로 처리하는 방식을 사용합니다. 

- 반면, **Transfusion**은 이미지를 연속적인 공간에서 유지하여 quantization 과정에서 발생하는 정보 손실을 방지합니다.

-   실험을 공정하게 하기 위해, 동일한 데이터와 계산 자원을 사용하여 VAE를 훈련시키며, Chameleon의 경우 양자화 레이어와 코드북 손실을 추가했습니다.

####  Data
-   거의 모든 실험에서 텍스트와 이미지 데이터를 1:1 비율로 샘플링합니다.
- 텍스트는 Llama 2 토크나이저를 사용하여 2T 토큰을 포함한 다양한 도메인의 데이터를 포함합니다.
-  이미지는 3억 8천만 개의 Shutterstock 이미지와 캡션 데이터셋을 사용, 256×256 픽셀로 크롭되고 리사이즈됩니다

#### Latent Image Representation
-   **VAE 훈련**: Esser et al. [2021]의 방법을 따르며, 파라미터 수는 86M이고, CNN 인코더와 디코더를 사용합니다. 이미지 크기 256×256을 32×32×8 크기의 텐서로 압축합니다. 각 잠재 픽셀은 원본 이미지에서 8×8 픽셀 패치를 나타냅니다. 총 1M 번의 학습 단계를 거칩니다.
-   **VQ-VAE 훈련**: VAE 훈련과 동일한 설정을 사용하지만, **$$ \mathcal{L}_{\text{KL}} $$** 대신 **codebook commitment loss**($$ \beta=0.25 $$)을 사용합니다. 이때 16,384개의 토큰 타입을 가진 코드북을 사용합니다.
#### Model Configuration
-   **모델 크기**: 0.16B, 0.37B, 0.76B, 1.4B, 7B 파라미터를 가진 모델들을 실험합니다. 각 설정은 **Llama**의 표준 설정을 따릅니다.

> **linear patch** 인코딩을 사용할 때 추가되는 파라미터는 전체 모델 파라미터의 0.5% 미만입니다. 
> **U-Net patch** 인코딩을 사용할 경우, 모든 설정에서 0.27B 파라미터가 추가되며, 7B 모델에서는 전체 파라미터의 3.8% 정도입니다.

![tab1_2](/posts/20240904_Transfusion/tab1_2.png){: width="800" height="300"}


#### Optimization

-   **최적화 알고리즘**: AdamW($$ \beta_1= $$0.9, $$ \beta_2= $$0.95, $$ \epsilon= $$1e-8)를 사용하며, 학습률은 3e-4에서 1.5e-5까지 cosine scheduler로 감소합니다. batch 당 4096개의 토큰을 총 250k 스텝 동안 총 0.5T개의 토큰을 학습합니다.
-   **대규모 실험**: 2T 토큰을 500k 스텝 동안 학습하며, 배치 크기는 4M 토큰입니다.
-   **정규화**: weight decay, 0.1와 gradient clipping(norm 1.0)을 적용합니다. $$  \lambda $$계수는 5로 설정했습니다. (parameter tuning 은 향후 연구과제로 남깁니다.)

#### Inference

-   **text mode**: 텍스트 생성을 위해 **greedy decoding**을 사용하며, **랭크 분류**는 Llama 평가에서 사용합니다.
-   **Image generation**: 1,000 timestep으로 훈련된 모델을 250 timestep으로 생성, Chameleon 실험에서는 CFG 계수 5를 사용하고, ablation 실험에서는 CFG 계수를 3으로 설정했습니다.


### 4.2. Controlled Comparison with Chameleon

-   **Transfusion**: 이 실험에서 간단한 선형 이미지 인코더/디코더(2×2 패치 크기)와 양방향 주의(attention)를 사용합니다.
-   **FLOPs 계산**: 성능과 계산량(FLOPs)의 관계를 비교하기 위해, **6ND**로 대략적인 FLOPs를 추정하고, 각 벤치마크에서 성능과 FLOPs 간의 스케일링 경향을 시각화합니다.
-   **컴퓨팅 효율성**: Transfusion과 Chameleon이 동일한 성능을 달성하는 데 필요한 FLOPs의 비율을 통해 상대적인 계산 효율성을 측정합니다.

![fig5](/posts/20240904_Transfusion/fig5.png){: width="700" height="400"}

-   **스케일링 경향**: **Figure 5**와 **Table 3**에서 Transfusion이 모든 벤치마크에서 Chameleon보다 더 나은 scaling law 보여줍니다. 특히 **이미지 생성**에서는, Transfusion이 Chameleon과 동일한 **FID** 점수를 34배 적은 계산으로 달성했습니다.
-   **텍스트 성능**: 텍스트 벤치마크에서도 Transfusion이 더 나은 성능을 보여주었으며, 이는 Transfusion과 Chameleon 모두 텍스트를 동일한 방식으로 모델링하는 점을 생각해보면 놀라운 결과라고 합니다.

![tab3](/posts/20240904_Transfusion/tab3.png){: width="800" height="400"}

이런 결과를 해석하기 위해 저자들은 여러가지 ablation을 진행한 결과 Chameleon에서 성능저하는 이미지 토큰과 architecture 수정으로 인한 stability modification이 원인이 되었다고 합니다. 

-   **competition between text and image tokens**: 텍스트와 이미지 토큰이 출력 분포에서 경쟁하면서 텍스트 성능에 부정적인 영향을 미쳤을 가능성이 있습니다.

-   **효율성 차이**: **확산 모델(diffusion)**이 이미지 생성을 더 효율적으로 처리하여 더 적은 파라미터로도 성능을 발휘하고, 이를 통해 Transfusion이 텍스트 모델링에 더 많은 용량을 할애할 수 있게 된 것으로 추측됩니다.

![tab4](/posts/20240904_Transfusion/tab4.png){: width="800" height="400"}


### 4.3. Architecture Ablations
이 섹션에서 저자들은 **Transfusion** 모델의 성능을 더욱 향상시키기 위한 다양한 아키텍처 변경 사항들을 실험적으로 분석합니다.

#### 4.3.1. Attention Masking
-   **intra-image bidirectional attention**가 중요한 역할을 한다는 점을 확인했습니다.
-   **표 5**에 따르면, **bidirectional attention**는 모든 벤치마크에서 성능을 향상시켰으며, 특히 linear enc/dec를 사용한 경우, 이미지 생성에서 FID 점수가 크게 개선되었습니다(61.3 → 20.3). **Linear enc/dec**를 사용하는 경우, 시퀀스 내의 패치 간 정보 흐름이 제한되기 때문에 성능 격차가 더 크게 나타났습니다. 
- 반면, **U-Net 블록**을 사용하는 경우, 양방향 주의가 내재되어 있어 이 격차가 줄어듭니다.

![tab5](/posts/20240904_Transfusion/tab5.png){: width="800" height="400"}

#### 4.3.2. Patch Size
 **Transfusion 모델**은 잠재 픽셀 패치의 크기를 다르게 설정할 수 있습니다. 더 큰 패치 크기는 학습 배치당 더 많은 이미지를 포함할 수 있으며 추론에 필요한 계산량을 줄여줍니다. **표 6**에 따르면, 
 
 - **Linear 인코딩**을 사용하는 모델의 경우 패치 수가 적어질수록 성능이 저하되었지만, 
 - **U-Net 인코딩**을 사용하는 모델은 더 큰 패치 크기에서 이미지 생성 성능이 향상되었습니다.

그러나, 패치 크기가 커질수록 텍스트 성능은 저하되었습니다. 이는 Transfusion 모델이 적은 패치로 이미지를 처리하기 위해 더 많은 자원(parameter)을 투입해야 하므로 성능 저하가 발생한 것으로 보입니다.

![tab6](/posts/20240904_Transfusion/tab6.png){: width="800" height="300"}

#### 4.3.3. Patch Encoding/Decoding Architecture
**U-Net block**이 **Linear layer**보다 더 좋은 성능을 보이는 이유는 **U-Net**의 **Inductive Bias** 덕분이라는 가설이 있습니다. 이를 확인하기 위해, 트랜스포머 파라미터를 7B로 확장하고 U-Net 레이어의 파라미터 수는 거의 동일하게 유지한 실험을 진행했습니다. 결과적으로, U-Net의 추가 파라미터는 전체 모델 파라미터의 3.8%만 차지했습니다.

**표 7**에 따르면, 트랜스포머 모델이 커질수록 U-Net의 상대적 이점은 줄어들지만, 여전히 성능 향상에 기여하고 있습니다. 예를 들어, **이미지 생성**에서 U-Net 인코더/디코더는 더 작은 모델로도 더 나은 FID 점수를 달성하게 해주었습니다.

![tab7](/posts/20240904_Transfusion/tab7.png){: width="800" height="300"}

#### 4.3.4. Image Noising

저자들의 실험에서는 이미지-캡션 쌍의 80%가 캡션을 먼저 제공하고, 이미지가 이를 조건으로 생성되도록 했습니다.(이미지 생성이 더 많은 데이터를 필요로 할 것이라 생각했답니다.) 

반면, 나머지 20%는 이미지를 먼저 제공한 후 캡션을 조건으로 설정했습니다. 저자들은 이렇게 이미지를 먼저 제공할 때 노이즈 수준을 제한할 때 어떤 변화가 있는지 실험적으러 확인했다고 합니다.

**표 8**에 따르면, 이미지를 먼저 제공할 때 노이즈 수준을 절반으로 제한하는 것이 **CIDEr 점수**를 크게 향상시켰으며, 다른 벤치마크에 미치는 영향은 1% 미만으로 나타났습니다.

![tab8](/posts/20240904_Transfusion/tab8.png){: width="800" height="300"}

### 4.4. Comparison with Image Generation Literature
이 섹션에서는 **Transfusion** 모델의 이미지 생성 성능을 최신 이미지 생성 모델들과 비교합니다. 이를 위해 70억(7B) 개의 파라미터를 가진 모델을 훈련시켰으며, 1조 개의 텍스트 토큰과 35억 개(3.5B)의 이미지 및 캡션으로 학습했습니다. 이 모델은 실험 제어에 집중한 이전 버전과 달리, 이미지 생성에 더 최적화된 디자인과 데이터 구성을 사용했습니다.

**결과**:

-   **표 9**에 따르면, **Transfusion**은 DeepFloyd와 같은 고성능 이미지 생성 모델과 유사한 성능을 보여주었으며, **SDXL**과 같은 기존의 모델들을 능가했습니다.
-   **SD 3** 모델과 비교했을 때, Transfusion은 성능이 약간 뒤처졌지만, 이는 **SD 3**이 **역번역을 통한 합성 이미지 캡션**을 사용해 성능을 6.5% 향상(0.433→0.498)시켰기 때문입니다. Transfusion은 자연 데이터를 사용했기 때문에 이 부분에서 비교적 성능이 낮게 나왔습니다.
-   Transfusion 모델은 텍스트 생성에서도 **Llama** 모델과 유사한 성능을 보였습니다.


![tab9](/posts/20240904_Transfusion/tab9.png){: width="800" height="300"}

![fig7](/posts/20240904_Transfusion/fig7.png){: width="700" height="300"}


### 4.5. Image Editing
학습한 modality text-text, image-text, and text-image외에도 image-image task가 가능한지 확인하기 위해 저자들은 8천 개의 이미지 editing example만으로 pretrained model 을 finetune했습니다. 

실험 결과, 모델은 주어진 지시에 따라 이미지를 성공적으로 편집했습니다. 이 실험은 제한적이었으나, Transfusion 모델이 새로운 모달리티 조합으로도 잘 적응할 수 있음을 보여주었습니다.


![fig6](/posts/20240904_Transfusion/fig6.png){: width="800" height="300"}


## 5. Conclusion
이 연구는 **discrete sequence modeling (next token prediction)**과 **continuous media generation (diffusion)** 간의 간극을 연결하는 방법을 탐구했습니다. 

**Transfusion** 모델은 텍스트와 이미지를 각각의 최적화 목표에 맞춰 단일 모델에서 훈련하는 새로운 방법을 제안했습니다. 

실험 결과, **Transfusion**은 효율적으로 확장되며, 파라미터 공유로 인한 성능 손실이 거의 없으면서도 모든 모달리티의 생성을 가능하게 했습니다.

