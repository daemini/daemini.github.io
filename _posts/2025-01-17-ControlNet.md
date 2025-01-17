---
title: "[Paper Reivew] Adding Conditional Control to Text-to-Image Diffusion Models"
description: Large pre-trained T2I freeze한 뒤, 원래 모델의 encoder layer를 복사해 strong backbone으로 사용하는 ControlNet제안.
toc: true
comments: true
# layout: default
math: true
date: 2025-01-17 16:40:00 +09:00
categories: [Deep Learning, Generative Model]
tags: [diffusion model, generative model, personalization]     # TAG names should always be lowercase
image: /posts/20251116_ControlNet/teaser.webp
alt : Thumbnail
author: Daemin
---

> ICCV 2023 [[Paper](https://arxiv.org/abs/2302.05543)] [[Github]](https://github.com/lllyasviel/ControlNet)]<br/>
> Lvmin Zhang, Anyi Rao, and Maneesh Agrawala <br/>
> Stanford University <br/>
> 26 Nov, 2023 <br/>


# TL;DR
large pre-trained T2I freeze한 뒤, 원래 모델의 encoder layer를 복사해 strong backbone으로 사용하는 ControlNet제안. 다양한 condition(Canny edge, depth map, etc.)을 사용해 ControlNet을 학습시키고, 이미지를 생성할 수 있음.

1. Pre-trained T2I model을 **freeze** & encoder 부분만 copy
2. Conditional dataset으로 **encoder만 fine-tuning**
3. 이때 block 간 연결 사이에 **zero-convolution** 사용해 harmful noise가 latent에 더해지는 것을 막아줌.



![fig1](/posts/20251116_ControlNet/fig1.png){: width="800" height="300"}

## 1. Introduction

**T2I diffusion model**은 text prompt를 조절하면서 원하는 이미지를 생성할 수 있지만.... 정확한 Pose, spatial layout을 text prompt 만으로 control 하기는 어렵습니다.

결국 이 문제는 Conditioning image에서 target image로의 mapping, 즉 Image-to-Image translation task로 생각할 수 있습니다. Spatial mask, image editing instructions, fine tuning...등 많은 연구들이 있습니다. 저자들이 말하길, 몇몇 문제는 training-free로 해결할 수 있지만, depth-to-image, pose-to-image 같은 많은 task는 **end-to-end learning**이 필요하다고 합니다.

하지만 Large T2I diffusion model에서 conditional control을 학습하는건 매우 어렵다고 합니다. 보통 pre-trained T2I model의 데이터셋 크기는 매우 큰 것에 비해 (e.g. LAION-5B), conditional dataset은 매우 작기 때문입니다. (e.g. 100K) 따라서 직접 모델을 Fine-tuning 하는 방법은 over-fitting, catastrophic forgetting 과 같은 문제를 야기할 수 있습니다. 

이런 문제를 해결하기 위해 저자들은 Conditional control을 학습하는 end-to-end Neural architecture, **ControlNet**을 제안합니다. 

1. Pre-trained T2I model을 freeze & encoder 부분만 copy.
2. Conditional dataset으로 encoder만 fine-tuning.
3. 이때 block 간 연결 사이에 zero-convolution 사용해 harmful noise가 latent에 더해지는 것을 막아줌.


## 2. Related Work

### 2.1. Finetuning Neural Networks
가장 쉽게 떠올릴 수 있는 방법으로, 추가적인 데이터를 이용해 네트워크를 finetune 하는 방법입니다. 하지만 overfitting, mode collapse, catastrophic forgetting 과 같은 문제가 발생할 수 있습니다.

-   **HyperNetwork**: 작은 네트워크를 학습시켜 큰 네트워크 가중치를 조정.
-   **Adapter**: 트랜스포머 모델에 새로운 모듈 추가로 맞춤화.
-   **Additive Learning**: 기존 가중치 고정 후 소규모 파라미터 추가.
-   **LoRA**: 저차원 공간에서 파라미터 offset 학습으로 catastrophic forgetting 방지.
-   **Zero-Initialized Layers**: 초기 학습 시 유해한 노이즈 방지, ControlNet에서 block을 연결할 때 사용


## 3. Method

### 3.1. ControlNet

![fig2](/posts/20251116_ControlNet/fig2.png){: width="800" height="300"}

ControlNet의 방법론은 사실 위 Figure가 끝입니다. 원래 network를 강력한 backbone으로 사용하면서, encoder 부분만 복사해 fine-tuning하는 것입니다. 이때 block사이 연결을 zero-conv 로 해주면 됩니다.

zero-conv 를 사용하는 이유는 training이 시작할 때 zero conv의 output은 0이므로, harmful noise가 hidden state에 주는 영향을 막으면서, gradient를 흘려주어 점차적으로 학습할 수 있게 만들어주는 역할입니다.

### 3.2. ControlNet for Text-to-Image Diffusion

예를들어 Stable Diffusion에 ControlNet을 적용해봅시다. Stable diffusion은 12개의 encoder, 1개의 middle, 12개의 decoder 로 이루어져 있습니다. 

ControlNet은 SD을 freeze하고,  12개의 encoder, 1개의 middle block을 복사합니다. 이때 backbone model과의 연결을 zero-conv로 연결해주는 것입니다. 


![fig3](/posts/20251116_ControlNet/fig3.png){: width="600" height="300"}


이때 Input conditioning을 latent space의 크기로 mapping 해주기 위해 작은 encoder도 필요하다고 합니다.




### 3.3. Training

ControlNet의 loss로는 다음과 같은 식을 사용합니다.

$$
\mathcal{L} = \mathbb{E}_{z_0, t, c_t, c_f, \epsilon \sim \mathcal{N}(0, 1)} \left[ \| \epsilon - \epsilon_\theta(z_t, t, c_t, c_f) \|_2^2 \right]
$$

학습 단계에서 random하게 50%의 확률로 text prompt $$ c_t $$를 empty string으로 바꿔버리는데, 이를 통해 모델에게 직접적으로 conditioning 이미지의 semantic 정보를 학습할 수 있도록 한다고 합니다.

여기서 재미있는 현상이 발생하는데, 저자들이 학습을 하는 과정에서 condition을 gradually 학습하는 것이 아니라, 어느 순간 갑자기 condition을 학습했다고 합니다. 저자들은 이를 "*sudden convergence phenomenon*"라 부릅니다.

![fig4](/posts/20251116_ControlNet/fig4.png){: width="600" height="300"}

### 3.4. Inference 

기존 Stable Diffusion에서 사용되는 CFG 는 다음과 같습니다.

$$
\epsilon_{prd} = \epsilon_{uc} + \beta_{cfg} (\epsilon_{c} - \epsilon_{uc})
$$

저자들은 CFG를 ControlNet에도 적용하려고 시도할 때, 

- $$ \epsilon_{uc} , \epsilon_{c} $$ 모두에게 conditioning.
- $$ \epsilon_{c}  $$ 만 conditioning.

두 가지 방법을 사용할 수 있는데, 이 두 방법 모두 challenging case의 경우 생성 품질이 그렇게 좋지 않았다고 합니다.

저자들은 이를 해결하기 위해 $$ \epsilon_{c}  $$ 에 Conditoning을 줄 때 resolution에 따른 weight를 걸어주었다고 합니다. 이를 **CFG with resolution weighting**이라고 합니다.

$$
w_i = 64/h_i
$$


![fig5](/posts/20251116_ControlNet/fig5.png){: width="800" height="300"}


## 4. Experiments

### 4.1. Qualitative Results

![fig7](/posts/20251116_ControlNet/fig7.png){: width="800" height="300"}





### 4.2. Ablative Study

1. zero conv를 일반적인 conv로 교체.
2. trainable copy을 사용하지 않고 단일 conv layer로 대체(ControlNet-lite).

![fig8](/posts/20251116_ControlNet/fig8.png){: width="800" height="300"}


-   ControlNet은 모든 프롬프트 설정에서 성공적으로 작동.
-   ControlNet-lite는 프롬프트 없는 설정 및 불충분한 프롬프트에서 실패.
-   zero conv를 대체하면 ControlNet의 성능이 크게 저하.


### 4.3. Comparison to Previous Methods

![fig9](/posts/20251116_ControlNet/fig9.png){: width="600" height="300"}

ControlNet은 다양한 조건 이미지를 강력하게 처리하며, 선명하고 깨끗한 결과를 달성.

### 4.5. Discussion

- **Influence of training dataset sizes** : 적은 데이터셋으로도 ControlNet이 robust한 특성을 가지며, 학습 데이터셋의 크기를 키울 수록 생성 이미지의 quality가 높아짐을 확인했습니다.

![fig10](/posts/20251116_ControlNet/fig10.png){: width="600" height="300"}

- **Capability to interpret contents** :

애매모호한 condition을 주더라도 모양을 해석하고 추측하여 이미지 생성.

![fig11](/posts/20251116_ControlNet/fig11.png){: width="600" height="300"}


- **Transferring to community models** :
ControlNet을 기존 SD를 변경하지 않으므로, SD의 변형에 직접적으로 적용 가능.


![fig12](/posts/20251116_ControlNet/fig12.png){: width="600" height="300"}



