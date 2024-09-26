---
title: "[Paper Reivew] A Survey on Personalized Content Synthesis with Diffusion Models-1"
description: 개인화 이미지 생성을 정리한 Survey 논문입니다.
toc: true
comments: true
# layout: default
math: true
date: 2024-09-25 18:10:00 +09:00
categories: [Deep Learning, Generative Model]
tags: [diffusion model, generative model,personalized content synthesis]     # TAG names should always be lowercase
image: /posts/20240924_Personalization/thumbnail1.jpeg
alt : Thumbnail
author: Daemin
---


> arXiv 2024. [[Paper]](https://arxiv.org/abs/2405.05538)  
> Xulu Zhang et. al.  
>Hong Kong Polytechnic University,,, etc.  
>9 May 2024  

양이 조금 많아서 나누어 작성했습니다 (꾸벅.)

> [1편, Methods ](https://daemini.github.io/posts/Personalization_1/) <- 현재 포스팅.  
[2편, Tasks](https://daemini.github.io/posts/Personalization_2/)    
[3편, Techniques](https://daemini.github.io/posts/Personalization_3/)  

# TL;DR
**Personalized Content Synthesis(PCS)** 를 optimization-based, learning-based로 구분하여, PCS분야의 여러가지 모델의 구조와 방법론을 정리한 **survey 논문**입니다.


## 1. Introduction

**Personalized Content Synthesis(PCS)** 는 학습 방법에 따라 optimization-based, learning-based로 구분할 수 있습니다.

1. **optimization-based** : 개인화 요청에 따라 diffusion model을 fine tuning하는 방법입니다.
2. **learning-based** : 어떠한 Subject of Interest(SoI)라도 다룰 수 있는 하나의 통합된 모델을 구축하는 것을 목표로 합니다. 



현재의 방법들이 놀라운 성능을 보여주고 있음에도 불구하고, reference image 수의 제한으로 인한 **overfitting** 문제가 가장 심각한 challenge라고 합니다. (출력에서 이미지의 배경을 그대로 반영하거나, Input prompt와 관련 없는 이미지가 생성)

또한 **image alignment** 와 **text fidelity**간 trade-off도 해결해야 할 문제로 남아있습니다.

## 2. Generic Framework

### 2.1 Optimization-based Framework
![fig4](/posts/20240924_Personalization/fig4.png){: width="600" height="300"}

#### Unique Modifier
Optimization-based Framework에서 중요한 측면 중 하나는 **SoI를 텍스트에 어떻게 표현하는지**입니다. 이런 modifier들을 inference시 text prompt에 넣어서 사용할 수 있습니다. 

1. **Learnable embedding** : 원래 dictionary에 존재하지 않는 pseudo token을 사용해 modifier로서 사용하는  방법입니다. 
2. **Plain Text** : SoI를 명시적으로 설명하는 텍스트를 사용하는 방법입니다. 예를 들어, *"Yello Cat"과 같이 사용자의 object를 직접적으로 나타낼 수 있지만, 기존 텍스트의 이미를 변경하여 다른 종류의 고양이를 생성하지 못한다는 문제점이 있습니다.
3. **Rare Token** : 자주 사용되지 않는 토큰을 사용해 일반적인 어휘에 미치는 영향을 최소화 하는 방법입니다. 하지만 이런 rare token은 유용한 정보를 제공하지 않아 텍스트 안에서 약한 표현력을 보이는 문제가 있습니다.

#### Training Prompt Construction
일반적으로 학습 프롬포트는 "_Photo of V*_"로 구성되지만, DreamBooth[^1]  의 저자들은 이러한 단순한 설명보다는 "*Photo of V* cat*"같이 unique modifier를 사용해 학습 시간을 줄이면서 성능을 더 높일 수 있었습니다. 또한, 더 나은 SoI와 관련 없는 개념의 분리를 위해 각 훈련 참조에 대해 더 구체적인 텍스트 프롬프트가 사용될 수 있습니다 “*Photo of V* cat on the chair*”. 이는 고품질 캡션이 정확한 텍스트 제어를 더욱 향상시키는 경향이  있음을 시사합니다.

#### Training Objective
Optimization-based 방법의 주요 목표는 $$ \theta $$을 개인화 요청에 따라 refine 하여 $$ \theta' $$를 얻는 것입니다.  이 과정은 **test-time fine-tuning**이라고 불리며, $$ \theta' $$ 을 조정하여 참조된 프롬프트에 따라 SoI를 재구성하는 것이 목적입니다.

$$ 
\begin{equation}
    L_{rec} = \mathbb{E}\left[w_{t}\| f_{\theta^{\prime}}(\mathbf{x}_{t},t,c)\mathbf{x}_0\|_{2}^2\right]
\end{equation}
$$

혹은 이와는 다르게, learnable parameter에 변화를 주는 방법이 있습니다.
- 토큰 임베딩 최적화 
	- [9, An image is worth one word]
	- [13, p+: Ex- tended textual conditioning in text-to-image generation]

- 전체 확산 모델 최적화 
	- [10, Dreambooth]


- 특정 매개변수 부분집합 최적화 
	- [14, Multi- concept customization of text-to-image diffusion]
	- [15, Key-locked rank one editing for text-to-image personalization]
	- [16, 1.  Svdiff: Compact parameter space for diffusion fine-tuning]


- 어댑터
	 - [17, A closer look at parameter- efficient tuning in diffusion models] 
	 - [18 Styledrop: Text-to-image synthesis of any style]

- 새로운 매개변수의 도입(LoRA) 
	- [19, Mix-of-show: Decentralized low- rank adaptation for multi-concept customization of diffusion models]
	- [20, Hyperdreambooth: Hypernet- works for fast personalization of text-to-image models]
	- [21, Omg: Occlusion-friendly person- alized multi-concept generation in diffusion models]

#### Inference

모델이 $$ \theta' $$ 로 finetune 된 후에는, 개인화된 이미지 생성을 위한 **추론 단계**로 넘어갑니다. SoI와 연관된 unique modifier를 포함한 새로운 입력 설명을 구성함으로써, 원하는 이미지를 쉽게 생성할 수 있습니다.

### 2.2. Learning-based Framework
#### Overview
최근 **Learning-based** 프레임워크는 테스트 시점에서의 fine tunning이 필요하지 않다는 장점 때문에, **개인화 콘텐츠 생성(PCS)**에서 많은 주목을 받고 있습니다. 이 방식의 기본 아이디어는 대규모 데이터셋을 활용하여 다양한 주제의 입력을 개인화할 수 있는 강력한 모델을 훈련하는 것입니다. 훈련 과정은 생성된 이미지와 실제 이미지 간의 reconstruction loss를 최소화하여 학습하는 것입니다. 하지만, 이렇게 강력한 모델을 훈련하는 것은 쉽지 않으며, 현재 방법에서는 3가지 주요 factor가 중요하다고 합니다.

1.  **effective architecture**: 테스트 시점에서 개인화를 용이하게 하기 위한 방법.
2.  **visual fidelity**를 보장하기 위해 SoI의 최대한의 정보를 어떻게 보존할 것인가.
3.  **appropriate size of data** training에 얼마만큼의 데이터를 사용할 것인가.

#### Architecture
일반적인 개인화 작업에서 두 가지의 정보를 제공합니다. 1개 이상의 참조 이미지와 텍스트 설명
이 두 가지 modality를 융합하는 방식에 따라 **placeholder-based** 와 **reference-conditioned architectures**로 나눌 수 있습니다. 


**1. Placeholder-based architecture** :

![fig5_top](/posts/20240924_Personalization/fig5_top.png){: width="600" height="300"}

_The framework utilizes an encoder to extract the image features as the feature vectors in the text embedding_

Optimization-based 방법에서 사용한 unique modifier에서 영감을 받아, placeholder를 도입해 SoI의 시각적 특성을 나타냅니다. Placeholder는 추출된 이미지 특징을 저장해 텍스트 임베딩과 concat됩니다. 결합된 특징은 이후 adapter, cross attention layer과 같이 학습 가능한 모듈을 통해 맥락적 연관성을 강화합니다.


**2. Reference-conditioned architecture** :

![fig5_bottom](/posts/20240924_Personalization/fig5_bottom.png){: width="600" height="300"}

_The framework initializes a new cross-attention module to fuse image features in the U-net_

U-Net backbone을 수정하여, adapter, cross attention layer와 같이 추가적인 layer로 추가적인 시각적 입력을 통합합니다. 

예를 들어 **IP-Adapter**[22, Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models]는 decoupled cross-attention 모듈을 훈련하여 이미지 특징과 텍스트 특징을 **별도로 처리**한 후, 소프트맥스 연산 후의 덧셈 결과를 최종 출력으로 정의합니다. 이 경우, 플레이스홀더는 필요하지 않습니다.

**3. Combination approach** :

**Subject-Diffusion**[23, Subject-diffusion: Open domain personalized text-to-image generation without test-time fine-tuning]과 같은 시스템은 **Placeholder-based**과 **Reference-conditioned architecture**을 모두 통합하여 두 가지 접근 방식의 장점을 결합하여 전반적인 개인화 능력을 강화합니다.

#### SoI Feature Representation
개인화 콘텐츠 생성에서 **SoI**의 대표적인 특징을 추출하는 것은 매우 중요합니다. 일반적인 접근 방식으로는 사전 훈련된 모델(**CLIP**, **BLIP**)을 사용하는 인코더를 활용하는 것입니다. 이러한 모델들은 global 특징을 잘 포착하는데 뛰어나지만, 종종 불필요한 정보까지 포함하여 출력 품질을 저하시킬 수 있습니다. (배경이 의도치 않게 포함되는 문제 등)

이를 해결하기 위해 일부 연구는 학습 과정에서 **SoI**에 초점을 맞추도록 추가적인 **prior knowledge**를 도입해, 불필요한 요소들을 배제하는 방법을 사용합니다. 예를 들어, **SoI**를 명확하게 구분하기 위한 **mask**를 적용하여 배경의 영향을 효과적으로 배제할 수 있습니다.

실제 응용에서는 **multiple input references**를 처리하는 것도 중요한 도전 과제입니다. 이는 여러 참조 이미지에서 추출된 특징들을 조합하여 시스템의 적응성을 강화하는 것이 필요합니다. 그러나 대부분의 현재 학습 기반 시스템은 단일 참조 입력만 처리할 수 있는 한계를 가지고 있습니다. 

일부 연구에서는 여러 참조 이미지에서 추출된 특징들을 
- 평균화 [Instantid: Zero-shot identity-preserving generation in seconds] 
- 스택[Pho- tomaker: Customizing realistic human photos via stacked id embed- ding]하여 
복합 SoI 표현을 형성하는 방법을 제안하고 있습니다.



#### Training Data
학습 기반 PCS(Personalized Content Synthesis) 모델을 훈련하려면 대규모 데이터셋이 필요합니다. 학습에 주로 사용되는 데이터는 두 가지 유형으로 나눌 수 있습니다:

1. **Triplet Data (Reference Image, Target Image, Target Caption)**
    -   이 데이터셋은 PCS의 목적과 직접적으로 일치하며, 참조 이미지와 개인화된 콘텐츠 간의 명확한 관계를 확립합니다. 하지만 **대규모 Triplet 샘플**을 수집하는 것은 어려운 과제입니다. 이를 해결하기 위해 여러 전략이 제안되었습니다:
        -   **데이터 증강(Data Augmentation)**
        -   **합성 샘플 생성(Synthetic Sample Generation)**
        -   **인식 가능한 SoI 활용**
3.  **Dual Data (참조 이미지, 참조 설명)**
    -   이러한 데이터셋은 **LAION** 및 **LAION-FACE** 와 같은 대규모 컬렉션을 포함하여 비교적 쉽게 구할 수 있습니다. 하지만 주요 단점은 이러한 데이터로 학습된 모델이 텍스트 설명보다는 **참조 이미지 재구성에 중점을 두는 경향**이 있다는 점입니다. 



## Next Posting..
다음 포스팅에서는 **Categorization of Personalization Tasks**을 중점적으로 다룰 예정입니다.


> [1편, Methods ](https://daemini.github.io/posts/Personalization_1/) <- 현재 포스팅.  
[2편, Tasks](https://daemini.github.io/posts/Personalization_2/)    
[3편, Techniques](https://daemini.github.io/posts/Personalization_3/)  

[^1]: [DreamBooth](https://daemini.github.io/posts/DreamBooth/)





