---
title: "[Paper Reivew] A Survey on Personalized Content Synthesis with Diffusion Models-3"
description: 개인화 이미지 생성을 정리한 Survey 논문입니다.
toc: true
comments: true
# layout: default
math: true
date: 2024-09-26 16:33:00 +09:00
categories: [Deep Learning, Generative Model]
tags: [diffusion model, generative model,personalized content synthesis]     # TAG names should always be lowercase
image: /posts/20240924_Personalization/thumbnail3.webp
alt : Thumbnail
author: Daemin
---

> arXiv 2024. [[Paper]](https://arxiv.org/abs/2405.05538)  
> Xulu Zhang et. al.  
>Hong Kong Polytechnic University,,, etc.  
>9 May 2024  

개인화 task들을 정리한 2편에 이어, 여러가지 테크닉에 대해 소개하는 포스팅입니다.

>[1편, Methods ](https://daemini.github.io/posts/Personalization_1/)  
[2편, Tasks](https://daemini.github.io/posts/Personalization_2/)   
[3편, Techniques](https://daemini.github.io/posts/Personalization_3/) <- 현재 포스팅

## 4. Techniques In Personalized Image Synthesis 
### 4.1. Attention-based Operation

**Attention based operation**은 모델 학습에서 중요한 기술로, 특히 특징을 효과적으로 처리하는 데 사용됩니다. 이 작업은 일반적으로 **Query-Key-Value (QKV)** 체계를 통해 데이터의 다른 부분에 모델이 집중하는 방식을 조정하는 것을 포함합니다. 그러나 이 과정에서 발생하는 문제 중 하나는 **SoI(관심 객체)**의 unique modifier가 attention map에서 지배적인 역할을 해, SoI에만 집중하고 다른 세부 사항을 무시하는 경향이 생길 수 있다는 점이라고 합니다.

- **Mix-of-Show** : **region-aware cross-attention**를 통해 컨텍스트 관련성을 높이는데, 이는 전역 프롬프트로 생성된 feature map을 각 객체에 해당하는 개별적인 지역 특징으로 대체하는 방식입니다.

- **DreamTuner** :  생성된 이미지의 특징을 Q, 생성된 특징의 연결된 데이터를 K, 그리고 참조 특징을 V로 사용하여 새로운 attention 레이어를 설계합니다. 

또 다른 연구 분야는 주의 레이어 내에서 **SoI 토큰의 영향력을 제한**하는 데 초점을 맞추고 있습니다. 

-  **Layout-Control**:  추가 학습 없이 주의 가중치를 레이아웃 주변으로 조정하는 방법을 제안합니다. 
- **Cones 2**: 원하지 않는 occupation을 방지하기 위해 negative attention map을 정의합니다. 
- **VICO** : 새로운 attention 레이어를 삽입하여 binary mask를 사용해 noise latent와 reference image 특징 사이의 attention map을 선택적으로 차단합니다.

### 4.2. Mask-guided Generation
**Mask-guided Generation**은 특정 객체의 위치와 윤곽을 나타내는 마스크(*Strong prior*)를 활용하여 생성 모델이 집중할 부분을 효과적으로 안내하는 강력한 전략입니다. 이러한 마스크는 segmentation 기술의 발전 덕분에 관심 객체(SoI)를 배경으로부터 정밀하게 분리할 수 있습니다. 이 전략을 기반으로 많은 연구들이 목표 객체에만 집중할 수 있도록 하여 불필요한 방해 요소를 제거합니다. 


마스크는 특정 특징 맵을 결합해 더 정보가 풍부한 **semantic patterns**을 구성할 수 있습니다. 
- **Subject-Diffusion** :  diffusion stages 전반에 걸쳐 **latent features에 마스킹**을 적용합니다.

- **AnyDoor** :  *High-frequency filter**를 추가로 사용하여 분할된 객체와 함께 세밀한 특징을 추출하고, 이를 이미지 생성 과정의 조건으로 사용합니다. 

- **DisenBooth**: 학습 가능한 마스크로 **identity-irrelevant embedding**을 정의하여, identity-preservation 임베딩과 identity-irrelevant 임베딩 사이 유사성을 극대화함으로써 불필요한 정보를 배제합니다. 

- **PACGen** ; SoI 억제 프롬프트(SKS PERSON)와 다양한 프롬프트(HIGH QUALITY, COLOR IMAGE) **두 가지 추가 프롬프트**를 사용하여 binary mask 와 함께 CFG로 활용합니다.

-  **Face-Diffuser**: 1) 사전 학습된 T2I Diffusion 모델과 2) learning-based personalized 모델이 예측한 **노이즈를 함께 사용하여 마스크를 결정**합니다. 각 모델은 자체적인 노이즈 예측을 수행하며, 최종 노이즈 출력은 마스크를 통한 *concatenation으로 생성됩니다.

> **Mask-guided Generation**는  생성 모델의 주의력을 구체적인 객체에 집중하게 하고, 불필요한 정보는 배제하여 정확하고 일관된 개인화된 콘텐츠 생성을 지원! 

### 4.3. Data Augmentation
**Data Augmentation**은 제한된 reference 이미지로 인해 발생하는 SoI의 완전한 의미 정보를 포착하는 데 어려움을 겪는 문제를 해결하기 위해 사용되는 전략입니다. 다양한 기법들은 SoI의 다양성을 강화하고 현실적이고 다양한 이미지를 생성하기 위해 연구 되었습니다.

-   **COTI** : 대규모 웹 크롤링 데이터 풀에서 높은 미적 품질의 의미 관련 샘플을 선택함으로써 **훈련 세트를 점진적으로 확장**하는 scorer network 제안

-   **SVDiff** : **여러 SoI가 혼합된 이미지를 새 훈련 데이터**로 manually 구성하여 모델이 복잡한 시나리오에 노출되도록 돕습니다. 

-   **BLIP-Diffusion** : foreground 객체를 나누고, 이를 **무작위 배경과 조합함**으로써 원래 텍스트-이미지 쌍을 instruction-followed 데이터셋으로 확장합니다.

-   **DreamIdentity** : 대규모 사전 학습된 확산 모델에 포함된 유명인들의 기존 지식을 활용하여 **원본 이미지**와 **편집된 얼굴 이미지**를 생성합니다.


-   **PACGen** : 공간적 위치가 정체성 정보와 얽혀 있다는 점을 보여주며, **Rescale, center crop, and relocation**가 이를 해결하는 효과적인 방법임을 제시합니다.

-   **StyleAdapter** :  패치를 섞어 불필요한 객체를 분리하고 원하는 스타일을 보존하는 방식을 선택합니다.

-   **Break-A-Scene** : Single reference image로부터, 여러 subject를 역으로 추출하는 것을 목표로 합니다. 

### 4.4. Regularization
1. **추가 데이터셋 활용**
 **추가 데이터셋**을 사용하여 SoI(Subject of Interest)와 동일한 범주의 이미지를 재구성하는 방법이 널리 사용됩니다. 이를 통해 개인화된 모델이 사전 학습된 지식을 보존하게 되어 Overfitting 문제를 완화할 수 있습니다.
-   **StyleBoost**는 스타일 개인화를 위해 **auxiliary dataset**을 도입합니다.
-   이후 연구에서는 **형태, 배경, 색상, 질감**을 명시한 세밀한 프롬프트를 포함한 정교한 데이터셋 구성 방식을 도입합니다.

 2. **사전 학습된 text prior**
사전 학습된 **대규모 데이터셋**에서 학습된 텍스트 프라이어를 활용하여 SoI 토큰이 다른 텍스트 설명과 자연스럽게 결합되도록 하는 방법도 있습니다. 이를 통해 **text-image alignment**을 높일 수 있습니다.

-   **Perfusion** : 텍스트 수준의 지식을 주입하기 위해 **Key projection**을 클래스 명사에 맞춰 제한하고, **visual fidelity**를 위해 SoI 이미지로 **Value Projection**을 조정합니다.
-   **Compositional Inversion**: 의미적으로 관련된 토큰을 **anchors**로 사용해 토큰 임베딩 탐색을 SoI와 높은 일치를 이루는 영역으로 제한합니다.
-  **Cones 2**: 1,000개의 문장에 포함된 클래스 명사를 재구성하여 **offset을 최소화**하는 방법을 제안합니다.
-  **VICO**: EOT token(end-of-text)이 SoI의 **의미적 일관성**을 유지한다고 발견하였으며, 이를 활용하기 위해 L2 loss를 사용하여 SoI 토큰과 EOT token간의 **attention similarity** 차이를 줄이는 방식을 제안합니다.



## 5. Evaluation
### 5.1. Evaluation Dataset
-   **DreamBooth** :  백팩, 동물, 자동차, 장난감 등 30개의 주제를 포함한 데이터셋
- **DreamBench-v2** : 220개의 테스트 프롬프트를 추가하여 확장
-   **Custom Diffusion**: 10개의 주제를 대상으로 각각 20개의 특정 테스트 프롬프트를 제공하며, 5개의 주제 쌍과 각 쌍에 대해 8개의 프롬프트로 구성된 다중 주제 조합 테스트도 포함
-   **Custom-101** : 평가 범위를 확장해 101개의 주제를 다룸
-   **Stellar** : 400명의 인간 identity를 기반으로 20,000개의 프롬프트를 제공하는 사람 중심의 평가를 목적으로 개발된 최근 데이터셋


### 5.2. Evaluation Metrics
개인화된 콘텐츠 합성의 목표는 fidelity를 유지하면서 텍스트 condition과의 alignment을 보장하는 것입니다.


-   **CLIP Score** : 텍스트 프롬프트의 의미가 생성된 이미지에 얼마나 잘 반영되는지를 측정

-   생성된 주제가 **SoI**와 얼마나 유사한지를 평가하기 위해 **CLIP** 및 **DINO** 같은 대규모 사전 학습된 모델을 기반으로 시각적 유사성을 측정
-   **FID**와 **IS** 같은 기존 평가 지표들은 이미지의 전반적인 품질과 일관성 평가

-   **Stellar**에서 사람 개인화에 맞춘 평가 지표를 개발하기도 했다고 합니다.**soft-penalized CLIP text 점수**, **아이덴티티 보존 점수**, **속성 보존 점수**, **아이덴티티 안정성 점수**, **객체 정확도**, **관계 충실도 점수** 등이 포함

## 6. Challenge and Outlook

### 6.1. Overfitting Problem
현재의 개인화 콘텐츠 합성(PCS) 시스템은 참조 이미지 세트의 수가 제한된 경우, Overfitting 문제를 겪는다는 주요 과제를 안고 있습니다. 이 과적합 문제는 두 가지 방식으로 나눌 수 있습니다.

1.  **SoI 편집 가능성 상실**: 개인화 모델은 참조에 있는 SoI를 고정된 모습으로 재현하는 경향이 있습니다. (예를 들어 동일한 포즈의 고양이만 생성)
2.  **불필요한 의미 포함**: 참조 이미지의 불필요한 요소가 출력 이미지에 포함되며, 이 요소들이 현재 문맥과 상관없는 배경이나 객체일 수 있습니다.

과적합 문제를 해결하기 위해 다양한 해결책들이 제안되었습니다. 많은 방법들이 불필요한 배경의 배제, attention 조작, 학습 가능한 파라미터의 regularization, 데이터 augmentation 등을 통해 과적합 문제를 완화하려고 시도했습니다. 특히 SoI의 외형이 고정적이지 않은 경우나 문맥 프롬프트가 참조 이미지의 불필요한 요소들과 유사한 의미적 상관관계를 가질 때 문제가 발생한다고 합니다.

### 6.2. Trade-off on Subject Fidelity and Text Alignment

-   **Subject Fidelity**를 높이기 위해서는 SoI의 세부적이고 구체적인 특징을 포착하고 재현해야 하며, 이는 매우 세밀한 특성을 학습하고 복제하는 것을 요구합니다.
-   반면에 **Text Alignment**은 SoI를 다양한 텍스트 설명에 맞게 유연하게 조정하는 것을 요구하며, 이는 포즈, 표정, 환경, 스타일의 변경을 포함할 수 있고, 이는 학습 중 재구성 과정과 충돌할 수 있습니다.


### 6.3. Standardization and Evaluation

현재 표준화된 테스트 데이터셋, 강력한 평가 지표의 부족 문제가 있습니다. 향후 연구는 다양한 **PCS** 모델의 측면을 테스트할 수 있는 포괄적이고 널리 **사용 될 수 있는 벤치마크**를 만들고, PCS 시스템의 성능을 보다 정확하게 반영할 수 있는 지표를 개발하는 것이 중요하다고 합니다.



## END!
Survey 논문은 처음 읽어보는데, 전반적인 흐름을 보기에는 좋은 것 같네요. 
다만 방법론이나 설명이 너무 간단하게만 설명이 나와있어서 무슨 말 하는지 모르겠는 부분이 꽤 있는데, 앞으로 천천히 채워나가겠습니다.

긴 글 읽어 주셔서 감사합니다 ! 

>[1편, Methods ](https://daemini.github.io/posts/Personalization_1/)  
[2편, Tasks](https://daemini.github.io/posts/Personalization_2/)   
[3편, Techniques](https://daemini.github.io/posts/Personalization_3/) <- 현재 포스팅











