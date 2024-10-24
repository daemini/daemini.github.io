---
title: "[Paper Reivew] A Survey on Personalized Content Synthesis with Diffusion Models-2"
description: 개인화 이미지 생성을 정리한 Survey 논문입니다.
toc: true
comments: true
# layout: default
math: true
date: 2024-09-26 16:33:00 +09:00
categories: [Deep Learning, Generative Model]
tags: [diffusion model, generative model,personalized content synthesis]     # TAG names should always be lowercase
image: /posts/20240924_Personalization/thumbnail2.webp
alt : Thumbnail
author: Daemin
---

> arXiv 2024. [[Paper]](https://arxiv.org/abs/2405.05538)  
> Xulu Zhang et. al.  
>Hong Kong Polytechnic University,,, etc.  
>9 May 2024  

개인화 방법론을 정리한 1편에 이어, 개인화의 대표적인 task와 모델을 소개하는 포스팅입니다.

  
>[1편, Methods ](https://daemini.github.io/posts/Personalization_1/)  
[2편, Tasks](https://daemini.github.io/posts/Personalization_2/) <- 현재 포스팅
[3편, Techniques](https://daemini.github.io/posts/Personalization_3/)  


## 3. Categorization of Personalization Tasks
![fig6](/posts/20240924_Personalization/fig6.png){: width="1000" height="500"}
### 3.1. Personalized Object Generation
Personalized Object Generation는 개인화에서 가장 기본이 되는 task입니다. Textual Inversion, DreamBooth, ELITE 크게 3가지 방법론으로 구분할 수 있습니다. 

#### **Textual Inversion**
Textual Inversion은 간단하지만 효과적인 방법입니다. 새로운 토큰을 토크나이저에 삽입해 SoI(관심대상)를 표현합니다. Noisy input에 SoI 참조를 재구성함으로써, pseudo token을 최적화합니다. 이 방식의 가장 큰 장점은 가장 적은 storage를 사용한다는 점입니다. (새로운 토큰을 저장하는 데 ~kB) 하지만 이 방식의 단점은 복잡한 시각적 특징을 소수의 파라미터로 압축하기 때문에 학습 시간이 오래걸리며, visual fidelity가 떨어질 수 있다고 합니다. 

이런 가상 토큰 임베딩을 다루는 최근 연구들을 다음과 같습니다. 

1. **P+** : U-Net 아키텍처의 다양한 레이어에서 *distinct textual condition*을 제공해 속성 제어를 개선
2. **NeTI** : P+방식을 발전시켜, *neural mapper*를 를 이용해 denosing time step과 U-Net구조에 맞게 토큰 임베딩을 adaptively 출력
3. **ProSpect**: 레이아웃, 색상, 구조, 질감과 같은 프롬프트 종류가 denosing step의 다른 단계에서 활성화되는 것을 발견, Time step에따라 여러 토큰 임베딩을 최적화하는 방법을 제시
4. **HiFiTuner** : 마스크 기반 손실 함수, 파라미터 정규화, 시간에 따른 임베딩, 가장 가까운 참조 이미지를 이용하며 다양한 기법들을 통합
5. **DreamArtist**: negative 와 positive prompt 임베딩을 동시에 사용
6. **InstructBooth**: 사람의 선호도 점수를 보상 모델로 사용하는 reinforcement learning framework 제시
7. **Gradient-free textual inversion** : 학습 가능한 토큰을 반복적으로 업데이트하는 방법으로 gradient- free evolutionary 알고리즘을 제안

#### **DreamBooth**
하지만 Optimization-based 방식에서는 토큰 임베딩에서, 모델을 fine-tuning하는 방식으로 유행이 바뀌었다고 합니다. 하지만 전체 모델을 fine-tuning하는 것은 상당한 storage cost가 필요하다는 단점이 있습니다.

1. **DreamBooth** : unique modifier를 사용해 SoI를 나타내고, 모델 전체를 fine-tuning 합니다.
2. **Custom Diffusion** : Cross attention layer의 주요 파라미터(key-value projection)를 fine-tuning하여 시각적 충실도와 저장 효율성 간의 균형을 맞추는 방법 제시.
3. **Perfusion** : Cross attention layer fine-tuning할 때, key 프로젝션의 업데이트 방향을 슈퍼 카테고리 토큰 임베딩으로, value 프로젝션을 학습 가능한 토큰 임베딩으로 정규화합니다.
4. **COMCAT**: Low-Rank Approximation를 통해 Attention matrix 저장 요구량을 6MB로 줄이면서도 높은 시각적 충실도를 유지하는 방법을 도입했습니다.

추가적으로 **Adapters**, **LoRA** 등의 방법들이 파라미터 효율성을 고려한 개인화된 생성에 점점 더 많이 사용되고 있습니다.


> 이러한 최적화 기반 접근법에서 중요한 점은, 가상 토큰 임베딩의 미세 조정이 확산 모델의 가중치 미세 조정과 호환된다는 것입니다. 예를 들어, 미세 조정된 프롬프트 임베딩은 후속 확산 모델 가중치 미세 조정을 위한 효과적인 초기값으로 간주될 수 있습니다. 또한, 두 부분은 서로 다른 학습률을 적용하여 동시에 최적화할 수 있습니다.

#### **ELITE**
**Re-Imagen**은 특정 프롬프트에 따라 텍스트-이미지 쌍에서 특징을 검색해 활용하는 retrieval-augmented generative 방법을 도입했는데, 이는 객체 개인화를 목표로 하지는 않지만, 이러한 프레임워크를 훈련하는 것의 가능성을 시사합니다. 

1. **ELITE** : 전역 참조 특징과 텍스트 임베딩을 결합하고, 관련 없는 배경을 제외한 지역적 특징을 포함하여 이미지 개인화
2. **InstantBooth** : CLIP 모델을 다시 학습시켜 이미지 특징과 패치 특징을 추출하고, 이를 어텐션 메커니즘과 학습 가능한 어댑터를 통해 확산 모델에 주입합니다.
3. **UMM-Diffusion** : reference image와 텍스트 프롬프트를 기반으로 fused feature를 만드는 multi-modal encoder 제안 (fused feature는 guidance signals처럼 mixed noise를 예측하는 것으로 생각가능)
4. **SuTI** : Re-Imagen과 비슷한 architecture를 사용하지만, 특정 객체 세트에 맞게 optimization-based 방식으로 생성된 많은 양의 training sample을 이용해 학습
5. **contrastive- based regularization technique**: 이미지 인코더에서 생성된 pseudo 임베딩을 가장 가까운 pre-trained token으로 push하는 방법 제시. Dual path attention 모듈 제안




또한 하나의 modality에서 확장되어, multi-modal LLM을 이용하는 연구들도 있습니다.
1. **BLIP-Diffusion** : BLIP2 모델을 활용하여 SoI 참조 이미지와 클래스 명사를 포함한 멀티모달 입력을 인코딩,  이 출력 임베딩을 문맥 설명과 결합해 이미지 생성에 사용합니다.
2. **Customization Assistant, KOSMOS-G** : Stable Diffusion의 텍스트 인코더를 pre-trained multi-modal LLM으로 대체

또한 최근에는 optimization-based and learning-based methods를 통합하려는 연구도 있습니다.  
- learning-based는 다양한 객체를 처리할 수 있는 일반적인 프레임워크를 제공하는 반면, 
- optimization-based는 특정 인스턴스에 맞춘 미세 조정을 통해 세밀한 디테일을 더욱 잘 보존

예를들어, **DreamTuner**는 SoI에 맞는 정확한 재구성을 위해 subject encdoer를 사전 학습하고, 두 번째 단계에서 참조 이미지와 유사한 정규화 이미지를 사용해 세밀한 디테일을 유지합니다.


### 3.2. Personalized Style Generation
Personalized Style Generation은 참조 이미지의 **미적 요소**를 개인화하는 작업으로, 여기서 "Style"은 화풍, 재질 질감, 색상 구성, 구조적 형태, 조명 기법, 문화적 영향 등을 포괄하는 다양한 예술적 요소를 포함합니다.

1. **StyleDrop** : 단일 참조 이미지에서 스타일을 효율적으로 캡처하기 위해 *어댑터 튜닝*을 활용. 인간 평가와 CLIP 점수와 같은 피드백 메커니즘을 활용한 반복적 훈련
2. **GAL** : 생성 모델에서 active learning 탐구하여, 합성 데이터 샘플링을 위한 uncertainty-based evaluation 전략을 제안하고, 추가 샘플과 원래 참조의 기여도를 균형 있게 맞추기 위한 가중치 체계를 도입했습니다.
3. **StyleAligned** : 이미지 batch 간 스타일 일관성을 유지하는 데 중점. 첫 번째 이미지를 참조로 사용하여 스타일 가이드라인을 제공하며, 이는 self-attention 레이어에서 추가적인 key와 value로 작용하여 배치 내 모든 이미지가 동일한 스타일 규칙을 따르도록 함
4. **StyleAdapter** : learning-based frame- work에서 dual-path cross-attention mechanism 사용. 여러 스타일 참조로부터 global 특징을 추출하고 통합하는 데 특화된 임베딩 모듈을 도입 -> 복합적인 스타일 정보를 효율적 처리

### 3.3. Personalized Face Generation
소수의 얼굴 이미지와 텍스트 프롬프트를 사용하여 다양한 ID 이미지를 생성하는 작업입니다. 일반 객체 개인화와 비교해, 인간이라는 **특정 클래스에 중점**을 둔 접근 방식으로, 얼굴 인식이나 랜드마크 탐지와 같은 잘 개발된 영역을 활용한 **대규모 인간 중심 데이터셋**을 사용하기 쉽다는 특징이 있습니다.


**Optimization-based**
1.  **PromptNet**: 입력 이미지와 noisy latent를 pseudo word embedding으로 인코딩하는 모델, 과적합 문제를 해결하기 위해, 노이즈와 context description을 CFG동안 균형있게 결합
    
2.  **HyperDreamBooth**: 대규모 데이터셋을 learning-based model로 학습한 후, Second-time fine-tuning을 통해 얼굴의 세부 표현과 충실도를 개선
    
3.  **유명인 얼굴 조합 모델**: 개인화된 ID를 사전 학습된 확산 모델의 유명인 얼굴로 변환하는 방식. 간단한 MLP를 통해 얼굴 특징을 celeb embedding space로 변환

**Learning-based**

1.  **Face0**: 얼굴 영역을 탐지하고 잘라서 세밀한 임베딩을 추출한 후, 샘플링 단계에서 얼굴 임베딩, 텍스트 임베딩, 얼굴-텍스트 결합 임베딩의 **노이즈 패턴을 가중치 기반으로 결합**
    
2.  **W+ Adapter**: StyleGAN의 W+ 공간에서 얼굴 특징을 텍스트 임베딩 공간으로 변환하기 위해 **mapping network, residual cross-attention modules**을 사용
    
3.   **FaceStudio**: 스타일화된 이미지, 얼굴 이미지, 텍스트 프롬프트를 포함한 **hybrid guidance**를 지원하는 cross-attention layer 적용
    
4.   **PhotoMaker**: 고품질 데이터를 수집하고 필터링하여 높은 품질의 데이터 수집.**ID 특징과 클래스 임베딩을 결합**하는 two layer MLP를 사용
    
5.   **PortraitBooth**: 텍스트 조건과 사전 학습된 얼굴 인식 모델의 얕은 특징을 결합하는 **간단한 MLP**를 사용하며, 얼굴 표현과 신원 유지(Identity Preservation)를 보장하기 위해 **expression token, mask-based cross-attention loss**추가
    
6.   **InstantID**: 얼굴 **랜드마크를 입력**으로 사용하여 강력한 가이드를 제공하는 **ControlNet** 변형 모델을 도입.


### 3.4. Multiple Subject Composition
사용자는 때때로 여러 SoI를 함께 구성하려는 의도를 가집니다. 하지만 모듈로 따로 fine tunning된 파라미터를 합치는 것이 어려워 도전적인 과제로 남아있다고 합니다. 

optimization method 방식으로,**Custom Diffusion**은 Cross attention key-Value Projection 가중치를 결합하여, 각 SoI 대한 재구성 성능을 최대화하는 것을 목표로 최적화 


- **StyleDrop** : 각 개인화된 확산 모델에서 노이즈 예측을 동적으로 요약
- **OMG**: 각 LoRA 조정 모델이 예측한 잠재 변수를 주제 마스크를 사용하여 공간적으로 합성

추가적으로 가장 간단한 해결책 중 하나는 모든 예상 주제를 포함한 데이터셋에서 **통합 모델**을 training하는 것입니다.
-**SVDiff**: Cut-Mix라는 데이터 증강 방법을 사용해 여러 주제를 함께 구성하고, SoI와 해당 토큰 간의 정렬을 보장하기 위해 location loss을 적용해 어텐션 맵을 정규화합니다

### 3.5. High-level Semantic Personalization
이미지 개인화 분야는 단순한 시각적 속성을 넘어 복잡한 의미 관계와 고차원 개념을 포함하는 방향으로 확장되고 있습니다. 이러한 추상적 요소를 이해하고 조작할 수 있는 모델의 역량을 강화하기 위해 여러 접근 방식이 개발되었습니다.

-   **ReVersion** : 참조 이미지에서 객체 관계를 반전시키는 것을 목표로 합니다. 이 방법은 contrastive loss를 사용하여 토큰 임베딩을 전치사, 명사, 동사 등 특정 품사 태그 클러스터로 최적화하도록 유도합니다. 또한, 학습 과정에서 더 큰 time stepd에서 노이즈를 추가하여 고차원 의미적 특징을 강조하여 추출할 수 있도록 확률을 높입니다.
    
-   **Lego** :  주로 형용사와 같은 더 일반적인 개념에 중점을 둡니다. 이 개념은 주제의 외형과 자주 얽혀 있는데, 이 형용사적 개념은 깨끗한 주제 이미지와 원하는 형용사를 포함한 이미지로 구성된 데이터셋에 contrastive loss를 적용해 학습할 수 있습니다.
    
-   **ADI** : 참조 이미지로부터 action-specific identifier를 학습하는 것을 목표로 합니다. 동작에만 집중되도록 하기 위해, ADI는 생성된 triplet sample(Reference Image, Target Image, Target Caption)으로부터 gradient 불변성을 추출하고 관련 없는 특징 채널을 마스킹하기 위한 thresholding을 사용합니다.


### 3.6. Attack and Defense
기술의 발전은 위험한 사용 가능성을 높이며, 이를 방지하기 위한 노력도 필요합니다. 

- **Anti-DreamBooth**: 참조 이미지에 미세한 노이즈를 추가하여, 이러한 이미지를 학습한 개인화 모델이 형편없는 결과만 생성하도록 합니다.

- **Backdooring textual inversion for concept censorship** : 미리 정의된 트리거 단어와 의미 없는 이미지를 짝지어 훈련에 포함시킵니다. 트리거 단어가 등장하면 생성된 이미지가 의도적으로 수정되어 보호 효과를 제공합니다.





### 3.7. Personalization on Extra Conditions
**Personalization on Extra Conditions**는 기본적인 이미지 생성 외에 추가적인 조건을 포함하여 콘텐츠를 개인화 하는 작업을 의미합니다. 가장 인기 있는 과제 중 하나는 **추가적인 source image를 기반으로** 개인화를 적용하는 것으로, 소스 이미지의 주제를 사용자 지정 객체(SoI)로 대체하는 방식입니다. 이는 개인화와 이미지 편집의 Cross domain 작업으로 볼 수 있습니다.

-   **PhotoSwap**: 먼저, diffusion model을 fine tune한 다음 **DDIM inversion**를 통해 소스 이미지의 배경을 보존하고, 소스 이미지 생성에서 파생된 intermediate feature map을 대체합니다.
    
-   **MagiCapture**: 얼굴 맞춤화 작업으로 확장한 연구입니다.
- **Virtual Try-on(VTON)** : 기술은 선택한 의류를 대상 인물에게 맞추는 응용 프로그램의 한 예입니다.

### 3.8. Personalized Video Generation
**Personalized 3D Generation**은 크게 세 가지 주요 목표로 구분됩니다: **appearance**, **motion**, and **both**


1.  **appearance**: 주로 **이미지**를 reference로 사용하며, **video diffusion 모델**이 foundation으로 활용됩니다. 
    
2.   **motion**: reference가 일관된 행동을 포함하는 **비디오 클립**으로 전환됩니다. 일반적인 접근 방식은 행동 클립을 재구성하기 위해 **video diffusion 모델을 fine tunning**하는 것입니다. 이렇게 참조 비디오 내에서 **appearance와 motion을 구별**하는 것은 어려운 과제라고 합니다. 


- **SAVE** : 학습 단계에서 외형을 동작 학습과 분리하기 위해 **외형 학습**을 적용합니다. 
- **VMC** : 훈련 시 **프롬프트 구성**에서 배경 정보를 제거하여 학습의 초점을 유지합니다.
    
3.  **외형과 동작을 통합하는 방법**에서는 두 가지 측면을 동시에 학습하는 복잡성을 해결하기 위한 혁신적인 방법이 사용됩니다. 
- **MotionDirector**: 공간적(spatial) 및 시간적(temporal) 손실을 활용하여 이 두 차원을 학습할 수 있게 합니다. 
- **DreamVideo**: **임의로 선택된 프레임의 residual features**을 통합하여 주제 정보를 강조합니다. 이 기법을 통해 fine tunning된 모듈이 주로 **동작 역학** 학습에 집중할 수 있도록 지원합니다.



### 3.9. Personalization 3D Generation

** Personalization 3D Generation**은 주로 **2D diffusion model**을 fine tunning하는 최적화 기반 방법으로 시작되며, 이후 이 모델이 **3D Neural Radiance Field (NeRF)** 모델의 최적화를 안내하는 방식으로 이루어집니다. 이 과정을 통해 각 사용자 지정 프롬프트에 맞는 3D 모델을 생성합니다.

-   **DreamFusion**: **Score Distillation Sampling (SDS)**을 도입하여 2D 확산 모델과 일치하는 이미지를 렌더링할 수 있는 3D 모델을 훈련합니다. 

- **DreamBooth3D**: DreamFusion으로 부터 발전한 방법으로, 세 가지 단계로 구성됩니다. 1) DreamBooth 모델에서 NeRF를 초기화하고 최적화한 후, 2) multi view 이미지를 렌더링, 3) 3D NeRF를 위한 DreamBooth를 fine tunning합니다.
    
-   **Consist3D** : 3D 모델 최적화 과정에서 **semantic 토큰**과 **geometric 토큰** 두 가지를 훈련하여 텍스트 임베딩을 향상시킵니다. 

- **TextureDreamer**: 다양한 3D 객체의 텍스처를 렌더링하기 위해 **양방향 반사 분포 함수(BRDF)** 필드에서 최적화된 텍스처 맵을 추출하는 데 중점을 둡니다.
    
또한, 3D 아바타 렌더링과 **동적 장면** 생성으로 연구가 확장되고 있습니다. **Animate124**와 **Dream-in-4D**는 **비디오 확산**을 4D 동적 장면 지원에 통합하여 3D 최적화 과정에서 활용합니다. **PAS**는 아바타 설정에 맞게 조정 가능한 3D 바디 포즈를 생성하고, **StyleAvatar3D**는 이미지를 기반으로 3D 아바타를 생성하며, **AvatarBooth**는 얼굴과 몸을 각각 따로 생성하기 위해 **두 개의 fine tunning된 diffusion model**을 사용합니다.





## Next Posting..
짧게 요약이 참 힘드네요,,
다음 포스팅에서는 개인화에서 사용되는 여러 테크닉을 다룹니다! 

>[1편, Methods ](https://daemini.github.io/posts/Personalization_1/)  
[2편, Tasks](https://daemini.github.io/posts/Personalization_2/) <- 현재 포스팅  
[3편, Techniques](https://daemini.github.io/posts/Personalization_3/)


