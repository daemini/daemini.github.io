---
title: "[Paper Reivew] Null-text Inversion for Editing Real Images using Guided Diffusion Models"
description: DDIM inversion의 trajectory를 pivot으로 삼아 null text embedding을 최적화하는 Pivotal inversion 방법론을 제안합니다.
toc: true
comments: true
# layout: default
math: true
date: 2024-12-09 18:30:00 +09:00
categories: [Deep Learning, Generative Model]
tags: [diffusion model, generative model, personalization, inversion]     # TAG names should always be lowercase
image: /posts/20241210_Null text inversion/teaser.jpeg
alt : Thumbnail
author: Daemin
---

> CVPR 2023. [[Paper](https://arxiv.org/abs/2211.09794)] [[Page](https://null-text-inversion.github.io/)]<br/> 
> Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, Daniel Cohen-Or <br/> 
> Google Research | Tel Aviv University <br/> 
> 17 Nov 2022 <br/> 


# TL;DR
Real image를 editing 하려면 inversion 과정이 필요합니다. 기존 DDIM inversion에 CFG를 적용하면 오차가 증폭되기 때문에, 이를 보완하고자 **pivotal inversion** 이라는 방법론을 제시합니다. 
Pivotal inversion은 DDIM inversion의 trajectory를 **pivot**으로 삼아 **null text embedding**을 최적화합니다. 

## 1. Introduction

![fig1](/posts/20241210_Null text inversion/fig1.png){: width="800" height="300"}


최근 텍스트 기반 이미지 편집을 하려면 **Inversion** 과정이 필요합니다. (그 이미지를 만드는 initial noise를 찾는 것)  이전 연구에서 DDIM inversion 방식이 제안되었지만, 이는 Unconditional diffusion model에서 디자인 되었기 때문에 CFG를 사용할 수 없다고 합니다. 

본 논문에서 저자들은 새로운 inversion scheme을 제안하여, 거의 완벽한 reconstruction 성능을 보였습니다. 

1. **Null-text optimization** : CFG는 conditional, unconditional diffusion step을 진행한 뒤 extrapolate 하는 방식입니다. 저자들은 unconditional 부분이 실질적인 영향을 가지는 것을 관찰하였고, empty text string을 저자들의 optimized embedding으로 바꾸는 **Null-text optimization**을 제안합니다.
2. **DDIM inversion** :  DDIM sampling을 거꾸로 하는 방식입니다. 각 step에 약간의 오차가 있긴 하지만, unconditional case에서는 잘 동작합니다. 하지만 CFG 방식을 사용하면 이 축적된 오차를 증폭시키기 때문에 문제가 된다고 합니다. 저자들은 DDIM inversion을 pivot으로 **Diffusion Pivotal Inversion**을 제안합니다. 이 Pivot 주위에서 최적화를 하기 때문에, 더 정확하게 동작한다고 합니다.

또한 저자들의 방법은 다른 방법들과 달리 model weight를 tuning하지 않으므로 모델 prior가 잘 유지된다고 합니다.


## 2. Related Work

Large-scale T2I model에서 개인화 된 object를 포함하는 이미지를 만들기 위해서는 Inversion 과정이 필요합니다. 

- [**Textual Inversion**](https://daemini.github.io/posts/Textual-Inversion/) : real image를 편집하는 성능이 안좋음.
- **SDEdit** : input image에 noise를 더하고, text-guided denoising step 진행. Input image의 detail을 잘 보존하지 못함.
- [**Blended Diffusion**](https://daemini.github.io/posts/Blended-Diffusion/) : Mask를 이용해 masking 영역만 편집하며 배경을 보존하는 방식. Mask가 필요하다는 단점. (저자들은 Text-only 방식을 target으로 하니까)
- **Text2live**: text 기반의 localized 편집 방법론을 제안했지만, Diffusion model 대신 CLIP을 사용해 복잡한 구조를 수정하는데 한계가 있음.
- [**Prompt-to-prompt**](https://daemini.github.io/posts/Prompt-To-Prompt/) : Cross-attention map을 injection 함으로써, Text 만으로 local, global 특징들을 조작할 수 있습니다. 하지만 이 방식도 **inversion**이 필요합니다.
- **Knn- diffusion** : Inversion 없이 local edit 하려는 시도를 했으나, Large-scale diffusion model에 비해 생성 퀄리티, 표현력이 떨어짐
- **DiffEdit** : DDIM inversion을 이용해 이미지 편집. 자동으로 생성되는 mask를 이용해서 distortion 막음.
- **Imagic, UniTune**: [Imagen](https://daemini.github.io/posts/Imagen/) 기반으로 좋은 editing 성능을 보였으나, Imagic은 새로운 편집마다 fine-tuning 과정이 필요하고, UniTune는 각 이미지마다 parameter search가 필요.

> 저자들의 방법론은 text-only, editing on real images, fine-tune 필요 x 


## 3. Method

Goal : Real Image, $$ \mathcal{I} $$를 text-guidance만을 이용해 편집된 이미지, $$ \mathcal{I}^* $$를 얻는 것! 이를 위해 Prompt-to-Prompt 방법론 사용. (source prompt, $$ \mathcal{P} $$와 edited prompt $$ \mathcal{P}^* $$로 editing guide)

이런 editing 방법은 $$ \mathcal{I} $$를 model output domain으로 invert 하는 과정이 선행되어야 합니다.  저자들의 핵심 관찰은 다음과 같습니다.

1. DDIM inversion은 CFG가 적용되면 성능이 좋지 않지만... good starting point!
2. Unconditional null text embedding 을 최적화하는 것이 reconstruction 하는데 도움이 된다.

### 3.1. Background and Preliminaries



#### T2I Diffusion model
Random noise vector $$ z_t $$, text prompt $$ \mathcal{P} $$를 입력으로 받아, Output Image $$ z_0 $$를 얻는 것. 이때 모델은 다음의 objective를 갖습니다. 

$$
\min_\theta \mathbb{E}_{z_0, \epsilon \sim \mathcal{N}(0, I), t \sim \text{Uniform}(1, T)} \|\epsilon - \epsilon_\theta(z_t, t, C)\|^2
$$
이때 $$ C = \psi(\mathcal{P}) $$는 text embedding 입니다. 

#### DDIM Sampling
Inference 일 때는, $$ T $$ step 동안 점차적으로 noise를 제거하는 과정을 거칩니다. Deterministic DDIM sampling 은 다음과 같습니다.

$$
z_{t-1} = \underbrace{\sqrt{\frac{\alpha_{t-1}}{\alpha_t}} z_t}_{\text{scaled previous variable}} + {\left( \sqrt{\frac{1}{\alpha_{t-1}} - 1} - \sqrt{\frac{1}{\alpha_t} - 1} \right) \cdot \underbrace{\epsilon_{\theta}(z_t, t, C)}_{\text{predicted noise}}}.
$$

#### Classifier-free guidance
Conditional, un-conditional 방식을 extrapolate하는 방법. Null text embedding $$ \varnothing = \psi("")$$라 하면 CFG는 다음과 같이 표현할 수 있습니다.

$$
\tilde{\epsilon}_{\theta}(z_t, t, C, \varnothing) = w \cdot \underbrace{\epsilon_{\theta}(z_t, t, C)}_{\color{red}{\text{Conditional}}} + (1 - w) \cdot \underbrace {\epsilon_{\theta}(z_t, t, \varnothing)}_{\color{blue}{\text{Unconditional}}}
$$



#### DDIM inversion
ODE를 거꾸로 뒤집은 형태가 DDIM Inversion 방법입니다. $$ z_0 \rightarrow z_T $$, 즉 original 이미지에서 noisy한 방향으로 진행됩니다.

$$
z_{t+1} = \sqrt{\frac{\alpha_{t+1}}{\alpha_t}} z_t + \left( \sqrt{\frac{1}{\alpha_{t+1}} - 1} - \sqrt{\frac{1}{\alpha_t} - 1} \right) \cdot \epsilon_{\theta}(z_t, t, C).
$$

### 3.2. Pivotal Inversion
기존 Inversion 방법은 random noise를 반복적으로 사용하여, 매번 이미지를 noise vector에 mapping -> 비효율적인 접근법

저자들은 single noise vector만을 사용하는 방법을 고민하다, pivotal noise vector 주위에서 local하게 optimization 하는 방법을 택했다고 합니다. 

SD 에서 CFG guidance $$ w $$ (default 7.5)를 사용하는데, DDIM inversion의 작은 오차가 증폭되어 noise가 Gaussian distribution에서 벗어날 수 있고, 이는 editability를 낮춘다고 합니다.

DDIM inversion이 image encoding $$ z_0 $$로부터 noise vector $$ z_T^* $$사이 trajectory를 제공해줍니다. 저자들은 DDIM inversion $$ w =1 $$을 정확하지는 않지만, rough approximation을 제공해주기 때문에 이를 **pivot trajectory**로 사용합니다. 이 pivot trajectory를 기준으로, $$ w>1 $$에서 최적화를 거칩니다. 

이때, 각 timestep마다 단계적으로 최적화를 거치는데, time step $$ t $$는 $$ t+1 $$의 최적화 결과를 시작점으로 사용합니다. 이렇게 함으로써 새로운 경로가 $$ z_0 $$에 근접하도록 설계했다고 합니다.
$$
\min \| z_{t-1}^* - \underbrace{z_{t-1}}_{\color{red}{\text{intermediate result}}} \|_2^2,
$$

![pivotal](/posts/20241210_Null text inversion/pivotal.png){: width="800" height="300"}



### 3.3. Null-text optimization

저자들의 핵심 관찰은 CFG에서 unconditional prediction이 결과에 미치는 영향이 크다는 것입니다. 저자들은 각 input image에 대해 null text embedding $$ \varnothing $$만을 최적화를 합니다. 

한 번의 inversion 과정으로, 여러 editing task가 가능하며, 모델 weight update 없이 효율적이며, 직관적 editing이 가능하다고 합니다.


1. **Global null-text optimization**: 
-   단일 **unconditional embedding ($$ \varnothing $$)** 최적화.  
-   간단하지만 reconstruction 품질에는 한계.

2. **Per-timestamp null-text optimization**:  
-   각 타임스탬프 t에 대해 **독립적인 null embedding ($$ \varnothing_t $$​)** 최적화.
-   $$ \varnothing_t $$는 이전 타임스텝의$$ \varnothing_{t+1} $$으로 초기화.
-   reconstruction 품질을 상당히 향상.


전체 알고리즘은 다음과 같습니다.

$$ w = 1 $$로 DDIM inversion $$ \rightarrow $$ $$ z_T^*, ..., z_0^*$$ 얻고, $$ \bar z_T = z_t $$로 initialize하고, 다음을 이용해 최적화를 진행합니다.

$$
\min_{\varnothing_t} \| z_{t-1}^* - \underbrace{z_{t-1}(\bar{z}_t, \varnothing_t, C)}_{\color{red}{\text{DDIM sampling results}}} \|_2^2,
$$

Early stopping을 적용하면 A100 GPU로 ~1 min 정도 시간이 걸리며, $$ \bar z_T = z_T^* $$, $$ \{\varnothing\}_{t=1}^{T} $$ 를 이용해 real image를 Edit 할 수 있다고 합니다.


![al](/posts/20241210_Null text inversion/al.png){: width="800" height="300"}

## 4. Ablation Study

![fig4](/posts/20241210_Null text inversion/fig4.png){: width="800" height="300"}

- **DDIM Inversion** : DDIM Inversion은 최적화 시작점으로 사용되며, Lower bound로 간주

- **VQAE** : 특정 모델에만 적용 가능하며, 일반 알고리즘 설계를 벗어남. Upper bound로 간주

- **Ours** : 적은 iteration으로도 높은 quality의 inversion

1. **Random Pivot** :DDIM inversion으로 초기화 대신 random Gaussian 노이즈를 사용하여 초기 경로 설정 시, 초기 오차가 커지고 수렴 속도가 느려짐.

2. **Robustness to different input captions** : 입력 캡션이 이미지와 잘 맞지 않아도(random caption 사용), 최적화는 VQAE 상한선에 근접한 재구성을 제공. 단, 텍스트 기반 편집에는 적절한 캡션이 필요하며, 자동 캡션 생성 모델을 사용해도 유효.
![fig12](/posts/20241210_Null text inversion/fig12.png){: width="800" height="300"}

3.  **Global Null-text Embedding**: 모든 타임스탬프에 대해 single **null-text embedding**을 최적화. 수렴이 느리고 성능이 낮다고 합니다.

4. **Textual Inversion** : text embedding $$ C = \psi(P) $$를 random noise로 초기화하고, textual inversion 방식으로 최적화. 수렴 속도가 느리고 재구성 품질이 저조.

5. **Textual inversion with a pivot** : Textual Inversion에 pivotal inversion을 결합. 재구성 품질이 크게 향상되어 제안된 방법과 유사한 성능 달성했지만, 저자들의 방법보다 editability는 낮다고 합니다.


6. **Null-text Optimization without Pivotal Inversion** : pivotal inversion을 사용하지 않고, 랜덤 노이즈 벡터를 사용하여 Null-text Optimization 수행. 
Null-text Optimization이 완전히 실패하며, DDIM Inversion보다도 낮은 성능을 보였다고 합니다. 이는 Null-text Optimization은 표현력이 제한적이므로 피벗 기반 접근에 의존하기 때문으로 생각할 수 있습니다.

![fig15](/posts/20241210_Null text inversion/fig15.png){: width="800" height="300"}


![fig14](/posts/20241210_Null text inversion/fig14.png){: width="800" height="300"}

## 5. Results

-   단 한 번의 Inversion으로 머리 색상, 안경, 표정, 배경, 조명 등을 수정 가능.
![fig2](/posts/20241210_Null text inversion/fig2.png){: width="500" height="300"}


-   특정 단어의 효과를 증폭하거나 약화 가능.
![fig5](/posts/20241210_Null text inversion/fig5.png){: width="500" height="300"}


### 5.1. Comparisons
#### Qualitative Comparison
![fig6](/posts/20241210_Null text inversion/fig6.png){: width="500" height="300"}

- **Text-only** 
	1.  **VQGAN+CLIP**  : 비현실적인 결과 생성.
	2.  **Text2Live** : 텍스처 편집은 가능하지만 구조적인 객체 편집(예: 보트 추가)에 실패.
	3.  **SDEdit** : 세부 사항이 포함된 얼굴과 같은 영역에서 아티팩트 발생:


![fig7](/posts/20241210_Null text inversion/fig7.png){: width="500" height="300"}

-   **Mask-based**: 
	4. **Glide**
	5. **Blended-Diffusion**
	6. **Stable Diffusion Inpaint**
    
> -   마스크 절차로 인해 중요한 구조적 정보 손실.
> -   inpainting의 한계로 세부 사항이 잘 보존되지 않음.

-   **model tuning based**: 
![fig17](/posts/20241210_Null text inversion/fig17.png){: width="800" height="300"}

	7. **Imagic**  (Imagen 모델 기반) : Imagen 모델 사용 시 높은 품질의 편집 가능하지만, 원본 세부 사항 보존 실패(예: 아기 ID, 배경의 컵).

#### Quantitative Comparison
Real image editing을 평가할만한 효과적인 metric이 없으므로, user study 진행

![tab1](/posts/20241210_Null text inversion/tab1.png){: width="500" height="300"}

### 5.2. Evaluating Additional Editing Technique

**Prompt-to-Prompt**뿐만 아니라 **SDEdit**와 결합했을 때도 성능 향상을 확인.
![fig8](/posts/20241210_Null text inversion/fig8.png){: width="500" height="300"}

제안된 방법이 아기의 ID와 같은 세부 사항을 더 잘 보존.



## 6. Limitations


1.   **추론 시간**:
	-   단일 이미지 역전환에 약 **1분 소요** (GPU 기준).
    -   이후 편집 작업은 10초 내외로 수행 가능하지만, **실시간 애플리케이션**에는 부적합.
2. **Stable Diffusion 및 Prompt-to-Prompt 관련 한계**: 
    -   **VQ Auto-encoder**: 특히 얼굴을 포함한 이미지에서 아티팩트 생성.

    -   **attention 맵 정확도**: Stable Diffusion의 attention 맵은 Imagen에 비해 정확도가 떨어지는데, 단어가 잘못된 영역과 연관될 수 있는 문제.

    -   **구조적 수정 제한**: 예: 앉아 있는 개를 서 있는 개로 바꾸는 작업은 어려움.

3.  **범용적 프레임워크 한계**:
	-   특정 모델이나 편집 기법의 문제는 제안된 방법과 독립적이며, 향후 개선될 가능성 있음.

## 7. Conclusions
저자들의 주요 기여는 다음과 같습니다.

1. **Pivotal Inversion**: DDIM Inversion을 사용해 노이즈 코드 시퀀스를 계산해 fixed pivot으로 사용
2. **Null-text optimization** : Null-text embedding을 최적화하여 Classifier-Free Guidance에서 발생하는 재구성 오류 보정.

$$ \rightarrow $$ 이를통해 real image를 모델 튜닝 없이 직관적인 텍스트 기반 편집을 가능하게 함.

