---
title: "[Paper Reivew] Multi-Concept Customization of Text-to-Image Diffusion"
description: 새로운 (single- or multiple-) concept을 Cross-attention layer의 Key, Value matrix만 fine-tune하여 학습할 수 있는 CustomDiffusion을 제안합니다.
toc: true
comments: true
# layout: default
math: true
date: 2025-01-07 18:10:00 +09:00
categories: [Deep Learning, Generative Model]
tags: [diffusion model, generative model, personalization, multi concept]     # TAG names should always be lowercase
image: /posts/20250107_CustomDiffusion/teaser.jpeg
alt : Thumbnail
author: Daemin
---

> CVPR 2023 [[Paper](https://arxiv.org/abs/2212.04488)] [[Page]](https://www.cs.cmu.edu/~custom-diffusion/)]<br/>
> Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, Jun-Yan Zhu <br/>
> Carnegie Mellon University, Tsinghua University,  Adobe Research<br/>
> 20 Jun, 2023 <br/>



# TL;DR
새로운 (single- or multiple-) concept을 Cross-attention layer의 Key, Value matrix만 fine-tune하여 학습할 수 있는 CustomDiffusion을 제안합니다.


## 1. Introduction
유저가 제공하는 몇 장의 이미지로, 새로운 이미지를 생성하는 것은 많이 연구되고있는 분야입니다. 이전 연구들의 문제점을 크게 보면 다음과 같습니다. 

1. Language drift :
예를들어 *"moongate"*라는 concept을 학습하면, *"moon"*, *"gate"*등의 concept을 잊는다고 합니다. 

3. Overfit에 취약


저자들은 이러한 문제를 해결하면서, 더 어려운 문제인 compositional fine-tuning 문제를 다루는 **CustomDiffusion**을 제안합니다. 아주 적은 parameter (Cross-attention layer의 **Key, value matrix**)만을 **fine-tune**하면서 효율적이고 성능이 좋은 방법이라고 주장합니다. 또한 model forgetting을 막기위해 저자들은 **target text와 유사한 real image**를 Retrieval하여 regularization set으로 활용하는 방법도 제안합니다. 또한 multiple-concept을 다룰 때, 동시에 학습하거나 따로 학습한 모델을 합치는 방법 모두 사용할 수 있다고 합니다. 

저자들의 	contribution을 요약하자면...

1. Cross-attention layer의 **Key, value matrix**만을 fine-tuning하여 효율적으로 concept 학습
2. Target text와 유사한 text를 갖는 real image를 regularization set으로 활용.
3. Jointly training 혹은 따로 학습한 weight를 optimization을 통해 Multiple concept 학습 가능.

![fig1](/posts/20250107_CustomDiffusion/fig1.png){: width="800" height="300"}


## 2. Related Work

생략

## 3. Method

![fig2](/posts/20250107_CustomDiffusion/fig2.png){: width="800" height="300"}
_핵심 아이디어 요약_

### 3.1. Single-Concept Fine-tuning
저자들이 Rate of change of weights, $$ \Delta_l = \frac{\|\theta_l' - \theta_l\|}{\|\theta_l\|}
 $$를 분석한 결과 전체 파라미터 수의 아주 작은 부분을 차지하고 있는 cross-attention layer의 변화가 가장 컸다고 합니다. 
 
 ![fig3](/posts/20250107_CustomDiffusion/fig3.png){: width="500" height="300"}

이런 관찰에서 저자들은 Cross-attention의 condition으로 들어가는 text feature가 Key, Value matrix에만 영향을 받으므로 Key, Value matrix만 fine-tuning했다고 합니다. 

 ![fig4](/posts/20250107_CustomDiffusion/fig4.png){: width="500" height="300"}

뒤의 실험 결과에서 Key, Value matrix만 fine-tuning해도 충분하다는 것을 보입니다. 

또한 저자들은 fIne-tuning 방법의 문제점인 language drift(*"moongate"*라는 concept을 학습하면, *"moon"*, *"gate"*등의 concept을 잊는 것)을 해결하기 위해 fine-tuning 과정에서 target image와 비슷한 image를 LAION-400 $$ \text{M} $$에서 retrieval하여 regularization에 사용했다고 합니다.

 ![fig5](/posts/20250107_CustomDiffusion/fig5.png){: width="500" height="300"}


### 3.2 . Multiple-Concept Compositional Fine-tuning

1. **Joint training on multiple concepts**
각 concept의 dataset을 모아 jointly training 하는 방식. 
2. **Constrained optimization to merge concepts**
이미 각 concept을 학습한 model이 있으면 이를 적절히 합쳐서 Multiple Concept을 학습할 수 있다고 합니다. 


$$
\widehat{W} = \arg\min_{W} \|WC_{\text{reg}}^\top - W_0C_{\text{reg}}^\top\|_F \quad \text{s.t.} \quad WC^\top = V,
$$

$$
\quad \text{where} \quad C = \begin{bmatrix} c_1 & \cdots & c_N \end{bmatrix}^\top
$$

$$
\text{and} \quad V = \begin{bmatrix} W_1c_1^\top & \cdots & W_Nc_N^\top \end{bmatrix}^\top.
$$


위 식은 Lagrange multipliers 방법을 통해 closed-form solution을 얻을 수 있다고 합니다.

$$
\widehat{W} = W_0 + \mathbf{v}^\top \mathbf{d}, 
\quad \text{where} \quad \mathbf{d} = C(C_{\text{reg}}^\top C_{\text{reg}})^{-1}, 
\quad \text{and} \quad \mathbf{v}^\top = (V - W_0C^\top)(\mathbf{d}C^\top)^{-1}.
$$

직관적으로 생각하면, 위 방법은 target caption에 있는 단어를 각 concept을 fine-tune한 모델의 결과에 일관적으로 mapping하도록 original matrix를 update하는 것이라고 합니다.

## 4. Experiments

### 4.1. Single-Concept Fine-tuning Results

- **Qualitative evaluation** 

 ![fig6](/posts/20250107_CustomDiffusion/fig6.png){: width="800" height="300"}

Textual Inversion보다 성능이 우수하며, DreamBooth와 유사한 성능을 보이지만 훈련 시간과 모델 저장 용량 측면에서 더 효율적입니다. (훈련 속도 약 5배 빠름, 75MB vs 3GB)

- **Quantitative evaluation** 
 ![fig8](/posts/20250107_CustomDiffusion/fig8.png){: width="800" height="300"}
 
Custom Diffusion은 모델의 모든 가중치를 세부 조정한 DreamBooth 방식과 유사한 성능을 보이며, 계산 및 시간 효율성이 더 높음.


 ![tab1](/posts/20250107_CustomDiffusion/tab1.png){: width="600" height="300"}
 
### 4.2. Multiple-Concept Fine-tuning Results

 ![fig7](/posts/20250107_CustomDiffusion/fig7.png){: width="800" height="300"}
 
### 4.3. Human Preference Study

-   **Text-alignment**: "어느 이미지가 텍스트와 더 일치하는가?"
-   **Image-alignment**: "어느 이미지가 목표 이미지와 더 잘 일치하는가?"

 ![tab2](/posts/20250107_CustomDiffusion/tab2.png){: width="600" height="300"}
 
### 4.4 Ablation and Applications
 ![tab3](/posts/20250107_CustomDiffusion/tab3.png){: width="600" height="300"}


- **Generated images as regularization (Ours w/ Gen)**

Regularization set으로 generated image 사용한 결과 성능이 떨어졌음. 저자들의 주장대로 real-image를 retrieval 하여 Regularization set으로 사용하는 것이 효과적.

- **Without regularization dataset (Ours w/o Reg)**
정규화 데이터셋 없이 학습을 진행한 결과 language drift가 심해졌다고 합니다. 

![fig5](/posts/20250107_CustomDiffusion/fig5.png){: width="600" height="300"}


- **Applications**

![fig9](/posts/20250107_CustomDiffusion/fig9.png){: width="800" height="300"}
특정 스타일에 대해 fine-tuning 가능.

![fig10](/posts/20250107_CustomDiffusion/fig10.png){: width="600" height="300"}
fine-tuned 모델을 이용해 image editing 가능.
