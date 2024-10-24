---
title: "[Paper Reivew] Style Aligned Image Generation via Shared Attention"
description: CLiC은 단일 이미지에서 local한 visual concept을 학습하고, 이를 다양한 목표 객체에 적용하는 In-Context Concept Learning 방법론을 제안합니다.
toc: true
comments: true
# layout: default
math: true
date: 2024-10-24 16:30:00 +09:00
categories: [Deep Learning, Generative Model]
tags: [diffusion model, generative model, personalization, consistant style, style transfer]     # TAG names should always be lowercase
image: /posts/20241024_StyleAlign/thumbnail.jpeg
alt : Thumbnail
author: Daemin
---


> CVPR 2024. Oral [[Page]](https://style-aligned-gen.github.io)  
> Amir Hertz, Andrey Voynov, Shlomi Fruchter, Daniel Cohen-Or   
> Google Research, Tel Aviv University     
> 11 Jan 2024      

# Abstract 
-   **StyleAligned**는 대규모 텍스트-이미지(T2I) 모델에서 **스타일 일관성**을 유지하기 위해 설계된 새로운 기법입니다.
-   기존 T2I 모델들은 동일한 스타일을 유지하는 이미지 생성이 어려워 **fine-tuning**과 **수동 조작**이 필요했습니다.
-   StyleAligned는 **attention sharing**을 통해 여러 이미지 간의 스타일을 정렬시키는 방법으로, **추가적인 최적화나 학습 없이** 스타일 일관성을 유지할 수 있습니다.


## TL;DR
- Fine tuning 없이 (간단한) attention sharing 연산을 통해 생성되는 이미지를 같은 스타일로 생성할 수 있습니다.
- Diffusion Inversion을 이용해 Style Transfer Task에도 활용할 수 있지만, Inversion 방식의 한계가 있다!


![teaser3](/posts/20241024_StyleAlign/teaser3.jpg){: width="800" height="300"}

## 1. Introduction

T2I Mode의 성능이 좋아지면서 제공한 이미지(Reference Image)와 같은 스타일의 이미지를 만들려는 연구가 진행되고 있습니다. 최근의 연구는 T2I 모델을 fine-tuning하는 방식을 사용하지만, 비효율적이라고 합니다.

저자들은 이미지들을 같은 스타일로 생성하는 방법론 **StyleAligned**를 제안합니다. **StyleAligned**는 fine-tuning이 필요하지 않으며, 간단히 attention sharing 연산을 추가하는 것 만으로도 일관적인 style로 이미지를 생성할 수 있다고 합니다. 게다가 Diffusion Inversion을 사용하면 저자들의 방법은 reference의 이미지와같은 style의 이미지를 만드는 것도 가능하다고 합니다.



## 2. Related Work

### 2.1. Attention Control in diffusion models
[Prompt-to-prompt] 연구에서는 cross, self attention map이 diffusion 과정에서 layout과 content를 결정한다는 것을 보였으며, attention map으로 이미지 생성을 control할 수 있다고 합니다. 

이후 attneion layer를 수정하면서 이미지의 fidelity, diversity를 향상시키려는 연구, attention control을 이용해 이미지 editting을 하려는 연구가 진행되고 있습니다.

특히, 저자들의 방법론은 다양한 structure에 적용할 수 있는 장점이 있다고 합니다.

### 2.2. Style Transfer
Style Transfer는 reference 이미지의 스타일로, target image를 만드는 task입니다. 방법론으로는 pre-trained 모델의 prior를 활용하기도, attention을 injecting하기도 하지만, 저자들의 방법론에서는 **AdaIN**을 활용했다고 합니다. 

> AdaIn(Adaptive Instance Normalization)을 간단히 설명하자면  reference의  이미지로 target 이미지를 normalization 하는 방법입니다. 

### 2.3. T2I Personalization
새로운 visual concept으로 이미지를 생성하기 위해 많은 연구들은 optimization technique을 찾는 것에 집중했습니다. 

StyleDrop의 경우 adapter를 fine tuning하여 style personalization을 하려고 시도했지만, single 이미지로 학습했기 때문에 일관적인 이미지를 생성하는데 어려움을 겪는다고 합니다.

이와 달리 저자들의 방법은 최적화 과정이 없이도 일관적인 이미지를 생성한다는 점에서 훌륭한 방법이라고 합니다.


## 3. Method
잠깐 정리하자면 저자들의 목표는 다음과 같습니다.

> **일관적인 Style**을 유지하면서,  text prompt $$ y_1 ... y_n $$와 일치하는 이미지 $$ \mathcal I_1 ... \mathcal I_n $$을 얻는 것! 


### 3.1. Naïve Approach
이를 달성하기 위한 naïve한 방법은 text-prompt에 같은 style을 적는 것일 겁니다. 

![text_prompt](/posts/20241024_StyleAlign/text_prompt.png){: width="800" height="300"}

뭐... 당연히(?) 잘 될리가 없습니다. 

이를 해결하기 위한 저자들의 핵심 방법론은 생성되는 이미지들 간 self-attention mechanism을 통해 target과 reference가 서로 communicate 하게 만드는 것입니다.

### 3.2. Full attention Sharing

$$ \mathcal I_i $$의 deep feature $$ \phi_i$$를 projection하면 $$ Q_i, K_i, V_i $$를 얻습니다. 이를 이용해 **full attention sharing**을 할 수 있습니다.

$$ 
Attention(Q_i, K_{1...n}, V_{1...n})
$$

이때 $$ K_{1...n} = \begin{bmatrix} K_1 \\ K_2 \\ \vdots \\ K_n \end{bmatrix} \quad \text{and} \quad V_{1...n} = \begin{bmatrix} V_1 \\ V_2 \\ \vdots \\ V_n \end{bmatrix}.
 $$


![bottom](/posts/20241024_StyleAlign/bottom.png){: width="800" height="300"}

하지만.... Content leakage가 발생합니다. (유니콘과 공룡색이 섞이는 등) 게다가 prompt set간의 이미지 diversity가 떨어진다는 문제가 있습니다. 

### 3.3. Shared Attention W/O AdaIN
> '서로 content가 섞이는 게 문제인거면 한 장의 attention만 뽑아서 공유하면 나머지 사진은 그 사진이랑 똑같은 Style로 나오는거 아닌가?!' 라는 생각으로 하지 않았을까..

Content leakage, less diversity 문제를 해결하기 위해 저자들은 batch의 이미지 중 한장의 attention만 sharing 하는 방식을 실험했다고 합니다.

![middle](/posts/20241024_StyleAlign/middle.png){: width="800" height="300"}

Diverse한 이미지를 만드는 것은 어느정도 가능했지만, Style이 일관적이지 않았습니다. 저자들은 Reference image에서 Target image로의 attention flow가 약했기 때문이라고 추측합니다.


### 3.4. Shared Attention W/ AdaIN
> 'attention flow를 강하게 해주면 되겠네!' AdaIN은 reference의 Q, K를 사용해 target의 Q, K를 Normalize하는 방법! 

![fig4](/posts/20241024_StyleAlign/fig4.png){: width="600" height="300"}

$$
\text{AdaIN}(x, y) = \sigma(y) \left( \frac{x - \mu(x)}{\sigma(x)} \right) + \mu(y)
$$

즉, $$ x $$를 normalization하고, $$ y $$ 평균과 표준편차를 이용해 정규화 하는 방법. (y의 style 정보를 x에 넣어주는 효과가 있다고 합니다.)

따라서 저자들은 reference의 Query, Key를 이용해 target의 Query, Key를 정규화 합니다.

$$ 
\hat Q_t = \text {AdaIN}(Q_t, Q_r),  \hat K_t = \text {AdaIN}(K_t, K_r)
$$

최종적인 Attention은 다음과 같습니다. 
$$ 
\text{Attention}(Q̂_t, K_{rt}^T, V_{rt}) \\
\quad \text{Where} \quad K_{rt} =\begin{bmatrix}
K_r \\K̂_t
\end{bmatrix} \quad \text{and}  \quad V_{rt} = \begin{bmatrix}
V_r \\V_t
\end{bmatrix}  
$$


![top](/posts/20241024_StyleAlign/top.png){: width="800" height="300"}

diverse 한 이미지를 얻으면서도 style이 일관적인 것을 확인할 수 있습니다.

## 4. Evaluation
저자들은 T2I Model로 Stable Diffusion XL 모델을 사용했다고 합니다. ChatGPT로 text-prompt를 만들어 사용.
1. CLIP : Text-Image Alignment 측정 
2. DINO : Reference Image와 Target Image간 유사도 측정 (Style Consistentcy)



### 4.1. Ablation Study
저자들은 Full Attention Sharing, Without Query-Key AdaIN, Full Method(StyleAlgined)로 ablation study를 진행했습니다.

![fig6](/posts/20241024_StyleAlign/fig6.png){: width="600" height="300"}

오른쪽 위로 갈 수록 스타일 일관성을 지키면서 text prompt에 맞는 이미지를 생성하는 좋은 모델입니다.

-   **Full Attention Sharing**:
모든 이미지 간에 완전한 주의 공유를 했을 때, 스타일은 일관되지만 각 이미지의 콘텐츠가 서로 혼합되어 콘텐츠 누출(Content Leakage)이 발생하였습니다. 
즉, 하나의 이미지에서 색이나 구조가 다른 이미지로 확산되어 이미지 간의 개별적인 다양성이 감소했습니다.
-   **Without Query-Key AdaIN**:
쿼리와 키에 AdaIN을 적용하지 않을 경우, 스타일 일관성이 떨어지고 이미지 세트 간에 통일된 스타일을 유지하는 데 어려움이 있었습니다.

-   **Full Method**:
제안된 방법인 참조 이미지와 타겟 이미지 간의 제한된 주의 공유 및 쿼리-키 AdaIN 적용을 통해, 이미지 세트 내에서 스타일 일관성을 유지하면서도 세트 간 다양성을 확보할 수 있었습니다.


## 4.2. Comparisions

-   **실험 방법**:
    -   **StyleDrop**과 **DreamBooth**는 각 세트의 첫 번째 이미지를 사용해 학습한 후, 추가 이미지들을 생성하는 방식으로 실험이 이루어졌습니다.
    -   StyleDrop의 두 가지 구현(SDXL 기반과 비공식 버전)과 DreamBooth의 LoRA 변형이 실험에 사용되었습니다.
-   **정성적 비교 (Qualitative Comparison)**:
    -   StyleAligned는 컬러 팔레트, 드로잉 스타일, 구도, 포즈와 같은 스타일 속성을 더욱 일관되게 유지하며 이미지 세트를 생성할 수 있었습니다.
    -   반면, 다른 방법들은 학습된 스타일의 콘텐츠가 새로 생성된 이미지에 반복적으로 나타나는 **콘텐츠 누출(Content Leakage)** 현상을 보였습니다. 예를 들어, DreamBooth와 StyleDrop에서 훈련된 참조 이미지의 일부 콘텐츠가 다른 이미지에서도 반복되었습니다.
-   **정량적 비교 (Quantitative Comparison)**:
    -   StyleAligned는 다른 방법들과 비교했을 때, 스타일 일관성과 텍스트 정합성 모두에서 우수한 평가를 받았습니다.
    -   사용자 연구에서도 StyleAligned가 다른 방법들에 비해 스타일 일관성 및 텍스트 정합성 측면에서 더 높은 점수를 받았습니다.


## 4.3. Additional Resutls

#### **Style Alignment Control**
-  저자들의 Shared-Attention을 Self-Attention 레이어 일부에만 적용하여 스타일 정렬의 정도를 제어할 수 있는 방법을 제시하였습니다.

![fig8](/posts/20241024_StyleAlign/fig8.png){: width="600" height="300"}


### **StyleAligned from an Input Image**

-   입력 이미지와 텍스트 설명을 사용해 **DDIM Inversion**을 적용하여 해당 이미지의 디노이징 궤적을 역으로 추정하였습니다. 이 역으로 추정된 궤적을 사용해 입력 이미지와 스타일이 일치하는 새로운 콘텐츠를 생성하였습니다.

> 예를 들어, “A render of a house with a yellow roof”이라는 프롬프트로 DDIM Inversion을 하고, 'roof'를 'car', 'cat', 'cactus' 등으로 바꿔서 style 일관성을 유지하는 이미지 생성

![fig9](/posts/20241024_StyleAlign/fig9.png){: width="600" height="300"}


-   이 방법은 최적화 없이도 특정 입력 이미지의 스타일을 기반으로 새로운 이미지를 생성할 수 있었으나, DDIM Inversion 과정에서 실패하거나 궤적 오류가 발생할 수 있는 점이 단점으로 지적되었습니다.

![fig13](/posts/20241024_StyleAlign/fig13.png){: width="800" height="300"}

### **Shared Self-Attention Visualization**
![fig10](/posts/20241024_StyleAlign/fig10.png){: width="600" height="300"}

Reference Image(기차) style로 만든 car, bull에서 self-attention map을 나타낸 것입니다. 주목할만한 점은, Query에 의미적으로 가까운 지점을 reference image에서 보고 있다는 점입니다. 이는 self-attention token이 단순한 전역 스타일 전이가 아니라, **의미적으로 적절한 방식**으로 스타일을 매칭한다는 것을 의미합니다.

![fig11](/posts/20241024_StyleAlign/fig11.png){: width="600" height="300"}

또한 Shared attention map의 주요 성분들을 시각화한 결과, 이미지 내 의미적으로 관련된 영역(예: 몸통, 머리, 배경 등)이 강조되는 것을 확인할 수 있었습니다.

**유명 이미지의 문제**:

매우 유명한 이미지에서 스타일 전이를 수행할 경우, 모델은 타겟 프롬프트를 완전히 무시하고 참조 이미지와 거의 동일한 결과를 생성하는 현상이 관찰되었습니다.

이는 유명한 이미지의 디노이징 출력이 매우 높은 신뢰도와 활성화 크기를 가지기 때문으로, **Self-Attention Sharing** 과정에서 참조 이미지의 Key가 대부분의 attention을 차지하는 현상이 발생했다고 합니다.

이를 해결하기 위해 attention scaling을 rescaling($$ \lambda < 1 $$)하여 사용했다고 합니다.

![fig15](/posts/20241024_StyleAlign/fig15.png){: width="600" height="300"}

$$ \lambda  = 1 $$ 일 때, Reference image와 거의 유사한 이미지를 얻지만, 적절히 rescale 했더니 비슷한 스타일의 다른 이미지를 생성할 수 있었습니다.


### **Integration with Other Methods**
-   StyleAligned는 훈련이나 최적화 없이도 **ControlNet**, **DreamBooth**, **MultiDiffusion**과 같은 다른 확산 기반 방법들과 쉽게 결합할 수 있었습니다.
-   예를 들어, ControlNet과 결합하여 깊이 맵에 조건부로 스타일 정렬 이미지를 생성하거나, MultiDiffusion을 사용해 여러 스타일을 공유하는 파노라마 이미지를 생성하는 등의 예가 제시되었습니다.

![controlnet](/posts/20241024_StyleAlign/controlnet1.jpg){: width="800" height="300"}
_controlnet_

![DreamBooth](/posts/20241024_StyleAlign/db.jpg){: width="800" height="300"}
_DreamBooth_

![multidiffusion](/posts/20241024_StyleAlign/multidiffusion.jpg){: width="800" height="300"}
_multidiffusion_


## 5. Conclusion
StyleAligned : Attention전 **AdaIN**을 적용하는 Attention Sharing을 제안해, fine-tune 없이 일관적인 style의 이미지를 생성할 수 있습니다.

In the future... 저자들은 StyleAligned의 확장성과 생성된 이미지 간의 형상 및 외형 유사성을 더 제어할 수 있는 방법을 연구할 것이라 합니다. 

특히 현재의 Diffusion inversion 방법의 한계를 극복하기 위해, 스타일 일관성을 가진 데이터를 생성 ->  텍스트-이미지 모델을 학습시키는 데 사용할 가능성을 모색할 계획이라고 합니다.


