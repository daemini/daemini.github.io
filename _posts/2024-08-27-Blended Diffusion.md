---
title: "[Paper Reivew] Blended Diffusion for Text-driven Editing of Natural Images"
description: 사전학습된 CLIP과 DDPM을 이용하여, 배경을 보존하면서 마스킹 부분을 text-prompt에 맞게 이미지를 생성하는 방법론을 제시한 논문입니다.
toc: true
comments: true
# layout: default
math: true
date: 2024-08-27 16:26:00 +09:00
categories: [Deep Learning, Generative Model]
tags: [diffusion model, generative model, blended, cvpr, t2i]     # TAG names should always be lowercase
image: /posts/20240827_Blended/thumbnail.jpeg
alt : Thumbnail
---


> CVPR 2022. [[Paper]](https://arxiv.org/abs/2111.14818) [[Github]](https://github.com/omriav/blended-diffusion) [[Page](https://omriavrahami.com/blended-diffusion-page/)] <br/>
> [Omri Avrahami](https://omriavrahami.com/)1,  [Dani Lischinski](https://www.cs.huji.ac.il/~danix/)1,  [Ohad Fried](https://www.ohadf.com/)2,<br/>
> The Hebrew University of Jerusalem | Reichman University<br/>
> 29 Nov 2021 <br/>



# TL;DR
**Natural language**와 **ROI mask**를 사용하여, **local editing**의 방법론을 제시한 논문입니다. **CLIP**을 활용하여 사용자가 제공한 텍스트 정보를 얻고, **DDPM**을 이용해 이미지를 생성합니다. 

masking 되어 수정된 부분(foreground)과 겉보기에는 수정이 안된 것 같은 이미지(background)를 자연스럽게 합치기 위해, **입력 이미지의 noisy한 버전**과 **local text-guided diffusion latent**를 공간적으로 **blend**하는 방법론입니다.

이와 같은 방법으로 저자들은 이전 연구들의 결과를 뛰어넘었으며, new object to an image, removing/replacing/altering existing objects, background replacement, image extrapolation와 같은 분야에 적용하며 실험 결과를 입증했습니다.

## 1. Introduction
기존 GAN-based text-driven 이미지 조작 모델들은 인상적인 결과를 보여주긴 했지만 다음과 같은 문제점이 지적되어왔다.

- 생성되는 이미지가 GAN이 학습한 도메인에 한정된다.
- 생성 품질과 편집 능력이 trade-off 관계에 있다.

이에 저자들은 _generic_ real-world natural **image**를 natural language **text guidance**를 이용하여 **region-based 편집**을 하는 첫 번째 **접근법을 제시**한다. 특히 저자들의 주요 목표는 다음과 같다.

1.  실제 이미지에서 작동
2.  특정 도메인으로 제한되지 않음
3.  사용자가 선택한 영역만 수정하고 다른 영역은 보존
4.  Globally 일관된 수정
5.  같은 입력에 대해 다양한 결과를 생성하는 능력

![fig1](/posts/20240827_Blended/fig1.png){: width="700" height="300"}
_Given an input image and a mask, we modify the masked area according to a guiding text prompt, without affecting the unmasked regions._

> 저자들이 연구를 할 때 이미지 편집 분야는 많은 관심을 받지 못하는 상황이였고, 가장 비슷한 연구조차 text-driven 방식이 아니라 classical한 분야였다고 합니다.
{: .prompt-info }

이를 위해 저자들은 Denoising Diffusion Probabilistic Models (**DDPM**)과 Contrastive Language-Image Pre- training (**ClIP**) 두 개의 **pre-trained model**을 이용했다고 한다. 

저자들은 naive하게 두 모델의 조합은 제대로 동작하지 않았기 때문에 input의 noisy한 버전을 CLIP-guided diffusion model에 blend하는 방법을 택했다고 한다.

## 2. Method
저자들의 목표는 나머지 region은 원래 이미지와 최대한 비슷하면서, ROI에서는 text-prompt와 일치하는 이미지를 만들고 싶은 것이다.

$$ 
x \odot (1-m) \approx \hat x \odot (1-m)
$$

### 2.1. Local CLIP-guided diffusion
저자들은 DDPM 접근법을 local text-driven editing에 적용시키는 것으로 저자들의 방법론을 시작한다. 

__Diffusion Models Beat GANs on Image Synthesis__(Dhariwal and Nichol, [이전 포스트](https://daemini.github.io/posts/Diffusion-Models-Beat-GANs-on-Image-Synthesis/))에서는 **classifier를 이용해 conditional 이미지 생성**의 성공적인 성능을 확인하였다. 이에 저자들은 pretrained CLIP model을 classifier로 활용하여 이미지 conditioning 할 수 있을 것이라 생각했다고 한다.

CLIP 모델은 깨끗한 이미지에 대해 학습되기 때문에, 저자들은 원본 이미지 $$ x_0 $$를 noisy latent $$ x_t $$로 다음과 같이 추정한다.

$$
\begin{equation}
\hat{x}_0 = \frac{x_t}{\sqrt{\vphantom{1} \bar{\alpha}_t}} - \frac{\sqrt{1 - \bar{\alpha}_t}\epsilon_\theta (x_t, t)}{\sqrt{\vphantom{1} \bar{\alpha}_t}}
\end{equation}
$$

또한 CLIP-based loss $$ \mathcal D_{CLIP} $$은 **CLIP의 text embedding**과 **estimated clean image $$ \hat x_0$$의 embedding**과의 **cosine distance**로 정의된다.

$$
\begin{equation}
\mathcal D_{CLIP} (x, d, m) = \mathcal D_c (CLIP_{img} (x \odot m), CLIP_{txt} (d))
\end{equation}
$$

비슷한 접근법이 CLIP-guided diffusion에도 사용되었는데, $$ x_t $$ 와 $$\hat x_0 $$의 linear combination으로 global guidance 를 제공하는 방식이다. 


이에 저자들은 input mask 하에서도 $$ \mathcal D_{CLIP} $$의 gradient 만을 이용해 local guidance가 가능하다고 생각했다. 하지만 background constraint가 없으므로, masked region에만 $$ \mathcal D_{CLIP}$$이 적용된다 하더라도, 전체 이미지에 영향을 미칠 수 있는 문제가 있었다. 따라서 저자들은 **background image를 보존**하기 위한 backgorund preserving loss $$ \mathcal D_{bg} $$를 제안해 **mask 밖 영역을 guide**하는 역할을 한다.

$$ 
\begin{equation}
\mathcal D_{bg} (x_1, x_2, m) = d (x_1 \odot (1-m), x_2 \odot (1-m)) \\
d (x_1, x_2) = \frac{1}{2} (MSE(x_1, x_2) + LPIPS (x_1, x_2))
\end{equation}
$$

최종적인 diffusion model guided loss는 다음과 같다.

$$
\begin{equation}
D_{CLIP} (\hat{x}_0, d, m) + \lambda D_{bg} (x, \hat{x}_0, m)
\end{equation}
$$

![al1](/posts/20240827_Blended/al1.png){: width="600" height="300"}

![fig3](/posts/20240827_Blended/fig3.png){: width="800" height="300"}
_Given an input image with a mask, and the prompt “a dog”: with λ set too low (λ = 100), the entire image changes completely, while if λ is too high (λ = 10000), the model fails to change the fore- ground (and the background preservation is not perfect). Using an intermediate value (λ = 1000) the model changes the foreground while resembling the original background (zoom for more details)_


### 2.2. Text-driven blended diffusion
이미지를 생성하는 과정은 각 time step마다 noisy한 manifold로 부터 less noisy manifold로 이동하는 과정이다. 배경 이미지는 유지하면서 masked region은 text prompt로 guiding하기 위해, 저자들은 **CLIP-guided process로 만들어진 noisy한 이미지**와 **대응하는 time step의 noisy version**을 **blend**하는 방법을 택했다고 한다.



저자들의 핵심 Insight는 각 스텝마다, 두 개의 noisy한 이미지를 blending한 결과는 일관성이 보장되진 않지만, blending 이후 denoising diffusion step이 일관성을 복구하는 방식일 것이라고 한다.

![al2](/posts/20240827_Blended/al2.png){: width="600" height="300"}

![fig4](/posts/20240827_Blended/fig4.jpg){: width="900" height="300"}

#### 2.2.1 Background preserving blending
배경을 보존하는 **naive한 방식**은 CLIP-guided diffusion process로 만든 이미지 $$ \hat x_0 $$를 어떠한 제약 조건 없이($$ \lambda = 0 $$ in Algorithm1.) 생성된 이미지의 배경을 $$ \hat{x} \odot m + x \odot (1-m) $$로 대체하는 것이다. 이러한 방식은 일관적이고 매끈한 이미지를 생성하는데 **실패**한다.

예전 연구에 따르면 두 이미지를 부드럽게 blending 하기 위해서는 각 Laplacian pyramid의 level에서 개별적으로 blending해야 한다는 것이다. 여기에서 영감을 받아 저자들은 **각 different noise step**에서 diffusion process를 따라 **이미지를 blending**한 것이다.

저자들의 주요 가설은 diffuison process의 각 step에서는 **noisy latent가** 특정 수준의 noise를 갖는 **natural image로 projection**될 것이다. 두 개의 noisy한 이미지를 **blending할 때, manifold 밖**으로 갈 수 있지만, **다음 diffusion step**에서 다음 manifold로 **projection** 시켜주기 때문에 **불일치가 개선**된다고 한다.

따라서 diffusion process의 각 단계에서, latent $$ x_t$$ ​에서 시작하여, 

(1) text prompt에 따라 방향을 조정하는 **CLIP-guided diffusion process**를 수행하여 latent **$$ x_{t-1, fg}$$​**를 얻는다. 

(2) 원본 이미지에서 *$$  x_{t-1, bg} $$​*라는 **noisy background** 버전을 얻는다. 

이 두 잠재 공간을 마스크를 사용해 다음과 같은 식으로 **blending**한다.

$$
\begin{equation}
x_{t-1} = x_{t-1, fg} \odot m + x_{t-1, bg} \odot (1-m)
\end{equation}
$$

이 과정을 반복한 뒤, 최종 단계에서 마스크 영역 밖의 모든 부분을 원본 이미지의 해당 부분으로 교체함으로써 배경을 강하게 보존하게 된다.

#### 2.2.2. Extending augmentations
**Adversarial example** 이란 이미지의 픽셀 값을 *직접* 최적화할 때 발생할 수 있는 잘 알려진 문제이다. 예를 들어, classifier를 속이기 위해 **잘못된 클래스에 대한 gradient 방향으로 픽셀을 약간 변경**하면, 인간이 인식하지 못할 정도의 작은 노이즈에도 불구하고 **이미지 분류가 잘못**되게 된다.

저자들은 이 문제를 완화하기 위해 diffusion process에서 추정된 중간 결과에 여러 augmentation을 수행하고, 각 augmentation에 대해 CLIP을 사용해 gradient 계산하는 방법을 제안한다.

>이 방식으로 CLIP을 "속이려면" 모든 증강에 대해 동일하게 작용해야 하므로, 이미지의 고수준 변화 없이 CLIP을 속이는 것이 더 어려워지며, 실제로, 단순한 증강 기법이 이 문제를 완화하는 데 효과가 있음을 확인하였다고 한다.

현재 추정된 결과 $$ \hat x_0 $$ 에 대해, CLIP loss의 gradient **직접 사용하는 대신**, 이 이미지를 몇 가지 **projectively transforemd copy**에 대해 gradient를 계산하고 이를 **평균**하는 전략이다. 저자들은 이 전략을 **"Extending Augmentation"** 이라고 부른다.

## 3. Results
### 3.1. Comparisons 
저자들이 제안한 방법과 
- (1) **PaintByWord**
- (2) **Local CLIP-guided diffusion** (Algorithm 1, $$ \lambda $$ = 1000)
- (3)  **VQGAN-CLIP + Paint By Word**

를 정량적, 정성적으로 비교한다.

![fig5](/posts/20240827_Blended/fig5.png){: width="700" height="300"}
_Comparison using examples from Paint By Word_

(당연하게도) 저자들의 결과가 나머지 (1)~(3)에 비해 뛰어난 것을 확인할 수 있다.

### 3.2. Ablation of extending augmentations
저자들의 "Extending augmentation"을 적용할 때와 적용하지 않았을 때의 비교이다.

![fig7](/posts/20240827_Blended/fig7.png){: width="700" height="300"}
_(1) without extending augmentations and (2) with them_

extending augmentations을 사용한 결과가 훨씬 자연스럽고 배경과 coherent함을 확인할 수 있다.

### 3.3. Applications
또한 저자들의 방법은 다양한 real-world 이미지에 적용될 수 있으며, 여러가지 application이 가능하다고 한다. 

#### **Text-driven object editing**
이미지를 텍스트 지시에 따라 수정할 수 있는데, 객체를 추가, 제거 또는 변경하는 작업이 가능하다.

![fig8](/posts/20240827_Blended/fig8.png){: width="700" height="300"}
_Given the same guiding text (top row: “a dog”, bottom row: “body of a standing dog”) our method generates multiple plausible results_

#### **Background replacement**
전경 객체를 편집하는 대신, 텍스트 지시를 사용해 배경을 교체할 수도 있다.

![fig17](/posts/20240827_Blended/fig17.png){: width="700" height="300"}

#### **Scribble-guided editing**
다른 이미지나 사용자가 제공한 Scribble을 가이드로 활용할 수 있다. 예를 들어, 사용자가 배경 이미지 위에 대략적인 형태를 그리면, 해당 스크리블을 덮는 마스크와 텍스트 프롬프트를 제공하여, 우리의 방법은 이 스크리블을 자연스러운 객체로 변환하고 프롬프트에 맞게 수정하는 방식이다.

![fig9](/posts/20240827_Blended/fig9.png){: width="700" height="300"}
_Users scribble a rough shape of the object they want to insert, mark the edited area, and provide a guiding text - “blanket”_


#### **Text-guided image extrapolation**
이미지를 텍스트 지시에 따라 경계 너머로 extrapolation 할 수 있다. 이때 변화는 점진적으로 이루어지며, 이미지를 제공하고 두 개의 텍스트 프롬프트를 사용해 이미지를 각각 한 방향으로 확장하는 예시를 보여준다. 

![fig10](/posts/20240827_Blended/fig10.png){: width="900" height="500"}
_The user provides an input image and two text descriptions: “hell” and “heaven”_

## 4. Limitations
저자들이 제시한 주요 Limitation은 다음과 같다.

1. Diffusion model의 **sequential한 특징**때문에 inference time이 오래 걸린다는 문제점이 있다.
2. 편집된 영역의 quality 만을 고려해 ranking을 메기는 방식으로, **전체 이미지를 고려해 생성하지 못할 때**가 있다고 한다.
3. CLIP의 약점과 편향성이 있다. (_typographic attack_)

![fig11](/posts/20240827_Blended/fig11.png){: width="900" height="500"}

## 5. Conclusions
본 논문에서 저자들의 주요 기여는 다음과 같다.

1. Natural language guidance를 사용해 다양한 real 이미지에 적용가능한 **general-purpose region-based** 편집 솔루션을 제안했다.

2. Background preservation 기술은 **수정되지 않은 영역이 완벽하게 보존**되도록 보장한다.

3. **Extending augmentation** 기법은 Adversarial result의 위험을 크게 줄여주며, 이를 통해 **gradient-based diffusion guidance**를 사용할 수 있음을 입증했다.

  

## **Reference**

[JiYeop Kim's blog](https://kimjy99.github.io/)를 참고하여 작성하였습니다.
