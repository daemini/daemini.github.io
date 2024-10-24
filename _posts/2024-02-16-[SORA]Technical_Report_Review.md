---
title: "[Report Review] SORA, OpenAI"
description: OpenAI의 SORA 발표 이후, 간단히 공부한 내용입니다.
toc: true
comments: true
pin : true
# layout: default
date: 2024-02-16 13:17:00 +09:00
categories: [Deep Learning, Generative Model]
tags: [generative model, computer vison, sora]     # TAG names should always be lowercase
image: posts/20240216_SORA/Thumbnail.png
alt : Demo Image
---


# SORA, OpenAI


**OpenAI**에서 5시간 전에(Feb 16, 2024) 나온 [SORA](https://openai.com/index/sora/)라고 하는 **text-to-video model** 을 공개했습니다.


들어가기에 앞서 여담으로는, Diffusion Transformer(DiT)로 졸업 논문 쓴 저자가 OpenAI로 가서 개발했다고 합니다.
저자의 이전 연구와 비슷하게, 생성 모델에 **Transformer architecture**를 사용했고, **high fidelity video**를 만들 수 있다고 주장하고 있으며,  특히 저자들은 model size를 키움으로써 physical world를 더 잘 simulate 할 수 있을 것이라 제안합니다.

이번 포스팅에서는 간단히 SORA의 technical report를 읽고 요약해보도록 하겠습니다.

## Video generation models as world simulators

Technical report에서는 두 가지에 집중합니다.

1. turning visual data of **all types into a unified representation** that enables large-scale training of generative models,

2.  **qualitative evaluation** of Sora’s capabilities and limitations.

다만 아쉽게도 모델이나 **구현은 공개되지 않았습니다.**

사실 이전에도 text-to-video model은 다양한 종류가 있었는데, RNN 기반, GAN 기반, autoregressive transformer, diffusion model 등이 있었죠.

하지만, 이런 모델은 동영상 길이가 짧거나, 정해진 크기의 비디오만 만들 수 있거나, 좁은 카테고리에만 집중되어 있다는 문제점을 갖고 있었습니다.

OpenAI에서 공개한 Sora는 visual data의 generalist 한 모델로 동영상 길이를 뿐만 아니라, aspect ratio(AR, 종횡비)나 resolution(해상도)를 조절할 수 있다고 합니다.


##  Turning visual data into patches

![Model Architecture](posts/20240216_SORA/model_architecture.png)

저자들은 internet-scale data로 training 함으로써 높은 일반화 성능을 가진 Large language model(LLM)에서 영감을 받았다고 합니다. LLM들이 **token**을 갖는 것처럼, Sora는 **visual patch**를 갖습니다. 

Patch는 visual data에서 효과적인 표현으로 선행 연구들이 있었으며, 저자들은 patch가 다양한 비디오나 이미지를 만드는 데 **highly-scalable and effective representation** 하다고 합니다.

High level에서, 저자들은 video encoder를 통해 낮은 차원의 latent space로 video를 patch로 변환하고, spactime patch들로 decomposing 한다고 합니다.

## Video compression network

비디오를 입력으로 받아 시간적, 공간적 정보를 압축하는 latent space로 압축하는 network입니다.
저자들은 또한 latent space에서 pixel space로 mapping 해주는 decoder도 학습했습니다.

## Spacetime Latent Patches

Latent vector로부터, Transformer의 token으로 사용되는 **spacetime patch**를 추출합니다. 이렇게 patch를 통해 학습함으로써, Sora는 **다양한 해상도, 종횡비, 길이**로 동영상을 만들어 낼 수 있습니다. 
또한 Inference time에서, 사용자는 randomly-initailized patch를 arrange 함으로써 비디오 사이즈를 조절할 수 있다고 합니다.

## Scaling transformers for video generation

![Training](posts/20240216_SORA/training_example.png)

Sora는 기본적으로 **diffusion model**입니다. 주어진 noisy patch로부터 original clean patch를 예측하도록 학습됩니다. 
특히 저자들은, SORA가 diffusion model중에서 transformer 구조를 갖는 **diffusion transformer**라 강조합니다. 

Transformer는 scaling property(모델 사이즈가 커짐에따라 성능이 올라가는 특성)를 갖는 것으로 알려져 있는데, language model, computer vison, image generation과 같은 다양한 domain에서 활용되고 있죠.

저자들은 이 연구에서, diffusion transformer가 **비디오에서도 잘 동작**함을 확인했습니다. Sample quality는 아래 사진에서 볼 수 있듯이 training compute이 증가할수록 확연하게 증가하는 것을 확인할 수 있습니다.

## Variable durations, resolutions, aspect ratios

이미지나 비디오 생성에서 이전 연구들의 접근법에서 standard size (e.g. 4 second, 256X256 resolution)로 resize, crop, trim 등을 수행했지만, 이번 연구에서 저자들은 **data의 native size**를 사용하는 것이 여러 이점을 제공한다고 합니다.

### Sampling flexibility
![sampling_flexibility](posts/20240216_SORA/sampling_flexibility.png)
SORA는 1920X1080부터, 1080X192(vertical) 사이에 있는 모든 resolution을 갖는 비디오를 생성할 수 있습니다. 이를 통해 Sora는 각 device의 native AR로 contents를 만들 수 있는 이점을 제공합니다.

### Improved framing and composition
저자들은 경험적으로 native AR을 사용했을 때, **composition, framing**이 향상됨을 확인했다고 합니다.

## Language understanding
Text-to-video model을 학습시키기 위해서는 매우 많은 양의 text captioning된 비디오가 필요하다. 저자들은 DALL·E 3에서 사용된 **re-captioning technique**을 적용했다고 한다. 

먼저 highly descriptive captioner model을 train 하고 이를 사용해 training에 사용되는 text caption을 만드는데 사용했다고 한다. 저자들은 highly descriptive caption이 **text fidelity**와 더불어 전반적인 비디오 **quality**를 올리는데 기여했다고 한다.

![Language_understanding](posts/20240216_SORA/Language_understanding.png)

## Prompting with images and videos

![Prompting](posts/20240216_SORA/Prompting.png)

SORA의 재밌는 점은 Text-to-video뿐만 아니라, Sora는 image나 **video를 input prompt**로 사용할 수 있습니다. 
아래 예시는 DALL · E 2, DALL·E 3로 만들어진 이미지를 동영상으로 만든 것입니다.

혹은 **비디오를 입력**으로 받아 디테일을 수정하거나, 동영상 앞뒤로 길이를 늘리는 작업도 가능하며, 비디오 간의 interpolation도 가능하다고 합니다.

![Prompting](posts/20240216_SORA/video_interpolation.png)

## Emerging simulation capabilities

저자들은 대규모로 학습할 때, 4가지 정도의 재미있는 특징들을 관측할 수 있었다고 합니다. 이런 특징들이 physical world로부터 사람이나 동물들의 특징들을 simulate 할 수 있도록 도와주는 역할을 한다고 설명합니다.

### 3D Consistency
SORA는 카메라의 dynamic camera motion을 생성할 수 있습니다. 카메라 Shift, rotate 동안 사람들이나 풍경들은 Consistently 3-D space를 따라 움직이는 영상을 잘 만들어내는 것을 볼 수 있습니다.

### Long-range coherence and object permanence.
비디오 생성의 어려운 점 중 하나는 긴 비디오에서 temporal consistency 유지하는  것입니다. SORA는 (항상 그런 것은 아니지만), short- and long-range dependencies를 효과적으로 표현할 수 있다고 합니다.

>예를 들어 다른 사물들에 가려져 잠시 동안 나오지 않던 물체나 사람들이 다시 나타날 때도 일관적으로 생성이 되어야 한다는 것.
{: .prompt-info }


### Interacting with the world.

화가가 캔버스에 붓 자국을 남기거나, 햄버거를 먹어서 먹은 자국을 만드는 것처럼 간단한 상호작용이 가능하다고 합니다.


![weird_videos](posts/20240216_SORA/weird_videos.png)


>Demo page에서 이 부분을 weakness로 꼽기도 했는데, 물리적으로 불가능한 비디오나, 여러 객체가 동시에 나타나기도 하며, bite mark가 남아있지 않는 점들이 있다.
{: .prompt-info }


### Simulating digital worlds.
SORA는 artificial process를 simulate 할 수 있습니다.. 재밌는 점은 이러한 능력은 zero-shot으로 prompt에 `Minecraft`를 입력하는 것으로 가능했다고 합니다.

![mincraft_demo](posts/20240216_SORA/mincraft_demo.png)

## Discussion
SORA는 아직 많은 한계점들이 있지만 이전 구글이나, 메타의 모델보다 월등한 성능을 보이는 것 같습니다. 
'ChatGPT처럼 Vision 분야에서 game changer가 되지 않을까?'라는 생각도 드는 인상깊은 모델이였습니다.

뭐 일단은 모델이 빨리 공개돼서 이것저것 실험해 볼 수 있으면 좋겠다는 생각을 하며,
이번 포스팅 마무리하겠습니다.
 
긴 글 읽어주셔서 감사합니다 ! 












