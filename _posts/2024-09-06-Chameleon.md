---
title: "[Paper Reivew] Chameleon: Mixed-Modal Early-Fusion Foundation Models (Chameleon)"
description: Chameleon은 텍스트와 이미지를 함께 처리하고 생성할 수 있는 token-based mixed-modal model으로, 기존 모델을 능가하는 성능을 보이며 안정적인 훈련을 위한 새로운 architecture 및 training 방법론을 제시합니다.
toc: true
comments: true
# layout: default
math: true
date: 2024-09-06 16:05:00 +09:00
categories: [Deep Learning, Generative Model]
tags: [diffusion model, generative model, token-based, meta, multimodal, t2i]     # TAG names should always be lowercase
image: /posts/20240905_Chameleon/thumbnail.webp
alt : Thumbnail
---


> arXiv 2024. [[Paper](https://www.arxiv.org/abs/2408.11039)] [[Github](https://github.com/lucidrains/transfusion-pytorch)]
> Chameleon Team
> Meta
> 17 May 2024

Transformer논문에서 비교한 Chameleon의 방법론도 궁금해져 읽게되었습니다.
[[Transformer, 이전 포스팅]](https://daemini.github.io/posts/Transfusion/)도 같이 보시면 더 좋을 것 같습니다 :) 

# TL;DR
**Chameleon**은 텍스트와 이미지를 함께 처리하고 생성할 수 있는 **token-based mixed-modal model**으로, 기존 모델을 능가하는 성능을 보이며 안정적인 훈련을 위한 새로운 architecture 및 training 방법론을 제시합니다. (3달 후, Transfusion한테 지지만..)

![fig1](/posts/20240905_Chameleon/fig1.png){: width="800" height="300"}


## 1.Introduction
최근 multimodal 기반 모델들은 다른 modality를 따로 모델링하며, 각각의 encoder/decoder를 갖는데, 이로인해 modality간의 통합 정보 처리 능력에 한계를 가질 수 있습니다. 이 논문에서 저자들은 **Chameleon**이라는 새로운 mixed-modal 기반 모델을 제안합니다. 기존 방법들과 달리 modality별 encoder/decoder를 갖지않으며, text, image에서 모두 token-based representation을 통해 하나의 transformer 구조에서 정보를 처리, 생성할 수 있다고 합니다.

**Chameleon**은 특히 훈련과정의 안정성과 확장성을 위해 몇 가지 방법을 도입하였습니다. 

1. Novel modifications to the transformer architecture : query-key normalization이나 layer norms 수정 등.
2. Supervsed finetuning 접근법 사용: text-only LLM에서 사용되던 방법을 Mixed-modal setting에서 사용

결과적으로 34B 파라미터의 모델을 성공적으로 training 할 수 있었으며, 기존 Text-only 모델을 능가하는 성능을 보였습니다. 특히, 인간 평가 실험에서도 **Chameleon**은 **Gemini-Pro** 및 **GPT-4V**와의 비교에서 탁월한 성능을 입증했습니다.

저자들의 핵심기여는 다음과 같습니다.

1.  텍스트와 이미지를 함께 처리할 수 있는 **토큰 기반 혼합 모달 모델, Chameleon**을 제시하여, 기존 다중 모달 모델의 한계를 넘어섬.
2.  **Transformer 아키텍처**에서 모달리티 간 통합을 안정적으로 학습하기 위한 새로운 훈련 기법과 아키텍처 혁신 도입.
3.  다양한 시각-언어 벤치마크에서 SOTA를 달성하고, 텍스트 전용 작업에서도 경쟁력 있는 성능을 유지.
4.  혼합 모달 추론 및 생성에서 **대규모 인간 평가**를 통해 **Chameleon**의 새로운 가능성을 입증.

## 2. Pre-Training
**Chameleon**은 텍스트뿐만 아니라, 이미지를 discrete 토큰으로 표현하며 AR transformer의 특성을 갖습니다. 학습 중 text-only 부터, text/image pair 등 다양한 형식으로 데이터를 처리할 수 있습니다.


### 2.1. Tokenization

#### **Image Tokenization**
저자들은 **Gafni et al. (2022)**을 기반으로 해 새로운 image tokenizer를 학습시켰는데, **512 × 512 크기의 이미지**를 **8192 크기의 코드북**에서 **1024개의 discrete 토큰**으로 변환합니다

#### **Tokenizer**
저자들은 새로운 BPE tokenizer를 학습을 위해,  이미지 코드북의 **8192개 토큰**도 포함하는**65,536개**의 토큰을 이용했다고 합니다. 

### 2.2. Pre-Training Data
저자들의 pre-training은 두 가지로 나누어 진행되었습니다. First stage가 훈련 시간의 80%를 차지하며, Second stage는 20%를 차지했습니다. 모든 *Text-to-Image* 쌍의 경우 50%의 확률로 이미지가 먼저 입력(i.e., captioning)으로 들어가게 설정했습니다.

#### **First Stage**
저자들은 매우 큰 규모의 unsupervised dataset을 first stage에 이용했습니다.

1. **텍스트 전용 데이터**  
**LLaMA-2**(Touvron et al., 2023)와 **CodeLLaMA**(Roziere et al., 2023)의 훈련 데이터를 포함한 다양한 텍스트 데이터셋이 사용되었습니다. 이로써 총 **2.9T 개의 텍스트 토큰**이 포함된 텍스트 전용 데이터셋을 구성했습니다.

2. **텍스트-이미지 데이터**  
공개된 데이터 소스와 라이센스된 데이터를 기반으로 텍스트-이미지 데이터를 수집하였고, 이미지들은 **512 × 512 크기로 리사이즈 및 센터 크롭**하여 토큰화했습니다. 총 **1.4B개의 텍스트-이미지 쌍**이 포함되어 **1.5T 개의 텍스트-이미지 토큰**이 생성되었습니다.

3. **텍스트/이미지 혼합 데이터**  
**웹에서 공개된 데이터 소스**에서 텍스트와 이미지가 혼합된 데이터를 수집하여 총 **400B 개의 혼합 토큰**을 구성했습니다. 이 데이터는 텍스트-이미지 데이터와 동일한 방식으로 필터링되었습니다.

#### **Second Stage**

**두 번째 단계**에서는 첫 번째 단계에서 사용된 데이터의 비중을 **50%로 줄이고**, **더 높은 품질**의 데이터셋을 혼합하여 사용했습니다. 

이 단계에서는 대규모 **instruction tuning** 데이터셋의 필터링된 부분 집합도 추가로 포함했다고 합니다.

### 2.3. Stability
저자들은 8B이상의 파라미터, 1T 개의 토큰을 처리하는데 어려움을 겪었다고 합니다. 특히 학습 후반부에서 불안정성이 나타나는 경우가 많았는데, 안정성을 위해 여러가지 architecture와 최적화 기법을 적용했다고 합니다.

#### **Architecture** 
Chameleon의 아키텍처는 **LLaMA-2**(Touvron et al., 2023)를 기반으로 하며, **RMSNorm**(Zhang and Sennrich, 2019), **SwiGLU**(Shazeer, 2020) 활성화 함수, **RoPE**(Su et al., 2021) 위치 임베딩을 사용합니다.


저자들은 일반적인 LLaMa 구조가 multi modality 학습에서 어려움을 겪는 문제의 원인으로 softmax 연산의 translation invariant property 때문이라고 합니다. Transformer에서 softmax연산이 일어나는 곳은, (1) attention mechanism, (2) softmax over the logits 입니다. 이를 해결하기 위해 Dehghani et al. (2023) and Wortsman et al. (2023)으로부터 **QK-Norm**을 도입합니다.  **QK-Norm**은 attention과정에서 query, key에 대해 layer norm을 적용함으로써, softmax입력이 커지는 것을 제어할 수 있습니다.

또한 저자들은 Qk-Norm이외에도, attention과 feed-forward 이후에도 dropout을 추가해 Chameleon-7B 모델을 안정적으로 학습했지만, Chameleon-34B는 추가적인 re-ordering까지 필요했다고 합니다. (이 경우 dropout은 제외)

![eq1](/posts/20240905_Chameleon/eq1.png){: width="800" height="300"}


![fig5](/posts/20240905_Chameleon/fig5.png){: width="800" height="300"}

#### **Optimization**

- **AdamW**(Loshchilov and Hutter, 2017) 옵티마이저를 사용하며, $$ \beta_1 $$을 0.9, $$  \beta_2 $$를 0.95로 설정하고, $$ \epsilon $$ 값을 $$ 10^{-5} $$로 설정했습니다. **4,000 스텝의 warp-up**과 **exponential decay schedule**을 사용했으며, **global gradient clipping**을 1.0으로 설정했습니다.

- **QK-Norm**을 적용했음에도 **최종 softmax에서 발생하는 logit shift 문제**는 해결되지 않았다고 합니다. 이를 해결하기 위해 **z-loss 정규화**(Chowdhery et al., 2022; Wortsman et al., 2023)를 적용하여 softmax의 **partition function Z**를 정규화했습니다.
$$
\begin{equation}
\sigma(x)_i = \frac{e^{x_i}}{Z} \text{, where } Z = \sum_{i} e^{x_i}
\end{equation}
$$


### 2.4. Inference
**Inference Stage**에서 mixed-modal 생성은 성능 관련 문제를 일으킬 수 있다고 합니다. 
1. 특히 **-   Data-dependencies per-step**이 주요 도전 과제로 작용합니다. 텍스트 또는 이미지 생성을 할 때마다 **GPU에서 CPU로 데이터 전송**이 이루어지며, 이를 통해 **제어 흐름을 조정**해야 합니다. 

2. 또한 **-   Masking for modality-constrained generation**을 통해 특정 모달리티만 생성되도록 관리해야 합니다.

3. **텍스트 생성**은 가변 길이이지만, **이미지 생성**은 고정 크기의 토큰 블록을 생성해야 한다는 차이가 있습니다.

저자들은 이런 문제를 해결하기 위해 PyTorch 기반 별도 파이프라인을 구축하고, xformers에서 제공하는 GPU kernel을 사용했다고 합니다. 

## 3. Alignment
저자들은 다른 최신의 연구에서처럼, **Supervised Fine Tuning(SFT)**을 위해 선별된 높은 퀄리티의 데이터셋을 사용했다고 합니다. 

### 3.1. Data
저자들은 SFT를 위한 dataset을 다음과 같은 카테고리로 분류했습니다. *Text, Code, Visual Chat, Image Generation, Interleaved Text/Image Generation, and Safety.*

**LLaMa-2**와 **CodeLLaMa**에서 이어받은 **텍스트 및 코드 데이터**, 그리고 자체적으로 curating한 **고품질 이미지 생성 데이터**로 이루어집니다. **시각적 대화**와 **텍스트/이미지 혼합 생성**을 위한 데이터는 외부 데이터 수집업체를 통해 수집되었으며, 안전성을 위해 **잠재적으로 위험한 프롬프트**에 대해 모델이 적절히 반응하도록 하는 데이터를 포함했다고합니다. 이는 **폭력**, **프라이버시**, **성적 콘텐츠** 등 민감한 주제를 다룹니다.

**혼합 모달리티** 데이터 수집은 특히 중요하며, 텍스트와 이미지 사이의 **잠재적 공격 벡터**에 대비할 수 있도록 다양한 데이터를 수집하여 모델의 안전성을 확보했습니다.


### 3.2. Fine-Tuning Strategy
#### **Data Balancing**
**Data Balancing**은 SFT단계에서 중요하며, 모달리티 간의 심각한 불균형이 발생하면 모델이 특정 모달리티의 과생성이나 미생성을 학습할 수 있다고 합니다.


#### **Optimization**

각 데이터셋 인스턴스는 프롬프트와 그에 대한 답변으로 구성되며, 가능한 많은 프롬프트와 답변을 시퀀스에 포함해 효율성을 높입니다. **AR training objective**를 사용하며, 프롬프트 토큰에 대해서는 손실을 선택적으로 마스킹하여 답변 토큰에 대해서만 최적화합니다.  추가적으로, 프롬프트에 포함된 이미지는 정보를 모두 제공할 수 있도록 padding& resized되며, 답변에 포함된 이미지는 시각적으로 좋은 품질을 보장하기 위해 center-crop했다고 합니다.

## 4. Human Evaluations and Safety Testing
### 4.1. Prompts for Evaluation
**Chameleon**은 기존 벤치마크로는 측정할 수 없는 새로운 혼합 모달리티 이해 및 생성 능력을 보유하고 있습니다. **인간 평가**는 다양한 프롬프트에 대해 모델이 얼마나 잘 반응하는지 확인하는 방식으로 수행됩니다. 인간 평가용 프롬프트는 **제3자 크라우드소싱 벤더**를 통해 수집되며, 다양한 실제 상황에서 사용자가 모델에게 기대하는 혼합 모달리티 응답을 상상하여 제공됩니다. 수집된 프롬프트는 명확성 여부와 이미지가 포함될 것으로 기대되는지를 기준으로 평가받으며, 최종적으로 1,048개의 프롬프트로 구성된 평가 세트가 완성되었습니다. 이 중 441개(42.1%)는 텍스트와 이미지를 모두 포함한 혼합 모달리티 프롬프트이며, 나머지 607개(57.9%)는 텍스트 전용 프롬프트입니다.



### 4.2. Baselines and Evaluations
Chameleon 34B를 OpenAI GPT-4V와 Google Gemini Pro와 비교했으며, 각 모델의 API를 호출하여 성능을 평가했습니다. GPT-4V와 Gemini Pro는 혼합 모달 프롬프트를 처리할 수 있지만, 텍스트 응답만 생성하므로, 이러한 응답에 이미지 캡션을 추가하고 OpenAI DALL-E 3으로 이미지를 생성하여 향상된 버전(GPT-4V+ 및 Gemini+)도 평가했습니다.

![fig9](/posts/20240905_Chameleon/fig9.png){: width="800" height="300"}



#### **Absolute Evaluation**
모델의 응답이 주어진 프롬프트를 얼마나 충실히 수행하는지에 대해 평가자가 응답을 평가했습니다. Chameleon은 55.2%의 완전한 작업 수행률을 기록하며, Gemini+(37.6%)와 GPT-4V+(44.7%)를 능가했습니다. Gemini(17.6%)와 GPT-4V(23.1%)의 경우, 텍스트만으로 혼합 모달 프롬프트를 처리한 탓에 응답이 완전한 작업 수행으로 간주되지 않았습니다. 작업 카테고리별로는 Chameleon이 브레인스토밍, 비교, 가상 시나리오에서 우수한 성능을 보였으나, 식별 및 추론에서는 개선이 필요하다고 평가되었습니다.
#### **Relative Evaluation**
프롬프트에 대한 Chameleon과 다른 베이스라인 모델의 응답을 비교한 결과, Chameleon이 Gemini+에 비해 41.5%의 승률을 기록했으며, GPT-4V+에 비해서도 35.8%의 승률을 기록했습니다. Chameleon의 응답은 원래의 Gemini와 GPT-4V에 비해 훨씬 우수한 성능을 보였으며, Gemini와 GPT-4V에 비해 각각 69.1%와 61.7%의 승률을 기록했습니다.

### 4.3. Inter-annotator Agreement

모든 질문은 3명의 평가자가 응답했으며, **대다수의 평가자 의견**을 최종 답변으로 간주했습니다. **평가자 간 일치도**를 분석한 결과, 단순하고 객관적인 질문(예: 유해한 콘텐츠 포함 여부)에서는 평가자 간 높은 일치도를 보였습니다. **상대 평가**에서 평가자 3명이 모두 일치한 경우는 약 28%에서 35%였으며, 약 10%의 경우에서는 의견 불일치가 발생했습니다.

![fig10](/posts/20240905_Chameleon/fig10.png){: width="800" height="300"}


### 4.4. Safety Testing
**안전성 테스트**는 모델이 자해, 폭력, 증오, 범죄 계획 등과 관련된 불안전한 콘텐츠를 생성할 수 있는 프롬프트를 대상으로 진행되었습니다. 프롬프트에는 텍스트 및 혼합 모달 입력이 포함되며, 안전하지 않은 텍스트, 이미지 또는 혼합 모달 출력 생성이 의도된 경우도 포함되었습니다. 평가자들은 각 프롬프트에 대한 모델의 응답을 안전 또는 불안전으로 분류했고, 경계선에 해당하는 응답에 대해선 '불확실' 옵션도 제공되었습니다. **테이블 5**에 따르면, Chameleon 7B 모델에서는 0.39%(78건)의 불안전한 응답이, 30B 모델에서는 0.095%(19건)의 불안전한 응답이 있었습니다.

**대화형 세션**에서는 내부 레드팀이 30B 모델을 대상으로 445개의 프롬프트-응답 상호작용을 수행했으며, 이 중 1.6%(7건)가 불안전한 응답으로, 4.5%(20건)가 불확실한 응답으로 분류되었습니다. **RLHF/RLAIF**를 사용한 추가 안전성 튜닝이 모델의 '탈옥' 및 악의적인 공격에 대한 방어력을 높일 수 있지만, 현재의 안전성 튜닝만으로도 연구 목적으로 사용 시 상당한 보호 효과를 제공함을 확인했습니다.

![tab5](/posts/20240905_Chameleon/tab5.png){: width="800" height="300"}

### 4.5. Discussion

Chameleon은 **혼합 모달 응답**을 요구하는 프롬프트 처리에서 **Gemini** 및 **GPT-4V**에 비해 경쟁력이 매우 뛰어납니다. 특히, Chameleon이 생성하는 이미지들은 대체로 문맥과 연관성이 높아, 텍스트와 이미지가 혼합된 문서가 사용자에게 매력적으로 다가옵니다.

그러나 **사람 평가의 한계**도 존재합니다. 첫째, 프롬프트는 실제 사용자와의 상호작용이 아닌 **크라우드소싱**을 통해 수집되었기 때문에, 데이터셋의 다양성이 부족할 수 있습니다. 둘째, 프롬프트가 혼합 모달 출력을 중심으로 구성되었기 때문에, **OCR**이나 **정보 그래프** 해석과 같은 특정 시각적 이해 작업이 평가에서 제외되었습니다. 마지막으로, 현재의 멀티모달 LLM API는 텍스트 응답만 제공하므로, Chameleon과 다른 **네이티브 혼합 모달 모델**과의 비교가 바람직하지만, 해당 부분이 한계로 작용했습니다.

## 5. Benchmark Evaluations
Chameleon 모델은 다양한 능력을 가진 모델로, 특정 범주에 대한 직접적인 비교 모델이 없기 때문에 각 범주의 최고 성능 모델들과 비교하여 평가를 진행했습니다.

### 5.1. Text

Chameleon의 텍스트 전용 성능을 다른 최신 대형 언어 모델들과 비교하여 평가했습니다. 다음은 주요 결과입니다:

-   **상식 추론 및 독해 능력**: Chameleon-7B 및 Chameleon-34B는 **Llama-2**와 경쟁할 만한 성능을 보였고, 특히 Chameleon-34B는 8개의 벤치마크 중 5개에서 Llama-2 70B를 능가했습니다. 상식 추론 및 독해 평가에서는 PIQA, SIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, OpenBookQA, BoolQ 벤치마크가 사용되었습니다.
    
-   **수학 및 세계 지식**: GSM8K (초등 수학 문제) 및 MATH 벤치마크에서 Chameleon-7B는 Llama-2와 경쟁하며 **Mistral 7B**와 유사한 성능을 보였습니다. Chameleon-34B는 Llama-2 70B를 능가하고 Mixtral 8x7B와 비슷한 성능을 기록했습니다. 특히, MATH 벤치마크에서는 Llama-2를 넘어서는 결과를 보여줬습니다.
    
-   **MMLU (세계/도메인 지식)**: Chameleon 모델은 MMLU 벤치마크에서도 Llama-2 모델을 능가했으며, 특히 Chameleon-34B는 **Mixtral 8x7B** 및 **Gemini-Pro**와 유사한 성능에 근접했습니다.


![tab6](/posts/20240905_Chameleon/tab6.png){: width="800" height="300"}

결론적으로 저자들의 Chameleon 모델은 LLaMa-2 모델을 전반적으로 능가하며, 특히 **Mistral 7B**와 **Mixtral 8x7B**와 경쟁할 수 있는 수준까지 성능을 끌어올렸습니다. 이는 LLaMa-2 데이터로 두 번의 에폭을 학습한 점, 고품질 데이터와 코드 데이터를 포함하여 성능을 개선한 점 등이 기여한 것으로 보입니다.

### 5.2. Image-To-Text
Chameleon 모델의 이미지-기반 텍스트 생성 능력을 평가하기 위해 이미지 캡션 생성 및 시각적 질문-응답(VQA) 작업에서 Chameleon-34B 모델을 테스트했습니다. 모델은 사전 학습된 상태와 특정 작업에 맞게 미세 조정된 상태에서 평가되었으며, Flamingo, IDEFICS, Llava-1.5, GPT-4V와 같은 최신 모델들과 비교되었습니다.

- 이미지 캡션 생성
  -   **COCO 및 Flickr30k**: Chameleon-34B는 COCO 데이터셋에서 Flamingo 및 IDEFICS의 80B 모델보다 우수한 성능을 보였으며, Flickr30k에서는 두 모델과 대등한 성능을 보였습니다.

  -  **미세 조정**: Chameleon-34B 모델의 멀티태스크 및 특정 작업용으로 미세 조정된 버전(SFT)은 모든 모델을 능가하는 성능을 보였습니다. Flickr30k에서도 Chameleon-34B-SFT 모델이 다른 모델보다 더 나은 성능을 기록했습니다.



- 시각적 질문-응답(VQA)

  -    **VQA-v2**: Chameleon-34B는 사전 학습된 상태에서 Flamingo와 IDEFICS의 32샷 성능과 대등한 성능을 보였으며, 멀티태스크 및 특정 작업 미세 조정 모델은 IDEFICS-80B 및 Gemini Pro와 유사한 성능을 보였습니다. 그러나 Flamingo-80B 및 GPT-4V와 같은 더 큰 모델보다 성능이 떨어졌습니다.



## 6. Conclusion

이 논문에서는 **Chameleon**이라는 새로운 토큰 기반 다중모달 모델을 소개했습니다. Chameleon은 이미지와 텍스트 토큰을 통합하여 학습하는 단일 모델로, 다양한 비전-언어 벤치마크에서 뛰어난 성능을 발휘하며, 새로운 혼합 모달 추론 및 생성 기능을 제공합니다.

Chameleon의 성공 요인은 완전히 **토큰 기반 아키텍처**에 있습니다. 이미지를 **이산 토큰**으로 변환하고, 텍스트와 이미지 데이터를 혼합하여 모델을 훈련함으로써, Chameleon은 서로 다른 모달리티 간의 정보를 자연스럽게 통합하고 **공동 추론**을 할 수 있습니다. 기존의 **Late-fusion** 방식이나 모달리티별로 분리된 인코더를 사용하는 모델에서는 불가능한 성과입니다. 또한, Chameleon은 **안정적이고 확장 가능한 학습**을 위한 새로운 기술을 도입하여, 이전에 한계로 여겨졌던 대규모 학습 문제를 해결했습니다.

Chameleon-34B 모델은 이미지 캡션 생성과 시각적 질문 응답(VQA) 작업에서 **Flamingo**와 **IDEFICS**와 같은 최신 모델보다 우수한 성능을 보였으며, 텍스트 전용 벤치마크에서도 경쟁력을 유지했습니다. 더 나아가, 혼합 모달의 **개방형 질문 답변** 작업에서도 강력한 성능을 발휘하며, 멀티모달 상호작용의 새로운 가능성을 제시했습니다.

![fig2](/posts/20240905_Chameleon/fig2.png){: width="800" height="300"}

![fig3](/posts/20240905_Chameleon/fig3.png){: width="800" height="300"}

![fig4](/posts/20240905_Chameleon/fig4.png){: width="800" height="300"}

