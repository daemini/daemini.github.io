---
title: jekyll 포스팅 튜토리얼
description: Jekyll Chripy template을 사용해 포스팅하는 튜토리얼입니다. 
toc: true
comments: true
# layout: default
date: 2024-06-19 13:17:00 +09:00
categories: [Blogging, Tutorial]
tags: [markdown, tutorial, jekyll chripy]     # TAG names should always be lowercase
image: /posts/20240618/devices-mockup.png
alt : Responsive rendering of Chirpy theme on multiple devices.
---


# Chripy template Tutorial

그냥 제가 나중에 필요할 때 보려고 포스팅합니다.

## Naming 

마크 다운 이름은 반드시 `yyyy-mm-dd-title.md` 형식으로 작성합니다.

## Front Matter
md 파일 제일 위에 [Front Matter](https://jekyllrb.com/docs/front-matter/)를 삽입합니다.


```markdown
---
title: TITLE
description: Short summary of the post.
comments: true/false
toc: true/false
date: YYYY-MM-DD HH:MM:SS +/-TTTT
categories: [TOP_CATEGORIE, SUB_CATEGORIE]
tags: [TAG]     # TAG names should always be lowercase
---
```

예시 : 

```markdown
---
title: Test Page
description: Short summary of the post.
comments: true
toc: false
date: 2024-06-18 13:20:00 +09:00
categories: [Animal, Cat]
tags: [cat, cute]     # TAG names should always be lowercase
---
```

### Preview Image

만약 preview 이미지를 넣고 싶으면 다음을 추가합니다.

```markdown
---
image:
  path: /path/to/image
  alt: image alternative text
---
```

### Pinned Post

Post를 고정하고 싶으면 다음을 추가합니다.

```markdown
---
pin: true
---
```

### Mathematics
수식을 사용하려면 `math` option을 `true` 로 바꿔줍니다.

```markdown
---
math: true
---
```


### Mermaid
[**Mermaid**](https://github.com/mermaid-js/mermaid) diagram tool을 이용하려면 `Mermaid` option을 `true` 로 바꿔줍니다.

```markdown
---
mermaid: true
---
```


## Image

### Default (with caption)

```markdown
![img-description](/path/to/image){: w="700" h="400" }
_Image Caption_
```

### position
디폴트는 `{: .normal }`이지만, `{: .left}`, and `{: .right}`.를 사용해 이미지의 위치를 바꿀 수 있습니다.

```markdown
![Desktop View](/path/to/image){: width="972" height="589" .w-50 .left}
```

![Desktop View](/posts/20240618/mockup.png){: width="972" height="589" }
_Full screen width and center alignment_

#### Left aligned

![Desktop View](/posts/20240618/mockup.png){: width="972" height="589" .w-75 .normal}

#### Float to left
![Desktop View](/posts/20240618/mockup.png){: width="972" height="589" .w-50 .left}
Praesent maximus aliquam sapien. Sed vel neque in dolor pulvinar auctor. Maecenas pharetra, sem sit amet interdum posuere, tellus lacus eleifend magna, ac lobortis felis ipsum id sapien. Proin ornare rutrum metus, ac convallis diam volutpat sit amet. Phasellus volutpat, elit sit amet tincidunt mollis, felis mi scelerisque mauris, ut facilisis leo magna accumsan sapien. In rutrum vehicula nisl eget tempor. Nullam maximus ullamcorper libero non maximus. Integer ultricies velit id convallis varius. Praesent eu nisl eu urna finibus ultrices id nec ex. Mauris ac mattis quam. Fusce aliquam est nec sapien bibendum, vitae malesuada ligula condimentum.

#### Float to right

![Desktop View](/posts/20240618/mockup.png){: width="972" height="589" .w-50 .right}
Praesent maximus aliquam sapien. Sed vel neque in dolor pulvinar auctor. Maecenas pharetra, sem sit amet interdum posuere, tellus lacus eleifend magna, ac lobortis felis ipsum id sapien. Proin ornare rutrum metus, ac convallis diam volutpat sit amet. Phasellus volutpat, elit sit amet tincidunt mollis, felis mi scelerisque mauris, ut facilisis leo magna accumsan sapien. In rutrum vehicula nisl eget tempor. Nullam maximus ullamcorper libero non maximus. Integer ultricies velit id convallis varius. Praesent eu nisl eu urna finibus ultrices id nec ex. Mauris ac mattis quam. Fusce aliquam est nec sapien bibendum, vitae malesuada ligula condimentum.



### Shadow 
`{: .shadow }` 로 그림자 옵션을 추가할 수 있습니다.

```markdown
![Desktop View](/assets/img/sample/mockup.png){: .shadow }
```
<!-- 
### video

아래 문법으로 내가 좋아하는 침아저씨도 불러올 수 있다.

```markdown
{% include embed/youtube.html id='hnanNlDbsE4' %}
``` -->

<!-- {% include embed/youtube.html id='hnanNlDbsE4' %} -->

## Footnote
`[^FOOTNOTE NAME]`을 이용해 Foot note 작성가능합니다.

예시:

```markdown
Click the hook will locate the footnote[^footnote], and here is another footnote[^fn-nth-2].
Footnote name can be anything![^f3]
``` 
출력 결과 :

Click the hook will locate the footnote[^footnote], and here is another footnote[^fn-nth-2].
Footnote name can be anything![^f3]





## Prompts

### Prompt type

`prompt-{type}` 를 이용해 prompt 타입을 바꿀 수 있습니다. 
type에는 `tip`, `info`, `warning`, and `danger` 가 있습니다.

```markdown
> Example line for prompt.
{: .prompt-info }
```

예시 : 

> An example showing the `tip` type prompt.
{: .prompt-tip }

> An example showing the `info` type prompt.
{: .prompt-info }

> An example showing the `warning` type prompt.
{: .prompt-warning }

> An example showing the `danger` type prompt.
{: .prompt-danger }


### Description list

Sun
: the star around which the earth orbits

Moon
: the natural satellite of the earth, visible by reflected light from the sun





## Reverse Footnote
[^footnote]: The footnote source
[^fn-nth-2]: The 2nd footnote source
[^f3]: The Third footnote source

