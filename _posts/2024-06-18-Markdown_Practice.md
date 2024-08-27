---
title: Markdown Practice
description: 마크다운 문법을 간단히 요약한 포스트입니다.
comments: true
toc: true
date: 2024-06-18 15:07:00 +09:00
categories: [Blogging, Tutorial]
tags: [markdown, tutorial]     # TAG names should always be lowercase
image: posts/20240618_Markdown_Practice/markdown.png
alt : Responsive rendering of Chirpy theme on multiple devices.
---


# Markdown Practice

마크다운 연습 페이지입니다.<br>
문법을 정확히 적기 보다는 제가 사용하기 편한 것들로 적절히 골랐으니,<br>
문법이 궁금하시다면 [Markdown Guide](https://www.markdownguide.org/getting-started/)를 참고하시기 바랍니다.<br>

## 제목

제목은 `#` 을 이용해 표현할 수 있습니다.

``` markdown
# 제목1
## 제목2
### 제목3
#### 제목4
##### 제목5
###### 제목6
```

## 강조
강조는 크게 이탤릭체(`*`), 볼드체(`**`)가 있습니다.

```markdown
이텔릭체는 *이렇게* 표현할 수 있습니다.
볼드체는 **요렇게** 표현할 수 있습니다.
```
출력결과 : 

이텔릭체는 *이렇게* 표현할 수 있습니다.<br>
볼드체는 **요렇게** 표현할 수 있습니다.

## 목록 

목록은 크게 순서가 있는 것과 없는 것으로 나눌 수 있습니다.

### 순서가 있는 목록
순서가 있는 목록은 `1.` `2.` `3.` 등으로 표현할  수 있습니다.

```markdown
1. 첫번째
2. 두번째
3. 세번째
```
출력 결과 : 

1. 첫번째
2. 두번째
3. 세번째

### 순서가 없는 목록 
순서가 없는 목록은 `-`기호로 표현할 수 있습니다.

```markdown
- 고양이
	- 귀여운 고양이
		- 엄청 귀여운 고양이
		- 왕 귀여운 고양이
	- 못생긴 고양이
- 강아지
- 거북이
```
출력 결과  :

- 고양이
	- 귀여운 고양이
		- 엄청 귀여운 고양이
		- 왕 귀여운 고양이
	- 못생긴 고양이
- 강아지
- 거북이

## 링크 

링크는 `[이름](링크 "설명")` 으로 표현합니다.

```markdown
[Youtube](https://www.youtube.com/ "유튜브나 보고 싶다.")
```
출력 결과 : 

[Youtube](https://www.youtube.com/ "유튜브나 보고 싶다.")

## 이미지 
이미지는 링크와 비슷하지만 `!`가 추가됩니다. 
`![대체텍스트](링크 "설명")` 로 사용합니다.

```markdown
![이건 이미지 입니다](posts/20240618/dog.png "링크 설명")
```

출력 결과 :

![이건 이미지 입니다](posts/20240618/dog.png "링크 설명")

이미지에 링크를 삽입할 수 있습니다.

이경우 간단히 이미지 md 문법을 링크 md 문법으로 감싸주면 됩니다.




```markdown
[![이미지 링크](posts/20240618/youtube_icon.png "링크 설명")](https://www.youtube.com/ "유튜브나 보고 싶다.")
``` 

출력 결과 : 

[![이미지 링크](posts/20240618/youtube_icon.png "링크 설명")](https://www.youtube.com/ "유튜브나 보고 싶다.")



## 표

테이블은 `헤더`와 `셀`로 구분할 수 있습니다.
헤더는 `---`으로 구분하며, `:`기호를 이용해 셀 내용을 정렬할 수 있습니다.

- `---`  좌측 정렬
- `:---:` 가운데 정렬
- `---:` 우측 정렬

```markdown
| 상속 유형 | public | protected | private |
| --- | :---: | :---: | ---:|
|`public`| public | protected |private |
|`protected`| protected | protected |private |
|`private`| private | private |private |
``` 
출력 결과 :

| 상속 유형 | public | protected | private |
| --- | :---: | :---: | ---:|
|`public`| public | protected |private |
|`protected`| protected | protected |private |
|`private`| private | private |private |


## 인용문 

인용문은 `>`기호를 이용해 표현합니다. (중첩이 가능합니다.)


```markdown
> 인생은 고통과 권태 사이에서 왔다 갔다 하는 시계추와 같다. 
> 행복은 욕망에 기생하는 것이다. 
> **Arthur Schopenhauer**

>  이렇게
> > 대충 흑백 사진에 아무 말이나 쓰면 명언 같다.
> >  > 이병건
``` 

출력 결과 : 

> 인생은 고통과 권태 사이에서 왔다 갔다 하는 시계추와 같다. <br>
> 행복은 욕망에 기생하는 것이다. <br>
> **Arthur Schopenhauer**

>  이렇게
> > 대충 흑백 사진에 아무 말이나 쓰면 명언 같다.
> >  > 이병건


## Todo list
`[]` 기호를 이용해 Todo list 를 만들 수 있습니다.

```markdown
- [x] 무작정 블로그 열기.
- [ ] 예쁘게 커스텀 하기.
- [x] 숨쉬기
``` 
출력 결과 : 

 - [x] 무작정 블로그 열기.
 - [ ] 예쁘게 커스텀 하기.
 - [x] 숨쉬기

## 수평선 

수평선은 `***`를 이용해 표현할 수 있습니다.

```markdown
수평선 그리기 시작~!
* * *
수평선 그리기 끝~!
``` 
출력 결과 :

수평선 그리기 시작~! <br>
* * *
수평선 그리기 끝~! 



## Reference 

이 글은 [HEROPY](https://www.heropy.dev/p/B74sNE) 님의 블로그를 바탕으로 작성했습니다.




