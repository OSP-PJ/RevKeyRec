# REVEYE - 상품 리뷰를 쉽게 이해하는 혁신적 웹서비스
  
  
  
**Members**
  -
- 🌟👩💻🌟 **이민지(PM)** : Project Manager, NLP, Back/Front, DB
- 🧑💻 **신연수** : Data crawling, vector embedding, 데이터 검수, DB
- 🧑💻 **장현석** : Front-end, 프론트 기획, DB
- 👨‍🦱💻 **태성우** : Data crawling, vector embedding, 데이터 검수, DB
- 🧑💻 **하태광** : Data crawling, NLP, Back-end, 데이터 검수



**Project Overview**
  -
• **E-commerce Platform의 발전과 발생하는 똑똑한 소비자의 고충**  
-> 방대한 양의 리뷰를 확인하는 작업은 상당한 피로를 유발
    
• **2021 Naver 키워드 리뷰 등장 - 플레이스의 정성적 리뷰 데이터를 직관적으로 파악 가능해짐**

• **24년 현재까지 상품(object)에 관해 서비스 확장은 되지 않음**

• **따라서 방대한 양의 리뷰를 읽을 필요 없이 상품의 핵심 장단점 키워드를 한눈에 파악하는 웹사이트 REVEYE 기획**

**Service Introduction**
  -

| ![이미지1](https://github.com/OSP-PJ/RevKeyRec/blob/feature-sw/image/1.jpg) | ![이미지2](https://github.com/OSP-PJ/RevKeyRec/blob/feature-sw/image/2.jpg) | ![이미지3](https://github.com/OSP-PJ/RevKeyRec/blob/feature-sw/image/3.jpg) |
|----------------------------------|----------------------------------|----------------------------------|
| ![이미지4](https://github.com/OSP-PJ/RevKeyRec/blob/feature-sw/image/4.jpg) | ![이미지5](https://github.com/OSP-PJ/RevKeyRec/blob/feature-sw/image/5.jpg) | ![이미지6](https://github.com/OSP-PJ/RevKeyRec/blob/feature-sw/image/6.jpg) |




**기술스택**
  -

### 🖥️ Front-End :
<img src="https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=React&logoColor=white"> <img src="https://img.shields.io/badge/javascript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=white">
<img src="https://img.shields.io/badge/html5-E34F26?style=for-the-badge&logo=html5&logoColor=white">
<img src="https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=CSS3&logoColor=white">

### 💻 Back-End :
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/fastapi-009688?style=for-the-badge&logo=fastapi&logoColor=white">
<img src="https://img.shields.io/badge/huggingface-FFD21E?style=for-the-badge&logo=huggingface-&logoColor=white">
<img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=GPT-4o&logoColor=white">
### 💻 Data-Base :
<img src="https://img.shields.io/badge/firebase-DD2C00?style=for-the-badge&logo=firebase&logoColor=white">  

### 🤝 Co-Work :
<img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white"> <img src="https://img.shields.io/badge/notion-000000?style=for-the-badge&logo=notion&logoColor=white" style="vertical-align: middle; display: inline;"> <a href="https://www.notion.so/c64c01d811b2400484ffda324078cf66?pvs=4">https://www.notion.so/c64c01d811b2400484ffda324078cf66?pvs=4</a>

| ![이미지1](https://github.com/OSP-PJ/RevKeyRec/blob/feature-sw/image/%EC%8A%AC%EB%9D%BC%EC%9D%B4%EB%93%9C17.JPG) | ![이미지2](https://github.com/OSP-PJ/RevKeyRec/blob/feature-sw/image/%EC%8A%AC%EB%9D%BC%EC%9D%B4%EB%93%9C18.JPG) | ![이미지3](https://github.com/OSP-PJ/RevKeyRec/blob/feature-sw/image/%EC%8A%AC%EB%9D%BC%EC%9D%B4%EB%93%9C19.JPG) |
|----------------------------------|----------------------------------|----------------------------------|



**Data_Base**
  -

>• **Base_Platfrom**
>->공정거래위원회(2022), 거래액 기준 시장 점유율 24.5%로 가장 높은 점유율을 기록
>
>• **오픈소스 코드를 통한 데이터 크롤링**  
>->99개의 상품 수집, 상품당 평균 800개의 리뷰 수집
>
>• **데이터 정제 및 가공**  
>-> Step1. GPT-4o 사용하여 한국어 문장의 불용어 제거 및 문장의 핵심 어구 추출  
>-> Step2. 텍스트 감성 분석 후 분류  
>-> Step3. 상품명 임베딩 - 모든 상품의 상품명을 임베딩하여 저장 - ko-sroberta-multitask 모델 사용
>
>• **데이터 베이스 설계**  
>->4가지 카테고리로 상품 분류 apple_data / IT / Others / Home



| ![이미지1](https://github.com/OSP-PJ/RevKeyRec/blob/feature-sw/image/gpt%EB%B6%88%EC%9A%A9%EC%96%B4%EC%B2%98%EB%A6%AC.jpg) | ![이미지2](https://github.com/OSP-PJ/RevKeyRec/blob/feature-sw/image/%EC%8A%AC%EB%9D%BC%EC%9D%B4%EB%93%9C26.JPG) | ![이미지3](https://github.com/OSP-PJ/RevKeyRec/blob/feature-sw/image/%EC%8A%AC%EB%9D%BC%EC%9D%B4%EB%93%9C27.JPG) |
|----------------------------------|----------------------------------|----------------------------------|
| ![이미지4](https://github.com/OSP-PJ/RevKeyRec/blob/feature-sw/image/%EC%8A%AC%EB%9D%BC%EC%9D%B4%EB%93%9C28.JPG) | ![이미지5](https://github.com/OSP-PJ/RevKeyRec/blob/feature-sw/image/%EC%8A%AC%EB%9D%BC%EC%9D%B4%EB%93%9C29.JPG) | ![이미지5](https://github.com/OSP-PJ/RevKeyRec/blob/feature-sw/image/%EC%8A%AC%EB%9D%BC%EC%9D%B4%EB%93%9C31.JPG) |



**Development Phase**
  -

### Front-end
>• **UX/UI:** 뉴모피즘, 글라스모피즘, 애니메이션
>  ->UI의 교과서 Apple사의 공식 웹사이트를 분석하여 심플함과 재미를 향상시킴  
>• **React:** 컴포넌트 기반 아키텍처 -> 애플리케이션을 컴포넌트를 기반으로 분리하여 코드를 재사용, 효율적인 개발 진행  

| ![이미지1](https://github.com/OSP-PJ/RevKeyRec/blob/feature-sw/image/%EC%8A%AC%EB%9D%BC%EC%9D%B4%EB%93%9C35.JPG) | ![이미지2](https://github.com/OSP-PJ/RevKeyRec/blob/feature-sw/image/%EC%8A%AC%EB%9D%BC%EC%9D%B4%EB%93%9C36.JPG) |
|----------------------------------|----------------------------------|
| ![이미지3](https://github.com/OSP-PJ/RevKeyRec/blob/feature-sw/image/%EC%8A%AC%EB%9D%BC%EC%9D%B4%EB%93%9C37.JPG) | ![이미지4](https://github.com/OSP-PJ/RevKeyRec/blob/feature-sw/image/%EC%8A%AC%EB%9D%BC%EC%9D%B4%EB%93%9C38.JPG) |


### Back-end
> **주요기능:벡터 검색 -> 정확하고 빠르게 검색 결과를 제공**  
> - `ko-sroberta-multitask` 모델을 이용하여 query 임베딩  
> - 상품명에 대한 정보를 순회하며 코사인 유사도 측정  
> - query와 유사도가 높은 상위 5개의 결과값을 Front에 전송
>
> **FastAPI**
> - 로컬서버구축

| ![이미지1](https://github.com/OSP-PJ/RevKeyRec/blob/feature-sw/image/%EC%8A%AC%EB%9D%BC%EC%9D%B4%EB%93%9C41.JPG) | ![이미지2](https://github.com/OSP-PJ/RevKeyRec/blob/feature-sw/image/%EC%8A%AC%EB%9D%BC%EC%9D%B4%EB%93%9C42.JPG) | ![이미지3](https://github.com/OSP-PJ/RevKeyRec/blob/feature-sw/image/%EC%8A%AC%EB%9D%BC%EC%9D%B4%EB%93%9C43.JPG) |
|----------------------------------|----------------------------------|----------------------------------|


## 기대효과


  
<table style="border-collapse: collapse; width: 100%; text-align: center; margin: 20px auto; table-layout: fixed;">
  <tr>
    <th style="border: 2px solid #000; padding: 60px; font-size: 36px;">CUSTOMER</th>
    <th style="border: 2px solid #000; padding: 60px; font-size: 36px;">SELLER</th>
  </tr>
  <tr>
    <td style="border: 2px solid #000; padding: 60px; font-size: 28px; line-height: 2;">
      이해 용이성<br>
      시간 절약<br>
      결정 지원
    </td>
    <td style="border: 2px solid #000; padding: 60px; font-size: 28px; line-height: 2;">
      피드백 분석<br>
      브랜드 관리<br>
      상품 개선
    </td>
  </tr>
</table>



