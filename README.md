# 유사 공연 추천 알고리즘
이 프로젝트는 공연 데이터를 기반으로 유사한 공연을 추천하는 알고리즘을 구현한 것입니다. 주로 공연의 설명(description)과 시작일(start_date), 지역(region) 등의 정보를 활용하여 유사한 공연을 찾아 추천합니다.
<br></br>
## 기술스택
<img src="https://img.shields.io/badge/Python-3.9-3776AB?style=flat&logo=Python&logoColor=F5F7F8"/>   <img src="https://img.shields.io/badge/scikitlearn-F7931E?style=flat&logo=scikitlearn&logoColor=F5F7F8"/>    <img src="https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=F5F7F8"/>    <img src="https://img.shields.io/badge/numpy-013243?style=flat&logo=numpy&logoColor=F5F7F8"/>    <img src="https://img.shields.io/badge/mongodb-47A248?style=flat&logo=mongodb&logoColor=F5F7F8"/>
<br></br>
## 개발기간
`2024.12.12 ~ 2024.12.16 (총 5일)`
<br></br>
## 기능설명
- **공연 데이터 분석**: 주어진 공연 데이터에서 중요한 특성(설명, 지역, 날짜 등)을 기반으로 공연을 분석합니다.
- **유사도 계산**: 각 공연 간의 유사도를 계산하여 유사한 공연들을 추천합니다.
- **추천 시스템**: 공연 설명을 벡터화하고, 코사인 유사도를 이용해 유사한 공연들을 추천합니다.
- **지역 및 날짜 필터링**: 유사 공연 추천 시, 지역과 날짜 기준으로 필터링하여 정확도를 높입니다.



