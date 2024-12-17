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
<br></br>
## get_top_similar_performances 함수
```python
def get_top_similar_performances(cosine_sim: np.ndarray, performances, top_n=3, threshold=0.98, days_limit=90)
```
- `cosine_sim (np.ndarray)`: 공연 간의 코사인 유사도 행렬입니다. 각 원소는 두 공연 간의 유사도를 나타냅니다.
- `performances`: 공연들의 정보가 포함된 리스트입니다. 각 공연은 딕셔너리 형태로, 최소한 start_date(공연 시작일)와 region(지역) 키를 포함해야 합니다.
- `top_n (int, 기본값=3)`: 각 공연에 대해 추천할 유사한 공연의 수를 제한합니다. 기본값은 3개로 설정되어 있습니다.
- `threshold (float, 기본값=0.98)`: 유사도 임계값입니다. 이 값보다 낮은 유사도를 가진 공연만 추천됩니다. 이유는 1은 자기자신을 의미하며 0.98~0.99 의 경우 지역만 다르고 같은 공연을 나타냅니다.
- `days_limit (int, 기본값=90)`: 기준 공연과 유사한 공연의 종료일이 기준 공연의 시작일로부터 90일 이내인 경우만 추천합니다.
<br></br>
이 알고리즘을 사용하려면 다음과 같은 Python 패키지가 필요합니다:
```bash
pip install numpy pandas scikit-learn gensim konlpy
```
<br></br>
## Contributors
`hamsunwoo`
<br></br>
## License
이 애플리케이션은 TU-tech 라이선스에 따라 라이선스가 부과됩니다.
<br></br>
## 문의
질문이나 제안사항이 있으면 언제든지 연락주세요:
- 이메일: TU-tech@tu-tech.com
- Github: `Mingk42`, `hahahellooo`, `hamsunwoo`, `oddsummer56`


