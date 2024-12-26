import string
from konlpy.tag import Okt
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta

okt = Okt()

# 전처리 함수
def preprocess(text):
    """
    텍스트를 소문자화하고 구두점을 제거한 후, 형태소 분석하여 명사만 추출하는 함수
    """
    if not text or not isinstance(text, str):
        return ["default"]  # 기본값 처리
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = okt.nouns(text)
    return tokens if tokens else ["default"]  # 빈 리스트 방지

def train_word2vec_model(descriptions):
    """
    전처리된 설명들로 Word2Vec 모델을 학습하는 함수
    """
    # 데이터 검증
    none_count = sum(1 for desc in descriptions if desc is None)
    empty_count = sum(1 for desc in descriptions if isinstance(desc, str) and not desc.strip())
    print(f"None 개수: {none_count}, 빈 문자열 개수: {empty_count}")

    # 전처리
    processed_descriptions = [preprocess(description) for description in descriptions]
    print(f"전처리된 데이터 크기: {len(processed_descriptions)}")

    # 전처리 후 빈 리스트 검증
    empty_processed = [desc for desc in processed_descriptions if not desc]
    print(f"전처리 후 빈 리스트 개수: {len(empty_processed)}")

    # 빈 리스트 기본값 추가
    processed_descriptions = [desc if desc else ["default"] for desc in processed_descriptions]
    print(f"Word2Vec 학습 데이터 크기: {len(processed_descriptions)}")


    model = Word2Vec(processed_descriptions, vector_size=100, window=5, min_count=1, workers=4)

    return model

# 3. 벡터 계산 함수
def get_average_vector(model, tokens):
    """
    주어진 단어들의 벡터 평균을 계산하는 함수
    """
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

def get_average_vector(model, tokens):
    """
    주어진 단어들의 벡터 평균을 계산하는 함수
    """
    vectors = []
    for word in tokens:
        if word in model.wv:
            vectors.append(model.wv[word])
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

def calculate_cosine_similarity(model, descriptions):
    """
    공연 설명별로 평균 벡터를 구하고, 코사인 유사도를 계산하는 함수
    """
    processed_descriptions = [preprocess(description) for description in descriptions]
    print(f"전처리된 설명 데이터 크기: {len(processed_descriptions)} / 원본 데이터 크기: {len(descriptions)}")
    
    # 각 공연 설명에 대한 평균 벡터 구하기
    description_vectors = np.array([get_average_vector(model, desc) for desc in processed_descriptions])
    print(f"벡터화된 데이터 크기: {len(description_vectors)}")
    print(f"0 벡터 개수: {sum(1 for vec in description_vectors if np.all(vec == 0))}")
    
    # 코사인 유사도 계산
    cosine_sim = cosine_similarity(description_vectors)
    print(f"코사인 유사도 계산 완료: {cosine_sim.shape}")

    return cosine_sim


def get_top_similar_performances(cosine_sim: np.ndarray, performances, top_n=3, threshold=0.98, days_limit=90):
    similar_performances = []

    # 각 공연에 대해 유사한 공연들을 찾기
    for idx, performance in enumerate(performances):
        # 유효성 검사: 인덱스 초과 시 기본값 처리
        if idx >= len(cosine_sim):
            print(f"Index {idx} is out of bounds for cosine_sim with size {len(cosine_sim)}")
            performance_similarities = np.zeros(len(performances))  # 기본값으로 유사도 0 배열 사용
        else:
            performance_similarities = cosine_sim[idx]
            
         # start_date가 None인 경우 건너뛰기
        performance_start_date_str = performance.get('start_date', None)
        if performance_start_date_str is None:
            continue

        performance_start_date = datetime.strptime(performance_start_date_str, '%Y.%m.%d')  # 공연 시작 날짜
        performance_similarities = cosine_sim[idx]  # 해당 공연과 다른 공연들의 유사도
        performance_region = performance.get('region', None)

        # 공연의 start_date 기준으로 90일 이내인 공연만 필터링
        date_limit = performance_start_date + timedelta(days=days_limit)
        
        # 유사도 높은 순으로 정렬하되, 자기 자신은 제외하고, threshold 이상의 유사도를 가진 공연들만 필터링
        similar_performances_idx = np.argsort(performance_similarities)[::-1]  # 유사도 높은 순으로 정렬
        top_similar = []
        
        for similar_idx in similar_performances_idx:
            # 자기 자신은 제외하고, 유사도 threshold 이하이어야 하며, 같은 지역이어야 함
            if similar_idx != idx and performance_similarities[similar_idx] < threshold:
                similar_performance = performances[similar_idx]

                similar_performance_start_date_str = similar_performance.get('start_date', None)

                # start_date가 None인 경우 건너뛰기
                if similar_performance_start_date_str is None:
                    continue

                similar_performance_start_date = datetime.strptime(similar_performance_start_date_str, '%Y.%m.%d') 
                #similar_performance_start_date = datetime.strptime(similar_performance['start_date'], '%Y.%m.%d')
                similar_performance_region = similar_performance.get('region', None)
                
                if performance_region != similar_performance_region:
                    continue
                
                # 지역이 동일하고, 90일 이내인 경우만 유사 공연으로 추가
                if performance_start_date <= similar_performance_start_date <= date_limit:
                    top_similar.append(similar_performance)
                
                # 이미 top_n개 만큼 찾았으면 추가 중지
                if len(top_similar) >= top_n:
                    break
        
        # 결과 리스트에 유사 공연 추가
        similar_performances.append(top_similar)
    
    return similar_performances
