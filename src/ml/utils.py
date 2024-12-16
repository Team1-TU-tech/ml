import string
from konlpy.tag import Okt
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

okt = Okt()

# 전처리 함수
def preprocess(text):
    """
    텍스트를 소문자화하고 구두점을 제거한 후, 형태소 분석하여 명사만 추출하는 함수
    """
    if text is None:
        return []  # None은 빈 리스트로 처리
    
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = okt.nouns(text)
    return tokens

def train_word2vec_model(descriptions):
    """
    전처리된 설명들로 Word2Vec 모델을 학습하는 함수
    """
    processed_descriptions = [preprocess(description) for description in descriptions]
    model = Word2Vec(processed_descriptions, vector_size=100, window=5, min_count=1, workers=4)
    return model

def get_word_vector(model, word):
    """
    주어진 단어의 벡터를 반환하는 함수
    """
    try:
        return model.wv[word]
    except KeyError:
        print(f"단어 '{word}'는 모델에 존재하지 않습니다.")
        return None

def filter_by_date(src_start_date, trg_start_date):
    """
    두 날짜를 비교하여 추천할 공연을 필터링하는 함수
    날짜가 동일하거나 가까운 범위 내의 공연을 추천 (예: 30일 이내)
    """
    try:
        src_start_date = datetime.strptime(src_start_date, '%Y.%m.%d')
        trg_start_date = datetime.strptime(trg_start_date, '%Y.%m.%d')
        return abs((src_start_date - trg_start_date).days) <= 270
    except Exception as e:
        print(f"날짜 형식 오류: {e}")
        return False

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
    
    # 각 공연 설명에 대한 평균 벡터 구하기
    description_vectors = [get_average_vector(model, description) for description in processed_descriptions]
    
    # 코사인 유사도 계산
    cosine_sim = cosine_similarity(description_vectors)
    
    return cosine_sim

def get_top_similar_performances(cosine_sim, performances, top_n=3):
    similar_performances = []
    
    # 각 공연에 대해 유사한 공연들을 찾기
    for idx, performance in enumerate(performances):
        if idx >= len(cosine_sim):  # idx가 cosine_sim의 범위를 벗어나지 않도록 체크
            print(f"Index {idx} is out of bounds for cosine_sim with size {len(cosine_sim)}")
            continue
        
        performance_similarities = cosine_sim[idx]  # 해당 공연과 다른 공연들의 유사도
        performance_start_date = datetime.strptime(performance['start_date'], '%Y.%m.%d')  # 공연 시작 날짜
        
        # 유사도 높은 순으로 정렬하되, 자기 자신은 제외
        similar_performances_idx = np.argsort(performance_similarities)[::-1][1:top_n+1]  # 유사도 높은 순

        # 유효한 인덱스만 선택 (범위를 벗어난 인덱스를 제외)
        similar_performances_idx = similar_performances_idx[similar_performances_idx < len(performances)]
        
        # 유효한 공연들의 유사도 정보 저장
        top_similar = []
        
        for i in similar_performances_idx:
            if i >= len(performances):
                continue  # 범위를 벗어난 인덱스는 건너뜀
            
            similar_performance = performances[i]
            similar_start_date = datetime.strptime(similar_performance['start_date'], '%Y.%m.%d')
            
            # 날짜 차이 계산
            days_difference = abs((performance_start_date - similar_start_date).days)
            
            # 날짜 차이가 0인 경우는 제외
            if days_difference == 0:
                continue
            
            top_similar.append({
                'performance_id': similar_performance['_id'],  # 공연 ID (MongoDB의 경우)
                'similarity_score': performance_similarities[i],  # 유사도 점수
                'date_difference': days_difference  # 날짜 차이
            })
        
        # 유사도와 날짜 차이를 기준으로 정렬 (유사도 우선, 날짜 차이는 두 번째)
        top_similar_sorted = sorted(top_similar, key=lambda x: (-x['similarity_score'], x['date_difference']))
        similar_performances.append(top_similar_sorted)
    
    return similar_performances