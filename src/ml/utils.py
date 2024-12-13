import string
from konlpy.tag import Okt
from gensim.models import Word2Vec
from datetime import datetime

okt = Okt()

# 전처리 함수
def preprocess(text):
    """
    텍스트를 소문자화하고 구두점을 제거한 후, 형태소 분석하여 명사만 추출하는 함수
    """
    # 소문자화 및 구두점 제거
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # 형태소 분석하여 명사만 추출
    tokens = okt.nouns(text)
    
    return tokens

def train_word2vec_model(descriptions):
    """
    전처리된 설명들로 Word2Vec 모델을 학습하는 함수
    """
    # 설명 리스트를 전처리
    processed_descriptions = [preprocess(description) for description in descriptions]
    
    # Word2Vec 모델 학습
    model = Word2Vec(processed_descriptions, vector_size=100, window=5, min_count=1, workers=4)
    """
    vector_size: 단어 벡터의 차원 수
    window: 문맥 윈도우 크기 (주변 단어를 얼마나 볼 것인지)
    min_count: 최소 등장 횟수 이상인 단어만 학습에 사용
    workers: 멀티 프로세싱을 위한 스레드 수
    """
    
    return model

def get_word_vector(model, word):
    """
    주어진 단어의 벡터를 반환하는 함수
    """
    try:
        word_vector = model.wv[word]
        return word_vector
    except KeyError:
        print(f"단어 '{word}'는 모델에 존재하지 않습니다.")
        return None

def filter_by_date(src_start_date, trg_start_date):
    """
    두 날짜를 비교하여 추천할 공연을 필터링하는 함수.
    날짜가 동일하거나 가까운 범위 내의 공연을 추천 (예: 30일 이내)
    """
    try:
        # 날짜 포맷: 'YYYY.MM.DD'로 가정
        src_start_date = datetime.strptime(src_start_date, '%Y.%m.%d')
        trg_start_date = datetime.strptime(trg_start_date, '%Y.%m.%d')
        
        # 예시로 30일 내에 시작하는 공연을 추천하도록 설정
        return abs((src_start_date - trg_start_date).days) <= 30
    except Exception as e:
        print(f"날짜 형식 오류: {e}")
        return False
