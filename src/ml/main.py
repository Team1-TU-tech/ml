from src.ml.db import connect_db
from src.ml.utils import *
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def get_data():
    #data = connect_db()

    data = [
        {
            'content': "Mnet ‘로드 투 킹덤: ACE OF ACE’를 통해 독보적인 서사와 영화 같은 반전이 담긴 무대로 강렬한 인상을 남긴 그룹 YOUNITE가 첫 번째 싱글 앨범 ‘Y’로 화려하게 돌아옵니다.\n최초로 공개될 타이틀곡 무대는 물론, 생동감 넘치고 역동적인 에너지를 담아낸 다채로운 퍼포먼스와 YOUNITE만의 독특한 '맛'을 선사할 첫 번째 싱글 앨범 ‘Y’ 컴백 쇼케이스!\n12월 11일 오후 8시, 강렬한 기억으로 남을 이 특별한 자리에 여러분을 초대합니다. YOUNITE와 함께 잊지 못할 순간을 만끽하세요",
            'start_date': '2024.01.01',
            'end_date':'2024.03.01'
        },
        {
            'content': "1994년 오리지널 애니메이션 영화 ‘라이온 킹’에는 슈퍼스타 ‘엘튼 존’, 과 작사가 ‘팀 라이스’, 그리고 영화 음악의 거장 ‘한스 짐머’등 주목할 만한 오스카 및 그래미 어워즈 수상자 팀의 잊을 수 없는 음악과, 남아프리카공화국 프로듀서 겸 작곡가 ‘레보 M’의 아프리카의 노래와 합창이 담겼다.\n디즈니의 '라이온 킹'은 미래의 왕이 탄생하는 아프리카 사바나로 여행을 떠납니다. 심바는 아버지 무파사 왕을 우상화하고 자신의 왕실 운명을 가슴에 새깁니다. 하지만 왕국의 모든 사람이 무파사의 새로운 아기의 도착을 축하하는 것은 아닙니다. 무파사의 동생이자 전 왕위 계승자인 스카는 자신만의 계획을 가지고 있습니다. 프라이드 록을 위한 전투는 배신과 비극, 드라마로 가득 차 있으며 결국 심바는 망명 생활을 하게 됩니다. 오랫동안 잃어버린 친구 날라와 새로 만난 호기심 많은 친구 품바와 티몬의 도움을 받아 심바는 자신의 과거를 직시하고 올바른 것이 무엇인지 되찾아야 합니다. 심바 역의 매튜 브로더릭, 날라 역의 모이라 켈리, 무파사 역의 제임스 얼 존스, 스카 역의 제레미 아이언스, 품바 역의 어니 사벨라, 티몬 역의 네이선 레인이 올스타에 출연합니다.\n디즈니 애니메이션의 걸작 ‘라이온 킹’은 아름다운 음악으로 뮤지컬로도 제작되어, 전세계 팬들에게 큰 사랑을 받고 있다.\n올 해, 개봉 30주년을 맞아 L.A Hollywood Bowl, 런던 Royal Albert Hall 등 세계 유수의 극장에서 기념 공연이 열렸다.\n대형 스크린으로 만나는 오리지널 ‘라이온 킹’과 풀 오케스트라와 합창단이 만들어 내는 감동을 온 가족과 함께 놓치지 말아야 할 것이다",
            'start_date':'2024.01.13',
            'end_date':'2024.04.01'
        },
        {
            'content': "“나의 밤을 온통 지배하는 자여. 썩어가는 것을 멈추고 대답하라!”\n윌리엄 셰익스피어와 크리스토퍼 말로의 만남과 숨겨진 이야기!\n동시대를 살았고, 서로가 서로에게 많은 영향을 끼쳤다고 하는 두 명의 위대한 작가,\n윌리엄 셰익스피어와 크리스토퍼 말로에 대해 사람들은 다양한 소문을 만들어 내기도, 다양한 해석을 내놓기도 했다.\n‘같은 시대를 살며, 서로의 작품에 영향을 미쳤지만,\n다른 작품의 길을 걸어 온 두 작가의 대화는 어떠했을까?’라는 생각에서 이 작품은 시작되었다.\n극 안에서 크리스토퍼는 윌리엄에게 숨겨둔 이야기, ‘나인’을 완성해달라고 제안하고\n윌리엄은 이를 받아들이고 나인과 함께 이야기를 만들어 간다.\n아름다운 이야기만 펼쳐질 것 같았던 시간 속에서,\n크리스토퍼는 이 모습을 지켜보며 다른 음모를 꾸미며 윌리엄과의 대화를 이어간다.\n[SYNOPSIS]\n1596년, 런던의 어느 날 밤.\n마법의 시간이 흐르고 붉은 달이 떠오르자\n윌리엄은 희미한 등불에 기대어 크리스토퍼의 무덤을 찾는다.\n마침내 짙은 어둠 속에서 크리스토퍼의 망령과 마주하는 윌리엄.\n윌리엄은 망령의 죽음을 둘러싼 무성한 소문들에 관해 묻는다.\n뜻밖의 질문이 재미있다는 듯\n윌리엄에게 숨겨둔 미완의 이야기,\n‘나인’을 완성해달라 제안하는 망령.\n윌리엄은 망령과의 거래가\n그를 어떤 세계로 이끌지 알지도 못한 채 이를 수락하고 만다.\n서로 달랐지만, 너무 닮았던\n두 시인이 언어라는 마법으로\n무대 위에 그리고자 했던 단 하나의 빛.\n그 영원한 빛에 관한 이야기.\n납처럼 무겁고\n깃털처럼 가벼운 펜을 들어 그려낸\n또 하나의 세계.\n그림자를 가진\n세상의 모든 이들에게 바치는 노래",
            'start_date':'2024.02.01',
            'end_date':'2024.05.01'
        }
    ]


# Word2Vec 모델 학습
data = get_data()
model = train_word2vec_model(data)

# 유사도를 기준으로 추천할 공연 리스트 생성
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

def calculate_cosine_similarity(model, data):
    """
    공연(performance) 줄거리별로 평균 벡터를 구하고, 코사인 유사도를 계산하는 함수
    """
    # 각 공연(performance) 줄거리에 대한 전처리된 토큰 리스트 구하기
    processed_performances = [preprocess(performance) for performance in data]

    # 각 공연(performance)의 평균 벡터 구하기
    performance_vectors = [get_average_vector(model, performance) for performance in processed_performances]

    # 코사인 유사도 계산
    cosine_sim = cosine_similarity(performance_vectors)

    return cosine_sim

# 코사인 유사도 계산
cosine_sim = calculate_cosine_similarity(model, data)

# 유사도를 기준으로 내림차순 정렬
similar_performances = list(enumerate(cosine_sim[0]))
sorted_similar_performances = sorted(similar_performances, key=lambda x: x[1], reverse=True)

# 추천 공연 출력
threshold = 0.2  # 유사도의 임계값 설정
print("추천 공연:")

# 각 공연의 시작 날짜 (예시: 'YYYY.MM.DD' 형식)
performance_start_dates = ['2024.01.01', '2024.01.10', '2024.02.15']

for i in sorted_similar_performances[1:4]:  # 자기 자신 제외
    src_start_date = performance_start_dates[0]  # 첫 번째 공연을 기준으로
    trg_start_date = performance_start_dates[i[0]]  # 현재 추천된 공연의 날짜
    if i[1] > threshold and filter_by_date(src_start_date, trg_start_date):
        print(f"공연 {i[0]}: 유사도 {i[1]:.4f}, 시작일: {trg_start_date}")