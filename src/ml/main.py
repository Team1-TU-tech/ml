from ml.db import *
from ml.utils import *
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 변수 로드

def update_performances_with_similarities(performances, similar_performances):
    try:
        client = MongoClient(mongo_uri)
        db = client['test']
        collection = db['similar']
        print("MongoDB connected successfully!")

    except Exception as e:
        print(f"MongoDB connection error: {e}")
    
    # 각 공연에 대해 유사한 공연들 정보를 업데이트
    for performance, top_similar in zip(performances, similar_performances):
        # 공연 ID 기준으로 유사 공연들 업데이트
        result = collection.update_one(
            {'_id': performance['_id']},  # 기준은 _id
            {
                '$set': {
                    'title': performance['title'],
                    'similar_performances': top_similar
                }
            },
            upsert=True  # 없으면 새로 삽입
        )
        if result.matched_count > 0:
            print(f"Updated performance: {performance['_id']}")
        else:
            print(f"Inserted new performance: {performance['_id']}")
    
    print("Performances updated successfully!")


def main():
    # 데이터 가져오기
    print("1. 데이터 가져오는 중...")
    performances = get_all_performances()
    print(f"데이터 로딩 완료: {len(performances)}개의 공연 데이터 로드")

    # 2. Word2Vec 모델 학습
    print("2. Word2Vec 모델 학습 중...")
    descriptions = [item['description'] for item in performances if item['description']]  # description이 None이 아닌 경우만 포함
    model = train_word2vec_model(descriptions)
    print("Word2Vec 모델 학습 완료")

    # 3. 공연 간 유사도 계산
    print("3. 공연 간 유사도 계산 중...")
    cosine_sim = calculate_cosine_similarity(model, descriptions)
    print(f"유사도 계산 완료: {len(cosine_sim)}개의 유사도 계산 완료")

    # 4. 유사도 기준으로 추천할 공연 리스트 생성
    print("4. 유사도 기준으로 추천할 공연 리스트 생성 중...")
    similar_performances = get_top_similar_performances(cosine_sim, performances)
    print(f"추천 공연 리스트 생성 완료: {len(similar_performances)}개의 공연에 대해 추천 리스트 생성 완료")

    # 5. MongoDB에 유사한 공연 정보 업데이트
    print("5. MongoDB에 유사한 공연 정보 업데이트 중...")
    update_performances_with_similarities(performances, similar_performances)
    print("MongoDB에 유사한 공연 정보 업데이트 완료")

if __name__ == "__main__":
    main()