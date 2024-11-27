from sentence_transformers import SentenceTransformer
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def search_similar_products(query, embeddings):
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')  
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings) #embeddomgs 는 2차원 리스트이므로 이상없이 작동 됨
    similarities = similarities[0]
    top_indices = np.argsort(similarities)[::-1][:5]
    #argsort 함수 사용, similarities 배열의 값들을 오름차순으로 정렬 -> [::-1]로 내림차순 정렬 -> 상위 5개 출력 (0~4번 index)
    
    similar_products = []
    for idx in top_indices: #값들의 원래 인덱스를 읽어옴
        if data['products'][idx] and 'name' in data['products'][idx]:
            similar_products.append({
                'name': data['products'][idx]['name'],
                'similarity': similarities[idx]
            }) #원래 인덱스, 즉 코사인 유도로 찾은 유사도가 가장 높은 값의 데이터들의 제품명, 유사도를 반환
    return similar_products #상위 5개에 대한 내용 리턴


with open("C:/RevKeyRec/RevKeyRec/PythonWorkSpace/Back/osp-revkeyrec-default-rtdb-export.json", 'r+', encoding='utf-8') as f:
    data = json.load(f)
    embeddings = [product['vector'] for product in data['products'] if product and 'vector' in product]
# 예시 검색 쿼리
query = str(input()) #APPLE 와 애플 두가지 잘 되는지 확인
similar_products = search_similar_products(query, embeddings) #함수 호출
print(similar_products)
