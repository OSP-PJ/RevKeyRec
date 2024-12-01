from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json


model = SentenceTransformer('jhgan/ko-sroberta-multitask')

path = "D:/학교/team/RevKeyRec/rsc/Phrases_Json/독거미 키보드 국내정품 AULA F98 PBT RGB .json" # 임의로 하나 집어 넣음
with open(path, 'r+', encoding='utf-8') as f:
    data = json.load(f)

review = data["data"] # 불용어 처리된 리뷰 불러 오기
lr = len(review)

embeddings = model.encode(review) #상품명 벡터화 진행 (인자는 iterable한 객체여야함)
cos_similar = cosine_similarity(embeddings) # 각 행은 행 번호에 해당하는 문장을 기준으로 각 문장에 대한 유사도를 저장 = 정방 메트릭스가 만들어짐

sort_similar_index = []
for i in range(lr) :
    sort_similar_index.append(np.argsort(cos_similar[i])[::-1]) # 각 행마다 유사도 기준으로 정렬한 인덱스를 반환

cccc = []
picked = []
cscs = []
for i in range(lr) :
    pr = []
    cs = []
    j = 0
    count = 0
    while count < 15 and j < lr:
        index = sort_similar_index[i][j]
        
        if index not in picked :
            if cos_similar[i][index] < 0.55 :
                break
            
            else :
                picked.append(index)
                cs.append(cos_similar[i][index])
                pr.append(review[index])
                count += 1
        j += 1
    if len(pr) > 0:
        cccc.append(pr)
        cscs.append(cs)
        
tc = sorted(cccc, key = len, reverse = True)
l = 0
for i in range(len(tc)) :
    print(tc[i])
    l += len(tc[i])
    
print(l)
# for review_list in cccc :    
#     if len(review_list) > 5 :
#         print(review_list)
#     else :
#         continue 

output_path = "D:/학교/team/RevKeyRec/rsc/Group_Json/독거미 키보드 국내정품 AULA F98 PBT RGB .json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(tc, f, ensure_ascii=False, indent=4)