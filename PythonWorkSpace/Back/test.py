import json
from sklearn.feature_extraction.text import TfidfVectorizer

# JSON 파일에서 utf8 형식으로 데이터 불러오기
with open("C:/RevKeyRec/RevKeyRec/PythonWorkSpace/Back/osp-revkeyrec-default-rtdb-export.json", 'r+', encoding='utf-8') as f:
    data = json.load(f)
# texts = [product['name'] for product in data.values()] 
    texts = [product['name'] for product in data['products'] if product and 'name' in product]
#지능형 리스트 사용!
#json의 구조가 딕셔너리 안에 products 라는 키에 대해 리스트로 값을 갖고 있으며, 리스트 안에 다시 딕셔너리 형태로 상품을 저장 중
#product가 None이 아니고, product 안에 'name' 키가 있는지 확인

    vectorizer = TfidfVectorizer()
    sample = vectorizer.fit_transform(texts) #주의! iterable한 객체여야함
#json 파일 vextor 키에 계산한 벡터값 넣어주기
    print(type(sample))

    array_sample = sample.toarray() #해당 벡터 출력함
    list_sample = array_sample.tolist() #json 파일에 저장하기 위해 리스트 형태로 변환 진행


    i = 0 #list_sample 의 인덱스에 접근하기 위한 변수
    for  product in data['products']: #각 상품에 접근
        if product and 'vector' in product: #NULL 이 아니고 상품 딕셔너리 안에 vector라는 키가 있다면
            product["vector"].clear() #내부를 초기화 하고
            product["vector"]=list_sample[i][:] #값 추가
            i+=1 #다음 인덱스로 넘어가기 위해 1 증가

    # 파일 포인터를 파일의 시작 위치로 이동
    f.seek(0)
    
    # 수정된 데이터를 파일에 덮어쓰기
    json.dump(data, f, ensure_ascii=False, indent=4)