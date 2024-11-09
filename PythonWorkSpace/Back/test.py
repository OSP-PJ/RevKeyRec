from sklearn.feature_extraction.text import TfidfVectorizer

texts=[]
vectorizer = TfidfVectorizer()
sample = vectorizer.fit_transform(texts) #주의! iterable한 객체여야함

print(sample.toarray()) #해당 벡터 출력함