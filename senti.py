import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# CSV 데이터 불러오기
csv_file = "news.csv"  # CSV 파일 경로
data = pd.read_csv("news.csv", header=None)

# KoBERT 토크나이저와 모델 로드
tokenizer = BertTokenizer.from_pretrained("snunlp/KR-Medium")
model = BertForSequenceClassification.from_pretrained("snunlp/KR-Medium")

# 감성 분석할 텍스트 열 선택
text_column = 1  # 감성 분석을 수행할 텍스트 열 이름
texts = data[text_column].tolist()

# 텍스트 토큰화 및 전처리
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# 추론 수행
with torch.no_grad():
    outputs = model(**inputs)

# 감성 점수 추출
sentiment_scores = outputs.logits

# 필요한 작업으로 감성 점수 처리
# 예를 들어, 점수를 해석하여 긍정/부정 감성을 판단할 수 있습니다.

# 결과 저장 또는 분석

for i, score in enumerate(sentiment_scores):
    negative_score = score[0]
    positive_score = score[1]

    # 감성 점수 분석
    if positive_score > negative_score:
        sentiment = "긍정적"
    elif negative_score > positive_score:
        sentiment = "부정적"
    else:
        sentiment = "중립적"

    print(f"텍스트 {i + 1}: 긍정 점수 - {positive_score:.4f}, 부정 점수 - {negative_score:.4f}, 감성 - {sentiment}")