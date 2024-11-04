#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cgi
import cgitb
import os
import pickle
import torch
import torch.nn as nn
from konlpy.tag import Okt
import sys
import io

cgitb.enable()

#  UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Content-Type: text/html; charset=utf-8\n")

# 형태소
okt = Okt()

# 모델 클래스
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = torch.mean(embedded, dim=1)
        output = self.fc(pooled)
        return output

# 텍스트를 시퀀스로 변환
def text_to_sequence(text, vocab):
    return [vocab.get(word, vocab['<UNK>']) for word in text.split()]

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'text_classification_model.pth')
VOCAB_PATH = os.path.join(os.path.dirname(__file__), 'vocab.pkl')
LABEL_ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'label_encoder.pkl')

with open(VOCAB_PATH, 'rb') as f:
    vocab = pickle.load(f)

with open(LABEL_ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)

device = torch.device('cpu')

# 모델 
model = TextClassifier(len(vocab), 100, len(le.classes_))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# 텍스트 처리 
def classify_text(text):
    morphemes = ' '.join(okt.morphs(text))  # 형태소 분석
    sequence = text_to_sequence(morphemes, vocab)
    sequence = [min(t, len(vocab) - 1) for t in sequence]  # 인덱스 범위 확인
    sequence = torch.tensor([sequence]).to(device)

    with torch.no_grad():
        output = model(sequence)
        _, predicted = torch.max(output.data, 1)  # 가장 높은 확률의 클래스를 예측
    
    return predicted.item()

# CGI로부터 입력된 값 처리
form = cgi.FieldStorage()
input_text = form.getvalue('text')

# HTML 출력 시작
print("""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Cache-Control" content ="no-cache,no-store,must-revalidate">
    <title>텍스트 분류기</title>
</head>
<body align="center">
    <h1>블로그 카테고리 예측</h1>
    <form method="post" action="/cgi-bin/class_11.py">
        <label for="text">판별하고 싶은 블로그 제목을 적어주세요!</label><br><br>
        <input type="text" id="text" name="text" size="100"><br><br>
        <input type="submit" value="예측">
    </form>
""")

# 텍스트 입력이 있을 때 분류를 실행
if input_text:
    # 입력된 텍스트 분류
    predicted_label = classify_text(input_text)
    
    # 예측된 결과를 해석
    if predicted_label == 0:
        result_text = "건강, 의학 블로그입니다."
    elif predicted_label == 1:
        result_text = "교육, 학문 블로그입니다."
    elif predicted_label == 2:
        result_text = "IT, 컴퓨터 블로그입니다."
    else:
        result_text = "알 수 없는 카테고리입니다."

    # 결과 출력
    print(f"<h2>입력 텍스트: {input_text}</h2>")
    print(f"<h3>예측된 블로그: {result_text}</h3>")
else:
    print("<h3>뭘까용?!</h3>")

print("""
</body>
</html>
""")


'''

- 0 : 건강,의학
- 1 : 교육,학문 
- 2 : it,컴퓨터


python -m http.server --cgi 8070

http://localhost:8070/cgi-bin/class_11.py
'''