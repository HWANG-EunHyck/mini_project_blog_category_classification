# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import cgi
# import cgitb
# import os
# import pickle
# import torch
# import torch.nn as nn
# from konlpy.tag import Okt

# # CGI 디버깅 활성화
# cgitb.enable()

# # 웹 출력 설정
# print("Content-Type: text/html\n")

# # 형태소 분석기
# okt = Okt()

# # 모델 클래스 정의
# class TextClassifier(nn.Module):
#     def __init__(self, vocab_size, embed_dim, num_classes):
#         super(TextClassifier, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
#         self.fc = nn.Linear(embed_dim, num_classes)

#     def forward(self, x):
#         embedded = self.embedding(x)
#         pooled = torch.mean(embedded, dim=1)
#         output = self.fc(pooled)
#         return output

# # 텍스트를 시퀀스로 변환하는 함수
# def text_to_sequence(text, vocab):
#     return [vocab.get(word, vocab['<UNK>']) for word in text.split()]

# # 저장된 모델 및 파일 경로
# MODEL_PATH = os.path.join(os.path.dirname(__file__), 'text_classification_model.pth')
# VOCAB_PATH = os.path.join(os.path.dirname(__file__), 'vocab.pkl')
# LABEL_ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'label_encoder.pkl')

# # 저장된 vocab과 label encoder 로드
# with open(VOCAB_PATH, 'rb') as f:
#     vocab = pickle.load(f)

# with open(LABEL_ENCODER_PATH, 'rb') as f:
#     le = pickle.load(f)

# # 장치 설정 (CPU)
# device = torch.device('cpu')

# # 모델 설정
# model = TextClassifier(len(vocab), 100, len(le.classes_))
# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# model.to(device)
# model.eval()

# # 입력된 텍스트 처리 함수
# def classify_text(text):
#     morphemes = ' '.join(okt.morphs(text))
#     sequence = text_to_sequence(morphemes, vocab)
#     sequence = [min(t, len(vocab) - 1) for t in sequence]  # 인덱스 범위 확인
#     sequence = torch.tensor([sequence]).to(device)

#     with torch.no_grad():
#         output = model(sequence)
#         _, predicted = torch.max(output.data, 1)
    
#     return le.inverse_transform(predicted.cpu().numpy())[0]

# # CGI로부터 입력된 값 처리
# form = cgi.FieldStorage()
# input_text = form.getvalue('text')

# if input_text:
#     # 입력된 텍스트 분류
#     predicted_label = classify_text(input_text)
    
#     # 결과 출력
#     print(f"<h1>입력 텍스트: {input_text}</h1>")
#     print(f"<h2>예측된 레이블: {predicted_label}</h2>")
# else:
#     print("<h1>Error: 텍스트가 입력되지 않았습니다.</h1>")

'''

- 0 : 건강,의학
- 1 : 교육,학문 
- 2 : it,컴퓨터


python -m http.server --cgi 8080

http://localhost:8080/cgi-bin/class.py?text=
'''