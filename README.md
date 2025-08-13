# Horse or Human Classifier (Sigmoid Model)

🧠 **소개**  
이 프로젝트는 **머신러닝(딥러닝)**을 이용해 이미지를 보고 그것이 **사람인지, 말인지** 구분하는 분류 프로그램입니다.  
`Sigmoid` 출력 구조를 가진 모델을 사용하여, 입력 이미지에 대해 **사람일 확률(0~1)**을 계산하고, 이를 기준으로 클래스를 판별합니다.  
UI를 통해 사용자가 이미지를 업로드하면, 분석 결과와 함께 신뢰도(%)를 직관적으로 제공합니다.

---

## ✨ 예시

### 1.사람 이미지 입력

    Sigmoid 출력: 0.9823

    결과: Human (98.23%)

### 2.말 이미지 입력

    Sigmoid 출력: 0.0001

    결과: Horse (99.99%)

    ![hores_or_human](https://github.com/user-attachments/assets/0bd08ae9-cc91-445b-81d5-c3ca85adb211)


---

## 🚀 사용 방법

### 1. 환경 설정
먼저, 필요한 패키지를 설치합니다.

```bash

pip install -r requirements.txt
```

---

### 2. 프로그램 실행
메인 스크립트를 실행합니다.

```bash
python main.py
```

---

### 3. UI 사용
실행 시 표시되는 UI 창에서 이미지 업로드 버튼을 클릭하여 사람 또는 말 이미지를 선택합니다.

선택한 이미지가 미리보기로 표시됩니다.

분석 버튼을 클릭하면, 결과와 확률이 함께 표시됩니다.

예: Horse (99.99%) 또는 Human (98.23%)

---

## ⚙️ 동작 방식

### 모델 구조
### 출력 형태: 
    Sigmoid (0~1 사이 값)

### 클래스 라벨:

    0에 가까울수록 → 말(Horse)

    1에 가까울수록 → 사람(Human)

### 임계값: 
    기본 0.5 (필요 시 튜닝 가능)

### 판별 로직 예시 (Python)

```
python

p = float(model.predict(x)[0][0])  # Sigmoid 출력값 (사람일 확률)
CLASS0, CLASS1 = "Horse", "Human"  # 0=말, 1=사람
threshold = 0.5

if p >= threshold:
    label = CLASS1
    confidence = p
else:
    label = CLASS0
    confidence = 1 - p

print(f"{label} ({confidence*100:.2f}%)")
```

---

## 📄 라이선스
이 프로젝트는 MIT 라이선스를 따릅니다.

---

## 📧 연락처

궁금한 점이나 제안 사항이 있으신가요?

[안진홍] - [ajh9703@gmail.com]

---
