# Boost up AI 2025 : 신약 개발 경진대회

**[Dacon] Boost up AI 2025 : Drug Discovery Competition** 화합물의 구조 정보를 이용하여 CYP3A4 효소 저해율(%inhibition)을 예측하는 AI 모델 개발

🔗 [대회 링크](https://dacon.io/competitions/official/236518/overview/description)

## 🏆 Result
* **Public Score**: 10위
* **Private Score**: 39위

## 📖 Overview
- **CYP3A4 효소 저해율**을 예측하기 위해 (GNN) 기반의 모델링을 수행.
- 대규모 분자 데이터셋(ZINC)을 이용한 사전 학습과 외부 데이터(PubChem) 활용.

### Key Strategies
1.  **Model Architecture**: `PNA (Principal Neighbourhood Aggregation)` 기반의 GNN 모델 사용
2.  **Self-Supervised Pre-training**:
    * **Dataset**: ZINC (약 2,300만 개)
    * **Method**: Contrastive Learning (Node/Edge Masking Augmentation 적용)
3.  **External Data Utilization**:
    * PubChem (AID 1851, 884, 885) 데이터를 수집 및 정제하여 학습 데이터 확장
    * Hill Equation 및 Potency 값을 활용하여 Inhibition 값으로 변환
4.  **Fine-tuning & Regularization**:
    * Mixup, EMA (Exponential Moving Average) 적용
    * Seed Ensemble을 통한 최종 예측

## 📂 Project Structure
```text
drug_discovery_2025/
│
├── src/                          
│   ├── __init__.py
│   ├── dataset.py                # SMILES -> Graph 변환 및 Dataset 클래스
│   └── augmentation.py           # Graph Augmentation (Masking) 함수
│
├── notebooks/
│   ├── 01_zinc_prep.ipynb          # Step 1: ZINC 데이터셋 전처리
│   ├── 02_external_data_prep.ipynb # Step 2: 외부 데이터(PubChem) 전처리
│   └── 03_finetuning_main.ipynb    # Step 3: 메인 학습 및 추론 (Fine-tuning)
│
├── scripts/                      # 사전 학습 스크립트
│   └── pretraining.py            # Step 1.5: PNA 모델 사전 학습 실행
│
└── data/                        
    ├── raw/                      
    ├── processed/                
    └── external/
``` 
        
