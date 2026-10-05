# 🏡 House Prices Prediction - Ames Housing (Kaggle)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg)](https://xgboost.ai/)
[![Kaggle Leaderboard #58](https://img.shields.io/badge/Kaggle%20Leaderboard-%2358%20(MAE%2012590.49)-20BEFF.svg)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Pipeline completo, modular e reproduzível de Ciência de Dados e Machine Learning para a competição **Housing Prices Competition for Kaggle Learn Users / Ames Housing Dataset**.

---

## 📌 Resumo Executivo & Resultados

O projeto foi estruturado em **duas fases independentes e comparáveis**, demonstrando a evolução de um modelo padrão de mercado até um comitê de elite capaz de atingir a **Posição #58 no Leaderboard Global do Kaggle** já na sua 1ª submissão:

| Fase | Arquitetura / Comitê | RMSLE (Holdout) | CV Médio (5-Fold) | Erro Médio (MAE Local) | Kaggle Oficial (MAE Test) | Submissão |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **🥇 Fase 1** | **Ensemble Inicial (50% XGBoost + 30% Ridge + 20% Random Forest)** | `0.1311` | `0.1292` | US$ 14,782.52 | ~US$ 14,800+ | [`submission_fase1_0.1292.csv`](submission_fase1_0.1292.csv) |
| **🚀 Fase 2** | **Comitê de Elite (25% XGBoost + 25% Huber GBR + 20% Lasso + 15% ENet + 15% Ridge)** | **`0.1146`** | **`0.1096`** 🔥 | **US$ 12,904.29** | **`12590.49181`** 🏆 (**#58**) | [`submission_elite.csv`](submission_elite.csv) |

> 🏆 **Resultado Oficial no Kaggle:** **Posição #58 no Mundo** com Score **12590.49181** (com **apenas 1 submissão**). Generalização exemplar: o erro real no conjunto de teste cego foi até ligeiramente menor do que o CV local, comprovando ausência de overfitting e rigor na validação cruzada.

---

## 🔍 Estudo de Caso Prático: O "Mistério" da Mansão Id 2550

O maior aprendizado deste projeto está na detecção e tratamento de anomalias estatísticas fundamentadas cientificamente:

- **O Problema:** No artigo do professor **Dean De Cock** (*Journal of Statistics Education*, 2011), é alertado que existem casas no conjunto de treino com área superior a 4.000 pés² vendidas em liquidação judicial por valores anormais (< US$ 200.000).
- **O Efeito no Modelo Normal (Fase 1):** Como essas anomalias permaneceram no treino, o modelo aprendeu que "mansões gigantes podem custar pouco". Diante do **Imóvel de Teste Id 2550** (uma mansão de mais de 1.000 m² de área construída e acabamento de luxo máximo nota 10), a Fase 1 subestimou grosseiramente seu valor em **US$ 368.223,04**.
- **A Solução na Fase 2 (Elite):** Com a exclusão fundamentada dos outliers e a criação da interação `Qualidade_TotalSF` (`OverallQual * TotalSF`), o modelo de Elite capturou o valor real de mercado da mansão, avaliando-a em **US$ 752.262,74** (+US$ 384.000 de correção)!

---

## 🛠️ Tecnologias e Técnicas Aplicadas

- **Linguagem & Ambiente:** Python 3, Jupyter Notebooks, Git.
- **Tratamento de Dados:** Pandas, NumPy, Scipy (`skewness correction` com `log1p` em 34 variáveis numéricas).
- **Engenharia de Atributos:** Criação de `TotalSF` (Área Total Construída), `TotalBath`, `IdadeImovel`, `IdadeReforma`, e interações multiplicativas de luxo (`Qualidade_TotalSF`).
- **Pré-processamento Sem Vazamento:** Imputação de dados faltantes (mediana por bairro aprendida no treino e aplicada no teste), One-Hot Encoding com alinhamento rigoroso.
- **Modelagem Matemática:**
  - Regularização L1 e L2: **Lasso**, **Ridge** e **ElasticNet** com `RobustScaler`.
  - Ensembles baseados em Árvores: **Random Forest**, **Gradient Boosting (Huber Loss)** e **XGBoost (Extreme Gradient Boosting)**.
  - **Stacking / Blending Ponderado:** Média ponderada calibrada via validação cruzada 5-Fold.

---

## 📁 Estrutura do Repositório

```text
house/
├── dados/
│   ├── dados_iniciais/                    # Arquivos brutos originais do Kaggle
│   │   ├── train.csv                      # Treino bruto (1.460 linhas)
│   │   ├── test.csv                       # Teste bruto (1.459 linhas)
│   │   ├── sample_submission.csv
│   │   └── data_description.txt
│   ├── dados_tratados/                    # [FASE 1] Dados pré-processados da versão 0.1292
│   │   ├── train_processado.csv           # 1.460 linhas x 239 colunas
│   │   └── test_processado.csv            # 1.459 linhas x 238 colunas
│   └── dados_tratados_elite/              # [FASE 2] Dados pré-processados da versão de Elite
│       ├── train_elite.csv                # 1.458 linhas x 240 colunas (outliers removidos)
│       └── test_elite.csv                 # 1.459 linhas x 239 colunas
│
├── notebooks/                             # Análise Exploratória e Visualização (Jupyter)
│   ├── 01_inspecao_inicial.ipynb          # Etapa 1: Dimensões, tipos e assimetria do SalePrice
│   └── 02_analise_valores_ausentes.ipynb  # Etapa 2: Mapeamento semântico de nulos ("Não Possui" vs Omissão)
│
├── scripts/                               # Scripts Automatizados e Modulares (Python)
│   ├── # --- FASE 1: Versão Original do Usuário (RMSLE = 0.1292) ---
│   ├── 03_preprocessamento.py             # Pré-processamento e Feature Engineering inicial
│   ├── 04_modelo_baseline.py              # Benchmarks (Naive, Ridge, Random Forest)
│   ├── 05_modelo_avancado.py              # XGBoost e Blending da Fase 1
│   ├── 06_gerar_submissao.py              # Gera submissão da Fase 1
│   │
│   ├── # --- FASE 2: Versão de Elite (RMSLE = 0.1096) ---
│   ├── 03_preprocessamento_elite.py       # Remoção de outliers, correção de skewness e interações
│   ├── 04_modelo_baseline_elite.py        # Baseline no dataset de elite
│   ├── 05_modelo_avancado_elite.py        # Treinamento do Comitê de 5 Modelos
│   ├── 06_gerar_submissao_elite.py        # Gera submission_elite.csv
│   │
│   └── comparar_resultados.py             # Utilitário: Compara previsões da Fase 1 e 2 lado a lado
│
├── modelos_salvos/                        # Modelos da Fase 1 serializados (.pkl)
├── modelos_salvos_elite/                  # Modelos de Elite serializados (.pkl)
├── dicionario_dados_ptbr.md               # Dicionário traduzido de todas as 81 variáveis em Português
├── submission_fase1_0.1292.csv            # Arquivo de submissão da Fase 1
├── submission_elite.csv                   # Arquivo de submissão de Elite (RMSLE = 0.1096)
├── requirements.txt                       # Dependências do projeto para pip install
└── README.md                              # Documentação oficial
```

---

## 🚀 Como Reproduzir este Projeto Localmente

### 1. Clonar o repositório e instalar dependências
```bash
git clone https://github.com/SEU_USUARIO/house-prices-ml.git
cd house-prices-ml
pip install -r requirements.txt
```

### 2. Comparar as previsões das duas fases na tela
```bash
py scripts/comparar_resultados.py
```

### 3. Executar o Pipeline de Elite do início ao fim
```bash
py scripts/03_preprocessamento_elite.py
py scripts/04_modelo_baseline_elite.py
py scripts/05_modelo_avancado_elite.py
py scripts/06_gerar_submissao_elite.py
```

---

## 👤 Autoria e Créditos

- **Fase 1 (Pipeline Inicial & RMSLE 0.1292):** Desenvolvido pelo autor do projeto.
- **Fase 2 (Otimização de Elite & RMSLE 0.1096):** Conduzido no **Google Antigravity IDE** utilizando o modelo **Gemini 3.8 Flash High**.
- **Dataset:** [House Prices - Advanced Regression Techniques (Kaggle)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

---
*Distribuído sob a licença MIT. Sinta-se à vontade para utilizar e contribuir!*
