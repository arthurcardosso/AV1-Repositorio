# Avaliação 1: Redes Neurais - Classificação KMNIST (Kana)

Este repositório contém o mini-projeto aplicado para a disciplina de Redes Neurais (NES), focado na classificação do dataset **KMNIST (Kuzushiji-MNIST)**.

O projeto compara duas arquiteturas de redes neurais (MLP vs CNN) e explora técnicas de interpretabilidade visual (Grad-CAM) para analisar erros em caracteres japoneses ambíguos.

**Aluno:** Arthur Felipe Cardoso dos Santos  
**Professor:** Eduardo Adame  
**Data:** Setembro/2025

---

## 📂 Estrutura do Repositório

```text
├── src/
│   ├── figures/           # Imagens geradas (curvas, matriz, grad-cam, demo)
│   ├── app.py             # Aplicação interativa (Streamlit)
│   ├── CNN_best.keras     # Modelo CNN treinado
│   └── MLP_best.keras     # Modelo MLP treinado
├── requirements.txt       # Dependências do projeto
└── README.md              # Documentação
```

## 🚀 Instalação e Dependências

Para executar este projeto, recomenda-se criar um ambiente virtual. As dependências principais são tensorflow, streamlit, numpy, matplotlib e tensorflow-datasets.

Instale tudo com o comando:

```bash
pip install -r requirements.txt
```

## 📊 Executando a Demo Interativa

O projeto inclui uma interface web onde é possível fazer upload de uma imagem KMNIST, ver a classificação em tempo real e visualizar o mapa de calor (Grad-CAM).

Para iniciar o app:

```bash
streamlit run src/app.py
```

O aplicativo abrirá automaticamente no seu navegador.

## 📈 Análise Visual

Matriz de Confusão: A CNN apresentou alta precisão, com erros concentrados em classes visualmente similares.

Interpretabilidade (Grad-CAM): A análise de erros revelou que o modelo foca em traços específicos (hastes e loops). Em casos de erro, o foco muitas vezes recai sobre fragmentos que lembram outra letra.

## 📝 Referências

Dataset: KMNIST (Kuzushiji-MNIST) via TensorFlow Datasets.
Template: Estrutura baseada nas diretrizes da Avaliação 1 (Prof. Eduardo Adame).
