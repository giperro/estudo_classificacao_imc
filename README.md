# Análise Preditiva: Previsão da Categoria de IMC com Modelos de Machine Learning


### Descrição do Projeto
O modelo de machine learning desenvolvido nesse estudo, visa prever a categoria de Índice de Massa Corporal (IMC) de uma pessoa, considerando não somente características físicas do indivíduo comumente utilizadas (peso e altura), como também características comportamentais que podem influenciar a saude do indivíduo. 
O objetivo central é desenvolver um classificador de IMC que se adeque às especificidades e estilos de vida de diferentes indivíduos.  

---
### Pacotes utilizados
Para a execução do estudo, foram necessários os seguintes pacotes:

| Pacote | Versão |
| :----------- | :-----------: |
| [pickle](https://pypi.org/project/pickle5/ "Pickle") | 2.2.1 |
| [pandas](https://pypi.org/project/pandas/ "Pandas") | 1.5.3 |
| [numpy](https://numpy.org/install/ "Numpy") | 1.24.3 |
| [seaborn](https://pypi.org/project/seaborn/ "Seaborn") | 0.12.2 |
| [plotly](https://pypi.org/project/plotly/ "Plotly") | 5.9.0 |
| [sklearn](https://pypi.org/project/scikit-learn/ "Scikit-learn") | 1.2.2 |
| [scipy](https://scipy.org/install/ "scipy") | 1.10.1 |
| [statsmodel](https://www.statsmodels.org/stable/ "Statsmodel") | 0.13.5 |
| [yellowbrick](https://pypi.org/project/yellowbrick/ "Yellowbrick") | 1.5 |


---
### Dataset
O [dataset](https://www.kaggle.com/datasets/aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster "Kaggle: Obesity or CVD Risk") utilizado para o treinamento do modelo contém as seguintes **variáveis preditoras**:

| Variável | Descrição | Tipo de variável |
| :----------- | :-----------: | :-----------: |
| Gender | Gênero: Male = 0, Female = 1 | binária |
| Age | Idade | numérica |
| Height | Altura | numérica |
| Weight | Peso | numérica |
| family_history_with_overweight | Histórico familiar com obesidade: Sim = 1, Não = 0 | binária|
| FAVC | Consumo de comidas de alto teor calórico: Sim = 1, Não = 0 | binária |
| FCVC | Frequência de consumo de vegetais | ordinal |
| NCP | Número de refeições principais | numérica |
| CAEC | Frequência de consumo de alimentos entre refeições principais | ordinal |
| Smoke | Fumante? Sim = 1, Não = 0 | binária |
| CH2O | Consumo diário de água | numérica |
| SCC | Monitora a ingestão de calorias? Sim = 1, Não = 0 | binária |
| FAF | Frequência de prática de atividade física | ordinal |
| TUE | Tempo gasto com dispositivos tecnológicos | numérica |
| CALC | Frequência de consumo de álcool | ordinal |
| MTRANS | Meio de transporte principal utilizado | nominal |

A **variável resposta** (NObeyesdad) dispõe das seguintes categorias:
- Insufficient_Weight
- Normal_Weight
- Obesity_Type_I
- Obesity_Type_II
- Obesity_Type_III
- Overweight_Level_I
- Overweight_Level_II

---
### Modelos Utilizados: principais conceitos
- **Naive Bayes:**
É o modelo de classificação que se baseia no Teorema de Bayes. Ele assume que as variáveis são independentes entre si. 
Ex: ao classificar e-mails como spam ou não spam, o Naive Bayes considera a probabilidade de cada palavra aparecer em um e-mail spam ou não spam de forma independente.

- **Árvore de Decisão:**
É o modelo de aprendizado supervisionado que representa um conjunto de regras de decisão em forma de árvore. Cada nó interno da árvore representa uma decisão baseada em um atributo, e cada folha representa o resultado da decisão. 
Ex: em um modelo para prever se um cliente vai alugar um apartamento, a árvore pode ter nós que representam o salário do cliente, preferências do cliente, preço do aluguel, etc.

- **Random Forest:**
É o modelo de aprendizagem que cria várias árvores de decisão durante o treinamento e combina suas previsões para obter uma previsão mais precisa e estável. 
Ex: em um modelo para prever o clima, o Random Forest pode combinar várias árvores de decisão que consideram diferentes variáveis meteorológicas.

- **KNN (K-Nearest Neighbors):**
É o modelo que classifica um ponto de dados com base nos pontos de dados mais próximos a ele. 
Ex: em um modelo para classificar animais, o KNN considera as características dos demais animais para determinar a classe de um animal desconhecido.

- **Regressão Logística:**
É o modelo usado para prever a probabilidade de um evento ocorrer com base em variáveis independentes (preditoras).
Ex: em um modelo para prever se um paciente tem uma determinada doença, a regressão logística pode ser usada para prever a probabilidade de o paciente ter a doença com base em seus sintomas.

- **SVM (Support Vector Machine):**
É um modelo de aprendizado supervisionado usado tanto para classificação quanto para regressão. Ele encontra o hiperplano que melhor separa os dados em classes diferentes. 
Ex: em um modelo para classificar imagens de gatos e cachorros, o SVM encontra o hiperplano que separa as características dos gatos das dos cachorros.

- **Redes Neurais:**
São modelos de aprendizado de máquina inspirados no funcionamento do cérebro humano. Elas consistem em neurônios artificiais organizados em camadas. Cada neurônio recebe entradas, realiza um cálculo e passa o resultado para os neurônios da próxima camada. 
Ex: em um modelo para reconhecimento de voz, uma rede neural pode ser treinada para reconhecer padrões sonoros associados a palavras específicas.

---
### Métricas de validação do modelo
- **Validação cruzada:** 
A validação cruzada é uma técnica usada para avaliar a capacidade de generalização de um modelo, dividindo o conjunto de dados em subconjuntos de treino e teste de forma iterativa. O modelo é treinado nos subconjuntos de treino e testado nos subconjuntos de teste, calculando métricas de desempenho como acurácia, precisão, etc. Isso ajuda a evitar problemas de sobreajuste (_overfitting_) e subajuste (_underfitting_).
- **Método de divisão de bases k-Folds:**
O método k-Folds divide o conjunto de dados em k partes iguais (ou quase iguais). O modelo é treinado em k-1 partes e testado na parte restante. Esse processo é repetido k vezes, com cada parte sendo usada como conjunto de teste uma vez. O resultado final é a média das métricas de desempenho calculadas em cada fold.
- **Teste de normalidade dos resultados:**
O teste de normalidade de Shapiro é usado para verificar se uma amostra de dados segue uma distribuição normal. A hipótese nula afirma que os dados seguem uma distribuição normal, enquanto a hipótese alternativa afirma que os dados não seguem uma distribuição normal. 
Se o teste de Shapiro mostrar que os _scores_ gerados no _cross validation_ não têm uma distribuição normal (ou seja, se a hipótese nula não for rejeitada), pode ser necessário considerar métodos estatísticos alternativos ou transformar os dados para garantir que eles atendam aos pressupostos do modelo.
- **Teste de hipóteses - ANOVA:**
O teste ANOVA (Análise de Variância) é usado para comparar as médias de três ou mais grupos diferentes para determinar se há diferença estatisticamente significativa entre eles. A H0 afirma que não há diferença significativa entre as médias dos grupos, enquanto a H1 afirma que há pelo menos uma diferença significativa entre as médias. Caso a H0 não seja rejeitada, entende-se que não há diferença significativa entre os modelos. Do contrário, infere-se que há um modelo superior aos demais.
- **Teste de Tukey (Tukey's Honest Significant Difference - HSD):**
O teste de Tukey é usado após um teste ANOVA significativo para identificar quais grupos têm médias significativamente diferentes entre si. Ele compara todas as combinações de grupos e calcula um intervalo de confiança para a diferença entre as médias, determinando se essa diferença é estatisticamente significativa. A hipótese nula (H0) do teste de Tukey é que não há diferença significativa entre as médias de todos os grupos comparados.

---
### Resultados
Tendo em vista que, o teste de Tukey não apontou uma diferença clara entre a Árvore de decisão e a Random Forest, o modelo foi selecionado a partir das seguintes premissas:
 Métrica de avaliação | Árvore de decisão | Random Forest 
:----------|-------------|----------
**R²**| 93,42% | 93,85% 
**Acurácia**| 95,32% | 95,49% 
**Teste de Tukey**| Não há diferença significativa | Não há diferença significativa 
**Tempo de treinamento e recursos computacionais**|Mais rápido, toma menos recursos| Mais demorado, complexo

Como o foco do projeto era encontrar uma solução prática para incorporar as particularidades de diferentes estilos de vida ao cálculo do IMC, entendeu-se que um modelo mais rápido seria o ideal, inclusive pelo fato de que não houve diferença significativa entre as médias dos modelos, assim como pela proximidade entre o desempenho dos modelos nas demais métricas. Portanto, o modelo escolhido foi a Árvore de decisão. 

Além disso, vale ressaltar que, de acordo com o modelo treinado, a ordem de importância entre os atributos estudados é a seguinte:
1. Peso 
2. Altura
3. Gênero
4. Idade
5. Consumo de alimentos altamente calóricos
6. Consumo diário de água
7. Frequência da prática de atividade física
8. Consumo de alimentos entre as refeições principais
9. Número de refeições principais
10. Histórico familiar com obesidade
11. Tempo gasto com dispositivos eletrônicos

_obs_: os demais atributos tiveram importância igual a zero.

---
### Aviso de Direitos Autorais
Este projeto é um trabalho original e protegido por direitos autorais. Não deve ser utilizado para fins comerciais. Você é livre para visualizar, estudar, aprender e se inspirar a partir deste código. O plágio é uma violação dos direitos autorais e ética acadêmica.

---
### Contato
Giovanna | [LinkedIn](www.linkedin.com/in/giovanna-rodrigues-2a988b143) | [E-mail](giovanna.rodrigues@unifesp.br) | Cel: (11) 93209-6371

