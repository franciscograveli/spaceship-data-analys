# Relatório Técnico - Spaceship Titanic

**Disciplina:** Aprendizado de Máquina - IF Sudeste MG (Campus Muriaé)
**Trabalho Prático II**
**Competição:** [Spaceship Titanic (Kaggle)](https://www.kaggle.com/competitions/spaceship-titanic)

## 1. O problema

Na competição, a nave Spaceship Titanic bate numa anomalia e parte dos passageiros é "transportada para outra dimensão".
Nossa tarefa é prever, para cada passageiro do teste, se ele foi transportado ou não (coluna `Transported`).
É um problema de **classificação binária**.

Os dados são:
- `train.csv`: 8.693 passageiros, com a resposta.
- `test.csv`: 4.277 passageiros, sem a resposta (é o que o Kaggle usa para dar a nota).

A classe está bem equilibrada: cerca de 50% foram transportados e 50% não.

## 2. O que fizemos nos dados

Juntamos treino e teste numa tabela só, para aplicar **exatamente o mesmo tratamento** nos dois. A resposta não foi usada em nenhum tratamento.

### 2.1 Análise exploratória (EDA)
O que descobrimos olhando os dados:
- Todas as colunas têm uns 2% de valores faltando.
- Quem está em sono criogênico (`CryoSleep`) é transportado bem mais (82%) e não gasta nada.
- Crianças até 12 anos são mais transportadas (70%). Os decks B e C também (cerca de 70%).
- Os gastos são muito desiguais: a maioria gasta pouco e poucos gastam muito.
- **Treino e teste não têm nenhum grupo de viagem em comum.** Por isso, na validação cruzada usamos `GroupKFold`, que mantém cada grupo inteiro numa mesma parte. Assim a validação se parece com o que acontece no teste do Kaggle.

### 2.2 Olhando o teste inteiro: os "bebês" e quem não gasta
Conferimos as regras e os padrões em todas as 12.970 linhas (treino e teste juntos), e não só nas primeiras. O professor comentou que passageiros que não gastam e **não estão em CryoSleep** poderiam ser fetos ou recém-nascidos. Os dados confirmam isso:

- **A idade 0 é diferente das outras.** Há 260 passageiros com idade 0, contra 94 com idade 1, 110 com 2 e 109 com 3. A contagem não cai aos poucos, então a idade 0 parece ser um marcador de recém-nascido.
- **Nenhum deles gasta:** os 260 têm gasto zero.
- **Viajam com a família:** 97% estão em grupo de 2 ou mais pessoas. 86% têm um adulto no mesmo grupo, 66% dividem a cabine com um adulto e 72% têm o mesmo sobrenome de um adulto do grupo.
- **Muitos não estão em CryoSleep** (59%), e mesmo assim não gastam.
- **São quase sempre transportados:** 82% no treino, estejam ou não em CryoSleep. As crianças de 1 a 12 anos ficam entre 64% e 70%, e os adultos fora do CryoSleep em 30%.

Olhando só quem **não gasta e não está em CryoSleep** (470 no treino e 197 no teste): 78% são crianças com menos de 13 anos, e 20% têm idade 0. Os adultos nesse grupo são raros (94 no treino e 36 no teste) e só 36% deles foram transportados. Já a **idade ausente** aparece em 270 passageiros, mas só 13 deles não gastam e estão fora do CryoSleep, então o problema é pequeno.

Também conferimos as regras de preenchimento nas 12.970 linhas, e nenhuma teve exceção:

| Regra | Exceções |
|---|---|
| Quem está em CryoSleep nunca gasta | 0 |
| Menores de 13 anos nunca gastam (aos 13 já gastam, 116 casos) | 0 |
| Menores de 18 anos nunca são VIP | 0 |
| Quem é da Terra nunca é VIP | 0 |
| Decks A, B, C e T são só de Europa, e o G é só da Terra | 0 |

**O que fizemos com isso:** nada mudou no modelo. Testamos marcar a idade 0 e a idade ausente (+0,0004 de acurácia) e preencher a idade ausente com uma regra (sem gasto e fora do CryoSleep vira criança, com gasto vira 13 anos ou mais: +0,0008). As duas diferenças ficaram dentro do ruído, porque o CatBoost já enxerga a idade 0 sozinho e são poucos casos.

**Uma limitação:** nossa regra "sem gasto significa CryoSleep" vale para adultos (94% deles sem gasto estão em CryoSleep), mas para crianças não, pois só 48% estão. Isso afeta 30 passageiros com CryoSleep ausente e menos de 13 anos (7 no teste), então o efeito no resultado é pequeno.

### 2.3 Variáveis novas
- `Grupo` e `TamanhoGrupo`: vêm do `PassengerId` (os 4 primeiros dígitos mostram quem viaja junto).
- `Deck`, `NumCabine` e `Lado`: vêm da coluna `Cabin`, que tem o formato deck/número/lado.
- `GastoTotal`: soma dos 5 gastos.
- `GastoLuxo` (RoomService + Spa + VRDeck) e `GastoBasico` (FoodCourt + ShoppingMall): na EDA vimos que os dois grupos de gasto se comportam de forma diferente em relação à resposta.

### 2.4 Valores ausentes (NaN)
Antes de usar a mediana ou a moda, usamos **regras do próprio problema**, que conferimos nos dados e valem em 100% dos casos:
- Quem está em CryoSleep não gasta nada, e quem gasta algo não está em CryoSleep.
- Menores de 13 anos não gastam nada.
- Quem viaja no mesmo grupo tem o mesmo planeta e o mesmo lado da cabine.
- Os decks A, B, C e T são só de Europa, e o deck G é só da Terra.
- Quem é da Terra nunca é VIP.

O que sobrou foi preenchido com a **mediana** (números) ou a **moda** (categorias). No final não ficou nenhum NaN.

### 2.5 Variáveis categóricas
`CryoSleep` e `VIP` viraram 1 e 0. `HomePlanet`, `Destination`, `Deck` e `Lado` viraram colunas de 0 e 1 (*one-hot encoding*).

### 2.6 Outliers
Usamos o método do **IQR** no `GastoTotal`: consideramos outlier quem gasta mais que `Q3 + 8 × IQR`. Só tiramos isso do **treino** (removemos 99 passageiros, cerca de 1%). No teste não dá para tirar ninguém, pois o Kaggle espera uma previsão para cada passageiro.

Por que o fator 8 e não o 1,5 ou o 3, que são os mais comuns? Muita gente gasta zero, então o IQR fica pequeno e os fatores menores acabam tirando passageiros que só gastam bastante, e não são erro. Testamos com validação cruzada e o fator 3 tirava 6% das linhas e **piorava** a acurácia em cerca de 0,4 a 1,2 ponto.

### 2.7 Padronização
Usamos o `StandardScaler` (média 0 e desvio padrão 1). Ele importa para Regressão Logística, KNN e SVM, que dependem da escala. O ajuste é feito só no treino, e o teste recebe a mesma transformação.

### 2.8 Seleção de variáveis
Treinamos uma Random Forest só para medir a importância de cada coluna e tiramos as de importância menor que 0,2% (ficamos com 26 das 29 colunas). Cortes maiores pioravam o resultado: com 0,5% a validação do CatBoost caiu de 0,813 para 0,810, e com 2% caiu para 0,796.
Para o CatBoost usamos todas as colunas, porque na validação cruzada ele ficou igual com ou sem as 3 colunas tiradas.

## 3. Os modelos e a otimização de hiperparâmetros

Para cada modelo usamos o `GridSearchCV` com validação cruzada de 5 partes (`GroupKFold`), escolhendo pela acurácia.

| Modelo | Parâmetros testados | Melhor |
|---|---|---|
| Regressão Logística | C: 0,1 / 1 / 10 | C = 0,1 |
| KNN | vizinhos: 5 / 11 / 21 | 11 |
| Naive Bayes | var_smoothing: 1e-9 / 1e-6 / 1e-3 | 1e-9 |
| SVM | C: 0,5 / 1 / 5 | C = 1 |
| Árvore de Decisão | profundidade: 3 / 5 / 7 / 10 | 5 |
| Random Forest | árvores: 100 / 300 e profundidade: 5 / 8 / 12 | 100 árvores, profundidade 12 |
| CatBoost | profundidade: 6 / 8 | 6 |

### O algoritmo extra: CatBoost
O CatBoost é um algoritmo de *gradient boosting*: ele treina várias árvores **uma depois da outra**, e cada árvore nova tenta corrigir os erros das anteriores. Na Random Forest, ao contrário, as árvores são treinadas separadas e se faz a média delas.

Ele tem duas diferenças em relação a outros métodos de boosting. Usa **árvores simétricas** (todos os nós de um mesmo nível fazem a mesma pergunta), o que ajuda a evitar que o modelo decore o treino. E usa o *ordered boosting*, que reduz o sobreajuste em bases pequenas como a nossa.

**Referência:** Prokhorenkova, L.; Gusev, G.; Vorobev, A.; Dorogush, A. V.; Gulin, A. *CatBoost: unbiased boosting with categorical features*. NeurIPS 2018. https://arxiv.org/abs/1706.09516

**Por que CatBoost e não XGBoost?** Começamos com o XGBoost (Chen e Guestrin, KDD 2016), que fez 0,80476 no Kaggle. Depois de criarmos as variáveis de gasto de luxo e básico, o CatBoost deu uma pontuação melhor (0,80804) e ficou melhor também na validação cruzada, então ele ficou como algoritmo extra.

## 4. Resultados

As métricas abaixo foram calculadas no **conjunto de treino**. A acurácia de validação cruzada mostra como o modelo se sai em dados que ele não viu. A pontuação do Kaggle é a do conjunto de teste.

| Modelo | Acurácia (treino) | Precisão | Especificidade | Recall | F1 | Acurácia (valid. cruzada) | Kaggle |
|---|---|---|---|---|---|---|---|
| Regressão Logística | 0,788 | 0,780 | 0,766 | 0,809 | 0,794 | 0,786 | 0,79261 |
| KNN | 0,829 | 0,843 | 0,845 | 0,814 | 0,828 | 0,786 | 0,78816 |
| Naive Bayes | 0,755 | 0,700 | 0,605 | 0,902 | 0,788 | 0,754 | 0,76548 |
| SVM | 0,818 | 0,818 | 0,812 | 0,824 | 0,821 | 0,802 | 0,80336 |
| Árvore de Decisão | 0,801 | 0,783 | 0,761 | 0,840 | 0,811 | 0,788 | 0,79611 |
| Random Forest | 0,909 | 0,910 | 0,908 | 0,911 | 0,911 | 0,804 | 0,80336 |
| **CatBoost** | 0,884 | 0,880 | 0,876 | 0,892 | 0,886 | **0,813** | **0,80804** |

### Colocação no ranking do Kaggle
O Kaggle só mostra a melhor posição da equipe, então as demais são a posição que cada pontuação teria hoje (são aproximadas, pois no empate o Kaggle coloca na frente quem enviou antes). O ranking tinha 2.149 equipes.

| Modelo | Kaggle | Posição |
|---|---|---|
| CatBoost | 0,80804 | 404 (oficial) |
| Random Forest | 0,80336 | ~919 |
| SVM | 0,80336 | ~919 |
| Árvore de Decisão | 0,79611 | ~1.478 |
| Regressão Logística | 0,79261 | ~1.599 |
| KNN | 0,78816 | ~1.804 |
| Naive Bayes | 0,76548 | ~1.964 |

*(Os prints do Kaggle com a colocação de cada método estão anexados no final.)*

## 5. O que a gente concluiu

- **O CatBoost foi o melhor**, tanto na validação cruzada (0,813) quanto no Kaggle (0,80804).
- **A Random Forest e o KNN decoram o treino.** A Random Forest acerta 91% no treino, mas só 80% na validação. O KNN passa de 82% para 78%. Por isso a acurácia no treino sozinha engana, e a validação cruzada é a medida mais confiável.
- **O Naive Bayes foi o pior.** Ele supõe que as colunas são independentes, e aqui elas não são (por exemplo, CryoSleep e os gastos andam juntos).
- **Tentamos várias ideias para melhorar e quase nada passou do ruído:** transformar os gastos em log, usar sobrenome (família), região da cabine, médias do grupo, votar entre vários modelos. Quase tudo ficou dentro de 0,1 ponto da acurácia, e juntar modelos até piorou. Só separar o gasto em luxo e básico ajudou um pouco.
- **Onde o modelo mais erra:** cerca de 74% dos erros são passageiros da Terra, principalmente os que estão em sono criogênico e as crianças. Nas colunas que temos não há nada que separe bem esses casos.
- **A pontuação do Kaggle tem ruído.** Com 4.277 linhas no teste, a pontuação varia cerca de ±0,006 só por sorte. Por isso, diferenças pequenas entre envios (como 0,805 e 0,808) não querem dizer muito. Na busca de profundidade do CatBoost, a profundidade 4 deu uma validação parecida com a 6 (0,815 e 0,813), e a diferença está dentro desse ruído.

## 6. Arquivos entregues
- `spaceship_titanic.ipynb`: código completo (roda no Google Colab).
- `submissoes/`: um arquivo `.csv` por modelo, enviado ao Kaggle.
- `quadro_comparativo.csv`: o quadro comparativo.
- Prints do Kaggle com a colocação de cada método.
