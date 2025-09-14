# Machine Learning - Differentiated Thyroid Cancer Recurrence
O projeto em questão consiste no treinamento de modelos de algoritmos de Inteligência Artificial (IA) baseados no *dataset* [Differentiated Thyroid Cancer Recurrence](https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence). O objetivo é fazer predições quanto à recorrência de câncer de tireoide a partir das *features* disponibilizadas (*Age, Gender, Pathology, Stage, ...*).

## Por que este *dataset*?
O *dataset* supramencionado foi escolhido por ser de tamanho gerenciável (383 instâncias), possuir dados bem ordenados e devidamente preenchidos, ter **features** e **target** bem definidos, além de apresentar uma utilidade considerável na medicina. Em suma, sua escolha se deu por atender aos requisitos deste projeto.

## Quais são as funcionalidades deste código?
Neste treinamento é possível:
* Ver as informações do *dataset* utilizado (*via biblioteca pandas*);
* Separar os dados para teste e treino;
* Realizar treinamento a partir de diversos algoritmos (Decision Tree Classifier, KNeighbors Classifier e Logistic Regression);
* Realizar predições da recorrência de câncer de tireoide para uma determinada entrada de dados;
* Medir variáveis de desempenho (acurácia, *recall* e F1);
* Visualizar resultados em forma de gráfico e matrizes de decisão.


## Resultados e Discussões
Abaixo estão dispostos os dados do experimento e suas respectivas descrições.

| Modelo                   | Acurácia   | Recall    | F1       |
| ------------------------ | ---------- | --------- | -------- |
| Decision Tree Classifier | 92,173 %   | 92,658 ¨% | 90,683 % |
| KNeighbors Classifier    | 87,826 %   | 81,005 %  | 83,477 % |
| Logistic Regression      | 95,652 %   | 94,108 %  | 94,535 % |

> Tabela comparativa das métricas de desempenho. Os dados foram aproximados para três casas decimais.

Utilizando as técnicas de medição disponibilizadas pelo `sklearn.metrics` (recurso do `Scikit-Learn`), foi possível descobrir as taxas de **acurácia**, ***recall*** e **F1** para cada modelo de treinamento.
> Em *train.py*, note que foram atribuídos os parâmetros `test_size=0.3` e `random_state=42` para o método `train_test_split()`. A alteração destes valores pode acarretar em mudanças nos parâmetros de *performance*.

Utilizando o `matplotlib`, foi produzido o gráfico seguinte para melhor visualização dos aspectos de cada modelo, o qual é composto por três grupos com três barras cada, sendo respectivamente:
* **Vermelho:** indica a acurácia;
* **Azul:** indica o *recall*;
* **Verde:** indica o F1.

![Gráfico comparativo de métricas de desempenho para os três algoritmos](/assets/bar-graphic-comparing-scores.png)

Ainda foi possível traçar as matrizes de confusão de cada modelo combinando o `ConfusionMatrixDisplay` com o `matplotlib`, as quais são apresentadas abaixo.

![Gráfico comparativo de métricas de desempenho para os três algoritmos](/assets/cm-decision-tree-classifier.png)

![Gráfico comparativo de métricas de desempenho para os três algoritmos](/assets/cm-kneighbors-classifier.png)

![Gráfico comparativo de métricas de desempenho para os três algoritmos](/assets/cm-logistic-regression.png)

Comparando a acurácia, o *recall* e o F1 de cada algoritmo, é possível verificar que, **para este *dataset* e para as condições deste experimento**, o modelo **Logistic Regression** apresentou melhores resultados que os demais, representando as porcentagens máximas para cada variável de *performance*.

Ademais, analisando as matrizes de confusão, percebe-se a ineficiência do algoritmo KNeighbors Classifier. Note que este foi o algoritmo que mais apresentou valores no terceiro quadrante (*True-Yes and Predict-No*), indicando que **muitos resultados positivos foram retornados como negativos**. Esta possibilidade é de maior gravidade, pois resultados deste tipo podem contribuir com a disseminação de doenças degenerativas, como o câncer de tireoide

Em contrapatida, os modelos Decision Tree Classifier e Logistic Regression apresentaram resultados mais condizentes na etapa de teste, estando mais próximos dos registros estruturados no *dataset*. 

Ressalta-se, entretanto, que a mudança em parâmetros de treinamento, o aumento das instâncias do *dataset* ou mesmo a inserção de novas *features* podem certamente mudar estes resultados. Pode-se, portanto, inferir que o contexto determina o melhor modelo, não sendo possível determinar a superioridade absoluta de um algoritmo em particular.
