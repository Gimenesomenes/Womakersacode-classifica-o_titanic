import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# Lendo o arquivo

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#print(train.head())
#print(test.head())

# Descrição das colunas
# Survived: Indica se o passageiro sobreviveu (1) ou não (2).
# Pclass: Classe do bilhete do passageiro (1°, 2°, 3° classe).
# Name: Nome do passageiro.
# Sex: Gênero do passageiro.
# Age: Idade do passageiro.
# SibSp: Número de irmãos/cônjuges a bordo.
# Parch: Número de pais/filhos a bordo.
# Ticket: Número do bilhete do passageiro.
# Fare: Tarifa paga pelo passageiro.
# Cabin: Número da cabine do passageiro (qdo disponível).
# Embarked: Porto de embarque do passageiro (C=Cherbourg, Q = Queenstown, S = Southampton).

# Variável target utilizada no modelo será a variável Survived.

print(train.columns)

print("Número de observaoções treino: ")
print(train.shape[0])
print("Número de observações teste: ")
print(test.shape[0])

print(train.describe(include=['object']))

print(train.describe())


# Estudando os valores nulos do Df
datadict = pd.DataFrame(train.dtypes)
datadict.columns = ['Dtypes'] 
datadict['Valores_Nulos'] = train.isnull().sum()
datadict['%_Valores_Nulos'] = (train.isnull().sum() / train.shape[0]) * 100
datadict['Nunique'] = train.nunique()

print(datadict)

# Análise inicial 

# Limpeza dos dados

# Idade

ax = train['Age'].hist(bins=15, density=True, stacked=True, color='blue', alpha=0.9)
train['Age'].plot(kind='density', color='red')
ax.set(xlabel='Age')
ax.set_title("Histograma: Idade (Age)")
plt.xlim(-10,85)
plt.show()

train['Age'].fillna(train['Age'].median(), inplace=True)

# Porto de embarque

print("Passageiros embarcados por porto (C = Cherbourg, Q = Queenstown, S = Southampton): ")
print(train['Embarked'].value_counts())
ax = sns.countplot(x='Embarked', data=train, palette='viridis')
ax.set_title("Contagens: Embarque ('Embarked')")
plt.show()


train['Embarked'].fillna(train['Embarked'].mode(), inplace=True)


# Cabines
# Na cabine nós observamos muitos valores nulos 77%, dessa forma não vale apena substituirmos por média ou mediana, assim será NA.

train['Cabin'].fillna("NA", inplace=True)

# Análise pre-modelagem 

print(train.describe())

# seleciona variáveis numéricas
num_df = train.select_dtypes(include=['float64', 'int64'])

# matriz de correlação

corr = num_df.corr()

# plot

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='viridis', fmt=".2f", linewidths=0.5)
plt.title('Matriz de correlação')
plt.show()

print("De acordo com a matriz de correlação a gente vê que a variável que tem maior correlação com nossa variável target é (Fare), assim como Parch e SibSp")

#sns.pairplot(num_df, diag_kind='hist')
#plt.show()

# Variáveis categóricas

# Sobreviventes:  

palette = sns.color_palette("viridis", len(train['Survived'].unique()))
plt.figure(figsize=(10,8))
sns.countplot(train, x='Survived', palette=palette).set_title("Contagem de Sobreviventes")

plt.show()

print(train.Survived.value_counts(normalize=True))

# Gênero

palette = sns.color_palette("viridis", len(train['Sex'].unique()))
plt.figure(figsize=(10,8))
sns.countplot(train, x='Sex', palette=palette).set_title("Contagem de Gênero")
plt.show()

print(train.Sex.value_counts(normalize=True))

# Embarque

palette = sns.color_palette("viridis", len(train['Embarked'].unique()))
plt.figure(figsize=(10,8))
sns.countplot(train, x='Embarked', palette=palette).set_title("Contagem de Portos Embarcados")
plt.show()

print(train.Embarked.value_counts(normalize=True))

print(train[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].groupby('Survived').agg({
    'Pclass':np.median,
    'Sex':pd.Series.mode,
    'Age':np.median,
    'SibSp':np.median,
    'Parch':np.median,
    'Fare':np.median,
    'Embarked':pd.Series.mode
}))


print(train.head())

# Base de test                                                                                                                                                                                                                                                                                             

test['Age'].fillna(test['Age'].median(), inplace=True)
test['Cabin'].fillna('NA', inplace=True)

# O fare aparece como nulo e vamos substituir pela mediana

test['Fare'].fillna(test['Fare'].median(), inplace=True)

print(test.head())

# Filtragem da base

train_filtrado = train[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare', 'Embarked']]

test_filtrado = test[['Pclass', 'Sex', 'Age', 'SibSp', 
                      'Parch', 'Fare', 'Embarked']] 

# limpeza excluindo survived por não estar nessa base

## Criando categorias

train_feature = pd.get_dummies(train_filtrado, columns= ['Pclass', 'Embarked', 'Sex'], dtype=int)
train_feature.drop('Sex_female', axis=1, inplace=True)

test_feature = pd.get_dummies(test_filtrado, columns=['Pclass', 'Embarked', 'Sex'], dtype=int)
test_feature.drop('Sex_female', axis=1, inplace=True)

# Criando a variável viajou sozinho

train_feature['viajou_sozinho'] = np.where((train_feature['SibSp'] + train_feature['Parch']) > 0, 0, 1)
train_feature.drop(['Parch', 'SibSp'], axis = 1, inplace=True)

test_feature['viajou_sozinho'] = np.where((test_feature['SibSp'] + test_feature['Parch']) > 0, 0, 1)
test_feature.drop(['Parch', 'SibSp'], axis = 1, inplace=True)

print(train_feature.head(), test_feature.head())


palette = sns.color_palette("deep6", len(train_feature['viajou_sozinho'].unique()))
plt.figure(figsize= (10,6))
sns.countplot(train_feature, x="viajou_sozinho", palette=palette).set_title('Contagem: viajou sozinho')
plt.show()


### Regressão logistica 

# leitura dos pacotes

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
from sklearn.linear_model import LogisticRegression

# cria vetores X e y com baso nos dados
#### X contem as variáveis explicativas


X = train_feature[['Age', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_C',
                   'Embarked_Q', 'Embarked_S', 'Sex_male', 'viajou_sozinho']]

#### y contem a variável de interesse

y = train_feature[['Survived']]


### vamos usar train_test split do pacote scklearn para a separação dos dados deixando 80% treino 20% teste


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


## fit do modelo

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

### Vamos agora aplicar o modelo filtrado na base de test

y_pred = logreg.predict(X_test)
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Mostrando output 

X_test['Previsao_Modelo'] = y_pred
X_test['Probabilidade_Sobreviver'] = y_pred_prob


print(X_test)


## Calculando a performance do modelo

## Calcular a matriz de confusão

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel("Classe Predita")
plt.ylabel("Classe Real")
plt.title("Matriz de Confusão")
plt.show()


# Calcular as métricas

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Acurácia", accuracy)
print("Precisão", precision)
print("Recall", recall)


### Obtendo as variáveis mais relevantes do modelo

# Obtendo os coeficientes do modelo (importância das características)
coefficients = logreg.coef_[0]
feature_names = X.columns

# Ordenando os índices das características pelos coeficientes 

indices = np.argsort(np.abs(coefficients))


# Top 10 características com maior importância

top_features = feature_names[indices][-10:]
top_importances = coefficients[indices][-10:]

# Plotando as top features com suas importâncias

plt.figure(figsize=(10, 6))
plt.barh(top_features, top_importances, color='skyblue')
plt.xlabel("Importância das características")
plt.ylabel("Características")
plt.title("Top 10 características com maior importância")
plt.show()

## Aplicando nos dados de teste para prever

previsao = logreg.predict(test_feature)
probabilidade = logreg.predict_proba(test_feature)[:,1]

test_feature['Previsao_Modelo'] = previsao
test_feature['Probabilidade_Sobreviver'] = probabilidade

print(test_feature)


palette = sns.color_palette("deep6", len(test_feature['Previsao_Modelo'].unique()))
plt.figure(figsize= (10,6))
sns.countplot(test_feature, x="Previsao_Modelo", palette=palette).set_title('Contagem: Previsao Modelo')
plt.show()










