---
marp: true
theme: default
title: PGCOMP-UFBA 2022 MATE34 - Atividade 01
author: Anderson Boa Morte
paginate: true
---
# PGCOMP-UFBA 2022 
## MATE34 - Atividade 01
No contexto de PLN - Processamento de Linguagem Natural, realizar um experimento de REN - Reconhecimento de Entidades Nomeadas. 

O experimento consiste em reconhecer as entidades 'pessoa' e 'localidade' em textos de contexto geral escritos na língua portuguesa.


---
# Descrição do experimento
Utilizamos a biblioteca NLP spaCy para realização do experimento. O spaCy possui 80 pipelines treinados para 24 linguas, incluindo 3 pipelines pre-treinados para a língua portuguesa, utilizamos o pt_core_news_lg (*). 

---
# Características do modelo pt_core_news_lg:
- **tipo**: core (*vocabulary, syntax, entities, vectors*)
- **gênero**: texto escrito (notícias, mídia)
- **tamanho**: lg (541 MB)
- **componentes**: tok2vec, morphologizer, parser, lemmatizer, senter, atribute_ruler, ner
- **vetores**: 500 mil chaves, 500 mil vetores exclusivos (300 dimensões)
- **dataset de treino inclui**: 
  - *corpus* Bosque (9.368 frases), nome formal: [UD-Portuguese-Bosque 2.8](https://www.puc-rio.br/ensinopesq/ccpg/pibic/relatorio_resumo2018/relatorios_pdf/ctch/LET/LET-Luisa%20Rocha.pdf)
---  
# Nomenclatura dos modelos spaCy:
 - **linguagem**: pt
 - **tipo**: core 
    - core: pipeline de proposito geral com *tagging*, *lemmatization*, *NER* etc. 
    - dep:  pipeline apenas com *tagging*, *lemmatization*
 - **genero**: web ou news
 - **size**: sm, md, lg ou trf.sm e trf não tem vetores estaticos de palavra
---
# Incrementando o reconhecimento de entidades do spaCy

O spaCy tem componentes que permitem o reconhecimento de entidades (NER).

O EntityRuler é um destes componentes, ele permite adicionar entidades nomeadas com base em dicionários de padrões, o que facilita a combinação de reconhecimento de entidades nomeadas baseado em regras e estatísticas para pipelines ainda mais poderosos.

Entity patterns são dicionários com duas chaves: "label", especificando o rótulo a ser atribuído à entidade se o padrão for correspondido, e "pattern", o padrão de correspondência. EntityRuler aceita dois tipos de padrões: Phrase patterns (para a string exata) e Token patterns (lista).

---
# Código-exemplo 1
```python
import spacy

# Carrega o modelo pre-treinado da língua Portuguesa do spaCy
nlp = spacy.load('pt_core_news_sm')

text = 'Foi arrasador quando primeiro descobrimos que nosso manuscrito de Galileu na realidade não é de Galileu,' \
    ' disse em entrevista a diretora interina das bibliotecas da universidade, Donna L. Hayward. Mas como a fina' \
    ' lidade... E esse é um teste para verificar se o spaCy consegue captuar o título do Dr. Jucelino. ' \
    'e um pouco mais de texto para identificar o reconhecimento de entidade localidade João Pessoa-PB, Salvador,' \
    'Ilhéus e Itaparica.' \
    
doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)
```
![Output](/img/entidades.png "Entidades reconhecidas")

---
# Código-exemplo 2
```python
import spacy

# Carrega o modelo pre-treinado da língua Portuguesa do spaCy
nlp = spacy.load('pt_core_news_sm')

text = 'Foi arrasador quando primeiro descobrimos que nosso manuscrito de Galileu na realidade não é de Galileu,' \
    ' disse em entrevista a diretora interina das bibliotecas da universidade, Donna L. Hayward. Mas como a fina' \
    ' lidade... E esse é um teste para verificar se o spaCy consegue captuar o título do Dr. Jucelino. ' \
    'e um pouco mais de texto para identificar o reconhecimento de entidade localidade João Pessoa-PB, Salvador,' \
    'Ilhéus e Itaparica.' \
    
doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)

options = {"compact": True, "bg": "#09a3d5",
           "color": "white", "font": "Source Sans Pro"}

spacy.displacy.serve(doc, style='ent', options=options)
```
![Output](/img/entidades_view.png "Entidades reconhecidas")

---
# Código-exemplo 3
```python
import spacy
from spacy.language import Language
from spacy.tokens import Span

nlp = spacy.load("pt_core_news_sm")

@Language.component("expand_person_entities")
def expand_person_entities(doc):
    new_ents = []
    for ent in doc.ents:
        if (ent.label_ == "PERSON" or ent.label_ == "MISC") and ent.start != 0:
            prev_token = doc[ent.start - 1]
            if prev_token.text in ("Dr", "Dr.", "Mr", "Mr.", "Ms", "Ms."):
                new_ent = Span(doc, ent.start - 1, ent.end, label=ent.label)
                new_ents.append(new_ent)
        else:
            new_ents.append(ent)
    doc.ents = new_ents
    return doc

# Add the component after the named entity recognizer
nlp.add_pipe("expand_person_entities", after="ner")

doc = nlp("Dr. Alex Smith chaired first board meeting of Acme Corp Inc.")
print([(ent.text, ent.label_) for ent in doc.ents])

```
---
# Código-exemplo 4

```python
import spacy

nlp = spacy.load('pt_core_news_sm')
ruler = nlp.add_pipe("entity_ruler")

patterns = [{"label": "PERSONA", 
             "pattern": [{"TEXT": {"REGEX": r"\d{3}"}}]
            }]
            
ruler.add_patterns(patterns) 

doc = nlp("This is Fred and his number is 123 to get an apple  pie") 
for ent in doc.ents:
    print(ent.text, ent.label_)
```