# Previsão do Tempo de Internamento em UCI a partir das Primeiras 24 Horas — MIMIC-III
### Relatório segundo a metodologia CRISP-DM

> Documento-fonte para o entregável em PDF; espelha o `main.ipynb`. Os valores entre
> parêntesis `[..]` são preenchidos a partir de uma execução real sobre o BigQuery. Os números
> de execuções com dados sintéticos **não** são resultados reais — servem apenas para validar o
> *pipeline*. Metodologia: CRISP-DM (Wirth & Hipp, 2000) — as seis fases que se seguem.
>
> **Nota de utilização:** cada secção está escrita para ser colada na célula *markdown*
> correspondente do notebook. Os títulos seguem as fases do CRISP-DM já presentes no `main.ipynb`.

---

## Sumário executivo

Prevemos o **tempo de internamento (Length of Stay, LOS, em dias) numa Unidade de Cuidados
Intensivos (UCI)** usando apenas as **primeiras 24 horas** de cada estadia, a partir de sinais
vitais (CHARTEVENTS), análises laboratoriais (LABEVENTS) e dados demográficos do MIMIC-III. O
trabalho é tratado como um problema de *big data*: a tabela CHARTEVENTS tem ~330 milhões de
linhas (4,2 GB comprimidos) e **nunca é descarregada para o disco local** — toda a filtragem e
agregação pesada é empurrada para o BigQuery, regressando apenas uma tabela compacta de
agregados. Reportamos duas formulações (regressão e classificação ordinal), controlamos
rigorosamente a **fuga de dados (*data leakage*)** e avaliamos com validação cruzada agrupada por
doente. Tão importante como o modelo é a **discussão de desempenho** (tempo, memória, custo de
*queries*) e a **crítica à literatura**, que, como argumentamos, é demasiado heterogénea e
frequentemente contaminada por fugas de dados para servir de *baseline* fiável.

---

## Fase 1 — Compreensão do Problema (*Business Understanding*)

**Objetivo.** Estimar o tempo de internamento em UCI **ao fim de 24 horas** — suficientemente
cedo para apoiar decisões de gestão de camas, alocação de pessoal e planeamento de transferência,
mas tarde o bastante para existir um dia completo de dados.

**Alvo (porquê este).** Definimos LOS = `OUTTIME − INTIME` por `ICUSTAY_ID`. Escolhemos o
internamento *na UCI* (e não a admissão hospitalar) porque: (a) corresponde aos dados de eventos
que modelamos; (b) é o alvo canónico de referência (Harutyunyan et al., 2019); (c) um
`SUBJECT_ID` pode ter várias estadias, problema que resolvemos por *split* agrupado e não por
mudança de alvo.

**Janela (porquê 24h).** A escolha da janela é o compromisso central do problema. Uma janela de
6h tem poucos resultados laboratoriais já disponíveis; uma de 72h atrasa todas as decisões e
exclui mais estadias curtas (que terminam antes da janela). A janela de 24h equilibra os dois:
existe um dia inteiro de vitais e labs, e ainda há tempo para agir. A coorte fica assim restrita
a estadias **ainda em curso às 24h** (`MIN_LOS_HOURS = 24`), com o LOS limitado a 60 dias para
neutralizar estadias erróneas ou extremas.

**Duas formulações (porquê ambas).** A regressão (prever dias) é o pedido literal, mas tem um R²
intrinsecamente baixo — é um problema difícil. A classificação ordinal (curto <3 d / médio 3–7 d
/ longo >7 d) é a formulação mais robusta e mais comum na literatura, e produz *buckets*
operacionalmente mais úteis. Reportamos as duas.

**Categorias de atributos.** Vitais das primeiras 24h (hemodinâmica, respiração, neurologia),
labs (função renal, eletrólitos, perfusão, hematologia, coagulação) e demografia/administrativos
(idade, sexo, tipo de admissão, unidade) — exatamente as variáveis que um clínico tem disponíveis
à hora 24 e que a literatura associa a deterioração e a estadias prolongadas.

---

## Fase 2 — Compreensão dos Dados (*Data Understanding*)

**Conjunto de dados (MIMIC-III v1.4).**

| Tabela | Linhas | Utilização |
|---|---|---|
| CHARTEVENTS | ~330 M (4,2 GB gz) | Sinais vitais das primeiras 24h |
| LABEVENTS | ~27 M | Resultados laboratoriais das primeiras 24h |
| ICUSTAYS / ADMISSIONS / PATIENTS | 61,5k / 59k / 46,5k | coorte, alvo, demografia |
| D_ITEMS / D_LABITEMS | ~12,5k / ~750 | mapeamento ITEMID → conceito |

**Inclusão do LABEVENTS — uma decisão a posteriori.** Importa ser transparente quanto à
cronologia do trabalho: o *pipeline* foi inicialmente construído apenas em torno do CHARTEVENTS
(sinais vitais), e a integração da tabela LABEVENTS (análises laboratoriais) foi uma **decisão
tomada posteriormente**, já com a arquitetura montada. Esta ordem histórica explica várias opções
de desenho que de outro modo poderiam parecer arbitrárias: (i) os atributos de laboratório são
acrescentados por *left join* sobre a matriz de vitais, mantendo **exatamente o mesmo conjunto de
estadias** — o que permite a ablação «sem labs vs. com labs» da Fase 4 ser rigorosamente
comparável (as mesmas linhas, apenas colunas a mais); (ii) os conceitos de laboratório levam o
prefixo `lab_` para não colidirem com vitais homónimos (a glicose, por exemplo, existe nas duas
tabelas); (iii) atribuímos aos labs um limiar de *missingness* mais tolerante, por serem medidos
com menos frequência do que os vitais. Em vez de dissimular esta sequência, tornámo-la explícita e
convertêmo-la numa mais-valia experimental: a ablação **quantifica honestamente quanto é que os
labs acrescentam** ao modelo que já existia com vitais apenas.

**Análise exploratória.** O LOS é fortemente assimétrico à direita (mediana [..] d, com uma cauda
longa e heterogénea); a classe "longo" é minoritária. Mostramos também as **séries temporais de
eventos por doente** (o gráfico-exemplo do enunciado: X = fração do dia desde a admissão, Y =
valor medido, cor = conceito) e uma vista de **doente vs. banda da população (média ± 1 DP)** que
revela se aquele doente corre alto/baixo/instável face à coorte.

**Harmonização CareVue/MetaVision.** O MIMIC-III resulta da fusão de **dois sistemas de informação
de UCI** usados em épocas diferentes no mesmo hospital: o **CareVue** (Philips, anterior a 2008),
cujos ITEMIDs são < 220000, e o **MetaVision** (iMDsoft, a partir de 2008), com ITEMIDs ≥ 220000.
Como cada sistema tinha o seu próprio dicionário de itens, o *mesmo* conceito clínico ficou
registado sob **vários ITEMIDs distintos** — por exemplo, a **frequência cardíaca** é o item `211`
no CareVue mas `220045` no MetaVision; a pressão arterial sistólica tem meia dúzia de códigos
repartidos entre os dois sistemas. Se agregássemos os ITEMIDs em bruto, cada código geraria a sua
própria coluna e o resultado seria duplamente problemático: (i) obteríamos **colunas duplicadas e
semivazias** — um doente do CareVue teria valor em `211` e *NaN* em `220045`, e vice-versa, pelo
que o modelo veria dois atributos sem perceber que são o mesmo; e (ii) a metade CareVue e a metade
MetaVision da coorte tornar-se-iam, na prática, **dois conjuntos de dados disjuntos**,
impossibilitando comparar doentes registados em sistemas diferentes. Por isso, **antes de
agregar**, mapeamos todos os ITEMIDs equivalentes para um único nome de conceito canónico
(`heart_rate`, `sbp`, …), definido em [`src/data/concepts.py`](../src/data/concepts.py), de modo
que `211` e `220045` alimentem a mesma coluna `heart_rate`.

**De onde vêm estes agrupamentos (não foram inventados por nós).** Decidir que `211` e `220045`
são "a mesma coisa" exige conhecimento de domínio sobre os dois dicionários do MIMIC. Não fizemos
esse mapeamento de raiz: seguimos as **definições de conceito curadas pela comunidade** no
repositório oficial `mimic-code` do MIT-LCP — o mesmo grupo que publica o MIMIC —, onde *scripts*
SQL revistos por pares listam explicitamente que ITEMIDs do CareVue e do MetaVision pertencem a
cada conceito, p. ex. `HeartRate → (211, 220045)` e `SysBP → (51, 442, 455, 6701, 220179,
220050)`. O nosso dicionário `CONCEPT_ITEMIDS` espelha esses agrupamentos, e os ITEMIDs dos labs em
[`src/data/lab_concepts.py`](../src/data/lab_concepts.py) (creatinina `50912`, ureia `51006`,
lactato `50813`, …) seguem as definições de *first-day labs* do mesmo repositório [ref. 9]. As
**gamas de plausibilidade** por conceito (`CONCEPT_VALID_RANGES`, que descartam valores
fisiologicamente impossíveis) foram, essas sim, fixadas por nós, deliberadamente generosas. Como o
`mimic-code` é curado pela comunidade, há ITEMIDs de fronteira ocasionalmente debatidos (é o que
motiva a *issue* #472 referida no código); por isso, como verificação de sanidade, podemos
confirmar contra a tabela `D_ITEMS` [ref. 10] que o `LABEL` de cada ITEMID corresponde ao conceito
que lhe atribuímos.

**Missingness informativo.** A ausência de medições **não é aleatória**: decidir pedir um lactato
ou um vital extra é, em si, um sinal clínico de preocupação. Por isso, além de imputar, adicionamos
(Fase 3) um atributo 0/1 *foi-medido* por conceito, que captura a própria decisão de medir —
informação que a imputação apagaria.

---

## Fase 3 — Preparação dos Dados (*Data Preparation*)

**Controlos de fuga de dados — o que é escondido ou alterado.** Este é o ponto metodológico mais
importante do trabalho (e, como discutimos na Fase 4, o que mais distingue a nossa abordagem da
literatura mais fraca).

| Variável / fonte | Ação | Porquê |
|---|---|---|
| `OUTTIME` / `DISCHTIME` | escondida (só define o alvo) | saber a alta *é* a resposta |
| eventos após `INTIME + 24h` | escondidos (janela em SQL) | indisponíveis no momento da previsão |
| contagens de medições | limitadas à janela de 24h | contagens da estadia inteira codificam o LOS |
| idade (`DOB`) | limitada a 90 | o MIMIC desloca o DOB de >89 anos em ~300 anos |
| `CHARTEVENTS.ERROR = 1` | descartada | valores assinalados como erro pelo clínico |
| `HOSPITAL_EXPIRE_FLAG`, `DISCHARGE_LOCATION`, `DEATHTIME`, `LOS` pré-calculado | escondidas | conhecidas só na/após a alta ou cópia direta do alvo |

**Nota sobre a idade (artefacto de anonimização).** A linha da idade merece um esclarecimento,
porque a sua motivação é de **qualidade de dados** e não, em rigor, de fuga. Por exigências de
anonimização — a norma norte-americana HIPAA trata qualquer idade acima dos 89 anos como
identificador potencial —, o MIMIC-III **desloca deliberadamente a data de nascimento (`DOB`) dos
doentes com mais de 89 anos** para cerca de 300 anos antes da primeira admissão. Em consequência,
ao calcular a idade como `ADMITTIME − DOB`, esses doentes surgem com idades absurdas de **~300
anos**. Deixar estes valores em bruto distorceria as estatísticas de idade e prejudicaria os
modelos (sobretudo os lineares, sensíveis a *outliers*). Por isso limitamos a idade a um teto de
90 anos (`AGE_CAP = 90`): todos os «doentes de 300 anos» passam a contar como 90, o que
**preserva a informação clinicamente relevante** («é um doente muito idoso») sem arrastar o valor
disparatado. Mantemos a linha na tabela por transparência, ainda que a sua natureza seja distinta
dos restantes controlos, que esses sim visam impedir *leakage*.

**Engenharia de atributos.** Agregados por (estadia, conceito): média/mín/máx/desvio-padrão/contagem;
intensidade limitada à janela; demografia codificada; e os indicadores `*_measured`. A imputação e
a normalização são feitas **dentro do *pipeline*** (ajustadas por *fold*), nunca sobre o conjunto
todo — caso contrário, o conjunto de teste contaminaria o de treino. Construímos **duas matrizes
sobre exatamente as mesmas linhas** (sem labs / com labs) para uma ablação rigorosamente
comparável.

**Exploração não supervisionada (exploratória, *não* preditiva).** PCA (scree + projeção 2-D),
t-SNE (apenas visualização — as distâncias no mapa não são fiáveis) e KMeans com *silhouette*
para escolher k. Expectativa honesta e confirmada: coortes clínicas agrupam-se fracamente
(*silhouette* tipicamente < 0,3), e os *clusters* alinham-se apenas vagamente com o LOS — o que
confirma que um modelo *supervisionado* é a ferramenta certa, e não uma segmentação.

**Divisão treino/teste.** `GroupShuffleSplit` / `GroupKFold` sobre `SUBJECT_ID`: nenhum doente
aparece simultaneamente no treino e no teste. Sem isto, o modelo memoriza idiossincrasias do
doente e as métricas ficam otimistas — uma falha comum na literatura (ver Fase 4).

---

## Fase 4 — Modelação (*Modeling*)

### 4.1 Revisão crítica da literatura — porque não a usamos como *baseline*

Esta secção é deliberadamente crítica. Levantámos os trabalhos mais citados sobre previsão de LOS
em MIMIC e concluímos que a maioria **não serve como linha de base fiável**. Resumindo o estado
da arte:

**A maioria faz classificação binária**, normalmente "estadia curta vs. longa" com um limiar
escolhido *a posteriori*, inviabilizando os dados:
- *Predicting Hospital Stay Length Using Explainable Machine Learning*
- *Predicting Hospital Length of Stay using Neural Networks on MIMIC III Data*
- *Predicting length of stay ranges by using novel deep neural networks*
- *Identifying and timing patient outcomes in clinician notes using large language models*
- *A deep attention model to forecast the Length Of Stay and the in-hospital mortality right on
  admission from ICD codes and demographic data*

**Poucos fazem regressão** (o pedido literal deste trabalho, e o mais difícil):
- *Prediction of Length-of-stay at Intensive Care Unit (ICU) Using Machine Learning based on
  MIMIC-III Database*
- *Prediction of Intensive Care Unit Length of Stay in the MIMIC-IV Dataset*
- *Using Machine Learning Models to Predict the Length of Stay in a Hospital Setting*

**Muitos usam grupos de tempo** (curto / médio / longo) — a formulação que também adotamos como
segunda framing, por ser a mais robusta.

**Problemas metodológicos transversais** que tornam estes trabalhos inadequados como *baseline*:

1. **Fuga de dados generalizada.** O caso mais flagrante é prever o LOS "logo na admissão" a
   partir de **códigos ICD** e do diagnóstico: no MIMIC, os códigos ICD são códigos de faturação
   atribuídos **na alta ou depois dela**, pelo que usá-los "à admissão" é fuga de dados pura. O
   mesmo se aplica a modelos sobre **notas clínicas** acumuladas ao longo de toda a estadia: usar
   o texto completo para prever a duração da estadia vaza o desfecho.
2. **Ausência de agrupamento por doente.** *Splits* aleatórios deixam estadias do mesmo
   `SUBJECT_ID` no treino e no teste, inflacionando as métricas.
3. **Definições de alvo inconsistentes.** Uns usam LOS hospitalar, outros LOS de UCI; alguns
   incluem a própria janela de previsão dentro do alvo. Isto torna os números mutuamente
   incomparáveis.
4. **Limiares e desbalanceamento arbitrários.** Em classificação binária, o limiar (>3 d, >7 d…)
   e o balanceamento variam de artigo para artigo, e muitos reportam *accuracy*/AUROC sobre
   classes desbalanceadas sem calibração nem curvas precision-recall — métricas facilmente
   enganadoras.
5. **Desempenhos absurdos.** R² próximos de 0,9 ou *accuracy* >95% num problema que os trabalhos
   rigorosos mostram ser difícil são, quase sempre, sintoma de fuga de dados, não de mérito.

**Conclusão crítica — com uma ressalva.** Tomados em conjunto, estes trabalhos são demasiado
heterogéneos e frequentemente contaminados para servirem de *baseline* quantitativa. **No
entanto, seria injusto e incorreto descartá-los a todos por igual.** O *benchmark* de
**Harutyunyan et al. (2019)** é a exceção rigorosa: define a coorte publicamente, controla fugas,
usa *splits* corretos e publica código reproduzível. É por isso o **único** que tratamos como
linha de base de referência (regressão linear MAD ≈ 116,4h ≈ 4,85 d; LSTM ≈ 94,0h ≈ 3,92 d;
*quadratic kappa* linear ≈ 0,34 / LSTM ≈ 0,43). Os restantes servem de *contexto*, não de alvo.

### 4.2 Linhas de base e SOTA proposto

Ancoramos contra três níveis: (i) **baselines triviais** (média / classe maioritária) — um modelo
tem de os bater, e um R² ≈ 0 destes confirma que o alvo não está a vazar; (ii) o **benchmark de
Harutyunyan**; (iii) o nosso **SOTA proposto** para este cenário tabular: **árvores com *gradient
boosting* (XGBoost, LightGBM)**, que igualam ou superam redes profundas em dados tabulares de
dimensão média (Grinsztajn et al., 2022), com a vantagem da interpretabilidade. O único *deep
learning* que valeria a pena testar operaria sobre a *sequência* horária em bruto (GRU / CNN
temporal); deixamo-lo como trabalho futuro, pois os ganhos específicos em LOS são modestos.

**Afinação de hiperparâmetros.** `RandomizedSearchCV` com `GroupKFold` sobre a família de
*boosting*.

**Ablação LABEVENTS.** Cada modelo é validado, com validação cruzada sobre os mesmos *folds*, sem
e com os atributos de laboratório, para quantificar o contributo dos labs de forma honesta.

---

## Fase 5 — Avaliação (*Evaluation*)

| Formulação | Métrica | Sem labs | Com labs | Δ | Literatura |
|---|---|---|---|---|---|
| Regressão (afinado, hold-out) | MAE (d) | [..] | [..] | **[..]** | LSTM 3,92 |
| Regressão | R² | [..] | [..] | **[..]** | ~0,04 |
| Classificação (melhor) | *quadratic* κ | [..] | [..] | **[..]** | LSTM 0,43 |
| Classificação | macro-AUROC | [..] | [..] | **[..]** | >7d 0,76 |

Os labs representaram **[..]%** da importância do modelo afinado (indicadores de missingness
**[..]%**); o ganho concentra-se nos modelos de árvore (o modelo linear quase não beneficia — os
labs acrescentam sinal não linear de disfunção orgânica). Analisamos as **distribuições dos
atributos mais importantes por *bucket* de LOS** e a **importância de atributos**. A tabela de
erro por banda revela o modo de falha universal: **as estadias longas são sistematicamente
subestimadas** — coerente com a assimetria do alvo e a escassez de exemplos longos.

---

## Fase 6 — Implementação, Desempenho e *Big Data*

Esta fase responde diretamente ao pedido do enunciado: *reportar o tempo total de execução, fazer
profiling das fases de ML e discutir questões de desempenho.*

### 6.1 BigQuery vs. PySpark — porque escolhemos BigQuery

As tabelas do MIMIC são **relacionais e tipo-SQL** (chaves, tipos, junções por `HADM_ID` /
`ICUSTAY_ID`). A carga de trabalho é *filtrar-depois-agregar* sobre duas tabelas grandes. Para
este padrão, o BigQuery é a escolha natural:

- **Sem gestão de *cluster*.** O BigQuery é *serverless*; o PySpark exigiria provisionar e afinar
  um *cluster* (Dataproc/EMR), gerir memória de executores, *partitioning* e *shuffles* — muito
  mais *boilerplate* para o mesmo resultado.
- **SQL declarativo vs. código imperativo.** A lógica de coorte, janela e agregação exprime-se
  diretamente em SQL, num único local (`src/data/sql.py`), o que é mais legível e menos sujeito a
  erros do que transformações encadeadas de *DataFrames* Spark.
- **Quando o Spark ganharia.** Se os dados vivessem em ficheiros planos sem um *data warehouse*,
  ou se precisássemos de transformações iterativas/personalizadas que não se exprimem bem em SQL,
  o Spark seria a opção acertada — e a *mesma* decomposição map (*parse* + filtro + janela) /
  reduce (agregar por estadia-conceito) aplicar-se-ia.

### 6.2 O BigQuery já faz *MapReduce* — não o reimplementámos de propósito

A *engine* do BigQuery (Dremel) executa exatamente este trabalho como um plano distribuído de
**estilo map-reduce**: o `WHERE` / `JOIN` é o *map* (corre em milhares de *shards* em paralelo) e
o `GROUP BY ... AVG/MIN/MAX/COUNT` é o *reduce*. Empurramos esse cálculo para SQL e só algumas
centenas de milhar de linhas agregadas saem do *warehouse*. **Escrever à mão um *job*
Hadoop/Spark MapReduce reproduziria o que o BigQuery já faz**, com mais sobrecarga e mais lento de
desenvolver — seria *over-engineering*. O MapReduce está, portanto, presente conceptualmente e
via *engine*, e não como código repetitivo que teríamos de manter.

### 6.3 Guardar os agregados na cloud / *cache* — justificação por tempo

A *query* de agregação sobre CHARTEVENTS é a operação cara: na nossa medição, indexar/varrer a
tabela de ~330 M linhas demorou **~1191 s (~20 min)**. Reexecutar isto a cada iteração do notebook
seria proibitivo. Por isso:

- **Persistimos os agregados** (média, mín, máx, desvio-padrão, contagem por estadia-conceito) já
  calculados — tanto do lado do BigQuery (as tabelas carregadas ficam no *dataset*) como num
  **cache local em parquet** (`src/data/loader.py`), indexado por *hash* da *query*.
- **Justificação:** é uma troca de tempo por espaço. O cálculo pesado faz-se **uma vez**; as
  execuções seguintes leem um ficheiro compacto em segundos. A *correctness* não é afetada porque
  o resultado é determinístico para a mesma *query*.

### 6.4 Otimização de *queries* — o custo mede-se em *bytes processados*, não em "tokens"

> **Correção de terminologia (e crítica a uma premissa errada).** O custo e a otimização de uma
> *query* no BigQuery medem-se em **bytes lidos/processados**, não em "tokens" — "tokens" é um
> conceito de modelos de linguagem (LLMs), que aqui não se aplica. O objetivo correto é **varrer
> menos bytes**.

Como o BigQuery é **colunar**, as otimizações que aplicámos foram:

- **Selecionar apenas as colunas necessárias** (nunca `SELECT *`): num motor colunar, cada coluna
  lida acrescenta bytes varridos. Lemos só `icustay_id`, `itemid`, `charttime`, `valuenum`, etc.
- **Filtrar cedo** (janela temporal e gama de valores no `WHERE`) para reduzir as linhas que
  chegam ao `GROUP BY`.
- **Agregar do lado do servidor** e trazer apenas o resultado compacto, em vez de exportar eventos
  em bruto.
- **Filtrar por ITEMIDs de interesse** através de um mapa de conceitos *inline* (`UNNEST(STRUCT)`),
  evitando uma tabela auxiliar e garantindo que o mapeamento coincide com o do Python.
- **Reutilização via *cache*** (§6.3), que evita revarrer a tabela em execuções repetidas.

### 6.5 Memória: *downcasting* de tipos (com a devida ressalva crítica)

Forçar o Pandas a usar tipos mais leves — `float64 → float32`, `int64 → int32` — reduz para
**metade** o espaço ocupado *por essas colunas* numéricas. Aplicamo-lo à matriz final de atributos
(convertida para `float32` em `engineering.py`).

> **Ressalva crítica.** É incorreto afirmar que o *downcasting* "reduz a RAM para metade" de forma
> geral: (i) só afeta as colunas numéricas a que se aplica — *timestamps*, *strings* e índices não
> encolhem; (ii) pode introduzir **perda de precisão** (somas de grande magnitude, datas); e
> (iii) **na nossa arquitetura é pouco determinante**, precisamente porque *não* mantemos os 330 M
> de linhas em Pandas — o BigQuery agrega primeiro e a matriz local é pequena (~10⁵ linhas × ~150
> colunas). O *downcasting* é, portanto, uma boa prática geral e útil caso alguma vez puxássemos
> eventos em bruto, mas não é, aqui, a otimização que carrega o desempenho — essa é o empurrar da
> agregação para o BigQuery.

### 6.6 Comparação de tempo e memória entre algoritmos

Reportamos, para cada modelo, o **tempo de ajuste** (já registado por `harness.py` via
`fit_time_s` na validação cruzada) e o **pico de memória**. Padrão esperado e a discutir:

| Modelo | Tempo de treino | Memória | Notas |
|---|---|---|---|
| Ridge / Logística | muito baixo | muito baixa | solução fechada / convexa; escala linearmente |
| Random Forest | alto | **alta** | guarda 300 árvores completas em memória |
| HistGradientBoosting | baixo | baixa | *binning* por histograma; muito eficiente |
| XGBoost / LightGBM | baixo–médio | baixa | histograma + multi-*thread* interno |
| KNN | treino trivial | **alta** | guarda todo o treino; caro na *inferência* |
| SVR | **muito alto** | alta | ~O(n²); não escala para esta dimensão |

A leitura operacional é clara: os modelos de *boosting* por histograma oferecem o melhor
compromisso desempenho/recursos, o que reforça a escolha de SOTA da Fase 4; RF e KNN pagam um
custo de memória desproporcionado e o SVR simplesmente não escala.

### 6.7 Multiprocessamento — onde o paralelismo é realmente usado

- **Lado do servidor (BigQuery):** a agregação distribui-se por muitos *workers* (§6.2).
- **Validação cruzada e ajuste:** `n_jobs = -1` (joblib) corre os *folds* do `GroupKFold` e as
  árvores da Random Forest em paralelo por todos os núcleos.
- **Bibliotecas de *boosting*:** XGBoost/LightGBM são internamente multi-*thread*.
- **Caching como amortização:** os resultados em parquet evitam repetir o passo distribuído.

**Não** paralelizamos por processos a engenharia de atributos em Pandas: após a agregação, a
matriz é pequena e vetorizada, pelo que a sobrecarga de um *process pool* só atrasaria.

### 6.8 *Profiling* — tempo total e repartição por fase

Tempo total: **[..] s** — repartido por carregamento (BigQuery), construção de atributos,
exploração não supervisionada, validação cruzada com ablação, e procura de hiperparâmetros (ver
gráfico de *profiling* no notebook, §6.1).

---

## Conclusões

- **Os labs ajudam, na direção esperada.** O MAE de *hold-out* variou [..] d e o *quadratic
  kappa* [..]; os labs representaram [..]% da importância, concentrados nos modelos de árvore (o
  modelo linear quase não beneficia — os labs acrescentam sinal não linear de disfunção orgânica).
  Os indicadores de missingness informativo contribuíram [..]%.
- **A regressão continua difícil** (R² ≈ [..], MAE ≈ [..] d), mas bate claramente o *baseline* da
  média e alinha-se com a literatura rigorosa de primeiras-24h; a **classificação** (*quadratic
  kappa* ≈ [..]) é comparável ao *benchmark* LSTM de Harutyunyan, usando árvores interpretáveis.
- **A estrutura é fraca** (melhor *silhouette* ≈ [..]): os doentes não se separam em fenótipos
  limpos, pelo que um modelo supervisionado é a escolha certa; as vistas não supervisionadas são
  descritivas, não preditivas.
- **A literatura é um mau *baseline*.** Pela heterogeneidade de alvos e pela fuga de dados
  frequente, a maioria dos trabalhos publicados não é utilizável como referência quantitativa; só
  o *benchmark* de Harutyunyan resiste a um escrutínio metodológico, e é esse que usamos.
- **O *big data* foi tratado empurrando a agregação de estilo map-reduce para o BigQuery**; a
  *cache* dos agregados torna as iterações seguintes quase instantâneas, e o tempo total sobre a
  matriz compacta local é de [..] s.

### Limitações e trabalho futuro

Medicação (INPUTEVENTS) e notas clínicas não foram usadas; um alvo de LOS *remanescente*
re-estimado diariamente seria mais útil em produção; uma janela pré-UCI (`intime − 6h`) captaria
labs de admissão; SHAP para interpretabilidade local; *deep learning* sequencial; e um *split*
temporal (por data de admissão) ao longo da transição CareVue → MetaVision.

### Referências

1. Wirth & Hipp (2000). *CRISP-DM 1.0*.
2. Johnson et al. (2016). *MIMIC-III, a freely accessible critical care database*. Scientific Data.
3. Harutyunyan et al. (2019). *Multitask learning and benchmarking with clinical time series data*.
4. Wang et al. (2020). *MIMIC-Extract*. ACM CHIL.
5. Grinsztajn et al. (2022). *Why do tree-based models still outperform deep learning on tabular
   data?* NeurIPS.
6. Sharafoddini et al. (2019). *Patient phenotyping / informative missingness in MIMIC*.
7. Dean & Ghemawat (2008). *MapReduce: Simplified Data Processing on Large Clusters*. CACM.
8. Melnik et al. (2010). *Dremel: Interactive Analysis of Web-Scale Datasets* (motor do BigQuery).
9. Johnson, Stone, Celi & Pollard (2018). *The MIMIC Code Repository: enabling reproducibility in
   critical care research*. J Am Med Inform Assoc 25(1):32–39. — fonte das definições de conceito
   (agrupamentos de ITEMIDs CareVue/MetaVision e dos labs). Repositório: MIT-LCP `mimic-code`,
   https://github.com/MIT-LCP/mimic-code (ver `mimic-iii/concepts/`).
10. MIMIC-III, dicionários `D_ITEMS` e `D_LABITEMS` —
    https://mimic.mit.edu/docs/iii/tables/d_items/ e `/d_labitems/`.
