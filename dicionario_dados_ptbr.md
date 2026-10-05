# Dicionário de Dados - Housing Prices Competition (Kaggle)
*Guia completo e traduzido para Português (Brasil)*

Este documento traduz e explica detalhadamente todas as **81 colunas** (1 identificador + 79 variáveis preditoras + 1 variável alvo) presentes no conjunto de dados da competição **Housing Prices Competition for Kaggle Learn Users** (Ames Housing Dataset).

> 💡 **Dica de ouro sobre dados faltantes (`NA`):**  
> Em várias colunas categóricas deste dataset (como `Alley`, `BsmtQual`, `FireplaceQu`, `GarageType`, `PoolQC`, `Fence`), a sigla **`NA` NÃO significa dado faltante (erro de preenchimento)**. Ela significa **"Não possui" / "Ausente"** (ex.: a casa não tem lareira, não tem piscina, não tem garagem). Ao carregar com `pandas.read_csv()`, trate esse detalhe para não descartar linhas válidas!

---

## Índice por Categorias

1. [Identificação e Variável Alvo](#1-identificação-e-variável-alvo)
2. [Terreno, Zoneamento e Localização](#2-terreno-zoneamento-e-localização)
3. [Estrutura Geral, Idade e Condição](#3-estrutura-geral-idade-e-condição)
4. [Telhado, Estrutura Externa e Fundação](#4-telhado-estrutura-externa-e-fundação)
5. [Porão / Subsolo (Basement)](#5-porão--subsolo-basement)
6. [Sistemas Internos: Elétrica, Aquecimento e Ar](#6-sistemas-internos-elétrica-aquecimento-e-ar)
7. [Áreas dos Andares e Cômodos](#7-áreas-dos-andares-e-cômodos)
8. [Lareiras (Fireplaces)](#8-lareiras-fireplaces)
9. [Garagem (Garage)](#9-garagem-garage)
10. [Decks, Varandas e Piscina](#10-decks-varandas-e-piscina)
11. [Recursos Diversos e Detalhes da Venda](#11-recursos-diversos-e-detalhes-da-venda)

---

## 1. Identificação e Variável Alvo

### `Id` (Identificador)
- **Tipo:** Numérico discreto
- **Descrição:** Número sequencial identificador único do imóvel.
- **Uso:** Não deve ser usado como feature preditiva. Usado para parear previsões no arquivo de submissão.

### `SalePrice` (Preço de Venda) — *Variável Alvo*
- **Tipo:** Numérico contínuo
- **Unidade:** Dólares americanos ($)
- **Descrição:** O preço final pelo qual a propriedade foi vendida. Esta é a variável que o modelo deve prever.
- **Nota:** A métrica de avaliação da competição avalia o logaritmo dos preços (RMSLE), por isso é comum treinar modelos prevendo `log(SalePrice)`.

---

## 2. Terreno, Zoneamento e Localização

### `MSZoning` (Classificação de Zoneamento)
- **Tipo:** Categórica nominal
- **Descrição:** Identifica a classificação geral de zoneamento onde o imóvel está situado:
  - `A`: Agrícola (*Agriculture*)
  - `C` ou `C (all)`: Comercial (*Commercial*)
  - `FV`: Residencial Vila Flutuante (*Floating Village Residential*)
  - `I`: Industrial (*Industrial*)
  - `RH`: Residencial de Alta Densidade (*Residential High Density*)
  - `RL`: Residencial de Baixa Densidade (*Residential Low Density*)
  - `RP`: Residencial Parque de Baixa Densidade (*Residential Low Density Park*)
  - `RM`: Residencial de Média Densidade (*Residential Medium Density*)

### `LotFrontage` (Frente do Lote)
- **Tipo:** Numérico contínuo
- **Unidade:** Pés lineares (1 pé ≈ 0,3048 m)
- **Descrição:** Largura linear da propriedade que faz limite/acesso com a rua.

### `LotArea` (Área do Terreno)
- **Tipo:** Numérico contínuo
- **Unidade:** Pés quadrados (*sq ft*, 1 sq ft ≈ 0,0929 m²)
- **Descrição:** Tamanho total da área do terreno do imóvel.

### `Street` (Tipo de Rua de Acesso)
- **Tipo:** Categórica nominal
- **Descrição:** Tipo de pavimentação da rua de acesso ao imóvel:
  - `Grvl`: Cascalho (*Gravel*)
  - `Pave`: Pavimentada / Asfaltada (*Paved*)

### `Alley` (Acesso por Beco/Travessa)
- **Tipo:** Categórica nominal
- **Descrição:** Tipo de acesso secundário por beco ou viela:
  - `Grvl`: Cascalho
  - `Pave`: Pavimentado
  - `NA`: Não possui acesso por beco (*No alley access*)

### `LotShape` (Formato do Terreno)
- **Tipo:** Categórica ordinal
- **Descrição:** Formato geométrico geral da propriedade:
  - `Reg`: Regular (retangular ou quadrado padrão)
  - `IR1`: Levemente irregular
  - `IR2`: Moderadamente irregular
  - `IR3`: Muito irregular

### `LandContour` (Nivelamento / Topografia do Terreno)
- **Tipo:** Categórica nominal
- **Descrição:** Nível e relevo do terreno:
  - `Lvl`: Plano / Nivelado (*Near Flat/Level*)
  - `Bnk`: Barranco (*Banked* - elevação rápida e significativa da rua até a casa)
  - `HLS`: Encosta de Colina (*Hillside* - inclinação lateral acentuada de um lado para o outro)
  - `Low`: Depressão / Baixio (*Depression*)

### `Utilities` (Serviços Públicos Disponíveis)
- **Tipo:** Categórica ordinal
- **Descrição:** Tipo de utilidades públicas conectadas ao imóvel:
  - `AllPub`: Todos os serviços públicos (Eletricidade, Gás, Água e Esgoto)
  - `NoSewr`: Eletricidade, Gás e Água (Fossa séptica no lugar de esgoto)
  - `NoSeWa`: Apenas Eletricidade e Gás (Sem água encanada pública)
  - `ELO`: Apenas Eletricidade (*Electricity only*)

### `LotConfig` (Configuração do Terreno)
- **Tipo:** Categórica nominal
- **Descrição:** Disposição do lote em relação à quadra/rua:
  - `Inside`: Lote interno (no meio da quadra)
  - `Corner`: Lote de esquina
  - `CulDSac`: Lote em rua sem saída com rotatória (*Cul-de-sac*)
  - `FR2`: Frente voltada para 2 lados da rua
  - `FR3`: Frente voltada para 3 lados da rua

### `LandSlope` (Declive do Terreno)
- **Tipo:** Categórica ordinal
- **Descrição:** Grau de inclinação do terreno:
  - `Gtl`: Inclinação suave (*Gentle slope*)
  - `Mod`: Inclinação moderada (*Moderate Slope*)
  - `Sev`: Inclinação severa/íngreme (*Severe Slope*)

### `Neighborhood` (Bairro)
- **Tipo:** Categórica nominal
- **Descrição:** Bairros e regiões geográficas dentro da cidade de Ames, Iowa:
  - `Blmngtn`: Bloomington Heights
  - `Blueste`: Bluestem
  - `BrDale`: Briardale
  - `BrkSide`: Brookside
  - `ClearCr`: Clear Creek
  - `CollgCr`: College Creek
  - `Crawfor`: Crawford
  - `Edwards`: Edwards
  - `Gilbert`: Gilbert
  - `IDOTRR`: Iowa DOT and Rail Road (próximo à ferrovia)
  - `MeadowV`: Meadow Village
  - `Mitchel`: Mitchell
  - `NAmes`: North Ames
  - `NoRidge`: Northridge
  - `NPkVill`: Northpark Villa
  - `NridgHt`: Northridge Heights
  - `NWAmes`: Northwest Ames
  - `OldTown`: Old Town (Centro histórico antigo)
  - `SWISU`: Sul e Oeste da Iowa State University
  - `Sawyer`: Sawyer
  - `SawyerW`: Sawyer West
  - `Somerst`: Somerset
  - `StoneBr`: Stone Brook
  - `Timber`: Timberland
  - `Veenker`: Veenker

### `Condition1` (Condição de Proximidade 1)
- **Tipo:** Categórica nominal
- **Descrição:** Proximidade com vias arteriais, ferrovias ou parques:
  - `Artery`: Adjacente a rua arterial (avenida movimentada)
  - `Feedr`: Adjacente a rua coletora (*feeder street*)
  - `Norm`: Normal (rua residencial típica)
  - `RRNn`: A menos de 60 metros (200 pés) da ferrovia Norte-Sul
  - `RRAn`: Adjacente à ferrovia Norte-Sul
  - `PosN`: Próximo a área de lazer positiva (parque, cinturão verde, etc.)
  - `PosA`: Adjacente a área de lazer positiva
  - `RRNe`: A menos de 60 metros da ferrovia Leste-Oeste
  - `RRAe`: Adjacente à ferrovia Leste-Oeste

### `Condition2` (Condição de Proximidade 2)
- **Tipo:** Categórica nominal
- **Descrição:** Mesmas categorias de `Condition1`, caso o imóvel faça divisa com mais de uma condição externa.

---

## 3. Estrutura Geral, Idade e Condição

### `MSSubClass` (Classe de Construção / Tipologia)
- **Tipo:** Categórica codificada numericamente
- **Descrição:** Identifica o tipo e a época arquitetônica da residência:
  - `20`: 1 Pavimento, 1946 ou mais nova, todos os estilos (*1-Story Newer*)
  - `30`: 1 Pavimento, 1945 ou mais antiga (*1-Story Older*)
  - `40`: 1 Pavimento com sótão acabado (*Finished Attic*), qualquer idade
  - `45`: 1 pavimento e meio, sótão inacabado
  - `50`: 1 pavimento e meio, sótão acabado
  - `60`: 2 Pavimentos, 1946 ou mais nova
  - `70`: 2 Pavimentos, 1945 ou mais antiga
  - `75`: 2 pavimentos e meio, qualquer idade
  - `80`: Níveis divididos ou multinível (*Split or Multi-Level*)
  - `85`: Sobrado com hall no nível intermediário (*Split Foyer*)
  - `90`: Duplex (residência geminada bifamiliar)
  - `120`: 1 Pavimento PUD (*Planned Unit Development* / Condomínio fechado planejado), 1946 ou mais nova
  - `150`: 1 pavimento e meio PUD
  - `160`: 2 Pavimentos PUD, 1946 ou mais nova
  - `180`: PUD Multinível (com split level/foyer)
  - `190`: Conversão familiar (casa unifamiliar convertida em bifamiliar)

### `BldgType` (Tipo de Habitação)
- **Tipo:** Categórica nominal
- **Descrição:** Tipo de moradia em relação aos vizinhos:
  - `1Fam`: Casa unifamiliar isolada (*Single-family Detached*)
  - `2FmCon`: Conversão bifamiliar (construída originalmente como casa única e dividida em duas)
  - `Duplx`: Duplex
  - `TwnhsE`: Sobrado / Casa geminada de ponta/canto (*Townhouse End Unit*)
  - `TwnhsI`: Sobrado / Casa geminada de meio (*Townhouse Inside Unit*)

### `HouseStyle` (Estilo Arquitetônico da Moradia)
- **Tipo:** Categórica nominal
- **Descrição:** Estilo construtivo dos andares:
  - `1Story`: Térrea (1 piso)
  - `1.5Fin`: 1 piso e meio (segundo nível concluído/habitável)
  - `1.5Unf`: 1 piso e meio (segundo nível inacabado)
  - `2Story`: Sobrado padrão (2 andares)
  - `2.5Fin`: 2 andares e meio (último piso concluído)
  - `2.5Unf`: 2 andares e meio (último piso inacabado)
  - `SFoyer`: Split Foyer (entrada na meia altura entre dois andares)
  - `SLvl`: Split Level (níveis intercalados com escadas curtas)

### `OverallQual` (Qualidade Geral dos Materiais e Acabamentos)
- **Tipo:** Numérico ordinal (Notas de 1 a 10)
- **Descrição:** Avalia a qualidade do material e acabamento geral da casa:
  - `10`: Excepcional / Luxo Superior (*Very Excellent*)
  - `9`: Excelente (*Excellent*)
  - `8`: Muito Boa (*Very Good*)
  - `7`: Boa (*Good*)
  - `6`: Acima da Média (*Above Average*)
  - `5`: Média (*Average*)
  - `4`: Abaixo da Média (*Below Average*)
  - `3`: Regular (*Fair*)
  - `2`: Ruim (*Poor*)
  - `1`: Muito Ruim (*Very Poor*)

### `OverallCond` (Condição Geral de Conservação)
- **Tipo:** Numérico ordinal (Notas de 1 a 10)
- **Descrição:** Avalia o estado de conservação atual da casa (mesma escala de 1 a 10, de Muito Ruim a Muito Excelente).

### `YearBuilt` (Ano de Construção Original)
- **Tipo:** Numérico discreto (Ano: YYYY)
- **Descrição:** Ano em que a casa foi originalmente construída.

### `YearRemodAdd` (Ano de Reforma ou Ampliação)
- **Tipo:** Numérico discreto (Ano: YYYY)
- **Descrição:** Ano em que foi realizada a última grande reforma ou adição. Se não houve reformas, coincide com o `YearBuilt`.

---

## 4. Telhado, Estrutura Externa e Fundação

### `RoofStyle` (Estilo do Telhado)
- **Tipo:** Categórica nominal
- **Descrição:** Formato do telhado:
  - `Flat`: Plano / Telhado reto
  - `Gable`: Duas águas (triangular convencional)
  - `Gambrel`: Estilo celeiro americano (dupla inclinação em cada lado)
  - `Hip`: Quatro águas
  - `Mansard`: Mansarda (estilo clássico francês)
  - `Shed`: Uma água (queda única)

### `RoofMatl` (Material do Telhado)
- **Tipo:** Categórica nominal
- **Descrição:** Tipo de cobertura do telhado:
  - `ClyTile`: Telha de cerâmica/argila (*Clay or Tile*)
  - `CompShg`: Telhas shingle asfálticas convencionais (*Standard Composite Shingle*)
  - `Membran`: Membrana sintética
  - `Metal`: Cobertura metálica
  - `Roll`: Manta asfáltica em rolo
  - `Tar&Grv`: Alcatrão e cascalho (*Tar & Gravel*)
  - `WdShake`: Ripas de madeira rústica (*Wood Shakes*)
  - `WdShngl`: Telhas de madeira serrada (*Wood Shingles*)

### `Exterior1st` (Revestimento Externo Principal)
- **Tipo:** Categórica nominal
- **Descrição:** Material primário de acabamento das paredes externas:
  - `AsbShng`: Telhas/Placas de amianto (*Asbestos Shingles*)
  - `AsphShn`: Telhas/Placas asfálticas
  - `BrkComm`: Tijolo comum (*Brick Common*)
  - `BrkFace`: Tijolo aparente à vista (*Brick Face*)
  - `CBlock`: Bloco de concreto (*Cinder Block*)
  - `CemntBd`: Placas cimentícias (*Cement Board*)
  - `HdBoard`: Chapas duras de fibra de madeira (*Hard Board*)
  - `ImStucc`: Imitação de estuque (*Imitation Stucco*)
  - `MetalSd`: Siding de alumínio/aço (*Metal Siding*)
  - `Other`: Outro material
  - `Plywood`: Compensado de madeira (*Plywood*)
  - `PreCast`: Painéis de concreto pré-moldado
  - `Stone`: Pedra
  - `Stucco`: Estuque / Emboço texturizado
  - `VinylSd`: Siding vinílico (muito popular nos EUA)
  - `Wd Sdng`: Tábuas de madeira (*Wood Siding*)
  - `WdShing`: Telhas pequenas de madeira decorativa (*Wood Shingles*)

### `Exterior2nd` (Revestimento Externo Secundário)
- **Tipo:** Categórica nominal
- **Descrição:** Segundo material de acabamento externo (caso a casa utilize mais de um material; usa as mesmas categorias de `Exterior1st`).

### `MasVnrType` (Tipo de Revestimento em Alvenaria)
- **Tipo:** Categórica nominal
- **Descrição:** Tipo de folheado/acabamento de alvenaria na fachada:
  - `BrkCmn`: Tijolo comum
  - `BrkFace`: Tijolo aparente
  - `CBlock`: Bloco de concreto
  - `None`: Sem revestimento de alvenaria
  - `Stone`: Pedra decorativa

### `MasVnrArea` (Área do Revestimento de Alvenaria)
- **Tipo:** Numérico contínuo
- **Unidade:** Pés quadrados (sq ft)
- **Descrição:** Área da fachada coberta com acabamento em pedra ou tijolo aparente. (0 se `MasVnrType == None`).

### `ExterQual` (Qualidade do Material Externo)
- **Tipo:** Categórica ordinal
- **Descrição:** Qualidade dos acabamentos exteriores:
  - `Ex`: Excelente (*Excellent*)
  - `Gd`: Bom (*Good*)
  - `TA`: Típico / Médio (*Average/Typical*)
  - `Fa`: Regular (*Fair*)
  - `Po`: Ruim (*Poor*)

### `ExterCond` (Condição de Conservação Externa)
- **Tipo:** Categórica ordinal
- **Descrição:** Estado atual de manutenção das paredes externas (`Ex`, `Gd`, `TA`, `Fa`, `Po`).

### `Foundation` (Tipo de Fundação / Alicerce)
- **Tipo:** Categórica nominal
- **Descrição:** Estrutura da base/fundação da residência:
  - `BrkTil`: Tijolo e telha cerâmica
  - `CBlock`: Blocos de concreto vazados (*Cinder Block*)
  - `PConc`: Concreto armado usinado moldado in loco (*Poured Concrete*)
  - `Slab`: Radier / Laje direta de concreto sobre o solo (*Slab*)
  - `Stone`: Fundação em alvenaria de pedra
  - `Wood`: Fundação em madeira tratada

---

## 5. Porão / Subsolo (Basement)

### `BsmtQual` (Altura / Pé-direito do Porão)
- **Tipo:** Categórica ordinal
- **Descrição:** Avalia a altura interna do teto do porão:
  - `Ex`: Excelente (acima de 2,54 m / 100+ polegadas)
  - `Gd`: Bom (2,28 m a 2,51 m / 90-99 polegadas)
  - `TA`: Típico / Normal (2,03 m a 2,26 m / 80-89 polegadas)
  - `Fa`: Regular (1,77 m a 2,00 m / 70-79 polegadas)
  - `Po`: Baixo / Ruim (menos de 1,77 m / <70 polegadas)
  - `NA`: Não possui porão (*No Basement*)

### `BsmtCond` (Condição Geral do Porão)
- **Tipo:** Categórica ordinal
- **Descrição:** Estado estrutural e umidade do porão:
  - `Ex`: Excelente
  - `Gd`: Bom
  - `TA`: Típico (umidade leve aceitável)
  - `Fa`: Regular (sinais de umidade ou pequenas trincas)
  - `Po`: Ruim (rachaduras graves, infiltração severa de água)
  - `NA`: Não possui porão

### `BsmtExposure` (Exposição e Entrada de Luz no Porão)
- **Tipo:** Categórica ordinal
- **Descrição:** Grau de saída ao nível do chão (*walkout*) ou janelas tipo jardim:
  - `Gd`: Boa exposição (janelas amplas e saída ao nível do jardim)
  - `Av`: Exposição média
  - `Mn`: Exposição mínima (apenas pequenas aberturas no alto)
  - `No`: Sem exposição (completamente subterrâneo)
  - `NA`: Não possui porão

### `BsmtFinType1` (Qualidade do Acabamento do Porão 1)
- **Tipo:** Categórica ordinal
- **Descrição:** Grau de acabamento da área principal do porão:
  - `GLQ`: Habitação de Alto Padrão (*Good Living Quarters*)
  - `ALQ`: Habitação Padrão Médio (*Average Living Quarters*)
  - `BLQ`: Habitação Abaixo da Média (*Below Average Living Quarters*)
  - `Rec`: Salão de jogos / Recreação típico (*Average Rec Room*)
  - `LwQ`: Acabamento de baixa qualidade (*Low Quality*)
  - `Unf`: Inacabado / Em osso (*Unfinished*)
  - `NA`: Não possui porão

### `BsmtFinSF1` (Área Acabada do Porão 1)
- **Tipo:** Numérico contínuo (sq ft)
- **Descrição:** Área em pés quadrados com o acabamento correspondente ao `BsmtFinType1`.

### `BsmtFinType2` (Qualidade do Acabamento do Porão 2)
- **Tipo:** Categórica ordinal
- **Descrição:** Grau de acabamento de uma segunda porção do porão (mesmas opções de `BsmtFinType1`).

### `BsmtFinSF2` (Área Acabada do Porão 2)
- **Tipo:** Numérico contínuo (sq ft)
- **Descrição:** Área em pés quadrados da segunda parte acabada do porão.

### `BsmtUnfSF` (Área Inacabada do Porão)
- **Tipo:** Numérico contínuo (sq ft)
- **Descrição:** Área em pés quadrados do porão que não possui acabamento (*unfinished area*).

### `TotalBsmtSF` (Área Total do Porão)
- **Tipo:** Numérico contínuo (sq ft)
- **Descrição:** Soma de todas as áreas do porão (`BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF`).

---

## 6. Sistemas Internos: Elétrica, Aquecimento e Ar

### `Heating` (Tipo de Aquecedor)
- **Tipo:** Categórica nominal
- **Descrição:** Sistema utilizado para aquecer a casa:
  - `Floor`: Aquecedor embutido no piso (*Floor Furnace*)
  - `GasA`: Fornalha a gás com ar forçado através de dutos (*Gas forced warm air*)
  - `GasW`: Radiadores a água quente ou vapor movidos a gás
  - `Grav`: Aquecedor por gravidade (sem ventilador)
  - `OthW`: Caldeira a água quente/vapor que não usa gás
  - `Wall`: Aquecedor de parede (*Wall furnace*)

### `HeatingQC` (Qualidade e Eficiência do Aquecimento)
- **Tipo:** Categórica ordinal
- **Descrição:** Avaliação da conservação e eficácia do aquecimento (`Ex`, `Gd`, `TA`, `Fa`, `Po`).

### `CentralAir` (Ar-Condicionado Central)
- **Tipo:** Categórica binária
- **Descrição:** Indica a presença de sistema de ar-condicionado central:
  - `Y`: Sim (*Yes*)
  - `N`: Não (*No*)

### `Electrical` (Padrão e Sistema Elétrico)
- **Tipo:** Categórica nominal
- **Descrição:** Tipo do quadro de força e fiação elétrica da residência:
  - `SBrkr`: Disjuntores modernos com fiação embutida padrão Romex (*Standard Circuit Breakers*)
  - `FuseA`: Caixa de fusíveis acima de 60 Amperes com cabeamento Romex (Média)
  - `FuseF`: Caixa de fusíveis de 60 Amperes com cabeamento misto (Regular)
  - `FuseP`: Caixa de fusíveis de 60 Amperes com fiação antiga tubo e botão (*knob & tube* - Ruim)
  - `Mix`: Sistema misto

---

## 7. Áreas dos Andares e Cômodos

### `1stFlrSF` (Área do 1º Andar / Térreo)
- **Tipo:** Numérico contínuo (sq ft)
- **Descrição:** Área habitável construída no primeiro pavimento.

### `2ndFlrSF` (Área do 2º Andar / Pavimento Superior)
- **Tipo:** Numérico contínuo (sq ft)
- **Descrição:** Área habitável construída no segundo pavimento.

### `LowQualFinSF` (Área com Acabamento de Baixa Qualidade)
- **Tipo:** Numérico contínuo (sq ft)
- **Descrição:** Metragem de áreas concluídas porém com padrão inferior de acabamento em todos os pisos.

### `GrLivArea` (Área Útil Habitável Acima do Solo)
- **Tipo:** Numérico contínuo (sq ft)
- **Descrição:** Área privativa total acima do nível da rua (*Ground Living Area*).  
  *Geralmente:* `GrLivArea = 1stFlrSF + 2ndFlrSF + LowQualFinSF`. É uma das variáveis mais correlacionadas com o preço de venda!

### `BsmtFullBath` (Banheiros Completos no Porão)
- **Tipo:** Numérico discreto
- **Descrição:** Quantidade de banheiros completos (com chuveiro/banheira) localizados no porão.

### `BsmtHalfBath` (Lavabos no Porão)
- **Tipo:** Numérico discreto
- **Descrição:** Quantidade de lavabos (vaso e pia, sem chuveiro) no porão.

### `FullBath` (Banheiros Completos Acima do Solo)
- **Tipo:** Numérico discreto
- **Descrição:** Quantidade de banheiros completos nos andares residenciais normais.

### `HalfBath` (Lavabos Acima do Solo)
- **Tipo:** Numérico discreto
- **Descrição:** Quantidade de lavabos nos andares superiores.

### `BedroomAbvGr` (Quartos Acima do Solo)
- **Tipo:** Numérico discreto
- **Descrição:** Número de quartos localizados nos andares acima da terra (*não inclui dormitórios improvisados no porão*).  
  *(Nota: no arquivo `data_description.txt` o nome aparece resumido como `Bedroom`).*

### `KitchenAbvGr` (Cozinhas Acima do Solo)
- **Tipo:** Numérico discreto
- **Descrição:** Quantidade de cozinhas nos andares principais (*geralmente 1, exceto em casas bifamiliares*).  
  *(Nota: no arquivo `data_description.txt` aparece resumido como `Kitchen`).*

### `KitchenQual` (Qualidade da Cozinha)
- **Tipo:** Categórica ordinal
- **Descrição:** Padrão de acabamento, bancadas e marcenaria da cozinha:
  - `Ex`: Excelente
  - `Gd`: Boa
  - `TA`: Típica / Média
  - `Fa`: Regular
  - `Po`: Ruim

### `TotRmsAbvGrd` (Total de Cômodos Acima do Solo)
- **Tipo:** Numérico discreto
- **Descrição:** Contagem total de cômodos da casa (*Total Rooms Above Ground*), **excluindo banheiros**.

### `Functional` (Funcionalidade e Danos Estruturais da Casa)
- **Tipo:** Categórica ordinal
- **Descrição:** Condição de habitabilidade e reduções de valor por danos:
  - `Typ`: Funcionalidade típica / sem deduções (*Typical Functionality*)
  - `Min1`: Deduções mínimas 1
  - `Min2`: Deduções mínimas 2
  - `Mod`: Deduções moderadas
  - `Maj1`: Deduções graves 1
  - `Maj2`: Deduções graves 2
  - `Sev`: Severamente danificada
  - `Sal`: Inabitável / Apenas demolição e salvamento de materiais (*Salvage only*)

---

## 8. Lareiras (Fireplaces)

### `Fireplaces` (Número de Lareiras)
- **Tipo:** Numérico discreto
- **Descrição:** Quantidade total de lareiras existentes na casa.

### `FireplaceQu` (Qualidade da Lareira Principal)
- **Tipo:** Categórica ordinal
- **Descrição:** Padrão construtivo e conservação da lareira:
  - `Ex`: Excelente (Lareira clássica em alvenaria nobre)
  - `Gd`: Boa (Lareira em alvenaria no piso principal)
  - `TA`: Média (Lareira pré-fabricada na sala ou de alvenaria no porão)
  - `Fa`: Regular (Lareira pré-fabricada no porão)
  - `Po`: Ruim (Fogão a lenha tipo "Ben Franklin")
  - `NA`: Não possui lareira (*No Fireplace*)

---

## 9. Garagem (Garage)

### `GarageType` (Localização e Tipo de Garagem)
- **Tipo:** Categórica nominal
- **Descrição:** Posicionamento da garagem em relação à casa:
  - `2Types`: Mais de um tipo de garagem
  - `Attchd`: Anexa / Ligada diretamente à casa (*Attached*)
  - `Basment`: Garagem no porão/subsolo
  - `BuiltIn`: Integrada (garagem dentro do volume da casa, comum ter cômodos acima dela)
  - `CarPort`: Abrigo coberto sem paredes laterais (*Carport*)
  - `Detchd`: Separada / Construção avulsa no quintal (*Detached*)
  - `NA`: Não possui garagem (*No Garage*)

### `GarageYrBlt` (Ano de Construção da Garagem)
- **Tipo:** Numérico discreto (Ano: YYYY)
- **Descrição:** Ano em que a garagem foi erguida.

### `GarageFinish` (Nível de Acabamento Interno da Garagem)
- **Tipo:** Categórica ordinal
- **Descrição:** Estado das paredes e teto internos da garagem:
  - `Fin`: Totalmente acabada com gesso/pintura (*Finished*)
  - `RFn`: Acabamento bruto / Emassada sem pintura fina (*Rough Finished*)
  - `Unf`: Inacabada / Estrutura aparente (*Unfinished*)
  - `NA`: Não possui garagem

### `GarageCars` (Capacidade de Veículos na Garagem)
- **Tipo:** Numérico discreto
- **Descrição:** Capacidade da garagem em número de carros (ex.: 1 vaga, 2 vagas, 3 vagas).

### `GarageArea` (Área da Garagem)
- **Tipo:** Numérico contínuo (sq ft)
- **Descrição:** Tamanho total da área da garagem em pés quadrados.

### `GarageQual` (Qualidade da Garagem)
- **Tipo:** Categórica ordinal
- **Descrição:** Qualidade dos materiais e construção da garagem (`Ex`, `Gd`, `TA`, `Fa`, `Po`, `NA`).

### `GarageCond` (Condição de Conservação da Garagem)
- **Tipo:** Categórica ordinal
- **Descrição:** Estado atual de manutenção da garagem (`Ex`, `Gd`, `TA`, `Fa`, `Po`, `NA`).

### `PavedDrive` (Pavimentação da Entrada da Garagem)
- **Tipo:** Categórica ordinal
- **Descrição:** Tipo de pavimento da rampa/acesso da garagem:
  - `Y`: Pavimentada / Asfaltada / Concretada (*Paved*)
  - `P`: Parcialmente pavimentada (*Partial Pavement*)
  - `N`: Terra batida ou Cascalho (*Dirt/Gravel*)

---

## 10. Decks, Varandas e Piscina

### `WoodDeckSF` (Área do Deck de Madeira)
- **Tipo:** Numérico contínuo (sq ft)
- **Descrição:** Metragem da área de deck de madeira ao ar livre.

### `OpenPorchSF` (Área da Varanda Aberta)
- **Tipo:** Numérico contínuo (sq ft)
- **Descrição:** Área da varanda coberta frontal ou lateral sem paredes.

### `EnclosedPorch` (Área da Varanda Fechada)
- **Tipo:** Numérico contínuo (sq ft)
- **Descrição:** Metragem da varanda fechada com paredes e janelas completas.

### `3SsnPorch` (Área da Varanda de 3 Estações)
- **Tipo:** Numérico contínuo (sq ft)
- **Descrição:** Varanda envidraçada para primavera, verão e outono (*Three-season porch*), mas sem isolamento térmico integral para o inverno rigoroso.

### `ScreenPorch` (Área da Varanda com Telas Mosquiteiras)
- **Tipo:** Numérico contínuo (sq ft)
- **Descrição:** Varanda protegida contra insetos com telas (*Screen porch*).

### `PoolArea` (Área da Piscina)
- **Tipo:** Numérico contínuo (sq ft)
- **Descrição:** Tamanho da superfície d'água da piscina em pés quadrados (0 se não tiver piscina).

### `PoolQC` (Qualidade da Piscina)
- **Tipo:** Categórica ordinal
- **Descrição:** Avaliação da piscina instalada:
  - `Ex`: Excelente
  - `Gd`: Boa
  - `TA`: Média
  - `Fa`: Regular
  - `NA`: Não possui piscina (*No Pool*)

### `Fence` (Qualidade da Cerca / Muro)
- **Tipo:** Categórica ordinal
- **Descrição:** Tipo e padrão de privacidade da cerca do lote:
  - `GdPrv`: Excelente Privacidade (cerca alta e opaca)
  - `MnPrv`: Privacidade Média/Mínima (cerca de madeira ou grade vazada)
  - `GdWo`: Madeira de boa qualidade (*Good Wood*)
  - `MnWw`: Arame ou madeira simples (*Minimum Wood/Wire*)
  - `NA`: Não possui cerca (*No Fence*)

---

## 11. Recursos Diversos e Detalhes da Venda

### `MiscFeature` (Recurso Adicional Extraordinário)
- **Tipo:** Categórica nominal
- **Descrição:** Benfeitorias ou características raras que não se enquadram em outros campos:
  - `Elev`: Elevador interno
  - `Gar2`: Segunda garagem independente
  - `Othr`: Outro recurso
  - `Shed`: Galpão / Depósito de ferramentas no jardim (> 9 m² / 100 sq ft)
  - `TenC`: Quadra de tênis particular
  - `NA`: Nenhum recurso adicional (*None*)

### `MiscVal` (Valor Financeiro do Recurso Adicional)
- **Tipo:** Numérico contínuo
- **Unidade:** Dólares ($)
- **Descrição:** Valor estimado do recurso apontado em `MiscFeature`.

### `MoSold` (Mês da Venda)
- **Tipo:** Numérico discreto (1 a 12)
- **Descrição:** Mês em que a venda foi fechada (ex.: 1 = Janeiro, 12 = Dezembro). Útil para capturar sazonalidade imobiliária.

### `YrSold` (Ano da Venda)
- **Tipo:** Numérico discreto (2006 a 2010)
- **Descrição:** Ano em que o imóvel foi vendido.

### `SaleType` (Tipo de Transação / Contrato de Venda)
- **Tipo:** Categórica nominal
- **Descrição:** Modalidade jurídica e financeira da escritura:
  - `WD`: Escritura de Garantia Convencional (*Warranty Deed - Conventional*)
  - `CWD`: Escritura com pagamento total à vista em dinheiro (*Cash*)
  - `VWD`: Escritura garantida por financiamento militar/veteranos (*VA Loan*)
  - `New`: Imóvel recém-construído vendido diretamente pela construtora
  - `COD`: Venda por decisão judicial ou inventário (*Court Officer Deed / Estate*)
  - `Con`: Contrato com 15% de entrada e parcelamento padrão
  - `ConLw`: Contrato com entrada baixa e juros reduzidos
  - `ConLI`: Contrato com juros baixos
  - `ConLD`: Contrato com entrada baixa
  - `Oth`: Outro tipo de contrato

### `SaleCondition` (Condição da Venda)
- **Tipo:** Categórica nominal
- **Descrição:** Situação de mercado sob a qual o imóvel foi comercializado:
  - `Normal`: Venda normal de mercado entre particulares
  - `Abnorml`: Venda fora dos padrões normais (leilão judicial por inadimplência, penhora, repasse emergencial)
  - `AdjLand`: Compra de terreno contíguo/vizinho
  - `Alloca`: Alocação (dois imóveis conjugados com escrituras separadas, ex.: apartamento com vaga escriturada avulsa)
  - `Family`: Venda realizada entre membros da mesma família (frequentemente com desconto)
  - `Partial`: Imóvel ainda em fase de conclusão de obra quando avaliado (comum em lançamentos)

---

## Dicionário Prático de Siglas Recorrentes

Muitas colunas de qualidade e condição utilizam as mesmas siglas ordinais:

| Sigla | Inglês | Tradução em Português | Pontuação Sugerida (Ordinal) |
| :---: | :--- | :--- | :---: |
| **Ex** | Excellent | Excelente / Padrão Luxo | 5 |
| **Gd** | Good | Bom / Acima da Média | 4 |
| **TA** | Typical / Average | Típico / Médio / Padrão Normal | 3 |
| **Fa** | Fair | Regular / Aceitável | 2 |
| **Po** | Poor | Ruim / Precário | 1 |
| **NA** | None / Not Present | Não possui o item | 0 |
