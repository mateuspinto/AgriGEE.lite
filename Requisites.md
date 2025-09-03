# Funcionais

## Lista de satélites:
- Todos os Landsats;
- Sentinel 1, 2 e 3;
- Modis Terra/Aqua (unidos);
- USDA Cropland;
- Canada Cropland;
- Mapbiomas Cropland;
- Europe Cropland;

## Bandas
- Bandas devem ser padronizadas com um nome especifico para cada uma delas, exemplo "red", "nir";
- Deve ser possível baixar todos os indices agrícolas (NDVI, EVI2, VV/VH ...)

## Visualização de dados

- Série temporal de indice agrícola escolhido;
- Falsa-cor de bandas escolhidas (padrão nir-swir1-red ou nir-red-red para satélites sem swir1);
- Série temporal do talhão em falsa cor ou RGB.

# Não funcionais

## Otimização de download
- Dividir geodataframe por grid + quantidade de samples por - chunk.
- QuadTree é overkill.

## Download de dados

### SITS
- Download de SITS mediana/média individual.
- Download de SITS mediana/média (computeFeatures) multithread;
- Download de SITS mediana/média (computeFeatures) assíncrono;
- Download de SITS mediana/média (computeFeatures) task;

### Imagem
- Download de imagem com pixel do tamanho definido pelo satélite individual (Tiff, Numpy).
- Download de imagem com pixel do tamanho definido pelo satélite multithread (Tiff, Numpy).
- Download de imagem com pixel do tamanho definido pelo satélite assíncrono (Tiff, Numpy).

## Detalhes de salvar por task
- Escolha de bucket/save-folder;
- Verificação se arquivo já existe antes de o recriar se for no bucket;

## Formatos de saída:
- Parquet ou NPZ. (sempre parquet?)
