# AgriGEE.lite

![mascote](https://github.com/user-attachments/assets/908400d6-c68f-4c26-98ae-0887cdb34d11)


AgriGEE.lite is an Earth Engine wrapper that allows easy download of **Analysis Ready Multimodal Data (ARD)**, focused on downloading **time series** of **agricultural** and native vegetation data.

For example, to download and view a time series of cloud-free Sentinel 2 imagery cropped to a specific field and date range, only a few lines of code are required. Here’s an example:

```python
import agrigee_lite as agl
import ee

ee.Initialize()

gdf = gpd.read_parquet("data/sample.parquet")
row = gdf.iloc[0]

satellite = agl.sat.Sentinel2(bands=["red", "green", "blue"])
agl.vis.images(row.geometry, row.start_date, row.end_date, satellite)
```

Through this example, it is already possible to understand the basic functioning of the lib. The **entire lib was designed to be used in conjunction with [GeoPandas](https://geopandas.org/en/stable/)**, and a basic knowledge of it is necessary.

![{Sentinel 2 RGB Agricultural Area from Mato Grosso, Brazil}](https://github.com/user-attachments/assets/f01c1b9b-03c2-43ff-9e00-3b687048659c)

You can also download aggregations, such as spatial median aggregations of indices. Here's an example with median from multiple satellites:

![{Multiple satellites EVI2 time series}](https://github.com/user-attachments/assets/dccd7d52-6047-4734-8d83-e6ea4de35808)

For more examples, see the examples folder.

Finally, the library features **multithreaded downloading**, which allows downloading on average **16-22 time series per second** (assuming 1-year series, cloud-free for Sentinel 2 Surface Reflectance).

The lib has 3 types of elements, which are divided into modules:

- agl.sat = Data sources, usually coming from satellites/sensors. When defining a sensor, it is possible to choose which bands you want to view/download, or whether you want to use atmospheric corrections or not. By default, all bands are downloaded, and all atmospheric corrections and harmonizations are used.

- agl.vis = Module that allows you to view data, either through time series or images.

- agl.get = Module that allows you to download data on a large scale.

## Available data sources (satellites, sensors, models and so on)

| **Name** | **Bands** | **Start Date** | **End Date** | **Regionality** | **Pixel Size** | **Revisit Time** | **Variations** |
|---|---|---|---|---|---|---|---|
| Sentinel 2 | Blue, Green, Red, Re1, Re2, Re3, Nir, Re4, Swir1, Swir2 | 2016-01-01 |  | Worldwide | 10 -- 60 | 5 days (with clouds), 8 days (wo) | Surface Reflectance, Top of Atmosphere |
| Landsat 5 | Blue, Green, Red, Nir, Swir1, Swir2 |  |  | Worldwide | 15 -- 30 | 16 days | Surface Reflectance, Top of Atmosphere; Tier 1 and Tier 2; |
| Landsat 7 | Blue, Green, Red, Nir, Swir1, Swir2 |  |  | Worldwide* | 15 -- 30 | 16 days | Surface Reflectance, Top of Atmosphere; Tier 1 and Tier 2; |
| Landsat 8 | Blue, Green, Red, Nir, Swir1, Swir2 |  |  | Worldwide | 15 -- 30 | 16 days | Surface Reflectance, Top of Atmosphere; Tier 1 and Tier 2; |
| Landsat 9 | Blue, Green, Red, Nir, Swir1, Swir2 |  |  | Worldwide | 15 -- 30 | 16 days | Surface Reflectance, Top of Atmosphere; Tier 1 and Tier 2; |
| MODIS Terra/Acqua | Red, Nir |  |  | Worldwide | 15 -- 30 | daily (with clouds) |  |
| Sentinel 1 | VV, VH |  |  | Worldwide* | 10** | 5 days**** | GRD, ARD*** |
| JAXOS PalSAR 1/2 | HH, HV |  |  | Worldwide | 25** | 15 days | GRD |
| Mapbiomas Brazil | 37 Land Usage Land Cover Classes | 1985-01-01 | 2023-12-31 | Brazil | 30 | 1 year |  |

## Motivations: what an average data scientist - me - thought when I started learning GEE

My journey with GEE started two and a half years ago. GEE is excellent, it allows you to use several different satellite data, but it is very complex to code. In addition to using A LOT of boilerplate code, the errors are extremely confusing, since the tool is executed server-side, most likely with a pure functional language (like Haskell). During my master's degree, I struggled a lot writing codes for GEE. Furthermore, harmonizing all satellites at the same time is difficult. Typically, each satellite has a different range of values ​​and cloud masks. Tired of having to rewrite similar codes, I decided to create a lib with a simple goal: using satellite data should be as simple as reading a CSV in Pandas, and you shouldn't need to be a Remote Sensing expert to achieve it.


## Objectives and target audience

The main objective of the lib is to be a simple, direct and high-performance way to download satellite data, both for academic and commercial use. Did you like it? Give it a star. Want to contribute? Open your Pull Request, I will be happy to include it.

## Questions possibly asked

### But isn't it just a case of using STAC? Why pay Google?

This is a tempting proposition, and it actually makes sense for large scale projects. However, processing satellite data locally can easily escalate to hell, especially for countries with huge agricultural areas like Australia, the United States or Brazil. So, you have to do the math to figure out whether it is cheaper or not to use GEE than to have your own processing infrastructure than STAC. However, GEE is completely free for students and non-commercial projects.

### "Hello, I am a Remote Sensing expert, and I believe that the term satellite is not the most appropriate, radars do not have bands and.... "

Yes, I was told that. The use of the term "satellite" instead of sensor, data source or something else is intended to simplify things, even though it is not the most correct term. Note that even Mapbiomas, a WONDERFUL project that is made using models and AMAZING PEOPLE (❤️) is called a satellite, and is treated exactly the same as Sentinel 2 or any Landsat. The same goes for the idea of ​​"bands" in a radar like Sentinel 1. The more standardized it is, the easier it is to keep the library code working. However, note that your help to the project is VERY WELCOME.

### The library mascot is cute! Did you make it?

Absolutely not, I'm terrible at drawings and anything. I made it using GPT4, and all the rights belong to God knows who. The base art is from the Odd-Eyes Venom Dragon card from the Yu-Gi-Oh card game. The inspiration has nothing to do with venom, but rather because it is a plant dragon (agriculture), it is a fusion card (multimodal data) and it has odd-eyes (like satellites, seeing the world through different eyes). If you're a cartoonist and want to design a new mascot, I'd be more than happy to make it official.

## Known Bugs

- QuadTree clustering functions produce absurd results when there are very uneven geographic density distributions, for example, thousands of points in one country and a few dozen in another. Some prior geospatial partitioning is recommended.

## TO-DO
- [x] Add Sentinel 2 as a satellite;
- [x] Add Landsats 5, 7, 8, 9 as a satellite;
- [x] Add Sentinel 1 GRD as a satellite;
- [x] Add Mapbiomas Brazil as a satellite (data source);
- [x] Add MODIS Terra/Acqua;
- [x] Add Satellite Image Time Series Aggregations online download;
- [x] Add Satellite Image Time Series Aggregations task download;
- [x] Add Images online download/visualization with matplotlib;
- [x] Add single/multiple SITS visualization
- [x] Add smart_open[gcs] for autorecovery SITS from GCS;
- [x] Add ALOS-2 PALSAR-2 radar;
- [ ] Add Images online visualization with plotly;
- [ ] Make cloud mask removable;
- [ ] Add all other Mapbiomas;
- [ ] Add Sentinel 1 ARD;
- [ ] Add Sentinel 3;
- [ ] Add jurassic Landsats (1-4);
- [ ] Add Landsat Pansharpening for image download;
