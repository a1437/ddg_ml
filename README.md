DDG prediction
**Structure-aware prediction of mutational effects in SKEMPI 2.0 database using deep learning

## Changelog
1/5 - Added ability to save model, minor fixes.  
27/4 - Added struture awareness  
26/4 - Initial commit, prediction only from facebook ESM embeddings  



## Dependencies
Running on python version 3.10. Notable  dependancies for running the model training script:  
```
conda install -c conda-forge boost=1.73  
conda install -c salilab dssp
```




## Dataset

| Dataset   | Download Script                                    |
| --------- | -------------------------------------------------- |
| [SKEMPI v2](https://life.bsc.es/pid/skempi2) | [`data/get_skempi_v2.sh`](./data/get_skempi_v2.sh) |



## Usage

### Codebook Pre-training and DDG Prediction

```
python -B train.py
```

The customized hyperparameters  are available in `./configs/param_config.json`.

