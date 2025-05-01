DDG prediction
**Structure-aware prediction of Mutational effects in SKEMPI 2.0 data base using deep learning

#Changelog

26/4 - Initial commit, prediction only from facebook ESM embeddings
27/4 - Added struture awareness
1/5 - Added ability to save model, minor fixes.



## Dependencies

The default PyTorch version is 1.8.1 and cudatoolkit version is 11.3. They can be changed in `environment.yml`.



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

