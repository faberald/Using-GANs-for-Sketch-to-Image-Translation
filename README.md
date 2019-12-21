# Using-GANs-for-Sketch-to-Image-Translation

This is a Pytorch implementation of Bicyclegan models on the edges2shoes dataset.

1. Bicyclegan
2. Bicyclegan with VGG Unet and encoder

## Dataset

The model is training using the edges2shoes dataset
To get the data

```
cd dataset
sh download_pix2pix_dataset.sh edges2shoes
```

## Training

```
cd models/bicyclegan
python3 bicyclegan.py --n_epochs 20 --dataset_name edges2shoes --data_size 40000
```

```
cd models/hpcyclegan-vgg
python3 hpcyclegan.py --n_epochs 20 --dataset_name edges2shoes --data_size 40000
```

## Trained Models
Bicyclegan 20 epochs, 35 epochs
