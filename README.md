# generative-manufacturing

[Initial commit]

## Requirements
- Python 3.10+ (I use 3.13) with: ```torch```, ```torchvision```, ```numpy```, ```scipy```, ```pillow```, ```tqdm```, ```matplotlib```
- download all code into the same structure as this repo (sample-data is optional)

## How to use
1. run **datagen MATLAB scripts** (note: I've included 50 samples from each class in the sample-data folder) - they will generate jpg + mat + fig files for each model labeled with numbers 1-1000 and 1000-2000 (warning: it will dump the files into whatever folder the script is in, and the fig popups make it difficult to use your computer for other tasks while it is running)
3. from **data-helpers** run ```python data-helpers/mk_manifest.py --root your-data-folder --output your-data-folder/all.json``` to generate the structure data JSON, AND ```python data-helpers/datasplit.py your-data-folder/all.json``` to generate ```your-data-folder/train.json``` and ```your-data-folder/validation.json```
4. from **gmdl package** and **gmdl.py** run ```python gmdl.py --data_root your-data-folder --encoder pointnet --contrastive_epochs 20 --phase2_epochs 30``` to train encoders and then ```python gmdl.py --train-decoders --weights train_log/default/model/latest.pt --data-root your-data-folder``` to train decoders
5. prediction of manufacturing process from image or mesh: ```python gmdl.py --predict --mesh test.mat --weights train_log/default/model/latest.pt``` ```python gmdl.py --predict --image test.jpg --weights train_log/default/model/latest.pt```
6. fit score: ```python gmdl.py --analyze --mesh test.mat --weights train_log/default/model/latest.pt```
7. design change recommendations and other capabilities: in progress 
