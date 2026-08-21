# Generative Manufacturing: using deep learning to classify, score, and improve 3D part designs

## Motivation
Can a neural network learn the intuition of a Mechanical Engineer? Upon inspection, the characteristics of a 3D design make it inherently more suitable for certain manufacturing processes. One way to perform this classifcation is by mapping each set of characteristics to a process, which is deterministic but tedious and may not capture the nuances of unique designs. Another way, which is more aligned with how human engineers learn, is simply viewing many examples of designs belonging to each class. Instead of manually specifying the features that differentiate designs for each manufacturing process, this pipeline learns which features are important on its own. These learned features can then be mapped back to 
manually specified characteristics in order to describe improvements in a human-understandable, actionable way.

## Technical Summary
* REQUIRES: Python 3.10+ with ```torch```, ```torchvision```, ```numpy```, ```scipy```, ```pillow```, ```tqdm```, ```matplotlib```
* Data format: ```your_data_folder/InjectionMolding/data1.jpg```, ```your_data_folder/InjectionMolding/data1.mat```
This pipeline uses a PointNet encoder for meshes and a CNN encoder for images to create embeddings for classification. Training incorporates InfoNCE contrastive loss to align the image and mesh embeddings. Then, the embeddings are used by a softmax ranking system to predict the best manufacturing process, and a KNN fit scorer to quantify how close the design is to others in a specified class. Finally, a linear decoder translates the Euclidean distance between the design and a specified class into physical changes in the design that can be implemented to improve its fit score, corresponding to a more manufacturable design.

## Usage 
- Data generation: run ```datagen/``` MATLAB scripts
- Data pre-processing: run ```python data-helpers/mk_manifest.py --root your-data-folder --output your-data-folder/all.json``` to generate the structure data JSON, AND ```python data-helpers/datasplit.py your-data-folder/all.json``` to generate ```your-data-folder/train.json``` and ```your-data-folder/validation.json```
- Train encoders: run ```python gmdl.py --data_root your-data-folder --encoder pointnet --contrastive_epochs 20 --phase2_epochs 30``` (or adjust number of epochs as desired)
- Train decoders: ```python gmdl.py --train-decoders --weights train_log/default/model/latest.pt --data-root your-data-folder```
- Prediction: ```python gmdl.py --predict --mesh test.mat --weights train_log/default/model/latest.pt``` or ```python gmdl.py --predict --image test.jpg --weights train_log/default/model/latest.pt```
- Fit scoring: ```python gmdl.py --analyze --mesh test.mat --weights train_log/default/model/latest.pt```
- Design improvements: implementation in progress

## References
Deep learning-based classification is inspired by [DeepMS](https://github.com/jaimemaqueda/DeepMS/tree/main)

MATLAB data generation scripts are originally by [HKS-CNN](https://github.com/ZhichaoWang970201/HKS-CNN/tree/main), with some modifications.
