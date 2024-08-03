# ChromCoder

> This is the official implementation of the paper
> "ChromCoder: Towards Novel Chromosome Representation
> for Chromosomal Structural Abnormality Detection"

### Introduction

![img.png](asset/overview.png)

### Installation

First, create an environment with python=3.8. 
If using conda, run the following command.

``conda create -n ChromCoder python=3.8``

Then switch to the ChromCoder environment, 
and run the following two commands.

``pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118``

``pip install -r requirements.txt``.

### Usage

API for abnormality detection is encapsulated in
[anomaly_detection.py](anomaly_detection.py).
And here is an example.

```python
import os

os.environ['CUDA_VISIBLE_DEVICES'] = 'X'  

from anomaly_detection import PatchDetector

detector = PatchDetector()
detector.build(use_gpu=True)

result = detector.detect(img1, 
                         img2, 
                         n,  
                         ctype='G',  
                         use_gpu=True)
```

Before applying, you need to create an 
instance of the PatchDetector class. Then, 
call the build method to load the trained 
parameters onto the GPU, and finally, call 
the run method to obtain the detection results.


### Pretraining PatchCoder & Preparing Ab-detector

To perform anomaly detection without an 
existing model in this project, follow 
the four steps below to obtain the model 
and threshold files. Once you have the model 
and threshold files, you can proceed with 
anomaly detection.

1.Run [train_CPC.py](train_CPC.py) to 
pre-train PatchCoder, generating model 
files in the TrainedModels/stl10 directory. 
(The dataset used for pre-training needs to 
be processed with 
[ChromPatcher](cms_dataset_helper/make_cms_patched_dataset.py).
)

2.(Optional) Run 
[train_classifier.py](train_classifier.py) to 
test the linear probing performance of the 
PatchCoder's backbone, which evaluates the 
effectiveness of feature extraction by the 
backbone.

3.Call the 
``_save_similarities_between_normal_patch_X ``
function in 
[anomaly_detection.py](anomaly_detection.py) 
to obtain statistical data on the similarities 
between all pairs of chromosomes in the dataset 
(X=G,A,L), generating statistical data 
files in the root directory.

Note that chromosomes are divided into G, A, and L
according to their length. Therefore, three 
different threshold matrix are needed for G, A,
and L individually.

4.Call the ``_save_patch_thresholds_X`` 
function in [anomaly_detection.py](anomaly_detection.py) 
to obtain the threshold T, generating threshold
files in the root directory (X=G,A,L).

### Namings & Tricks

Here we wish to clarify the connection between
the naming of variables and files in our code 
and the concepts discussed in the paper. 
First, ChromPatcher corresponds to the 
preprocessing of images and is encapsulated 
in the CMS_Trans class. 
Second, PatchCoder is encapsulated in the 
CPC class. 
Third, Ab-detector is encapsulated in 
the PatchDetector class.

Additionally, there is a very useful trick used
in this project. Chromosome images are very
different from the images in the real world.
Therefore, we set the **Normalization Parameter**
according to the feature of chromosome images.
