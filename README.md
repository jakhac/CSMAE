# Exploring Masked Autoencoders for Sensor-Agnostic Image Retrieval in Remote Sensing

![Alt text](csmae.png?raw=true "Model: Cross-Sensor Masked Autoencoders")

This repository contains the code of the paper [Exploring Masked Autoencoders for Sensor-Agnostic Image Retrieval in Remote Sensing](https://arxiv.org/abs/2401.07782). This work has been done at the [Remote Sensing Image Analysis group](https://rsim.berlin/) by [Jakob Hackstein](https://rsim.berlin/team/members/jakob-hackstein), [Gencer Sumbul](https://people.epfl.ch/gencer.sumbul?lang=en), [Kai Norman Clasen](https://rsim.berlin/team/members/kai-norman-clasen) and [Begüm Demir](https://rsim.berlin/team/members/begum-demir).

If you use this code, please cite our paper given below:

> J. Hackstein, G. Sumbul, K. N. Clasen, B. Demir, "[Exploring Masked Autoencoders for Sensor-Agnostic Image Retrieval in Remote Sensing](https://arxiv.org/abs/2401.07782)", IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2024.3517150, 2024

```bibtex
@article{hackstein2024exploring,
    author={Hackstein, Jakob and Sumbul, Gencer and Clasen, Kai Norman and Demir, Begüm},
    journal={IEEE Transactions on Geoscience and Remote Sensing}, 
    title={Exploring Masked Autoencoders for Sensor-Agnostic Image Retrieval in Remote Sensing},
    year={2024},
    doi={10.1109/TGRS.2024.3517150}
}
```


## Training CSMAE models

1. First, set up a python (conda) environment based on the `environment.yaml` file. 

2. Training and model parameters can be adjusted via yaml-files. The paper introduces four different CSMAE variants, termed _CECD_, _CESD_, _SECD_ and _SESD_, and each variant has a pre-defined `csmae_<variant>.yaml` file already. You can also modify them according to your needs and configure new CSMAE models. To do so, check the explanations for relevant parameters in existing `csmae.yaml` files.

3. Independent on your yaml-file, two entries have to be completed:

    - The training progress is tracked on [Weights & Biases](https://wandb.ai/). To this end, the `wandb.entity` and `wandb.project` fields have to be entered in the `wandb` attribute.

    - For training, [BigEarthNet-MM](https://bigearth.net/) is required. The dataloader requires the LMDB format which is explained [here](http://docs.kai-tub.tech/bigearthnet_encoder/intro.html). Finally, the `data.root_dir` should point to the directory containing the LMDB file and `data.split_dir` should point to the directory containing CSV-file splits of the dataset.

4. Then, pre-training can be started by running `train.py` with two flags required by Hydra:
    ```bash
    python train.py --config-path ./ --config-name csmae_<variant>.yaml
    ```

5. Checkpoints of trained models with config files are stored under `./trained_models`.

## Evaluation of CSMAE models

To compute image retrieval results, run the `retrieval.py` script. The two required flags are
- name of the folder, which contains the model checkpoint to be evaluated
- the GPU device number used for inference.

For instance, a model stored under `./trained_models/abcd1234/` can be evaluated with

```bash
python retrieval.py abcd1234 0
```

## Model Weights

We share model weights for the best-performing CSMAE variants [here](https://tubcloud.tu-berlin.de/s/E4RcbGjzSrjBq7R). To load weights into a backbone, see the following code snippet.

```python
import torch
from src.csmae_backbone import CSMAEBackbone
from omegaconf import OmegaConf

csmae_variant = 'sesd'
cfg = OmegaConf.load(f'./checkpoints/{csmae_variant}/cfg.yaml')
model = CSMAEBackbone(**cfg.kwargs)

state_dict = torch.load(f'./checkpoints/{csmae_variant}/weights.ckpt', map_location="cpu")['state_dict']
for k in list(state_dict.keys()):
    if "backbone" in k:
        state_dict[k.replace("backbone.", "")] = state_dict[k]
    del state_dict[k]

model.load_state_dict(state_dict, strict=True)
```

## Acknowledgement

This work is supported by the European Research Council
(ERC) through the ERC-2017-STG BigEarth Project under
Grant 759764 and by the European Space Agency (ESA)
through the Demonstrator Precursor Digital Assistant Interface
For Digital Twin Earth (DA4DTE) Project and by the German Ministry for
Economic Affairs and Climate Action through the AI-Cube
Project under Grant 50EE2012B.

The code for pre-training is inspired by [solo-learn](https://github.com/vturrisi/solo-learn) and the code for dataloading partly stems from [ConfigILM](https://github.com/lhackel-tub/ConfigILM).


## Authors
**Jakob Hackstein**
https://rsim.berlin/team/members/jakob-hackstein

**Gencer Sumbul**
https://people.epfl.ch/gencer.sumbul?lang=en

**Kai Norman Clasen**
https://rsim.berlin/team/members/kai-norman-clasen 

**Begüm Demir**
https://rsim.berlin/team/members/begum-demir

For questions, requests and concerns, please contact [Jakob Hackstein via mail](mailto:hackstein@tu-berlin.de)

## License
The code in this repository is licensed under the **MIT License**:
```
MIT License

Copyright (c) 2024 Jakob Hackstein

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

