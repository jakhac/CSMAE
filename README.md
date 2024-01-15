# Exploring Masked Autoencoders for Sensor-Agnostic Image Retrieval in Remote Sensing

![Alt text](CSMAE.png?raw=true "Model: Cross-Sensor Masked Autoencoders")

This repository contains code of the paper [Exploring Masked Autoencoders for Sensor-Agnostic Image Retrieval in Remote Sensing](TODO). This work has been done at the [Remote Sensing Image Analysis group](https://rsim.berlin/) by [Jakob Hackstein](https://rsim.berlin/team/members/jakob-hackstein), [Gencer Sumbul](https://people.epfl.ch/gencer.sumbul?lang=en), [Kai Norman Clasen](https://rsim.berlin/team/members/kai-norman-clasen) and [Begüm Demir](https://rsim.berlin/team/members/begum-demir).

If you use this code, please cite our paper given below:

TODO - which bibtex file?
<!-- > L. Hackel and K. N. Clasen and M. Ravanbakhsh and B. Demіr, "LiT-4-RSVQA: Lightweight Transformer-based Visual Question Answering in Remote Sensing", IEEE International Geoscience and Remote Sensing Symposium, Pasadena, California, 2023.

```bibtex
@INPROCEEDINGS{10281674,
    author={Hackel, Leonard and Clasen, Kai Norman and Ravanbakhsh, Mahdyar and Demir, Begüm},
    booktitle={IGARSS 2023 - 2023 IEEE International Geoscience and Remote Sensing Symposium}, 
    title={LIT-4-RSVQA: Lightweight Transformer-Based Visual Question Answering in Remote Sensing}, 
    year={2023},
    volume={},
    number={},
    pages={2231-2234},
    doi={10.1109/IGARSS52108.2023.10281674}
}
``` -->


## Training CSMAE models

First, set up a python (conda) environment according the `environment.yaml` file. Modify the `csmae.yaml` according to your needs to configure different CSMAE models. You can use the current configuration which corresponds to our best performing model. However, a few entries specific to your system are required:

- The training progess is tracked on [Weights & Biases](https://wandb.ai/). To this end, the entity and project has to be entered in the `wandb` attribute.

- For training, [BigEarthNet-MM](https://bigearth.net/) is required. The dataloader requires a LMDB format which is explained [here](http://docs.kai-tub.tech/bigearthnet_encoder/intro.html). Finally, the `data.root_dir` should point to the directiory containing the LMDB file and `data.split_dir` should point the directory with csv-file splits of the dataset.


Then, pre-training can be started by running `train.py` with two flags required by hydra:

```bash
python train.py --config-path ./ --config-name csmae.yaml
```

Model weights of trained models with config files are stored under `trained_models`.

## Acknowledgement

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

Copyright (c) 2023 Jakob Hackstein, Gencer Sumbul

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

