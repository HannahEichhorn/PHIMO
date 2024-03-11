# PHIMO: Physics-Informed Deep Learning for Motion-Corrected Reconstruction of Quantitative Brain MRI

**Hannah Eichhorn**, Veronika Spieker, Kerstin Hammernik, Elisa Saks, Kilian Weiss, Christine Preibisch, Julia A. Schnabel

- Submitted to [MICCAI 2024](https://www.ismrm.org/24m/) | [preprint]()

- Accepted at [ISMRM 2024](https://www.ismrm.org/24m/) | [abstract](https://www.ismrm.org/24/accepted_abstracts.pdf)



## Citation
If you use this code, please cite our preprint or abstract:

```
@article{eichhorn2024miccai,
      title={Physics-Informed Deep Learning for Motion-Corrected Reconstruction of Quantitative Brain {MRI}}, 
      author={Hannah Eichhorn and Veronika Spieker and Kerstin Hammernik and Elisa Sacks and Kilian Weiss and Christine Preibisch and Julia A. Schnabel},
    journal={arXiv e-prints},
      year={2024},
}
```
```
@InProceedings{eichhorn2024ismrm,
      title={PHIMO: Physics-Informed Motion Correction of {GRE} {MRI} for {T2*} Quantification}, 
      author={Hannah Eichhorn and Kerstin Hammernik and Veronika Spieker and Elisa Sacks and Kilian Weiss and Christine Preibisch and Julia A. Schnabel},
      booktitle="Proceedings of the 2024 ISMRM & ISMRT Annual Meeting & Exhibition",
      year={2024},
}
```

## Contents of this repository:

- `ismrm-abstract`: code belonging to the ISMRM abstract
- `miccai`: code belonging to the MICCAI paper - *Code will be published soon.*

All computations were performed using Python 3.8.12 and PyTorch 2.0.1, , using an adapted version of the [IML-CompAI Framework](https://github.com/compai-lab/iml-dl) and the [MERLIN Framework](https://github.com/midas-tum/merlin).


## Setup:

1. Create a virtual environment with the required packages:
    ```
    cd ${TARGET_DIR}/PHIMO
    conda env create -f conda_environment.yml
    source activate ismrm_2024 *or* conda activate ismrm_2024
    ```

2. Install pytorch with cuda:
    ```
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    pip install torchinfo
    conda install -c conda-forge pytorch-lightning
    ```

3. For setting up wandb please refer to the [IML-CompAI Framework](https://github.com/compai-lab/iml-dl).
