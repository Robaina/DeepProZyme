#DeepEC_v2 reconstruction
##Procedure

**Note**: 
This source code was developed in Linux, and has been tested in Ubuntu 16.04 with Python 3.6.

1. Clone the repository

        git clone https://github.com/kaistsystemsbiology/DeepZyme.git

2. Create and activate virtual environment

        conda env create -f environment.yml
        conda activate th_env

3. To use gpus properly, install the pytorch and cuda for your gpus. This code was tested on pytorch=1.7.0 using cuda version 10.2

4. To use homologous search using DIAMOND, install [diamond](https://github.com/bbuchfink/diamond/wiki). We recommed to use conda

        conda install -c bioconda diamond


##Example


- Run DeepEC

        python run_deepec_v2.py -i ./example/mdh_ecoli.fa -o ./example/results -g cpu -b 128 -cpu 2
        python run_deepec_v2.py -i ./example/mdh_ecoli.fa -o ./example/results -g cuda:3 -b 128 -cpu 2