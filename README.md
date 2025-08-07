# DeepProZyme  enzimatic tool

<p align="center">
<img src="https://user-images.githubusercontent.com/10931299/117299954-313a7300-ae77-11eb-9036-a29e38235519.png" width="250"/>
</p>


## 📖 Description

This is a forked version of the original DeepProZyme project, which can be found here: https://github.com/kaistsystemsbiology/DeepProZyme


DeepProZyme is a tool for predicting enzyme commission (EC) numbers from protein sequences. This version includes additional features and improvements.


## 🛠️ Installation


**Note**: This source code was developed in Linux and has been tested on Ubuntu 16.04 with Python 3.6.


1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/DeepProZyme.git
    cd DeepProZyme
    ```

2.  **Create and activate conda environment:**

    ```bash
    conda env create -f envs/environment.yml
    conda activate deepectransformer
    ```

3.  **Install PyTorch and CUDA:**

    To use GPUs properly, install PyTorch and CUDA for your specific hardware. This code was tested on `pytorch=1.7.0` with CUDA version 10.2.


4.  **Download the ProtBert model:**

    ```bash
    python src/download_protbert.py
    ```


## ▶️ How to Run


-   **Run DeepECtransformer:**

    ```bash
    python src/run_deepectransformer.py -i ./example/mdh_ecoli.fa -o ./example/results -g cpu -b 128 -cpu 2
    ```

    Or with a GPU:

    ```bash
    python src/run_deepectransformer.py -i ./example/mdh_ecoli.fa -o ./example/results -g cuda:3 -b 128 -cpu 2
    ```


## 🧪 Testing

A comprehensive test suite is available to verify the tool's functionality:

```bash
python tests/test_deepec.py
```

The test includes:
- **Execution Test**: Verifies that DeepEC runs successfully with the provided parameters
- **Output Format Validation**: Checks that the result file contains the correct columns and structure
- **Consistency Test**: Compares new results against existing expected results

### Test Data
Test data is located in `tests/data/TARA_ARC_108_MAG_00080.fasta` and expected results in `tests/results/DeepECv2_result.txt`.

### Test Features
- Automatic timeout protection (30 minutes)
- Detailed error reporting
- Output format validation
- Sequence consistency checking

## 🐳 Docker


You can also use Docker to run the tool. Dockerfiles for both CPU and GPU are provided.


-   **CPU:**

    ```bash
    docker build -t deepenzyme -f Dockerfile.cpu .
    docker run -it --rm -v $(pwd)/example:/DeepProZyme/example deepenzyme python src/run_deepectransformer.py -i ./example/mdh_ecoli.fa -o ./example/results -g cpu -b 128 -cpu 2
    ```


-   **GPU:**

    ```bash
    docker build -t deepenzyme -f Dockerfile.gpu .
    docker run -it --rm --gpus all -v $(pwd)/example:/DeepProZyme/example deepenzyme python src/run_deepectransformer.py -i ./example/mdh_ecoli.fa -o ./example/results -g cuda:0 -b 128 -cpu 2
    ```


## 🙏 Acknowledgements


This project is a fork of the original DeepProZyme project from the KAIST Systems Biology group.


-   **Original Project:** [https://github.com/kaistsystemsbiology/DeepProZyme](https://github.com/kaistsystemsbiology/DeepProZyme)