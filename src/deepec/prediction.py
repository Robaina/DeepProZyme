import os
import re
import shutil
import subprocess
from subprocess import CalledProcessError
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Any

from .data_loader import DeepECDataset
from .utils import run_neural_net, save_dl_result
from .homology import read_best_blast_result, merge_predictions


def initialize_model(model_path: str, n_threads: int = 2) -> torch.nn.Module:
    """
    Initialize and load the BERT model for protein sequence analysis.

    Args:
        model_path (str): The file path to the pre-trained model.

    Returns:
        torch.nn.Module: The loaded PyTorch model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(n_threads)
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.bert.config.output_attentions = False
    model.eval()
    return model


def preprocess_sequences(sequences: List[str], max_length: int = 1000) -> List[str]:
    """
    Preprocess a list of protein sequences for model prediction.

    This function processes each sequence in the list by removing any whitespace,
    truncating it to a maximum length, and ensuring that it contains only valid
    amino acid characters.

    Args:
        sequences (List[str]): A list of raw protein sequences.
        max_length (int, optional): Maximum allowed length of each sequence. Defaults to 1000.

    Returns:
        List[str]: A list of preprocessed protein sequences. Sequences with invalid characters are excluded.
    """
    # Define valid amino acid characters
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")

    processed_sequences = []
    for seq in sequences:
        seq = re.sub(r"\s", "", seq)
        seq = seq[:max_length]
        if set(seq).issubset(valid_aas):
            processed_sequences.append(seq)

    return processed_sequences


def run_network_and_save_results(
    model: torch.nn.Module,
    dataloader: DataLoader,
    threshold: float,
    device: torch.device,
    output_dir: str,
    input_ids: List[str] = None,
) -> Tuple[Dict, List]:
    """
    Run the neural network on the provided data and save the results.

    Args:
        model: The neural network model.
        dataloader: DataLoader for the protein sequences.
        threshold: Threshold for prediction.
        device: The device to run the model on.
        output_dir: Directory to save temporary results.

    Returns:
        Tuple[Dict, List]: Predictions and list of failed cases.
    """
    y_pred, y_score = run_neural_net(model, dataloader, threshold, device=device)
    explainECs = model.explainECs if hasattr(model, "explainECs") else []

    if input_ids is None:
        input_ids = [f"Query_{i}" for i in range(len(y_pred))]
    if not os.path.exists(os.path.join(output_dir, "tmp")):
        os.makedirs(os.path.join(output_dir, "tmp"))

    failed_cases = save_dl_result(
        y_pred, y_score, input_ids, explainECs, os.path.join(output_dir, "tmp")
    )
    return y_pred, failed_cases


def handle_failed_cases(
    failed_cases: List[str],
    input_seqs: List[str],
    id2ind: Dict[str, int],
    diamond_db: str,
    output_dir: str,
    threads: int = 4,
) -> bool:
    """
    Handle failed cases by running BLASTP.

    Args:
        failed_cases: List of sequences that failed in DL prediction.
        input_seqs: List of all input sequences.
        id2ind: Mapping from sequence ID to index in input_seqs.
        diamond_db: Path to the Diamond database.
        output_dir: Directory to save output files.
        threads: Number of threads to use for Diamond BLASTP.

    Returns:
        bool: True if BLASTP was run, False otherwise.
    """
    if len(failed_cases) == 0:
        return False

    blastp_input = os.path.join(output_dir, "tmp", "temp_seq.fa")
    blastp_tmp_output = os.path.join(output_dir, "tmp", "blast_tmp_result.txt")
    blastp_output = os.path.join(output_dir, "tmp", "blast_result.txt")

    with open(blastp_input, "w") as fp:
        for seq_id in failed_cases:
            idx = id2ind[seq_id]
            seq = input_seqs[idx]
            fp.write(f">{seq_id}\n{seq}\n")

    run_diamond_blastp(
        input_file=blastp_input,
        output_file=blastp_tmp_output,
        db_path=diamond_db,
        threads=threads,
    )
    blastp_pred = read_best_blast_result(blastp_tmp_output)

    with open(blastp_output, "w") as fp:
        fp.write("sequence_ID\tprediction\n")
        for seq_id in blastp_pred:
            ec = blastp_pred[seq_id][0]
            fp.write(f"{seq_id}\t{ec}\n")
    return True


def run_diamond_blastp(
    input_file: str, output_file: str, db_path: str, threads: int
) -> None:
    """
    Run Diamond BLASTP.

    Args:
        input_file (str): Path to the input file.
        output_file (str): Path to save the output.
        db_path (str): Path to the Diamond database.
        threads (int): Number of threads to use.
    """
    command = [
        "diamond",
        "blastp",
        "-d",
        db_path,
        "-q",
        input_file,
        "-o",
        output_file,
        "--threads",
        str(threads),
        "--id",
        "50",
        "--outfmt",
        "6",  # tabular format
    ]

    try:
        subprocess.run(
            command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except CalledProcessError as e:
        print("Error occurred while running Diamond BLASTP:")
        print(
            e.stderr.decode()
        )  # This will print the error message from Diamond BLASTP
        raise  # Re-raise the exception to handle it as per your error handling policy


def prepare_protein_dataloader(
    input_seqs: List[str], model: Any, batch_size: int, input_ids: List[str] = None
) -> Tuple[DataLoader, Dict[str, int]]:
    """
    Prepare the DataLoader for protein sequences.

    Args:
        input_seqs (List[str]): A list of protein sequences.
        model: The neural network model with an attribute 'explainECs'.
        batch_size (int): Batch size for the DataLoader.
        input_ids (List[str], optional): A list of sequence IDs. Defaults to None.

    Returns:
        Tuple[DataLoader, Dict[str, int]]: A DataLoader for the dataset and a mapping from sequence IDs to their indices.
    """
    if input_ids is None:
        input_ids = ["Query_" + str(i) for i in range(len(input_seqs))]
    id2ind = {seq_id: i for i, seq_id in enumerate(input_ids)}
    pseudo_labels = np.zeros((len(input_seqs)))

    # Assuming explainECs is a list of EC numbers that model can explain
    explainECs = model.explainECs if hasattr(model, "explainECs") else []

    proteinDataset = DeepECDataset(
        data_X=input_seqs, data_Y=pseudo_labels, explainECs=explainECs, pred=True
    )
    proteinDataloader = DataLoader(proteinDataset, batch_size=batch_size, shuffle=False)

    return proteinDataloader, id2ind


def merge_and_finalize_predictions(output_dir: str, dl_success: bool) -> None:
    """
    Merge DL and BLAST predictions and finalize the results.

    Args:
        output_dir: Directory where the results are stored.
        dl_success: Boolean indicating if DL predictions were successful.
    """
    dl_pred_file = os.path.join(output_dir, "tmp", "DL_prediction_result.txt")
    blast_pred_file = os.path.join(output_dir, "tmp", "blast_result.txt")
    final_output_file = os.path.join(output_dir, "DeepECv2_result.txt")

    if dl_success:
        shutil.copy(dl_pred_file, output_dir)
        os.rename(
            os.path.join(output_dir, "DL_prediction_result.txt"), final_output_file
        )
    else:
        merge_predictions(dl_pred_file, blast_pred_file, output_dir)
