import os
import re
import shutil
import subprocess
from subprocess import CalledProcessError
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Any

from .data_loader import DeepECDataset
from .utils import run_neural_net, save_dl_result
from .homology import read_best_blast_result, merge_predictions


def initialize_model(model_path: str = None, n_threads: int = 2) -> torch.nn.Module:
    """
    Initialize and load the BERT model for protein sequence analysis.

    Args:
        model_path (str): The file path to the pre-trained model. If None, the default model is used.

    Returns:
        torch.nn.Module: The loaded PyTorch model.
    """
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "model", "model.pth")
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


def run_deepec_and_save_results(
    model: torch.nn.Module,
    dataloader: DataLoader,
    threshold: float,
    device: torch.device,
    output_file: str,
    input_ids: List[str] = None,
) -> Tuple[Dict, List]:
    """
    Run the neural network on the provided data and save the results.

    Args:
        model: The neural network model.
        dataloader: DataLoader for the protein sequences.
        threshold: Threshold for prediction.
        device: The device to run the model on.
        output_file: Path to save the output file.
        input_ids: List of sequence IDs. Defaults to None.

    Returns:
        Tuple[Dict, List]: Predictions and list of failed cases.
    """
    y_pred, y_score = run_neural_net(model, dataloader, threshold, device=device)
    explainECs = model.explainECs if hasattr(model, "explainECs") else []

    if input_ids is None:
        input_ids = [f"Query_{i}" for i in range(len(y_pred))]
    failed_cases = save_dl_result(y_pred, y_score, input_ids, explainECs, output_file)
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


def extract_sequences_from_fasta(file_path: str, sanitize: bool = False) -> list:
    """
    Parse a FASTA file and extract sequence IDs and protein sequences. Optionally, sanitize the sequences
    by removing asterisks from the end.

    Args:
        file_path (str): Path to the FASTA file.
        sanitize (bool): If True, removes asterisks from the end of sequences. Defaults to False.

    Returns:
        List[Tuple[str, str]]: A list of tuples, where each tuple contains a sequence ID and its protein sequence.
    """
    with open(file_path, "r") as file:
        sequences = []
        seq_id = ""
        seq = ""
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if seq_id and seq:
                    sequences.append((seq_id, seq.rstrip("*") if sanitize else seq))
                seq_id = line.split()[0][1:]
                seq = ""
            else:
                seq += line
        if seq_id and seq:
            sequences.append((seq_id, seq.rstrip("*") if sanitize else seq))
    return sequences


def filter_predictions_by_score(
    input_file: str, output_file: str, score_threshold: float, merge_ec: bool = False
) -> None:
    """
    Filters rows in a file based on a minimum score threshold and writes the result to another file.
    Optionally, merges rows with the same sequence ID, presenting EC numbers and scores as comma-separated strings.

    Args:
        input_file (str): Path to the input file containing the data to be filtered.
        output_file (str): Path where the filtered (and optionally merged) data will be saved.
        score_threshold (float): The minimum score threshold. Rows with scores below this threshold will be excluded.
        merge_ec (bool): If True, merges rows with the same sequence ID. Defaults to False.

    Returns:
        None: This function writes the result to a file and does not return any value.
    """
    # Read the data from the input file
    data = pd.read_csv(input_file, sep="\t")
    filtered_data = data[data["score"] >= score_threshold]
    if merge_ec:
        filtered_data = (
            filtered_data.groupby("sequence_ID")
            .agg(
                {
                    "prediction": lambda x: ",".join(x),
                    "score": lambda x: ",".join(x.astype(str)),
                }
            )
            .reset_index()
        )
    filtered_data.to_csv(output_file, sep="\t", index=False)


def merge_ec_numbers(input_file: str, output_file: str) -> None:
    """
    Merges rows with the same sequence ID such that EC numbers and scores are presented as comma-separated strings.

    Args:
        input_file (str): Path to the input file containing sequence IDs, predictions (EC numbers), and scores.
        output_file (str): Path where the merged data will be saved.

    Returns:
        None: This function writes the result to a file and does not return any value.
    """
    data = pd.read_csv(input_file, sep="\t")
    merged_data = (
        data.groupby("sequence_ID")
        .agg(
            {
                "prediction": lambda x: ",".join(x),
                "score": lambda x: ",".join(x.astype(str)),
            }
        )
        .reset_index()
    )
    merged_data.to_csv(output_file, sep="\t", index=False)


def predict_ec_numbers(
    fasta_file_path: str,
    output_file: str,
    deepec_checkpt_file: str = None,
    n_threads: int = 12,
    batch_size: int = 128,
    score_threshold: float = None,
):
    """
    Process protein sequences from a FASTA file, predict enzyme activities using a deep learning model,
    handle failed cases with BLASTP, save the results, and filter predictions by score.

    Args:
        fasta_file_path (str): Path to the FASTA file with protein sequences.
        output_file (str): Path to save the prediction results.
        checkpt_file (str): Path to the deep learning model checkpoint file. If None (default), the default model is used.
        n_threads (int, optional): Number of threads for model initialization. Defaults to 12.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 128.
        score_threshold (float, optional): Score threshold for filtering predictions.
    """
    fasta_sequences = extract_sequences_from_fasta(fasta_file_path, sanitize=True)
    query_seqs = [seq[1] for seq in fasta_sequences]
    query_ids = [seq[0] for seq in fasta_sequences]
    input_seqs = preprocess_sequences(query_seqs, max_length=1000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_model(model_path=deepec_checkpt_file, n_threads=n_threads)
    model.to(device)
    threshold = model.thresholds.to(device)

    proteinDataloader, _ = prepare_protein_dataloader(
        input_seqs, model, batch_size=batch_size, input_ids=query_ids
    )
    run_deepec_and_save_results(
        model, proteinDataloader, threshold, device, output_file, input_ids=query_ids
    )

    if score_threshold is not None:
        filter_predictions_by_score(
            output_file,
            output_file,
            score_threshold=score_threshold,
            merge_ec=True,
        )
