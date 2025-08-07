import os
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from deepec.process_data import read_EC_actual_Fasta
from deepec.data_loader import DeepECDataset
from deepec.utils import argument_parser, run_neural_net, save_dl_result
from deepec.homology import run_blastp, read_best_blast_result, merge_predictions



logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')


if __name__ == '__main__':
    parser = argument_parser()
    options = parser.parse_args()

    output_dir = options.output_dir
    input_data_file = options.seq_file

    device = options.gpu
    batch_size = options.batch_size
    num_cpu = options.cpu_num
    tokenizer = options.tokenizer

    torch.set_num_threads(num_cpu)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(output_dir + '/artifacts'):
        os.makedirs((output_dir+'/artifacts'))

    
    model = torch.load('./model/model.pth')
    model = model.to(device)
    explainECs = model.explainECs
    pred_thrd = model.thresholds
    
    input_seqs, input_ids = read_EC_actual_Fasta(input_data_file)
    id2ind = {seq_id: i for i, seq_id in enumerate(input_ids)}
    pseudo_labels = np.zeros((len(input_seqs)))

    proteinDataset = DeepECDataset(
        data_X=input_seqs,
        data_Y=pseudo_labels,
        explainECs=explainECs,
        pred=True,
        tokenizer_name=tokenizer,
        )
    proteinDataloader = DataLoader(proteinDataset, batch_size=batch_size, shuffle=False)

    y_pred, y_score = run_neural_net(model, proteinDataloader, pred_thrd, device=device)
    failed_cases = save_dl_result(y_pred, y_score, input_ids, explainECs, output_dir+'/artifacts/DL_prediction_result.txt')

    if len(failed_cases) > 0:
        blastp_input = f'{output_dir}/artifacts/temp_seq.fa'
        blastp_tmp_output = f'{output_dir}/artifacts/blast_tmp_result.txt'
        blastp_output = f'{output_dir}/artifacts/blast_result.txt'

        with open(blastp_input, 'w') as fp:
            for seq_id in failed_cases:
                idx = id2ind[seq_id]
                seq = input_seqs[idx]
                fp.write(f'>{seq_id}\n{seq}\n')

        run_blastp(blastp_input, blastp_tmp_output, './model/swissprot_enzyme_diamond', threads=num_cpu)
        blastp_pred = read_best_blast_result(blastp_tmp_output)
        
        with open(blastp_output, 'w') as fp:
            fp.write('sequence_ID\tec_number\tscore\tmethod\n')
            for seq_id in blastp_pred:
                ec = blastp_pred[seq_id][0]
                score = blastp_pred[seq_id][1]
                fp.write(f'{seq_id}\t{ec}\t{score:0.4f}\tblastp\n')

        merge_predictions(f'{output_dir}/artifacts/DL_prediction_result.txt', blastp_output, output_dir)
    
    else:
        # Process DeepEC-only results with new format
        dl_data = {}
        with open(output_dir+'/artifacts/DL_prediction_result.txt', "r") as f1:
            f1.readline()  # skip header
            for line in f1:
                if not line.strip():
                    continue
                parts = line.strip().split("\t")
                seq_id, ec, score, method = parts
                if seq_id not in dl_data:
                    dl_data[seq_id] = []
                dl_data[seq_id].append((ec, float(score), method))

        with open(f"{output_dir}/DeepECv2_result.txt", "w") as fp:
            fp.write("sequence_ID\tec_numbers\tdeepec_ecs\tdeepec_scores\tblastp_ecs\tblastp_scores\n")
            
            for seq_id in sorted(dl_data.keys()):
                deepec_ecs = []
                deepec_scores = []
                
                for ec, score, method in dl_data[seq_id]:
                    ec_clean = ec.replace("EC:", "")
                    deepec_ecs.append(ec_clean)
                    deepec_scores.append(f"{score:.4f}")
                
                combined_ecs = ",".join(deepec_ecs)
                deepec_ec_str = ",".join(deepec_ecs)
                deepec_score_str = ",".join(deepec_scores)
                
                fp.write(f"{seq_id}\t{combined_ecs}\t{deepec_ec_str}\t{deepec_score_str}\t\t\n")