import logging
import subprocess


def run_blastp(target_fasta, blastp_result, db_dir, threads=1):
    logging.info("BLASTp prediction starts on the dataset")
    subprocess.call(
        "diamond blastp -d %s -q %s -o %s --threads %s --id 50 --outfmt 6 qseqid sseqid evalue score qlen slen length pident"
        % (db_dir, target_fasta, blastp_result, threads),
        shell=True,
        stderr=subprocess.STDOUT,
    )
    logging.info("BLASTp prediction ended on the dataset")


def read_best_blast_result(blastp_result):
    query_db_set_info = {}
    with open(blastp_result, "r") as fp:
        for line in fp:
            sptlist = line.strip().split("\t")
            query_id = sptlist[0].strip()
            db_id = sptlist[1].strip()

            ec_number = db_id.split("|")[1].strip()
            score = float(sptlist[3].strip())
            qlen = sptlist[4].strip()
            length = sptlist[6].strip()
            length = float(length)
            pident = float(sptlist[-1].strip())

            ec_number = ec_number.split(";")
            ec_numbers = []
            for item in ec_number:
                if "EC:" in item:
                    ec_numbers.append(item)
                else:
                    ec_numbers.append(f"EC:{item}")
            ec_numbers.sort()
            ec_numbers = ";".join(ec_numbers)

            if pident < 50:
                continue
            coverage = length / float(qlen) * 100
            if coverage >= 75:
                if query_id not in query_db_set_info:
                    query_db_set_info[query_id] = [ec_numbers, score]
                else:
                    p_score = query_db_set_info[query_id][1]
                    if score > p_score:
                        query_db_set_info[query_id] = [ec_numbers, score]
    return query_db_set_info


def merge_predictions(dl_pred_result, blastp_pred_result, output_dir):
    # Read DeepEC predictions
    dl_data = {}
    with open(dl_pred_result, "r") as f1:
        f1.readline()  # skip header
        for line in f1:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            seq_id, ec, score, method = parts
            if seq_id not in dl_data:
                dl_data[seq_id] = []
            dl_data[seq_id].append((ec, float(score), method))

    # Read BLASTP predictions
    blastp_data = {}
    with open(blastp_pred_result, "r") as f2:
        f2.readline()  # skip header
        for line in f2:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            seq_id, ec, score, method = parts
            if seq_id not in blastp_data:
                blastp_data[seq_id] = []
            blastp_data[seq_id].append((ec, float(score), method))

    # Get all sequence IDs
    all_seq_ids = set(dl_data.keys()) | set(blastp_data.keys())

    with open(f"{output_dir}/DeepECv2_result.txt", "w") as fp:
        fp.write("sequence_ID\tec_numbers\tdeepec_ecs\tdeepec_scores\tblastp_ecs\tblastp_scores\n")
        
        for seq_id in sorted(all_seq_ids):
            # Collect all EC numbers for combined column
            all_ecs = []
            deepec_ecs = []
            deepec_scores = []
            blastp_ecs = []
            blastp_scores = []
            
            # Process DeepEC results
            if seq_id in dl_data:
                for ec, score, method in dl_data[seq_id]:
                    ec_clean = ec.replace("EC:", "")
                    all_ecs.append(ec_clean)
                    deepec_ecs.append(ec_clean)
                    deepec_scores.append(f"{score:.4f}")
            
            # Process BLASTP results (only for failed DeepEC cases)
            if seq_id in blastp_data:
                for ec, score, method in blastp_data[seq_id]:
                    ec_clean = ec.replace("EC:", "")
                    if seq_id not in dl_data:  # Only add if DeepEC failed
                        all_ecs.append(ec_clean)
                    blastp_ecs.append(ec_clean)
                    blastp_scores.append(f"{score:.4f}")
            
            # Format output columns
            combined_ecs = ",".join(all_ecs) if all_ecs else ""
            deepec_ec_str = ",".join(deepec_ecs) if deepec_ecs else ""
            deepec_score_str = ",".join(deepec_scores) if deepec_scores else ""
            blastp_ec_str = ",".join(blastp_ecs) if blastp_ecs else ""
            blastp_score_str = ",".join(blastp_scores) if blastp_scores else ""
            
            fp.write(f"{seq_id}\t{combined_ecs}\t{deepec_ec_str}\t{deepec_score_str}\t{blastp_ec_str}\t{blastp_score_str}\n")
