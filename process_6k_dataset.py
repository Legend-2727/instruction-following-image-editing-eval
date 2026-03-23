import json
import os

input_file = "/home/user/ML-Project(42_46)/en_to_hindi.json"
out_dir = "data/hf_snapshots/xlingual_picobanana_multilingual_6k"
os.makedirs(out_dir, exist_ok=True)

out_metadata = os.path.join(out_dir, "metadata.jsonl")

labels_dir = "artifacts/multilingual_data_6k"
os.makedirs(labels_dir, exist_ok=True)
out_labels = os.path.join(labels_dir, "labels.jsonl")

success_count = 0
reject_count = 0

with open(input_file, 'r', encoding='utf-8') as fin, \
     open(out_metadata, 'w', encoding='utf-8') as fmd, \
     open(out_labels, 'w', encoding='utf-8') as flb:
    for line in fin:
        if not line.strip(): continue
        data = json.loads(line)
        
        # Build new metadata row
        row = {
            "id": data["id"],
            "instruction_en": data["instruction_en"],
            "summarized_text": data["summarized_text"],
            "edit_type": data["edit_type"],
            "source_type": data["source_type"],
            "source_path": data["source_path"],
            "target_path": data["target_path"],
            "instruction_hi": data["translation_en_to_hi"],
            "instruction_bn": data["translation_en_to_bn"]
            # No nepali
        }
        
        fmd.write(json.dumps(row, ensure_ascii=False) + "\n")
        
        # Build label proxy
        # sft -> Success, preference_rejected -> Partial/No proxy
        source_type = data["source_type"].lower()
        if 'sft' in source_type:
            adherence = "Success"
            success_count += 1
        else:
            adherence = "Partial/No"
            reject_count += 1
            
        label_row = {
            "sample_id": data["id"],
            "adherence_label": adherence,
            "taxonomy_labels": []
        }
        flb.write(json.dumps(label_row) + "\n")

print(f"Dataset generated at {out_metadata}")
print(f"Labels generated at {out_labels}")
print(f"Success: {success_count}, Rejected proxy: {reject_count}")
print(f"Total processed: {success_count + reject_count}")
