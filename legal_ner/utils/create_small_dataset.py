import json

def filter_entries_by_labels(input_json, output_json, target_labels):
    with open(input_json, 'r', encoding='utf-8') as file:
        data = json.load(file)

    filtered_data = []

    for entry in data:
        annotations = entry.get('annotations', [])
        labels = []
        for annotation in annotations:
            for result in annotation.get('result', []):
                labels.append(result.get('value', {}).get('labels')[0])
        # Map labels to "OTHER" if not in target_labels
        mapped_labels = ["OTHER" if label not in target_labels else label for label in labels]
        # Check if any label is in target_labels
        if any(label in target_labels for label in mapped_labels):
            # Update the labels in the entry
            """for annotation in entry.get('annotations', []):
                for result, mapped in zip(annotation.get('result', []), mapped_labels):
                    result['value']['labels'] = [mapped]"""
            filtered_data.append(entry)
    with open(output_json, 'w', encoding='utf-8') as file:
        json.dump(filtered_data, file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_json_path = "legal_ner/data/NER_TRAIN/NER_TRAIN_ALL.json"
    output_json_path = "legal_ner/data/NER_TRAIN/NER_TRAIN_SMALL2.json"

    target_labels = {"ORG", "GPE", "PRECEDENT"}  # Replace with your target labels

    filter_entries_by_labels(input_json_path, output_json_path, target_labels)

    input_json_path = "legal_ner/data/NER_DEV/NER_DEV_ALL.json"
    output_json_path = "legal_ner/data/NER_DEV/NER_DEV_SMALL2.json"

    filter_entries_by_labels(input_json_path, output_json_path, target_labels)
