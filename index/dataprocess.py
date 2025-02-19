# import numpy as np
# import torch
# import torch.utils.data as data
import os
import gc
import torch
import datasets
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel
import pickle


target_length = 3

def data_map_embedding_function(batch, tokenizer, model, model_name):
    new_batch = {}
    
    def pad_sequences(seq, target_len, pad_value=-1):
        return seq + [pad_value] * (target_len - len(seq))
    new_batch['labels'] = [pad_sequences(l, target_length) for l in batch['label']]
    

    if model_name == "bert-base-uncased":
        tokens = tokenizer(batch['token'], padding=True, truncation=True, max_length = 512,return_tensors='pt')
        tokens = {key: value.to("cuda:0") for key, value in tokens.items()}
        new_batch['embedding'] = model(**tokens).last_hidden_state[:, 0].cpu().detach()
    elif model_name == "bge-large-en-v1.5":
        tokens = tokenizer(batch['token'], padding=True, truncation=True, return_tensors='pt').to("cuda:0")

        with torch.no_grad():
            new_batch['embedding'] = torch.nn.functional.normalize(model(**tokens)[0][:, 0].cpu().detach(), p=2, dim=1)
    elif model_name == "SFR-Embedding-2_R":
        tokens = tokenizer(batch['token'], max_length=4096, padding=True, truncation=True, return_tensors="pt")
        tokens = {key: value.to("cuda:0") for key, value in tokens.items()}

        left_padding = (tokens['attention_mask'][:, -1].sum() == tokens['attention_mask'].shape[0])
        if left_padding:
            new_batch['embedding'] = model(**tokens).last_hidden_state[:, -1].cpu().detach()
        else:
            sequence_lengths = tokens['attention_mask'].cpu().sum(dim=1) - 1
            last_hidden_states = model(**tokens).last_hidden_state.cpu().detach()
            batch_size = last_hidden_states.shape[0]
            new_batch['embedding'] = last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    del tokens
    torch.cuda.empty_cache()
    gc.collect()
    return new_batch

def generate_mapping(data_dir, labels):
    mapping = {}
    if type(labels) == torch.tensor:
        l1_values = sorted(set(labels[:, 0]))
    elif type(labels) == list:
        l1_values = sorted(set([label[0] for label in labels]))
    
    if max(l1_values) != len(l1_values) - 1: #说明第一级分类标签id不是从0开始的顺序编号，需要做映射
        tmp = {}
        for idx, v in enumerate(l1_values):
            tmp[v] = idx
        mapping['level-0-mapping'] = tmp
    
    cur_idx = 0
    for i in l1_values:
        #获得第一级分类标签为i，的第二类标签值
        if type(labels) == torch.tensor:
            l2_values = labels[labels[:, 0] == i, 1]
        else:
            l2_values = [label[1] for label in labels if label[0] == i]

        l2_sorted_values = sorted(set(l2_values))
        
        tmp = {}
        for idx, v in enumerate(l2_sorted_values):
            tmp[v] = cur_idx + idx
        cur_idx += idx + 1
        mapping[i] = tmp
    
    print(mapping)
    #{0: {7: 0, 12: 1, 23: 2, 28: 3, 37: 4, 41: 5, 46: 6, 49: 7, 50: 8, 57: 9, 91: 10, 99: 11, 101: 12, 110: 13, 111: 14, 125: 15, 140: 16}, 1: {8: 0, 11: 1, 16: 2, 19: 3, 21: 4, 22: 5, 27: 6, 35: 7, 36: 8, 38: 9, 39: 10, 40: 11, 43: 12, 44: 13, 47: 14, 52: 15, 53: 16, 54: 17, 63: 18, 65: 19, 68: 20, 69: 21, 70: 22, 72: 23, 73: 24, 74: 25, 75: 26, 76: 27, 78: 28, 80: 29, 85: 30, 89: 31, 106: 32, 107: 33, 108: 34, 109: 35, 112: 36, 114: 37, 115: 38, 116: 39, 117: 40, 121: 41, 124: 42, 126: 43, 127: 44, 128: 45, 129: 46, 131: 47, 132: 48, 134: 49, 135: 50, 138: 51, 139: 52}, 2: {9: 0, 33: 1, 45: 2, 51: 3, 56: 4, 66: 5, 71: 6, 82: 7, 83: 8, 118: 9, 120: 10}, 3: {10: 0, 24: 1, 29: 2, 55: 3, 59: 4, 60: 5, 62: 6, 79: 7, 84: 8, 90: 9, 95: 10, 98: 11, 102: 12, 103: 13, 113: 14}, 4: {13: 0, 17: 1, 18: 2, 20: 3, 25: 4, 26: 5, 31: 6, 32: 7, 42: 8}, 5: {14: 0, 34: 1, 48: 2, 61: 3, 67: 4, 77: 5, 81: 6, 92: 7, 94: 8}, 6: {15: 0, 30: 1, 64: 2, 86: 3, 87: 4, 88: 5, 93: 6, 96: 7, 97: 8, 100: 9, 104: 10, 105: 11, 119: 12, 122: 13, 123: 14, 130: 15, 133: 16, 136: 17, 137: 18}}
    with open('{}/labels_mapping.pkl'.format(data_dir), 'wb') as f:
        pickle.dump(mapping, f)

model_path={
    'bert-base-uncased':  "/mnt/disk5/hewenting_nfs_serverdir/models/google-bert:bert-base-uncased",
    "bge-large-en-v1.5": "/mnt/disk5/hewenting_nfs_serverdir/models/baai:bge-large-en-v1.5",
    "SFR-Embedding-2_R": "/mnt/disk5/hewenting_nfs_serverdir/models/Salesforce:SFR-Embedding-2_R"
}

def data_process(dataset, model_name="bert-base-uncased", data_dir=None, with_category=False):
    if data_dir is None:
        data_dir = os.path.join("/home/hewenting/data_preprocess/", dataset)
    save_dir = os.path.join(data_dir, model_name)
    
    train_filename = '{}/{}_train_withcategory.json'.format(data_dir, dataset)
    
    has_test_data = True
    has_dev_data = True
    if dataset == "Amazon-531": #这个数据集只有train和dev
        has_test_data=False
    elif dataset == "Amazon-392-hwt": #这个数据集只有train
        has_test_data=False
        has_dev_data=False
        if with_category:
            save_dir = os.path.join(data_dir, f'{model_name}_withcategory')
            train_filename = '{}/{}_train_withcategory.json'.format(data_dir, dataset)
        else:
            save_dir = os.path.join(data_dir, f'{model_name}_withoutcategory')
            train_filename = '{}/{}_train_withoutcategory.json'.format(data_dir, dataset)
    

    if os.path.exists(save_dir):
        print(f"----->data_process load_from_disk {save_dir}")
        dataset = datasets.load_from_disk(save_dir)
        # generate_mapping(data_dir, dataset['train']['labels'])
        # print(type(dataset['train']['embedding']), len(dataset['train']['embedding'][0]))
    else:
        print(f"----->data_process load_from_text & embedding, save to {save_dir}")
        print("train_filename=",train_filename)
        
        data_files={'train': train_filename}
        if has_dev_data:
            data_files['dev'] = '{}/{}_dev.json'.format(data_dir, dataset)
        if has_test_data:
            data_files['test'] = '{}/{}_test.json'.format(data_dir, dataset)
        dataset = datasets.load_dataset('json',  data_files=data_files)
        
        if model_name not in model_path.keys():
            raise Exception("Do not support this model")
        tokenizer = AutoTokenizer.from_pretrained(model_path[model_name], use_fast=False)
        model = AutoModel.from_pretrained(model_path[model_name]).cuda()
        model.eval()

        dataset = dataset.map(lambda x: data_map_embedding_function(x, tokenizer, model,  model_name), batched=True, batch_size=16)
        dataset.save_to_disk(save_dir)
        generate_mapping(data_dir, dataset['train']['labels'])
    
    with open('{}/labels_mapping.pkl'.format(data_dir), 'rb') as f:
        labels_mapping = pickle.load(f)
    
    dataset = dataset.remove_columns('label')
    dataset['train'].set_format('torch', columns=['embedding', 'labels'], output_all_columns=True)
    if has_dev_data:
        dataset['dev'].set_format('torch', columns=['embedding', 'labels'], output_all_columns=True)
    if has_test_data:
        dataset['test'].set_format('torch', columns=['embedding', 'labels'], output_all_columns=True)
    
    label_dict = {}
    if os.path.exists(os.path.join(data_dir, 'value_dict.pt')):
        label_dict = torch.load(os.path.join(data_dir, 'value_dict.pt'))   #data/WebOfScience/value_dict.pt
        label_dict = {i: v for i, v in label_dict.items()} #label_dict共141个key,其中0-6是父类,label_dict={0: 'CS', 1: 'Medical', 2: 'Civil', 3: 'ECE', 4: 'biochemistry', 5: 'MAE', 6: 'Psychology', 7: 'Symbolic computation', 8: "Alzheimer's Disease", 9: 'Green Building', 10: 'Electric motor', 11: "Parkinson's Disease", 12: 'Computer vision', 13: 'Molecular biology', 14: 'Fluid mechanics', 15: 'Prenatal development', 16: 'Sprains and Strains', 17: 'Enzymology', 18: 'Southern blotting', 19: 'Cancer', 20: 'Northern blotting', 21: 'Sports Injuries', 22: 'Senior Health', 23: 'Computer graphics', 24: 'Digital control', 25: 'Human Metabolism', 26: 'Polymerase chain reaction', 27: 'Multiple Sclerosis', 28: 'Operating systems', 29: 'Microcontroller', 30: 'Attention', 31: 'Immunology', 32: 'Genetics', 33: 'Water Pollution', 34: 'Hydraulics', 35: 'Hepatitis C', 36: 'Weight Loss', 37: 'Machine learning', 38: 'Low Testosterone', 39: 'Fungal Infection', 40: 'Diabetes', 41: 'Data structures', 42: 'Cell biology', 43: 'Parenting', 44: 'Birth Control', 45: 'Smart Material', 46: 'network security', 47: 'Heart Disease', 48: 'computer-aided design', 49: 'Image processing', 50: 'Parallel computing', 51: 'Ambient Intelligence', 52: 'Allergies', 53: 'Menopause', 54: 'Emergency Contraception', 55: 'Electrical network', 56: 'Construction Management', 57: 'Distributed computing', 58: 'Electrical generator', 59: 'Electricity', 60: 'Operational amplifier', 61: 'Manufacturing engineering', 62: 'Analog signal processing', 63: 'Skin Care', 64: 'Eating disorders', 65: 'Myelofibrosis', 66: 'Suspension Bridge', 67: 'Machine design', 68: 'Hypothyroidism', 69: 'Headache', 70: 'Overactive Bladder', 71: 'Geotextile', 72: 'Irritable Bowel Syndrome', 73: 'Polycythemia Vera', 74: 'Atrial Fibrillation', 75: 'Smoking Cessation', 76: 'Lymphoma', 77: 'Thermodynamics', 78: 'Asthma', 79: 'State space representation', 80: 'Bipolar Disorder', 81: 'Materials Engineering', 82: 'Stealth Technology', 83: 'Solar Energy', 84: 'Signal-flow graph', 85: "Crohn's Disease", 86: 'Borderline personality disorder', 87: 'Prosocial behavior', 88: 'False memories', 89: 'Idiopathic Pulmonary Fibrosis', 90: 'Electrical circuits', 91: 'Algorithm design', 92: 'Strength of materials', 93: 'Problem-solving', 94: 'Internal combustion engine', 95: 'Lorentz force law', 96: 'Prejudice', 97: 'Antisocial personality disorder', 98: 'System identification', 99: 'Computer programming', 100: 'Nonverbal communication', 101: 'Relational databases', 102: 'PID controller', 103: 'Voltage law', 104: 'Leadership', 105: 'Child abuse', 106: 'Mental Health', 107: 'Dementia', 108: 'Rheumatoid Arthritis', 109: 'Osteoporosis', 110: 'Software engineering', 111: 'Bioinformatics', 112: 'Medicare', 113: 'Control engineering', 114: 'Psoriatic Arthritis', 115: 'Addiction', 116: 'Atopic Dermatitis', 117: 'Digestive Health', 118: 'Remote Sensing', 119: 'Gender roles', 120: 'Rainwater Harvesting', 121: 'Healthy Sleep', 122: 'Depression', 123: 'Social cognition', 124: 'Anxiety', 125: 'Cryptography', 126: 'Psoriasis', 127: 'Ankylosing Spondylitis', 128: "Children's Health", 129: 'Stress Management', 130: 'Seasonal affective disorder', 131: 'HIV/AIDS', 132: 'Migraine', 133: 'Person perception', 134: 'Osteoarthritis', 135: 'Hereditary Angioedema', 136: 'Media violence', 137: 'Schizophrenia', 138: 'Kidney Health', 139: 'Autism', 140: 'Structured Storage'}
    elif os.path.exists(os.path.join(data_dir, 'train/labels.txt')):
        with open(os.path.join(data_dir, 'train/labels.txt'), 'r', encoding='utf-8') as file:
            for line in file:
                id, label = line.strip().split('\t')
                label_dict[int(id)] = label
                
    return dataset, labels_mapping, label_dict


if __name__ == '__main__':
    embedding_model = "bge-large-en-v1.5"
    # dataset_name = "WebOfScience"
    # data_dir = "/mnt/disk5/hewenting_nfs_serverdir/githubs/HPT/data/WebOfScience/results_with_keywords"
    
    dataset_name = "Amazon-531"

    dataset, labels_mapping, label_dict = data_process(dataset_name, model_name=embedding_model, data_dir = None)