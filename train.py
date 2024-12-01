import os
import argparse
import subprocess
import random
from tqdm import tqdm
import torch
from statistics import mean
from collections import defaultdict
import sys
import modeling
import data
import modeling_util

SEED = 42
LR = 0.001
BERT_LR = 2e-5
MAX_EPOCH = 10
BATCH_SIZE = 16
BATCHES_PER_EPOCH = 512
GRAD_ACC_SIZE = 2
VALIDATION_METRIC = 'P_20'
PATIENCE = 4
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)


MODEL_MAP = {
    'vanilla_bert': modeling.VanillaBertRanker,
   
}


def main(model, dataset, train_pairs, qrels_train, valid_run, qrels_valid, model_out_dir):
  

    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    non_bert_params = {'params': [v for k, v in params if not k.startswith('bert.')]}
    bert_params = {'params': [v for k, v in params if k.startswith('bert.')], 'lr': BERT_LR}
    optimizer = torch.optim.Adam([non_bert_params, bert_params], lr=LR)

    epoch = 0
    top_valid_score = None
    print(f'Starting training, upto {MAX_EPOCH} epochs, patience {PATIENCE} LR={LR} BERT_LR={BERT_LR}', flush=True)
    for epoch in range(MAX_EPOCH):

        loss = train_iteration(model, optimizer, dataset, train_pairs, qrels_train)
        print(f'epoch = {epoch} loss={loss}')

        valid_score = validate(model, dataset, valid_run, qrels_valid, epoch)
        print(f'validation epoch={epoch} score={valid_score}')

        if top_valid_score is None or valid_score > top_valid_score:
            top_valid_score = valid_score
            print('new top validation score, saving weights', flush=True)
            model.save(os.path.join(model_out_dir, 'weights.p'))
            top_valid_score_epoch = epoch
        if top_valid_score is not None and epoch - top_valid_score_epoch > PATIENCE:
            print(f'no validation improvement since {top_valid_score_epoch}, early stopping', flush=True)
            break

    if top_valid_score_epoch != epoch:
        model.load(os.path.join(model_out_dir, 'weights.p'))
    return (model, top_valid_score_epoch)


def train_iteration(model, optimizer, dataset, train_pairs, qrels):
    
    total = 0
    model.train()
    total_loss = 0.
    with tqdm('training', total=BATCH_SIZE * BATCHES_PER_EPOCH, ncols=80, desc='train', leave=False) as pbar:
        for record in data.iter_train_pairs(model, dataset, train_pairs, qrels, GRAD_ACC_SIZE):
            scores = model(record['query_tok'],
                           record['query_mask'],
                           record['doc_tok'],
                           record['doc_mask'])
            count = len(record['query_id']) // 2
            scores = scores.reshape(count, 2)
            loss = torch.mean(1. - scores.softmax(dim=1)[:, 0])
            loss.backward()
            total_loss += loss.item()
            total += count
            if total % BATCH_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad()
            pbar.update(count)
            if total >= BATCH_SIZE * BATCHES_PER_EPOCH:
                return total_loss

def validate(model, dataset, run, valid_qrels, epoch):
    run_filtred = {query_id: top_docs for query_id, top_docs in run.items() if any(doc_id in valid_qrels.get(query_id, {}) for doc_id in top_docs)}
    run_scores = run_model(model, dataset, run_filtred)
    rank = {qid: {docid: float(score) for docid, score in sorted(docs.items(), key=lambda x: x[1], reverse=True)[:100]} for qid, docs in run_scores.items()}
    mrr_at_100 = modeling_util.compute_mrr_at_100(rank, valid_qrels)
    print("MRR@100:", mrr_at_100)
    
    return mrr_at_100


def run_model(model, dataset, run, desc='valid'):

    run_dev = {qid: run[qid] for qid in random.sample(run.keys(), int(len(run) * 0.03))}
    rerank_run = defaultdict(dict)
    with torch.no_grad(), tqdm(total=sum(len(r) for r in run_dev.values()), ncols=80, desc=desc, leave=False) as pbar:
        model.eval()
        for records in data.iter_valid_records(model, dataset, run_dev, BATCH_SIZE):
            scores = model(records['query_tok'],
                           records['query_mask'],
                           records['doc_tok'],
                           records['doc_mask'])
            for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                rerank_run[qid][did] = score.item()
            pbar.update(len(records['query_id']))
    return rerank_run
    


def main_cli():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--model', choices=MODEL_MAP.keys(), default='vanilla_bert')
    parser.add_argument('--datafiles', type=argparse.FileType('rt'), nargs='+')
    parser.add_argument('--qrels', type=argparse.FileType('rt'))
    parser.add_argument('--qrels_valid', type=argparse.FileType('rt'))
    parser.add_argument('--train_pairs', type=argparse.FileType('rt'))
    parser.add_argument('--valid_run', type=argparse.FileType('rt'))
    parser.add_argument('--initial_bert_weights', type=argparse.FileType('rb'))
    parser.add_argument('--model_out_dir')
    args = parser.parse_args()
    model = MODEL_MAP[args.model]().cuda()
    dataset = data.read_datafiles(args.datafiles)
    qrels = data.read_qrels_dict(args.qrels)
    qrels_valid = data.read_qrels_dict(args.qrels_valid)
    train_pairs = data.read_pairs_dict(args.train_pairs)
    valid_run = data.read_run_dict(args.valid_run)
   
    model.load(args.initial_bert_weights.name)
    main(model, dataset, train_pairs, qrels, valid_run, qrels_valid, args.model_out_dir)


if __name__ == '__main__':
    main_cli()