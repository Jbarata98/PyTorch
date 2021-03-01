from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from bert_score import score as bert_sc
from eval import *
from bleurt import score as bleurt_sc
import statistics
from itertools import repeat


EVALUATE = False
JSON_results = 'hypothesis'
JSON_refs = 'references'
bleurt_checkpoint = "bleurt/test_checkpoint" #uses Tiny Bleurt

def create_json(hyp,refs):
    hyp_dict, ref_dict = {},{}
    img_count_refs,img_count_hyps = 0,0
    for ref_caps in refs:
        ref_dict[str(img_count_refs)] = [item for sublist in ref_caps for item in sublist]
        img_count_refs+=1
    for hyp_caps in hyp:
        hyp_dict[str(img_count_hyps)] = [hyp_caps]
        img_count_hyps+=1

    with open(JSON_results + '.json', 'w') as fp:
        json.dump(hyp_dict, fp)
    with open(JSON_refs +'.json', 'w') as fp:
        json.dump(ref_dict, fp)

def open_json():
    with open(JSON_refs + '.json', 'r') as file:
        gts = json.load(file)
    with open(JSON_results + '.json', 'r') as file:
        res = json.load(file)

    return gts,res

def bleu(gts,res):
    scorer = Bleu(n=4)

    score, scores = scorer.compute_score(gts, res)

    print('BLEU(1-4) = %s' % score)

def cider(gts,res):
    scorer = Cider()
    # scorer += (hypo[0], ref1)
    (score, scores) = scorer.compute_score(gts, res)
    print('CIDEr = %s' % score)

def meteor(gts,res):
    scorer = Meteor()
    score, scores = scorer.compute_score(gts, res)
    print('METEOR = %s' % score)

def rouge(gts,res):
    scorer = Rouge()
    score, scores = scorer.compute_score(gts, res)
    print('ROUGE = %s' % score)

def spice(gts, res):
    scorer = Spice()
    score, scores = scorer.compute_score(gts, res)
    print('SPICE = %s' % score)

def bert_based(gts,res):
    refs, cands = [], []
    for refers in gts.values():
        sub_refs = []
        for ref in refers:
            sub_refs.append(ref + '.')
        refs.append(sub_refs)
    for cand in res.values():
        cands.append(cand[0] + '.')
    #
    P, R, F1 = bert_sc(cands, refs, lang='en', verbose=True)
    print('BERTScore = %s' % F1.mean().item())

    refs_bleurt = ''
    refs_bleurt_list =[]
    for ref in refs:
        for ref_cap in ref:
            refs_bleurt_list.append(ref_cap)

    cands = [x for item in cands for x in repeat(item,5)]

    scorer = bleurt_sc.BleurtScorer(bleurt_checkpoint)

    scores = scorer.score(refs_bleurt_list, cands, batch_size=None)
    assert type(scores) == list
    print('BLEURT = %s' % statistics.mean(scores))

def main():
    gts,res = open_json()
    bleu(gts,res)
    cider(gts,res)
    meteor(gts,res)
    rouge(gts,res)
    spice(gts,res)
    bert_based(gts,res)

if EVALUATE:

    refs, hyps = evaluate(beam_size)
    create_json(hyps, refs)

main()











