from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
import json
from bert_score import score
from eval import *

EVALUATE = False
JSON_results = 'hypothesis'
JSON_refs = 'references'


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
    refs = []

    cands =[]
    for refers in gts.values():
        sub_refs = []
        for ref in refers:
            sub_refs.append(ref)
        refs.append(sub_refs)
    print(len(refs))
    for cand in res.values():
        cands.append(cand[0] +'.')
    print(len(cands))

    P, R, F1 = score(cands, refs, lang='en', verbose=True)
    print(F1.mean())






def main():
    gts,res = open_json()
    # bleu(gts,res)
    # cider(gts,res)
    # meteor(gts,res)
    # rouge(gts,res)
    # spice(gts,res)
    bert_based(gts,res)

if EVALUATE:

    refs, hyps = evaluate(beam_size)
    create_json(hyps, refs)

main()











