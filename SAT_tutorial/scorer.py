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
out_file = open("evaluation_results.txt", "w")


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

    out_file.write('BLEU(1-4) = %s' % score + '\n')

    return score

def cider(gts,res):
    scorer = Cider()
    # scorer += (hypo[0], ref1)
    (score, scores) = scorer.compute_score(gts, res)
    out_file.write('CIDEr = %s' % score + '\n')

def meteor(gts,res):
    scorer = Meteor()
    score, scores = scorer.compute_score(gts, res)
    out_file.write('METEOR = %s' % score + '\n')
    return score

def rouge(gts,res):
    scorer = Rouge()
    score, scores = scorer.compute_score(gts, res)
    out_file.write('ROUGE = %s' % score + '\n')
    return score

def spice(gts, res):
    scorer = Spice()
    score, scores = scorer.compute_score(gts, res)
    out_file.write('SPICE = %s' % score + '\n')
    return score

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
    out_file.write('BERTScore = %s' % F1.mean().item() + "\n")
    BERTScore = F1.mean().item()

    refs_bleurt = ''
    refs_bleurt_list =[]
    for ref in refs:
        for ref_cap in ref:
            refs_bleurt_list.append(ref_cap)

    cands = [x for item in cands for x in repeat(item,5)]

    scorer = bleurt_sc.BleurtScorer(bleurt_checkpoint)

    scores = scorer.score(refs_bleurt_list, cands, batch_size=None)
    assert type(scores) == list
    out_file.write('BLEURT = %s' % statistics.mean(scores))
    BLEURT = statistics.mean(scores)
    return BERTScore, BLEURT

def main():
    gts,res = open_json()
    bleu(gts,res)
    cider(gts,res)
    meteor(gts,res)
    rouge(gts,res)
    spice(gts,res)
    bert_based(gts,res)
    out_file.close()

if EVALUATE:

    refs, hyps = evaluate(beam_size)
    create_json(hyps, refs)

main()











