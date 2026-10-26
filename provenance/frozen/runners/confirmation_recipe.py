"""Outcome-independent recipe for 60 content groups; metadata, never fitting input."""
from dataclasses import replace
from itertools import product
import hashlib
import json
import numpy as np

SPANS = (1, 2, 4, 6, 8)
JPEG_QUALITIES = (95, 90, 85, 80, 75, 70, 65, 60)
SEED = 2026091501


def declared_editor_prior():
    from ai_operator import InpaintSpec, DeclaredPrior, box_mask
    return DeclaredPrior(tuple(InpaintSpec(prompt, box_mask(*box, feather_pixels=8), seed=seed,
                                           num_inference_steps=25)
                               for prompt, box, seed in product(
                                   ('a patch of red flowers', 'a small blue ceramic vase'),
                                   ((48,64,144,176), (96,80,208,208)), (112,317))),
                         declaration_id='opweave-confirmation-editor-prior-20260915-v1')


def recipe(index, schedules, condition, seed=SEED):
    from history_recovery.data.records import Operation
    from evaluate_chainprint_stages import select_all_families
    from evaluation import make_bank
    if condition not in ('traditional', 'ai_mixed'):
        raise ValueError('Unknown condition')
    rng = np.random.default_rng(seed + index)
    available = select_all_families(schedules, seed, 30000+index, 'J')
    jpeg = replace(next(op for op in available if op.code == 'J'),
                   params={'quality':76, 'subsampling':'4:4:4'})
    other = next(op for op in available if op.code != 'J')
    operations = [('traditional', jpeg), ('traditional', other)]
    prior = declared_editor_prior()
    if condition == 'traditional':
        operations += [('traditional', Operation('R', {'scale':scale,'interpolation':'bicubic','restore_size':True}, seed))
                       for scale in (0.625, 0.75)]
    else:
        # Both configurations share prompt/mask and differ in fixed generation seed.
        pair = 2 * (index % 4)
        operations += [('ai_inpaint', prior.specs[pair]), ('ai_inpaint', prior.specs[pair+1])]
    order = rng.permutation(4).tolist()
    inverse = {old:new for new,old in enumerate(order)}
    operations = [operations[k] for k in order]
    # Eight actual operations, with order-sensitive alternatives and operator reuse.
    word = [inverse[g] for g in (0,2,1,3,0,2,1,2)]
    spans = []
    for length in SPANS:
        truth = word[-length:]
        count = 4 if length == 1 else 16
        bank, truth_id = make_bank(truth, count, 4, seed+index*100+length)
        # Guarantee a same-category/different-configuration competitor.
        competitor = list(truth); competitor[-1] = inverse[3]
        if competitor not in bank:
            replace_id = next(j for j in range(len(bank)-1,-1,-1) if j != truth_id)
            bank[replace_id] = competitor
        spans.append(dict(length=length, source_step=8-length, bank=bank, truth=truth_id,
                          same_category_competitor=bank.index(competitor)))
    manifest = dict(group=index, condition=condition, true_word=word,
                    operations=[{'kind':k,'configuration':op.to_dict() if k=='traditional' else op.manifest()}
                                for k,op in operations], spans=spans,
                    information_boundary='Renderer/evaluator audit only; estimator receives paired pixels',
                    prior_sha256=prior.manifest()['sha256'])
    manifest['sha256'] = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    return operations, manifest
