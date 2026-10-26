"""ChainPrint: exactly enumerated, controlled theory experiment (not image accuracy).

Finite hypotheses are fixed before support, and support eligibility never sees
query endpoints. The two support interventions have equal K, equal cardinality,
equal fit-error spectra and equal global functional diameter.
"""
from __future__ import annotations
import csv
import hashlib
import argparse
from . import support_history_geometry as geometry
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parent / 'generated'

CONFIG = {
 'name': 'ChainPrint controlled support-direction intervention',
 'scope': 'Deterministic real-valued 2D operation family; NOT real-image performance, learned-method accuracy, or continuous-space certification.',
 'operation': 'F_(a,b)(u,v)=((1+a)u+b*v,v)',
 'translation': 'T_d(u,v)=(u+d,v)',
 'true_parameter': [0.2, 0.0],
 'a_grid': [0.05, 0.35, 61], 'b_grid': [-0.15, 0.15, 61],
 'support_inputs': {'constrain_a': [[1,0],[-1,0]], 'constrain_b': [[0,1],[0,-1]]},
 'support_max_abs_scalar_error': 0.01000000001,
 'query_angles_degrees': [0,45,90],
 'translation_levels': [0.05,0.1,0.2,0.4,0.6,0.8,1.0,1.2],
 'shared_F_occurrences': [1,3,5,7],
 'histories': ['F^r then T_d (generating)', 'T_d then F^r'],
 'loss': 'mean squared error over both output coordinates',
 'tie_atol': 1e-12,
 'global_probes': [[1,0],[-1,0],[0,1],[0,-1]],
 'sampling_budgets': [8,16,32,64,128,305],
 'sampling_seeds': list(range(20)),
 'no_statistical_population': 'Cartesian case grid and repeated model subsets are not independent test images; no p-values or population CIs.',
 'reference': 'Candidate 0 is fixed for paired-loss reporting; this label is not used for support eligibility.',
}

def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def F(x, theta):
    x = np.broadcast_to(np.asarray(x, float), (len(theta),2)).copy()
    x[:,0] = (1+theta[:,0])*x[:,0] + theta[:,1]*x[:,1]
    return x

def history(x, theta, d, r, translate_first):
    state = np.broadcast_to(np.asarray(x,float),(len(theta),2)).copy()
    if translate_first: state[:,0] += d
    for _ in range(r): state = F(state,theta)
    if not translate_first: state[:,0] += d
    return state

def main(*, plots=True):
    started = time.perf_counter()
    if ROOT.exists():
        raise FileExistsError('Use a fresh control output directory')
    ROOT.mkdir(parents=True)
    # Written before any execution of the declared experiment.
    (ROOT/'controlled_design.json').write_text(json.dumps(CONFIG, indent=2), encoding='utf-8')
    a,b = np.meshgrid(np.linspace(*CONFIG['a_grid']),np.linspace(*CONFIG['b_grid']),indexing='ij')
    theta = np.column_stack((a.ravel(),b.ravel()))
    truth = np.array([CONFIG['true_parameter']])
    arms = list(CONFIG['support_inputs'])
    masks, fit_spectra, control_rows = [], [], []
    probes = np.array(CONFIG['global_probes'],float)
    for arm in arms:
        inputs = np.array(CONFIG['support_inputs'][arm],float)
        pred = np.stack([F(x,theta) for x in inputs],axis=1)
        target = np.stack([F(x,truth)[0] for x in inputs],axis=0)
        residual = pred-target
        fit = np.sqrt(np.mean(residual**2,axis=(1,2)))
        mask = np.max(np.abs(residual),axis=(1,2)) <= CONFIG['support_max_abs_scalar_error']
        masks.append(mask); fit_spectra.append(np.sort(fit[mask]))
        selected = theta[mask]
        behavior = np.stack([F(x,selected) for x in probes],axis=1)
        pair_dist = np.sqrt(np.mean((behavior[:,None]-behavior[None,:])**2,axis=(2,3)))
        control_rows.append(dict(arm=arm,K=len(inputs),global_models=len(theta),compatible_models=int(mask.sum()),
          support_error_min=float(fit[mask].min()),support_error_max=float(fit[mask].max()),
          global_behavior_diameter=float(pair_dist.max())))
    masks = np.stack(masks)
    assert masks.sum(axis=1).tolist() == [305,305]
    assert np.allclose(fit_spectra[0],fit_spectra[1],atol=1e-15)
    assert abs(control_rows[0]['global_behavior_diameter']-control_rows[1]['global_behavior_diameter']) < 1e-15
    cases, summaries, raw_losses, raw_predictions, raw_endpoints = [], [], [], [], []
    counterexamples, sampling = [], []
    for angle in CONFIG['query_angles_degrees']:
      query = np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))])
      query[np.abs(query)<1e-14]=0.
      for r in CONFIG['shared_F_occurrences']:
       for d in CONFIG['translation_levels']:
        cid=len(cases)
        outputs=np.stack([history(query,theta,d,r,first) for first in (False,True)],axis=1)
        true_outputs=np.stack([history(query,truth,d,r,first)[0] for first in (False,True)])
        y=true_outputs[0]
        loss=np.mean((outputs-y)**2,axis=2)
        separation=float(np.sqrt(np.mean((true_outputs[1]-y)**2)))
        case=dict(case_id=cid,query_angle_deg=angle,query_u=float(query[0]),query_v=float(query[1]),F_repeats=r,
                  chain_length=r+1,translation=d,candidate_count=2,true_candidate_rmse_gap=separation)
        cases.append(case); raw_losses.append(loss); raw_predictions.append(outputs);raw_endpoints.append(y)
        for arm_idx,arm in enumerate(arms):
          ids=np.flatnonzero(masks[arm_idx]); selected=loss[ids]
          diagnostics=geometry.finite_pool_comparison(loss,0,masks[arm_idx],tie_atol=CONFIG['tie_atol'])
          margin=selected[:,1]-selected[:,0]
          correct=margin>CONFIG['tie_atol'];wrong=margin < -CONFIG['tie_atol'];tie=~(correct|wrong)
          row={**case,'arm':arm,'compatible_models':len(ids),'K':2,
             'global_behavior_diameter':control_rows[arm_idx]['global_behavior_diameter'],
             'min_margin':float(margin.min()),'max_margin':float(margin.max()),
             'correct_models':int(correct.sum()),'wrong_models':int(wrong.sum()),'tied_models':int(tie.sum()),
             'fraction_models_correct':float(correct.mean()),
             'stable_correct_finite_pool':int(diagnostics['stable_in_evaluated_pool']),
             'candidate0_prediction_diameter':float(np.sqrt(np.mean((outputs[ids,0].max(0)-outputs[ids,0].min(0))**2))),
             'candidate1_prediction_diameter':float(np.sqrt(np.mean((outputs[ids,1].max(0)-outputs[ids,1].min(0))**2))),
          }
          summaries.append(row)
          if wrong.any():
            mid=int(ids[np.argmin(margin)])
            counterexamples.append({**case,'arm':arm,'model_id':mid,'a':float(theta[mid,0]),'b':float(theta[mid,1]),
              'loss_generating':float(loss[mid,0]),'loss_reversed':float(loss[mid,1]),'margin':float(margin.min())})
          for seed in CONFIG['sampling_seeds']:
            perm=np.random.default_rng(seed).permutation(len(ids))
            for budget in CONFIG['sampling_budgets']:
              sub=margin[perm[:budget]]
              sc=bool(np.all(sub>CONFIG['tie_atol']))
              sampling.append(dict(case_id=cid,arm=arm,seed=seed,model_budget=budget,
                stable_correct_sampled=int(sc),stable_correct_full=row['stable_correct_finite_pool'],
                false_consensus=int(sc and not row['stable_correct_finite_pool']),
                full_pool_has_reversal=int(wrong.any()),full_pool_has_tie=int(tie.any()),
                minimum_sampled_margin=float(sub.min())))
    # Explicit paired checks concern design balance, not a preselected outcome.
    for i in range(0,len(summaries),2):
      for key in ('K','compatible_models','true_candidate_rmse_gap','candidate_count','chain_length'):
        assert summaries[i][key] == summaries[i+1][key]
    np.savez_compressed(ROOT/'raw_finite_enumeration.npz',theta=theta,eligible=masks,
       losses=np.stack(raw_losses),predicted_endpoints=np.stack(raw_predictions),observed_endpoints=np.stack(raw_endpoints),
       support_rmse_spectra=np.stack(fit_spectra),arm_names=np.array(arms))
    write_csv(ROOT/'cases.csv',cases);write_csv(ROOT/'summary.csv',summaries)
    write_csv(ROOT/'design_balance.csv',control_rows);write_csv(ROOT/'counterexamples.csv',counterexamples)
    write_csv(ROOT/'sampling_sensitivity.csv',sampling)
    if plots:
        make_plot(summaries)
    aggregate={}
    for angle in CONFIG['query_angles_degrees']:
      aggregate[str(angle)]={arm:dict(stable_cases=sum(x['stable_correct_finite_pool'] for x in summaries if x['query_angle_deg']==angle and x['arm']==arm),
           total_cases=sum(1 for x in summaries if x['query_angle_deg']==angle and x['arm']==arm)) for arm in arms}
    false_by_budget={str(b):sum(x['false_consensus'] for x in sampling if x['model_budget']==b) for b in CONFIG['sampling_budgets']}
    missed_reversals={str(b):sum(x['false_consensus']*x['full_pool_has_reversal'] for x in sampling if x['model_budget']==b) for b in CONFIG['sampling_budgets']}
    result=dict(case_pairs=len(cases),summary_rows=len(summaries),enumerated_global_models=len(theta),
        compatible_models_per_arm=305,design_balance_passed=True,aggregate_by_query_direction=aggregate,
        sampled_false_consensus_counts=false_by_budget,sampling_rows_per_budget=len(cases)*len(arms)*len(CONFIG['sampling_seeds']),
        sampled_consensus_with_missed_reversal_counts=missed_reversals,
        seconds=time.perf_counter()-started,scope=CONFIG['scope'])
    (ROOT/'run_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.iterdir() if p.is_file() and p.name!='manifest.json'}
    (ROOT/'manifest.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

def make_plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(10.5,2.9),layout='constrained')
    colors={'constrain_a':'#2D57A1','constrain_b':'#D98B51'}
    # These two panels intentionally have identical y scales and zero reference.
    for ax,angle in zip(axes[:2],[90,0]):
      for arm in ('constrain_a','constrain_b'):
        subset=[x for x in rows if x['query_angle_deg']==angle and x['F_repeats']==1 and x['arm']==arm]
        ax.plot([x['translation'] for x in subset],[x['fraction_models_correct']*100 for x in subset],
          marker='o' if arm=='constrain_a' else 's',markersize=3.5,color=colors[arm],label='Examples along '+('u' if arm=='constrain_a' else 'v'))
      ax.set(xlabel='Translation d (controls separation)',ylabel='Correct hypotheses (%)',ylim=(-2,102),title=f'({"a" if angle==90 else "b"}) Query along {"v" if angle==90 else "u"}')
      ax.grid(axis='y',alpha=.15)
    subset=[x for x in rows if x['query_angle_deg']==90 and x['translation']==.6]
    for arm in ('constrain_a','constrain_b'):
      data=[x for x in subset if x['arm']==arm]
      x=np.array([r['chain_length'] for r in data],float)+(0 if arm=='constrain_a' else .16)
      lo=np.array([r['min_margin'] for r in data]);hi=np.array([r['max_margin'] for r in data])
      axes[2].vlines(x,lo,hi,color=colors[arm],linewidth=2)
      axes[2].scatter(x,lo,color=colors[arm],s=14,marker='o' if arm=='constrain_a' else 's')
    axes[2].axhline(0,color='#333333',linewidth=.8,linestyle='--')
    axes[2].set(xlabel='History length (shared repeated F)',ylabel='Rival loss minus generating loss',title='(c) All-hypothesis margin range',xticks=[2,4,6,8])
    axes[0].legend(frameon=False,fontsize=7,loc='lower right')
    fig.suptitle('Controlled 2D operation experiment: equal examples, 305 hypotheses, identical paired candidates',fontsize=9)
    fig.savefig(ROOT/'controlled_direction_preview.png',dpi=220)
    fig.savefig(ROOT/'controlled_direction_preview.svg')
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT,
                        help='Output directory (default: generated beside this script)')
    args = parser.parse_args()
    ROOT = args.output.resolve()
    main()

