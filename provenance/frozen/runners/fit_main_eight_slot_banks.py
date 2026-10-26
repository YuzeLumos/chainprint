"""Support-only full eight-slot fits and one necessary actual-model invariance check."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,fcntl,hashlib,json,os,shutil,sys,time,traceback
EXPECTED='d1312922c4e42c7179d410676039bd7c7959e41201ad20c8015c7d66f694576c'
ENTRY='db76716fa902691025e8536552405c4ad01976f50c5832aadc29a5305653898e'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def main():
    ap=argparse.ArgumentParser()
    for k in ('root','manifest','pool','output'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();root,pool,out=map(Path,(a.root,a.pool,a.output))
    if out.exists():raise FileExistsError('Never overwrite support fits')
    if sha(a.manifest)!=EXPECTED:raise ValueError('Frozen source design changed')
    if shutil.disk_usage(out.parent).free<2*1024**3:raise OSError('Need2GiB reserve')
    lock=(out.parent/'main_eight_slot_cpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=read(read(root/'stage1_protocol.json')['original_protocol']);repo=Path(old['repo'])
    sys.path[:0]=[str(root/'code'),str(root),*old['helper_roots'],str(repo/'scripts'),str(repo/'src')]
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import numpy as np,torch
    from gain_block import GainReplay
    from confirmation_recipe import declared_editor_prior
    from opweave_frozen_20260916 import OpWeaveEstimator,VERSION
    import opweave_frozen_20260916
    from final_local_extension import final_parameter_hash
    from supplement_experiment_data import PairedData
    from run_chainprint_supplement import load_candidate
    from history_recovery.v2_training import configure_determinism
    configure_determinism(2026091701);torch.set_num_threads(4)
    if sha(opweave_frozen_20260916.__file__)!=ENTRY:raise ValueError('Frozen scientific method changed')
    shared,manifest=read(pool/'SHARED_POOL_MANIFEST.json'),read(a.manifest)
    if shared['parent_manifest_sha256']!=EXPECTED:raise ValueError('Pool mismatch')
    prior=declared_editor_prior();base=GainReplay(torch.device('cpu'))
    with np.load(pool/'all_prior_examples_evaluator_only.npz',allow_pickle=False) as z:inputs,outputs=z['before'],z['after']
    memo={(hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),s.fingerprint):outputs[k,j]
        for k,x in enumerate(inputs) for j,s in enumerate(prior.specs)}
    class SupportReplay:
        device=base.device
        def image(self,x):return base.image(x).to(dtype=torch.float64)
        def apply(self,image,op):
            if op[0]!='ai_inpaint':return base.apply(image,op)
            key=(hashlib.sha256(np.ascontiguousarray(image.detach().cpu().numpy().astype(np.float32)).tobytes()).hexdigest(),op[1].fingerprint)
            if key not in memo:raise ValueError('Only actually computed exact support/prior outputs may be memoized')
            return self.image(memo[key])
    data=PairedData(read(repo/old['second_protocol']),0)
    candidate=load_candidate(data.first,0,data.config,data.contract,torch.device('cpu'))
    estimator=OpWeaveEstimator(base,SupportReplay(),candidate,prior)
    out.mkdir();started=time.monotonic()
    write(out/'RUN_CONTRACT.json',dict(schema='chainprint-eight-anonymous-slot-main-fits-v1',method_version=VERSION,
        method_entry_sha256=ENTRY,source_manifest_sha256=EXPECTED,shared_pool_sha256=sha(pool/'SHARED_POOL_MANIFEST.json'),
        support_prior_memo_sha256=sha(pool/'all_prior_examples_evaluator_only.npz'),declared_prior=prior.manifest(),
        code_sha256=sha(__file__),K=8,local_ids=8,seed=2026091701,global_weight_updates=0,
        inputs='Only anonymous support pixels; every complete editor prior at every slot',
        batching='Contiguous four+four after frozen independent anonymous permutation; same frozen estimator, pixel-keyed support memo',
        invariance_check='First actual eight-slot fit versus fixed cross-batch permutation; exact restored parameter hashes required before formal scoring',
        timing='Fit records retained; cached editor calls and invariance overhead separated; not formal Table3 setup time',
        started_utc=datetime.now(timezone.utc).isoformat(),no_paper_edits=True))
    def fit8(before,after):
        models=[];records=[]
        for start in (0,4):
            fitted,record=estimator.fit(before,np.ascontiguousarray(after[:,start:start+4]))
            models.extend(fitted);records.append(record)
        return models,records
    bankrecords=[];cache={}
    for cell in range(6):
        design=next(d for d in manifest['designs_evaluator_only'] if d['config_cell']==cell)
        mapping=design['canonical_to_anonymous_evaluator_only'];bank=shared['banks'][cell];path=Path(bank['path'])
        if sha(path)!=bank['sha256']:raise ValueError('Example pixels changed')
        with np.load(path,allow_pickle=False) as z:before,canonical=z['before'],z['after']
        after=np.empty_like(canonical)
        for c,g in enumerate(mapping):after[:,g]=canonical[:,c]
        keys=[hashlib.sha256(before.tobytes()+np.ascontiguousarray(after[:,g]).tobytes()).hexdigest() for g in range(8)]
        models=[];records=[]
        for start in (0,4):
            subset=keys[start:start+4]
            if all(k in cache for k in subset):fitted=[cache[k] for k in subset];record=dict(status='EXACT_SUPPORT_PIXEL_MEMO',hashes=subset)
            else:
                write(out/'STATUS.json',dict(status='FROZEN_FOUR_SLOT_REFITTING',cell=cell,start=start,pid=os.getpid(),gpu_used=False))
                fitted,record=estimator.fit(before,np.ascontiguousarray(after[:,start:start+4]))
                for k,m in zip(subset,fitted):cache[k]=m
            models.extend(fitted);records.append(record)
        if cell==0:
            write(out/'STATUS.json',dict(status='ACTUAL_CROSS_BATCH_PERMUTATION_CHECK',cell=cell,pid=os.getpid(),gpu_used=False))
            perm=[4,0,5,1,6,2,7,3];alternate,alternate_records=fit8(before,np.ascontiguousarray(after[:,perm]))
            restored=[alternate[perm.index(g)] for g in range(8)]
            exact=final_parameter_hash(restored)==final_parameter_hash(models)
            write(out/'SLOT_INVARIANCE.json',dict(status='PASS' if exact else 'FAIL',permutation=perm,
                original_hash=final_parameter_hash(models),restored_hash=final_parameter_hash(restored),
                original_records=records,alternate_records=alternate_records,performance_evidence=False))
            if not exact:raise AssertionError('Actual anonymous regrouping changed operator parameters; do not score formal eight-slot episodes')
        modelpath=out/f'cell_{cell}_anonymous_models.pt';torch.save(models,modelpath)
        write(out/f'cell_{cell}_fit.json',dict(cell=cell,local_mapping_evaluator_only=mapping,stages=records,model_hash=final_parameter_hash(models)))
        bankrecords.append(dict(cell=cell,model_path=str(modelpath),model_file_sha256=sha(modelpath),
            mapping_evaluator_only=mapping,column_pixel_keys=keys,model_hash=final_parameter_hash(models)))
        write(out/'PROGRESS.json',dict(completed_banks=cell+1,pid=os.getpid()))
    status=dict(status='COMPLETE_FROZEN_EIGHT_SLOT_FITS',banks=bankrecords,invariance='PASS',
        source_manifest_sha256=EXPECTED,shared_pool_sha256=sha(pool/'SHARED_POOL_MANIFEST.json'),
        seconds=time.monotonic()-started,pid=os.getpid(),finished_utc=datetime.now(timezone.utc).isoformat())
    write(out/'aggregate.json',status);write(out/'STATUS.json',status);print(json.dumps(status),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as err:
        if '--output' in sys.argv and not isinstance(err,FileExistsError):
            out=Path(sys.argv[sys.argv.index('--output')+1])
            if out.exists():write(out/'FAILED.json',dict(type=type(err).__name__,error=str(err),traceback=traceback.format_exc()))
        raise
