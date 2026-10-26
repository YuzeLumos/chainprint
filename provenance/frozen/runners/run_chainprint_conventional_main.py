"""Prospective T1 conventional rows, using the frozen support-only estimator."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,fcntl,hashlib,json,os,shutil,subprocess,sys,time,traceback
EXPECTED="d1312922c4e42c7179d410676039bd7c7959e41201ad20c8015c7d66f694576c"
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8");tmp.replace(p)
def main():
    ap=argparse.ArgumentParser()
    for key in ("root","manifest","pool","output"):ap.add_argument("--"+key,required=True)
    ap.add_argument("--wait-for-pool",action="store_true");a=ap.parse_args()
    root,pool,out=Path(a.root),Path(a.pool),Path(a.output)
    if out.exists():raise FileExistsError("Never duplicate or overwrite formal units")
    wait_start=time.monotonic()
    while not (pool/"SHARED_POOL_MANIFEST.json").exists():
        if not a.wait_for_pool or (pool/"FAILED.json").exists() or time.monotonic()-wait_start>3700:
            raise RuntimeError("Shared pool unavailable/failed; no duplicate generation")
        time.sleep(10)
    lock=(root/"gpu0_experiment.lock").open("a")
    for attempt in range(20):
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:
            if attempt==19:raise
            time.sleep(5)
    uuid=subprocess.check_output(["nvidia-smi","--id=0","--query-gpu=uuid","--format=csv,noheader"],text=True).strip()
    apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"],text=True)
    if any(line.strip().startswith(uuid+",") for line in apps.splitlines()):raise RuntimeError("GPU0 occupied; do not preempt")
    if sha(a.manifest)!=EXPECTED:raise ValueError("Source design changed")
    if shutil.disk_usage(out.parent).free<2*1024**3:raise OSError("Need2GiB reserve")
    manifest=read(a.manifest);shared=read(pool/"SHARED_POOL_MANIFEST.json")
    if shared["parent_manifest_sha256"]!=EXPECTED:raise ValueError("Pool/design mismatch")
    old=read(read(root/"stage1_protocol.json")["original_protocol"]);repo=Path(old["repo"])
    sys.path[:0]=[str(root/"code"),str(root),*old["helper_roots"],str(repo/"scripts"),str(repo/"src")]
    os.environ.update(CUDA_VISIBLE_DEVICES="0",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",OMP_NUM_THREADS="4",CUBLAS_WORKSPACE_CONFIG=":4096:8")
    import numpy as np,torch
    from PIL import Image
    from gain_block import GainReplay
    from ai_operator import DiffusersInpaintBackend,AIReplay
    from available_inference import MemoryAvailableAIReplay,compose_available,available_distances,ranking_with_availability
    from chainprint_shared_residual import ResidualReplay
    from confirmation_recipe import declared_editor_prior
    from opweave_frozen_20260916 import OpWeaveEstimator,VERSION
    import opweave_frozen_20260916
    from supplement_experiment_data import PairedData
    from run_chainprint_supplement import load_candidate
    from source_resolver import read_prepared_pixels
    from history_recovery.v2_training import configure_determinism
    from history_recovery.data.records import Operation
    from history_recovery.data.operations import apply_operation
    configure_determinism(2026091701);torch.set_num_threads(4)
    entryhash="db76716fa902691025e8536552405c4ad01976f50c5832aadc29a5305653898e"
    if sha(opweave_frozen_20260916.__file__)!=entryhash:raise ValueError("Frozen method changed")
    out.mkdir();started=time.monotonic()
    data=PairedData(read(repo/old["second_protocol"]),0)
    candidate=load_candidate(data.first,0,data.config,data.contract,torch.device("cpu"))
    backend=DiffusersInpaintBackend(cache_dir=str(root/"hf_cache"),local_files_only=True)
    base=GainReplay(torch.device("cpu"));mixed=MemoryAvailableAIReplay(AIReplay(backend,base),capacity=256)
    replay=ResidualReplay(mixed);prior=declared_editor_prior();estimator=OpWeaveEstimator(base,mixed,candidate,prior)
    write(out/"RUN_CONTRACT.json",dict(schema="chainprint-t1-conventional-formal-v2",method="ChainPrint",method_version=VERSION,
          source_manifest_sha256=EXPECTED,shared_pool_sha256=sha(pool/"SHARED_POOL_MANIFEST.json"),
          method_entry_sha256=entryhash,code_sha256=sha(__file__),declared_prior=prior.manifest(),
          source_counts=manifest["source_counts"],condition="conventional",lengths=[2,4,8],K=8,N=16,seed=2026091701,
          evidence="PROSPECTIVE_FORMAL_MAIN_TABLE",global_weights_updated=False,
          cache="Support-column pixel hashes only, no semantic IDs; first fit uses independently shuffled local IDs",
          method_fit_inputs="Anonymous before/after pairs only; unused slots trimmed from public candidate ID union",
          units="Equal source-content groups; do not count lengths/candidates as independent n",
          error_policy="All16 method scores required; incomplete scores count as primary failure, never exclude by method result",
          truth="Frozen uniform stage-eligible history; stable seeded candidate order, tie size retained",
          old_historical3493_index_available=False,no_paper_edits=True,started_utc=datetime.now(timezone.utc).isoformat(),
          previous_diagnostic_results_not_reused=True))
    modelcache={};bankcache={};records=[]
    def fingerprint(before,column):return hashlib.sha256(before.tobytes()+np.ascontiguousarray(column).tobytes()).hexdigest()
    def render(image,operation):
        rgb=np.rint(np.clip(image,0,1)*255).astype(np.uint8).transpose(1,2,0)
        with Image.fromarray(rgb) as im:
            result=apply_operation(im,operation,source_id="t1-private-endpoint-generation",parent_history_id="evaluator-only")
            value=np.asarray(result.image).copy().transpose(2,0,1).astype(np.float32)/255;result.image.close()
        return value
    for design in manifest["designs_evaluator_only"]:
        group,cell=design["group"],design["config_cell"];mapping=design["canonical_to_anonymous_evaluator_only"]
        if cell not in bankcache:
            bankrecord=shared["banks"][cell];path=Path(bankrecord["path"])
            if sha(path)!=bankrecord["sha256"]:raise ValueError("Example pixels changed")
            with np.load(path,allow_pickle=False) as z:bankcache[cell]=(z["before"],z["after"])
        before,canonical_after=bankcache[cell]
        panels=[p for p in design["panels"] if p["condition"]=="conventional"]
        used=sorted({g for p in panels for word in p["method_bank"] for g in word});compact={g:i for i,g in enumerate(used)}
        anonymous_after=np.empty_like(canonical_after)
        for canonical,slot in enumerate(mapping):anonymous_after[:,slot]=canonical_after[:,canonical]
        after=np.ascontiguousarray(anonymous_after[:,used]);keys=[fingerprint(before,after[:,i]) for i in range(len(used))]
        if not all(k in modelcache for k in keys):
            write(out/"STATUS.json",dict(status="FITTING",group=group,cell=cell,completed=len(records),pid=os.getpid()))
            models,fit=estimator.fit(before,after)
            torch.save(models,out/f"cell_{cell}_first_models.pt");write(out/f"cell_{cell}_first_fit.json",fit)
            for key,model in zip(keys,models):modelcache[key]=model
        models=[modelcache[k] for k in keys]
        parameters=shared["banks"][cell]["traditional_parameters_evaluator_only"]
        ops=[Operation(p["code"],p["params"],p.get("seed",2026091701)) for p in parameters]
        for dataset in manifest["dataset_order"]:
            row=manifest["datasets"][dataset][group];source=read_prepared_pixels(row)
            for panel in panels:
                target=source
                for g in panel["true_canonical"]:target=render(target,ops[g])
                words=[[compact[g] for g in word] for word in panel["method_bank"]]
                write(out/"STATUS.json",dict(status="SCORING",dataset=dataset,group=group,span=panel["span"],completed=len(records),pid=os.getpid()))
                tick=time.perf_counter();predictions,_=compose_available(replay,source,models,words)
                losses,available=available_distances(predictions,target);ranking=ranking_with_availability(losses,available,panel["truth_index"])
                complete=bool(np.asarray(available).all());metrics=ranking["metrics"]
                r=dict(method="ChainPrint",dataset=dataset,condition="conventional",span=panel["span"],fold=panel["fold"],
                       seed=2026091701,group=group,source_id=row["source_id"],source_root=row["root"],config_cell=cell,
                       K=8,N=16,candidate_words=panel["method_bank"],truth_index=panel["truth_index"],
                       source_manifest_sha256=EXPECTED,query_seconds=time.perf_counter()-tick,
                       generation_complete=True,method_complete=complete,scores=[-float(v) if bool(ok) else None for v,ok in zip(losses,available)],
                       raw_ranking=ranking,primary_top1=float(metrics["top1"]) if complete else 0.,
                       primary_mrr=float(metrics["mrr"]) if complete else 0.)
                records.append(r)
                with (out/"episode_records.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(r,allow_nan=False)+"\n")
                endpointpath=out/f"query_{dataset}_{group:03d}_L{panel['span']}.npz"
                np.savez_compressed(endpointpath,earlier=source,later=target)
        if group%10==9:write(out/"PROGRESS.json",dict(completed_records=len(records),completed_sources_per_dataset=group+1))
    rows=[]
    for dataset in manifest["dataset_order"]:
        for L in (2,4,8):
            subset=[r for r in records if r["dataset"]==dataset and r["span"]==L]
            if len(subset)!=60:raise ValueError("Incomplete formal matrix")
            rows.append(dict(method="ChainPrint",dataset=dataset,condition="conventional",L=L,K=8,N=16,
                             independent_sources=60,top1_percent=100*np.mean([r["primary_top1"] for r in subset]),
                             mrr=np.mean([r["primary_mrr"] for r in subset]),method_complete=sum(r["method_complete"] for r in subset)))
    status=dict(status="COMPLETE_CONVENTIONAL_FORMAL",rows=rows,episode_count=len(records),
                seconds=time.monotonic()-started,records_sha256=sha(out/"episode_records.jsonl"),pid=os.getpid(),
                finished_utc=datetime.now(timezone.utc).isoformat(),ai_rows="PENDING",other_methods="PENDING")
    write(out/"aggregate.json",status);write(out/"STATUS.json",status);print(json.dumps(status),flush=True)

if __name__=="__main__":
    try:main()
    except Exception as err:
        if "--output" in sys.argv and not isinstance(err,FileExistsError):
            out=Path(sys.argv[sys.argv.index("--output")+1])
            if out.exists():write(out/"FAILED.json",dict(type=type(err).__name__,error=str(err),traceback=traceback.format_exc()))
        raise
