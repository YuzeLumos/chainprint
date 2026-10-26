"""Missing CP-AI/Joint G8 support models only; no query/truth/editor interface."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from formal_efficiency_runtime_v1 import read, sha, write, now
from native_k_support_bindings_v1 import (FIT_KS, METHODS, MAPPING,
    prefix_available, make_joint, validate_record, validate_banks)


def main():
    ap = argparse.ArgumentParser()
    for key in ('protocol', 'output'):
        ap.add_argument('--'+key, type=Path, required=True)
    ap.add_argument('--cpu-lock-fds', type=int, nargs=2, required=True)
    a = ap.parse_args(); p = read(a.protocol)
    b, root, out = Path(p['base']), Path(p['root']), a.output
    if out.exists():
        raise FileExistsError('Never overwrite or duplicate support fits')
    for fd, name in zip(a.cpu_lock_fds, p['cpu_locks']):
        if Path(os.readlink('/proc/self/fd/'+str(fd))) != Path(name):
            raise ValueError('Inherited original CPU lock required')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for q, h in {**p['input_sha256'], **p['code_sha256']}.items():
        if sha(q) != h:
            raise ValueError('Frozen input/code changed '+q)
    out.mkdir(); start = time.monotonic()
    def bounds():
        if time.monotonic()-start > p['maximum_child_seconds']:
            raise TimeoutError('Frozen child24h bound')
        if shutil.disk_usage(b).free < p['min_free_bytes']:
            raise OSError('Preserve original2GiB reserve')
        if sum(q.stat().st_size for q in out.rglob('*') if q.is_file()) > p['output_cap_bytes']:
            raise OSError('Own256MiB supportfit cap')
    bounds()
    old = read(read(root/'stage1_protocol.json')['original_protocol'])
    repo = Path(old['repo']); code = Path(__file__).parent
    sys.path[:0] = [str(code), str(root/'code'), str(root), *old['helper_roots'],
                   str(repo/'scripts'), str(repo/'src')]
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import numpy as np
    import torch
    from confirmation_recipe import declared_editor_prior
    from ai_operator import CheckerRejectedSample
    from variable_k_estimator_v1 import make_estimator
    from run_real_author_k_reference_v1 import load_registered_pools
    from history_recovery.v2_training import configure_determinism
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    prior = declared_editor_prior()
    if prior.manifest()['sha256'] != p['prior_sha256']:
        raise ValueError('Original finite public prior changed')
    before16, banks, masks, pixels = load_registered_pools(b)
    with np.load(b/'shared_support_pool_v2/all_prior_examples_evaluator_only.npz', allow_pickle=False) as z:
        memo_before, memo_after = z['before'].copy(), z['after'].copy()
    if not np.array_equal(before16[:8], memo_before) or memo_after.shape[:2] != (8, 8):
        raise ValueError('Original first8 all-prior support memo mismatch')
    memo = {(hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(), s.fingerprint): memo_after[i,j]
            for i,x in enumerate(memo_before) for j,s in enumerate(prior.specs)}
    unavailable = set()
    extra = read(b/'support16_extra8_prior_assets_v2/EXTRA8_PRIOR_MANIFEST.json')
    for r in extra['requests']:
        i,j = r['support_index'], r['prior_index']
        key = hashlib.sha256(before16[i].tobytes()).hexdigest(), prior.specs[j].fingerprint
        if key != (r['before_pixels_sha256'], r['spec_sha256']):
            raise ValueError('Actual support/prior request not exact pixel/spec match')
        if r['generation_complete']:
            if sha(r['path']) != r['sha256']:
                raise ValueError('Actual extra prior output changed')
            with np.load(r['path'], allow_pickle=False) as z:
                value = z['after'].copy()
            if value.shape != before16[i].shape or not np.isfinite(value).all() or hashlib.sha256(value.tobytes()).hexdigest() != r['after_pixels_sha256']:
                raise ValueError('Invalid actual registered prior output')
            memo[key] = value
        else:
            unavailable.add(key)
    if len(extra['requests']) != 64 or len(unavailable) != 5:
        raise ValueError('Keep all64 real attempts and5 registered refusals')
    class SupportReplay:
        def __init__(self, physical):
            self.base, self.device = physical, physical.device
        def image(self, x):
            return self.base.image(x).double()
        def apply(self, image, op):
            if op[0] != 'ai_inpaint':
                return self.base.apply(image, op)
            key = (hashlib.sha256(np.ascontiguousarray(image.detach().cpu().numpy().astype(np.float32)).tobytes()).hexdigest(), op[1].fingerprint)
            if key in unavailable or (key in memo and not np.isfinite(memo[key]).all()):
                raise CheckerRejectedSample('Original actual support/prior checker refusal, not retried')
            if key not in memo:
                raise ValueError('Exact actual support/prior pixels absent; never generate or substitute')
            return self.image(memo[key])
    write(out/'SUPPORT_PREFIX_PIXEL_BINDING.json', dict(banks=pixels, masks=masks,
        before16_pixels_sha256=hashlib.sha256(before16.tobytes()).hexdigest(),
        actual_extra_outputs=59, registered_extra_refusals=5,
        no_replication_fill_drop_retry_or_new_editor_calls=True))
    write(out/'RUN_CONTRACT.json', dict(protocol=p, protocol_sha256=sha(a.protocol),
        started_utc=now(), pid=os.getpid(), device='cpu', threads=1,
        environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__),
        declared_prior=prior.manifest(), support_prefix_pixel_binding_sha256=sha(out/'SUPPORT_PREFIX_PIXEL_BINDING.json'),
        anonymous_mapping_evaluator_only=list(MAPPING), planned_fit_bank_attempts=84,
        global_neural_weight_updates=0, support_specific_parameter_fits=True,
        queries_truth_histories_or_candidates_seen_by_estimator=False,
        editor_calls=0, K8_new_fit_banks=0, performance_evidence=False,
        no_paper_edits=True, all16_complete=False))
    rows = []; actual_chunks = 0; reused_chunks = 0
    for k in FIT_KS:
        bounds(); configure_determinism(p['support_seed'])
        cp, parameter_hash, cp_audit = make_estimator(root, k, prior, SupportReplay)
        physical = cp.traditional_replay
        joint, joint_audit = make_joint(code, root, k, physical, SupportReplay(physical), prior)
        write(out/f'K{k}_INTERFACE_AUDIT.json', dict(ChainPrint=cp_audit, Joint=joint_audit))
        before = np.ascontiguousarray(before16[:k])
        for method, estimator in zip(METHODS, (cp, joint)):
            cache = {}
            for cell in range(6):
                bounds(); available = prefix_available(masks[cell], k)
                missing = [[i,j] for i in range(k) for j,v in enumerate(masks[cell][i]) if not v]
                row = dict(method=method, K=k, cell=cell, support_dictionary_G=8,
                    support_prefix_indices=list(range(k)), support_prefix_available=available,
                    missing_support_pairs=missing, canonical_to_anonymous_evaluator_only=list(MAPPING),
                    K8_exact_parent_reuse=False, performance_evidence=False)
                if not available:
                    row.update(status='REGISTERED_G8_SUPPORT_PREFIX_INCOMPLETE', model_path=None,
                               model_file_sha256=None, model_hash=None, column_pixel_keys=None)
                    rows.append(row); write(out/'PROGRESS.json',dict(completed_bank_attempts=len(rows), planned=84, banks=rows))
                    continue
                canonical = np.stack([np.stack(r) for r in banks[cell][:k]])
                after = np.empty_like(canonical)
                for c,g in enumerate(MAPPING):
                    after[:,g] = canonical[:,c]
                after = np.ascontiguousarray(after)
                keys = [hashlib.sha256(before.tobytes()+np.ascontiguousarray(after[:,g]).tobytes()).hexdigest() for g in range(8)]
                models = []; stages = []
                for offset in (0,4):
                    bounds(); subset = keys[offset:offset+4]
                    write(out/'STATUS.json',dict(status='ACTUAL_SUPPORT_ONLY_FULL_PRIOR_FIT',
                        K=k, method=method, cell=cell, chunk=offset, completed_bank_attempts=len(rows),
                        planned=84, gpu_used=False, performance_evidence=False, pid=os.getpid(), utc=now()))
                    if all(key in cache for key in subset):
                        fitted = [cache[key][0] for key in subset]
                        record = dict(status='EXACT_SUPPORT_COLUMN_PIXEL_MEMO',
                            column_pixel_keys=subset, source_fit_records=[cache[key][1] for key in subset])
                        reused_chunks += 1
                    else:
                        if method == 'ChainPrint':
                            fitted, record = estimator.fit(before, np.ascontiguousarray(after[:,offset:offset+4]), check_budget=bounds)
                        else:
                            fitted, record = estimator.fit(before, np.ascontiguousarray(after[:,offset:offset+4]))
                        validate_record(method,k,record)
                        if len(fitted) != 4:
                            raise AssertionError('Original independent4+4 batching must retain all8 slots')
                        actual_chunks += 1
                        ref = f'K{k}_{METHODS.index(method)}_cell{cell}_chunk{offset}_fit.json'
                        for key, model in zip(subset, fitted):
                            if key in cache and parameter_hash(cache[key][0]) != parameter_hash(model):
                                raise AssertionError('Same anonymous column fit depends on other slots')
                            cache[key] = model, ref
                    ref = f'K{k}_{METHODS.index(method)}_cell{cell}_chunk{offset}_fit.json'
                    write(out/ref, record); stages.append(dict(path=str(out/ref), sha256=sha(out/ref)))
                    models.extend(fitted); bounds()
                path = out/f'K{k}_{METHODS.index(method)}_cell{cell}_anonymous_models.pt'
                torch.save(models,path)
                row.update(status='FITTED', model_path=str(path), model_file_sha256=sha(path),
                    model_hash=parameter_hash(models), column_pixel_keys=keys,
                    support_pixel_sha256=hashlib.sha256(before.tobytes()+after.tobytes()).hexdigest(),
                    stage_fit_records=stages)
                rows.append(row); write(out/'PROGRESS.json',dict(completed_bank_attempts=len(rows), planned=84, banks=rows))
    validate_banks(rows,masks); bounds()
    for q,h in {**p['input_sha256'], **p['code_sha256']}.items():
        if sha(q) != h:
            raise ValueError('Frozen input/code changed during new support fits '+q)
    result = dict(status='COMPLETE84_NATIVE_K_SUPPORT_BANK_ATTEMPTS_FAILURES_RETAINED',
        banks=rows, masks=masks, protocol_sha256=sha(a.protocol),
        actual_fit_chunks=actual_chunks, exact_pixel_memo_chunks=reused_chunks,
        incomplete_prefix_banks=sum(not r['support_prefix_available'] for r in rows),
        support_specific_parameter_fits=True, global_neural_weight_updates=0,
        query_scores=0, editor_calls=0, gpu_allocations=0, K8_new_fits=0,
        performance_evidence=False, six_method_K_performance_complete=False,
        all16_complete=False, finished_utc=now(), pid=os.getpid())
    write(out/'aggregate.json',result); write(out/'STATUS.json',result)
    print(json.dumps(dict(status=result['status'], actual_fit_chunks=actual_chunks)), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        if '--output' in sys.argv and not isinstance(err, FileExistsError):
            out = Path(sys.argv[sys.argv.index('--output')+1])
            if out.exists():
                write(out/'FAILED.json',dict(error=str(err), type=type(err).__name__,
                    traceback=traceback.format_exc(), pid=os.getpid(), utc=now()))
        raise
