"""Anonymous traditional/AI estimation with a common candidate library at every slot."""
import numpy as np
import torch


class MixedEstimator:
    def __init__(self, method, traditional_replay, mixed_replay, candidate, prior):
        if method not in ('pooled', 'pooled_corrected', 'joint', 'joint_corrected'):
            raise ValueError('Unknown frozen default method')
        self.method, self.base, self.replay = method, traditional_replay, mixed_replay
        self.candidate, self.prior = candidate, prior

    def fit(self, before, after):
        from joint_fit import _fit_joint, FAMILIES, _residual_fit
        from ai_operator import fit_ai_operators, CheckerRejectedSample
        from chainprint_shared_residual import features, corrected
        if len(before) != 8 or len(after) != 8:
            raise ValueError('This confirmation requires eight independent support contents')
        # Frozen traditional fits use float64 coefficients. Preserve that image
        # convention when mixed supports originate from the float32 resolver.
        images = [self.base.image(x) for x in before]
        targets = [[self.base.image(y) for y in row] for row in after]
        g_count = len(targets[0])
        _, ai_records = fit_ai_operators(self.replay, before, after, self.prior)
        ai_ops = [('ai_inpaint', spec) for spec in self.prior.specs]
        ai_losses = [np.asarray([[0.0 if v is None else v for v in row]
                                 for row in r['support_mse_8bit']],dtype=np.float64) for r in ai_records]
        ai_available = [np.asarray(r['support_available'],dtype=bool) for r in ai_records]
        traditional_candidates, traditional_losses = None, None
        if self.method.startswith('pooled'):
            prepared = self.candidate.prepare_support(support_before=before, support_after=after)
            fits, diagnostics = [], []
            for k in range(8):
                f, d = self.base.refine(before[k], after[k], prepared.fits[k], prepared.diagnostics[k])
                fits.append(f); diagnostics.append(d)
            traditional_candidates = [[fits[k][g] for k in range(8)] + [
                ('spatial', {'codec_quality': diagnostics[k][g]['codec_support_fit']['quality'],
                             'codec_subsampling': diagnostics[k][g]['codec_support_fit']['subsampling']})
                for k in range(8)] for g in range(g_count)]
            with torch.no_grad():
                traditional_losses = [np.asarray([[float(((self.base.apply(x, op).double()-targets[k][g])*255).square().mean())
                                                   for k,x in enumerate(images)] for op in proposals])
                                      for g, proposals in enumerate(traditional_candidates)]

        def select(include):
            # Joint parameters and family are refitted from exactly include.
            if self.method.startswith('joint'):
                fitted, details = _fit_joint(self.base, [before[k] for k in include], [after[k] for k in include], FAMILIES)
            chosen, records = [], []
            for g in range(g_count):
                # Availability on a held-out content must not select its model.
                eligible_ai = [j for j in range(len(ai_ops)) if bool(ai_available[g][j,include].all())]
                valid_ai_ops = [ai_ops[j] for j in eligible_ai]
                valid_ai_losses = ai_losses[g][np.ix_(eligible_ai,include)].mean(1).tolist()
                if self.method.startswith('joint'):
                    proposals = [fitted[g]] + valid_ai_ops
                    base_loss = min(details[g]['mean_support_mse_8bit'])
                    losses = [base_loss] + valid_ai_losses
                    origins = [include] + [[] for _ in valid_ai_ops]
                else:
                    eligible = [j for j in range(16) if j % 8 in include]
                    proposals = [traditional_candidates[g][j] for j in eligible] + valid_ai_ops
                    losses = traditional_losses[g][np.ix_(eligible, include)].mean(1).tolist() + valid_ai_losses
                    origins = [[j % 8] for j in eligible] + [[] for _ in valid_ai_ops]
                index = int(np.argmin(losses))
                chosen.append(proposals[index])
                records.append(dict(selected=index, kind=proposals[index][0], mean_support_mse_8bit=losses,
                                    proposal_source_contents=origins, fitting_contents=include,
                                    eligible_ai_prior_indices=eligible_ai))
            return chosen, records

        models, selection = select(list(range(8)))
        validation = []
        if self.method.endswith('corrected'):
            baseline, adjusted = [[] for _ in models], [[] for _ in models]
            fold_choices = []
            for held in range(8):
                include = [k for k in range(8) if k != held]
                fold_models, fold_record = select(include)
                fold_choices.append(fold_record)
                for g, model in enumerate(fold_models):
                    with torch.no_grad():
                        coefficient = _residual_fit(self.replay, [images[k] for k in include], [targets[k][g] for k in include], model)
                        try:
                            prediction = self.replay.apply(images[held], model)
                        except CheckerRejectedSample:
                            baseline[g].append(None); adjusted[g].append(None)
                            continue
                        fixed = corrected(prediction, features(images[held], prediction), coefficient)
                        baseline[g].append(float(((prediction.double()-targets[held][g])*255).square().mean()))
                        adjusted[g].append(float(((fixed.double()-targets[held][g])*255).square().mean()))
            for g, model in enumerate(models):
                evaluable = [value is not None for value in baseline[g]]
                base = np.asarray([v if v is not None else 0.0 for v in baseline[g]])
                new = np.asarray([v if v is not None else 0.0 for v in adjusted[g]])
                adopt = bool(all(evaluable) and np.all(new <= base) and new.mean() < base.mean())
                if adopt:
                    coefficient = _residual_fit(self.replay, images, [t[g] for t in targets], model)
                    models[g] = ('shared_residual', (model, coefficient))
                validation.append(dict(adopted=adopt, held_baseline_mse_8bit=baseline[g],
                                       held_corrected_mse_8bit=adjusted[g],held_available=evaluable,
                                       fold_selection=[r[g] for r in fold_choices]))
        return models, dict(method=self.method, selection=selection, correction=validation,
                            prior=self.prior.manifest(), ai_support_records=ai_records,
                            information_boundary='Only eight paired anonymous examples and declared common prior; no renderer labels, query endpoints, or truth chains. Each fold uses availability on included columns only.')
