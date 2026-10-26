"""Explicit down/up resampling-class prior, identified from support pixels.

This is a stronger structural prior than generic spatial kernels. Neither
operation identities nor true sizes/filters are accepted as inference inputs.
"""
import numpy as np
from PIL import Image
import torch
from .chainprint_radix_offset import RadixOffsetLikelihood


class ResamplingGeometryLikelihood(RadixOffsetLikelihood):
    filters={'bicubic':Image.Resampling.BICUBIC,'lanczos':Image.Resampling.LANCZOS}

    @staticmethod
    def pil_image(value):
        rgb=np.rint(np.asarray(value).transpose(1,2,0).astype(np.float64)*255).clip(0,255).astype(np.uint8)
        return Image.fromarray(rgb)

    def roundtrip(self,image,size,filter_name):
        kernel=self.filters[filter_name]
        return image.resize(size,resample=kernel).resize(image.size,resample=kernel)

    @torch.no_grad()
    def fit(self,support_before,support_after):
        fitted,diagnostics=super().fit(support_before,support_after)
        image=self.pil_image(support_before)
        w,h=image.size
        yy,xx=np.indices((h,w));fitmask=(yy+xx)%4!=0
        sample=np.linspace(0,h*w-1,min(1024,h*w),dtype=np.int64)
        sample=sample[fitmask.ravel()[sample]]
        if not len(sample):raise ValueError('No resampling fitting pixels')
        targets=np.rint(np.asarray(support_after,dtype=np.float64)*255)
        active=[i for i,d in enumerate(diagnostics) if not d['block_quantization_selected']]
        for d in diagnostics:d['resampling_selected']=False
        if not active:return fitted,diagnostics
        sampled=targets[active].reshape(len(active),3,-1)[:,:,sample]
        best=np.full(len(active),np.inf);parameters=[None]*len(active)
        seen=set()
        # Exhaustive latent image dimensions are fitted model parameters, not
        # development-selected hyperparameters. Fit only prescribed train pixels.
        for name in self.filters:
            for latent in range(1,max(h,w)):
                scale=latent/max(h,w)
                size=(max(1,round(w*scale)),max(1,round(h*scale)))
                if (name,size) in seen:continue
                seen.add((name,size))
                prediction=np.asarray(self.roundtrip(image,size,name)).transpose(2,0,1)
                values=prediction.reshape(3,-1)[:,sample].astype(np.float64)
                errors=np.square(sampled-values[None]).mean((1,2))
                for j in np.flatnonzero(errors<best):
                    best[j]=errors[j];parameters[j]=(name,size)
        for j,i in enumerate(active):
            d=diagnostics[i];name,size=parameters[j]
            prediction=np.asarray(self.roundtrip(image,size,name)).transpose(2,0,1).astype(np.float64)
            error=float(np.square(prediction[:,~fitmask]-targets[i][:,~fitmask]).mean())
            if d['color_response_selected']:baseline=d['color_validation_mse_8bit']
            elif d['selected']=='pointwise':baseline=d['pointwise_validation_mse_8bit']
            elif d['deadzone_selected']:baseline=d['deadzone_validation_mse_8bit']
            else:baseline=d['linear_validation_mse_8bit']
            chosen=error<baseline
            d.update(resampling_selected=chosen,resampling_train_mse_8bit=float(best[j]),
                     resampling_held_mse_8bit=error,resampling_baseline_held_mse_8bit=baseline,
                     resampling_fit_pixels=len(sample))
            if chosen:
                value={'resampling_size':list(size),'resampling_source_size':[w,h],'resampling_filter':name}
                fitted[i]=('spatial',value)
                d.update(selected='spatial',response_class='support_fitted_resampling',**value)
        return fitted,diagnostics

    def apply(self,before,operator):
        kind,value=operator
        if kind=='spatial' and isinstance(value,dict) and 'resampling_size' in value:
            image=self.pil_image(before.detach().cpu().numpy())
            if list(image.size)!=value['resampling_source_size']:
                raise ValueError('Resampling support/query dimensions must match')
            result=np.asarray(self.roundtrip(image,tuple(value['resampling_size']),value['resampling_filter'])).copy()
            return torch.tensor(result.transpose(2,0,1),dtype=before.dtype,device=self.device)/255
        return super().apply(before,operator)

    @staticmethod
    def route_from_support(diagnostics,candidates):
        active={g for word in candidates for g in word}
        return any(diagnostics[g]['block_quantization_selected'] or diagnostics[g].get('resampling_selected',False)
                   for g in active)
