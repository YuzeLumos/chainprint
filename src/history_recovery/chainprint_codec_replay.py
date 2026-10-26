"""Calibrate a declared JPEG codec prior from anonymous pixel pairs only.

This is a stronger codec prior than approximate block quantization. It is not
discovery of an unknown compression algorithm. Names, true parameters, encoded
files and query targets are never accepted by fitting. All local roles undergo
the same fitting and support-only validation.
"""
from io import BytesIO
import numpy as np
from PIL import Image
import torch
from .chainprint_quantized_separable_likelihood import QuantizedSeparableLikelihood


class SupportCodecReplay(QuantizedSeparableLikelihood):
    @staticmethod
    def codec_bytes(image,quality,subsampling):
        with BytesIO() as stream:
            image.save(stream,format='JPEG',quality=int(quality),subsampling=int(subsampling),
                       optimize=False,progressive=False)
            stream.seek(0)
            with Image.open(stream) as decoded:return np.asarray(decoded.convert('RGB')).copy().transpose(2,0,1)

    def infer_codec(self,before,after):
        """Fit on three quarters of source pixels; validate on the remaining quarter."""
        image=self.pil_image(before);targets=np.rint(np.asarray(after,dtype=np.float64)*255)
        _,h,w=targets.shape[1:];yy,xx=np.indices((h,w));train=(yy+xx)%4!=0
        # All hypotheses use identical observed input pixels. Validation output
        # pixels never choose quality or chroma subsampling.
        sampled=np.flatnonzero(train.ravel())
        sampled=sampled[np.linspace(0,len(sampled)-1,min(1024,len(sampled)),dtype=np.int64)]
        observed=targets.reshape(len(targets),3,-1)[:,:,sampled]
        errors=np.full(len(targets),np.inf);params=[None]*len(targets);ties=np.zeros(len(targets),dtype=int)
        for subsampling in (0,1,2):
            for quality in range(1,101):
                pred=self.codec_bytes(image,quality,subsampling)
                values=pred.reshape(3,-1)[:,sampled].astype(np.float64)
                loss=np.square(values[None]-observed).mean((1,2))
                improved=loss<errors;equal=loss==errors
                for i in np.flatnonzero(improved):errors[i]=loss[i];params[i]=(quality,subsampling);ties[i]=1
                ties[equal]+=1
        result=[]
        for i,parameter in enumerate(params):
            prediction=self.codec_bytes(image,*parameter).astype(np.float64)
            train_error=float(np.square(prediction[:,train]-targets[i][:,train]).mean())
            held_error=float(np.square(prediction[:,~train]-targets[i][:,~train]).mean())
            # Exact support byte agreement prevents a low-error approximate
            # JPEG fit from replacing an unrelated operator on smooth images.
            result.append({'quality':parameter[0],'subsampling':parameter[1],
                'train_mse_8bit':train_error,'held_mse_8bit':held_error,'fit_ties':int(ties[i]),
                'accepted':train_error==0.0 and held_error==0.0})
        return result

    @torch.no_grad()
    def refine(self,before,after,fitted,diagnostics):
        fitted=list(fitted);diagnostics=[dict(d) for d in diagnostics]
        for i,estimate in enumerate(self.infer_codec(before,after)):
            diagnostics[i].update(codec_support_fit=estimate,codec_selected=estimate['accepted'])
            if estimate['accepted']:
                fitted[i]=('spatial',{'codec_quality':estimate['quality'],'codec_subsampling':estimate['subsampling']})
                diagnostics[i].update(selected='spatial',response_class='support_fitted_jpeg_codec',block_quantization_selected=True)
        return fitted,diagnostics

    @torch.no_grad()
    def fit(self,support_before,support_after):
        fitted,diagnostics=super().fit(support_before,support_after)
        return self.refine(support_before,support_after,fitted,diagnostics)

    def apply(self,before,operator):
        kind,value=operator
        if kind=='spatial' and isinstance(value,dict) and 'codec_quality' in value:
            image=self.pil_image(before.detach().cpu().numpy())
            result=self.codec_bytes(image,value['codec_quality'],value['codec_subsampling'])
            return torch.as_tensor(result,device=self.device,dtype=before.dtype)/255
        return super().apply(before,operator)
