import json,time
from pathlib import Path
import numpy as np,torch,onnxruntime as ort
from timestral import TimestralForecaster
torch.set_num_threads(4)
m=TimestralForecaster.from_pretrained('mistralai/Timestral-5M',local_files_only=True)
s=ort.InferenceSession('browser-model/timestral.onnx',providers=['CPUExecutionProvider'])
print('INPUTS',[(x.name,x.shape) for x in s.get_inputs()],flush=True)
curves=json.loads(Path('browser-model/fixtures.json').read_text())['curves']
contexts=[(x['name'],x['request']['context']) for x in curves]+[('constant',[.5]*64),('zero',[0]*64),('one',[1]*64),('noise',np.random.default_rng(42).uniform(0,1,64).tolist())]
r=[]
for name,context in contexts:
 x=np.asarray(context,dtype=np.float32).reshape(1,1,64)
 f=m.forecast(x,prediction_length=64)
 actual=s.run(None,{'context':x})
 expected=[f.quantile(.1).numpy(),f.median.numpy(),f.quantile(.9).numpy()]
 diff=max(float(np.max(np.abs(a-b))) for a,b in zip(actual,expected))
 print(name,diff,flush=True)
 assert all(np.allclose(a,b,rtol=2e-4,atol=2e-5) for a,b in zip(actual,expected)),name
 r.append({'name':name,'max_abs_diff':diff})
Path('browser-model/equivalence.json').write_text(json.dumps(r,indent=2))
