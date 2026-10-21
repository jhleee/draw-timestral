import hashlib,json,struct,shutil,base64
from pathlib import Path
import onnx
from timestral.loading import resolve_checkpoint
p=resolve_checkpoint('mistralai/Timestral-5M',local_files_only=True)
b=(p/'model.safetensors').read_bytes();hsize=struct.unpack('<Q',b[:8])[0];head=json.loads(b[8:8+hsize]);start=8+hsize
lookup={}
for k,v in head.items():
 if k=='__metadata__':continue
 a,z=v['data_offsets'];raw=b[start+a:start+z];lookup[hashlib.sha256(raw).digest()]=(start+a,len(raw),k)
m=onnx.load('browser-model/timestral.onnx');n=0
for t in m.graph.initializer:
 if not t.raw_data:continue
 match=lookup.get(hashlib.sha256(t.raw_data).digest())
 if not match:continue
 offset,length,key=match
 t.ClearField('raw_data');t.data_location=onnx.TensorProto.EXTERNAL
 for k,v in [('location','official-model.safetensors'),('offset',str(offset)),('length',str(length))]:
  e=t.external_data.add();e.key=k;e.value=v
 n+=length
onnx.save(m,'browser-model/browser-graph.onnx')
shutil.copyfile(p/'model.safetensors','browser-model/official-model.safetensors')
Path('browser-model/browser-graph.b64').write_text(base64.b64encode(Path('browser-model/browser-graph.onnx').read_bytes()).decode())
Path('browser-model/weights-info.json').write_text(json.dumps({'revision':p.name,'url':f'https://huggingface.co/mistralai/Timestral-5M/resolve/{p.name}/model.safetensors','sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'externalized_bytes':n,'graph_bytes':Path('browser-model/browser-graph.onnx').stat().st_size},indent=2))
print(Path('browser-model/weights-info.json').read_text())
