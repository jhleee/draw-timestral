import torch
from timestral import TimestralForecaster
from pathlib import Path
torch.set_num_threads(4)
torch.onnx.register_custom_op_symbolic('aten::asinh',lambda g,x:g.op('Asinh',x).setType(x.type()),18)
torch.onnx.register_custom_op_symbolic('aten::sinh',lambda g,x:g.op('Sinh',x).setType(x.type()),18)
from torch.onnx import symbolic_helper
@symbolic_helper.parse_args('v','i')
def cummax_symbolic(g,x,dim):
 sizes=x.type().sizes();dim=len(sizes)-1
 print('CUMMAX_SYMBOLIC',sizes,dim,x,flush=True)
 n=sizes[dim]
 if n is None: raise RuntimeError('cummax requires fixed dimension')
 vals=[]
 def const(v):return g.op('Constant',value_t=torch.tensor(v,dtype=torch.int64))
 for i in range(n):
  part=g.op('Slice',g.op('Cast',x,to_i=6),const([0]),const([i+1]),const([dim]))
  vals.append(g.op('ReduceMax',part,const([dim]),keepdims_i=1))
 out=g.op('Cast',g.op('Concat',*vals,axis_i=dim),to_i=9)
 out.setType(x.type())
 return out,g.op('ConstantOfShape',g.op('Shape',out),value_t=torch.tensor([0],dtype=torch.int64))
torch.onnx.register_custom_op_symbolic('aten::cummax',cummax_symbolic,18)
def repeat_symbolic(g,x,repeats,dim=None,output_size=None):
 def const(v):return g.op('Constant',value_t=torch.tensor(v,dtype=torch.int64))
 if symbolic_helper._is_none(dim):
  x=g.op('Reshape',x,const([-1]));axis=0
 else:axis=symbolic_helper._parse_arg(dim,'i')
 size=g.op('Gather',g.op('Shape',x),const(axis),axis_i=0)
 repeats=g.op('Cast',repeats,to_i=7)
 indices=g.op('Div',g.op('Range',const(0),g.op('Mul',size,repeats),const(1)),repeats)
 out=g.op('Gather',x,indices,axis_i=axis)
 rank=symbolic_helper._get_tensor_rank(x)
 if rank:out.setType(x.type().with_sizes([None]*rank))
 return out
torch.onnx.register_custom_op_symbolic('aten::repeat_interleave',repeat_symbolic,18)
from torch.onnx import symbolic_opset9
@symbolic_helper.parse_args('v','i','i','i')
def unfold_symbolic(g,x,dim,size,step):
 rank=symbolic_helper._get_tensor_rank(x)
 if dim<0:dim+=rank
 return symbolic_opset9.unfold(g,x,dim,size,step)
torch.onnx.register_custom_op_symbolic('aten::unfold',unfold_symbolic,18)
f=TimestralForecaster.from_pretrained('mistralai/Timestral-5M',device='cpu',revision='bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f')
class Wrapper(torch.nn.Module):
 def __init__(self):
  super().__init__();self.model=f.model
 def forward(self,x):
  p=f.forecast(x,prediction_length=64)
  return p.quantile(.1),p.median,p.quantile(.9)
w=Wrapper().eval()
torch.onnx.export(w,torch.linspace(.1,.9,64).reshape(1,1,64),'browser-model/timestral.onnx',input_names=['context'],output_names=['q10','median','q90'],opset_version=18,do_constant_folding=False,dynamo=False)
print('EXPORTED',Path('browser-model/timestral.onnx').stat().st_size)
