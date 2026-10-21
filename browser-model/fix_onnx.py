import onnx
from onnx import helper,TensorProto
p='browser-model/timestral.onnx';m=onnx.load(p);nodes=[]
for node in m.graph.node:
 if node.op_type=='EyeLike':
  dtype=next((a.i for a in node.attribute if a.name=='dtype'),None)
  if dtype in (None,TensorProto.BOOL):
   src=node.input[0];dst=node.output[0];prefix=node.name
   nodes.append(helper.make_node('Cast',[src],[prefix+'_float_input'],name=prefix+'_input_cast',to=TensorProto.FLOAT))
   node.input[0]=prefix+'_float_input';node.output[0]=prefix+'_float_output'
   del node.attribute[:]
   node.attribute.append(helper.make_attribute('dtype',TensorProto.FLOAT))
   nodes.append(node);nodes.append(helper.make_node('Cast',[prefix+'_float_output'],[dst],name=prefix+'_output_cast',to=TensorProto.BOOL));continue
 nodes.append(node)
del m.graph.node[:];m.graph.node.extend(nodes);onnx.save(m,p)
