import * as ort from 'onnxruntime-web/wasm';
import {readFileSync,writeFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;
ort.env.wasm.wasmPaths=pathToFileURL(process.cwd()+'/node_modules/onnxruntime-web/dist/').href;
const graph=new Uint8Array(readFileSync('browser-model/browser-graph.onnx'));
const weights=new Uint8Array(readFileSync('browser-model/official-model.safetensors'));
console.time('load');
const s=await ort.InferenceSession.create(graph,{executionProviders:['wasm'],externalData:[{path:'official-model.safetensors',data:weights}],graphOptimizationLevel:'all'});
console.timeEnd('load');
const curves=JSON.parse(readFileSync('browser-model/fixtures.json','utf8')).curves;
const results=[];
for(const c of curves){
 const x=new ort.Tensor('float32',Float32Array.from(c.request.context),[1,1,64]);const t=performance.now();
 const out=await s.run({context:x});
 const p={};for(const name of ['q10','median','q90'])p[name]=Array.from(out[name].data);
 if(Object.values(p).some(a=>a.length!==64||!a.every(Number.isFinite)))throw Error('Invalid output');
 results.push({name:c.name,inference_ms:performance.now()-t,prediction:p});
 console.log(c.name,results.at(-1).inference_ms,p.median.slice(0,3));
}
writeFileSync('browser-model/wasm-results.json',JSON.stringify(results,null,2));
