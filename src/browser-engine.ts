import * as ort from 'onnxruntime-web/wasm';
import graphBase64 from '../browser-model/browser-graph.gz.b64?raw';
import info from '../browser-model/weights-info.json';
import type {Prediction} from './curve';
let modelPromise:Promise<ort.InferenceSession>|undefined;
export function loadBrowserModel(progress:(text:string)=>void):Promise<ort.InferenceSession>{
 if(modelPromise)return modelPromise;
 modelPromise=(async()=>{
  progress('모델 다운로드 중 · 20.8MB');
  ort.env.wasm.numThreads=1;
  ort.env.wasm.proxy=false;
  ort.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
  const response=await fetch(info.url,{signal:AbortSignal.timeout(120000)});
  if(!response.ok)throw Error(`공식 모델 다운로드 실패: HTTP ${response.status}`);
  const reader=response.body?.getReader();let weights:Uint8Array;
  if(reader){const chunks:Uint8Array[]=[];let total=0;while(true){const {done,value}=await reader.read();if(done)break;chunks.push(value);total+=value.length;progress(`모델 다운로드 · ${(total/1e6).toFixed(1)} / ${(info.bytes/1e6).toFixed(1)}MB`)}weights=new Uint8Array(total);let offset=0;for(const chunk of chunks){weights.set(chunk,offset);offset+=chunk.length}}
  else weights=new Uint8Array(await response.arrayBuffer());
  if(weights.length!==info.bytes)throw Error('모델 파일 크기가 공식 가중치와 일치하지 않습니다.');
  const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',weights.buffer as ArrayBuffer))).map(x=>x.toString(16).padStart(2,'0')).join('');
  if(digest!==info.sha256)throw Error('공식 모델 SHA-256 검증 실패');
  progress('모델 초기화 중 · WebAssembly');
  const zipped=Uint8Array.from(atob(graphBase64.trim()),x=>x.charCodeAt(0));
  const graph=new Uint8Array(await new Response(new Blob([zipped]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
  return ort.InferenceSession.create(graph,{executionProviders:['wasm'],externalData:[{path:'official-model.safetensors',data:weights}],graphOptimizationLevel:'all'});
 })();
 modelPromise.catch(()=>{modelPromise=undefined});return modelPromise;
}
export async function browserPredict(context:number[]):Promise<{prediction:Prediction;inference_ms:number}>{
 if(context.length!==64||!context.every(v=>Number.isFinite(v)&&v>=0&&v<=1))throw Error('관측값은 0–1 범위 숫자 64개여야 합니다.');
 const session=await loadBrowserModel(()=>{});const start=performance.now();
 const result=await session.run({context:new ort.Tensor('float32',Float32Array.from(context),[1,1,64])});
 const prediction={} as Prediction;
 for(const k of ['median','q10','q90'] as const){const t=result[k];if(!t||t.data.length!==64)throw Error('모델 출력 형상 오류');prediction[k]=Array.from(t.data as Float32Array)}
 return {prediction,inference_ms:performance.now()-start};
}
