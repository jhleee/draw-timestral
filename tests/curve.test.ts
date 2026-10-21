import test from 'node:test';
import assert from 'node:assert/strict';
import {resample,baseline,metrics,validatePrediction} from '../src/curve.ts';
test('interpolation preserves a straight line, exactly 128 samples and 64/64 partition',()=>{
 const y=resample(Array.from({length:21},(_,i)=>({x:i/20,y:.2+.5*i/20})));
 assert.equal(y.length,128);y.forEach((v,i)=>assert.ok(Math.abs(v-(.2+.5*i/127))<1e-12));
 assert.equal(y.slice(0,64).length,64);assert.equal(y.slice(64).length,64);
});
test('incomplete and reversing strokes are rejected',()=>{
 assert.throws(()=>resample([{x:0,y:0},{x:.8,y:1}]));
 assert.throws(()=>resample(Array.from({length:10},(_,i)=>({x:i===5?.2:i/9,y:.5}))));
});
test('constant and repeated x positions interpolate without NaN',()=>{
 const p=Array.from({length:10},(_,i)=>({x:i/9,y:.5}));p.splice(3,0,{x:2/9,y:.5});
 assert.ok(resample(p).every(v=>v===.5));
});
test('linear baseline recovers the known affine continuation without clipping',()=>{
 const p=baseline(Array.from({length:64},(_,i)=>i/127));
 p.forEach((v,i)=>assert.ok(Math.abs(v-(i+64)/127)<1e-12));
});
test('metrics have defined inclusive quantile boundaries',()=>{
 const p={median:Array(64).fill(.5),q10:Array(64).fill(.2),q90:Array(64).fill(.8)};
 const m=metrics(Array(64).fill(.8),p);assert.ok(Math.abs(m.mae-.3)<1e-12);assert.equal(m.coverage,1);assert.equal(m.exceed,0);
 assert.throws(()=>validatePrediction({...p,median:[NaN]}));
 assert.throws(()=>validatePrediction({...p,q10:Array(64).fill(.9)}));
});
