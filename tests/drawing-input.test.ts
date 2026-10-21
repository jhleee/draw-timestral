import test from 'node:test';
import assert from 'node:assert/strict';
import {attachDrawingInput} from '../src/drawing-input.ts';
import {resample} from '../src/curve.ts';
function event(type:string,data:Record<string,unknown>){const e=new Event(type,{cancelable:true});Object.assign(e,data);return e}
test('native touch id 0 draws a complete resampleable curve and cancels browser scrolling',()=>{
 const win=Object.assign(new EventTarget(),{ontouchstart:null});Object.assign(globalThis,{window:win});
 const c=new EventTarget(),points:{x:number;y:number}[]=[];let ended=0,cancelled=0;
 const cleanup=attachDrawingInput(c as HTMLCanvasElement,{start(p){points.push({x:p.clientX,y:p.clientY});return true},move(p){points.push({x:p.clientX,y:p.clientY})},end(){ended++},cancel(){cancelled++}});
 const touch=(identifier:number,x:number)=>({identifier,clientX:x,clientY:.2+.5*x});
 const start=event('touchstart',{touches:[touch(0,0)],changedTouches:[touch(0,0)]});c.dispatchEvent(start);assert.equal(start.defaultPrevented,true);
 // Ignore duplicate Pointer Events emitted for the same finger.
 c.dispatchEvent(event('pointerdown',{pointerType:'touch',pointerId:4,isPrimary:true,button:0,clientX:0,clientY:0}));
 for(let i=1;i<21;i++){const e=event('touchmove',{changedTouches:[touch(0,i/21)]});c.dispatchEvent(e);assert.equal(e.defaultPrevented,true)}
 c.dispatchEvent(event('touchend',{changedTouches:[touch(7,.5)]}));assert.equal(ended,0);
 c.dispatchEvent(event('touchend',{changedTouches:[touch(0,1)]}));assert.equal(ended,1);assert.equal(cancelled,0);
 const samples=resample(points);assert.equal(samples.length,128);samples.forEach((v,i)=>assert.ok(Math.abs(v-(.2+.5*i/127))<1e-12));
 cleanup();c.dispatchEvent(start);assert.equal(points.length,22);
});
test('cancelled touch stays invalid; pen still works when pointer capture throws',()=>{
 const win=Object.assign(new EventTarget(),{ontouchstart:null});Object.assign(globalThis,{window:win});
 const c=Object.assign(new EventTarget(),{setPointerCapture(){throw Error('capture unavailable')}});let ended=0,cancelled=0,moved=0,started=0;
 const cleanup=attachDrawingInput(c as unknown as HTMLCanvasElement,{start(){started++;return true},move(){moved++},end(){ended++},cancel(){cancelled++}});
 const t={identifier:0,clientX:0,clientY:0};c.dispatchEvent(event('touchstart',{touches:[t],changedTouches:[t]}));c.dispatchEvent(event('touchcancel',{changedTouches:[t]}));assert.equal(cancelled,1);assert.equal(ended,0);
 c.dispatchEvent(event('pointerdown',{pointerType:'mouse',button:2,pointerId:1}));assert.equal(started,1);
 c.dispatchEvent(event('pointerdown',{pointerType:'pen',button:0,pointerId:2,isPrimary:true}));
 win.dispatchEvent(event('pointermove',{pointerId:2,clientX:1,clientY:1}));win.dispatchEvent(event('pointerup',{pointerId:2,clientX:2,clientY:1}));assert.equal(started,2);assert.equal(ended,1);assert.equal(moved,3);
 cleanup();
});
