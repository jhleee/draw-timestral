export type Contact={clientX:number;clientY:number};
type Callbacks={start:(point:Contact)=>boolean;move:(point:Contact)=>void;end:()=>void;cancel:()=>void};
// Native, non-passive touch listeners keep a drawing gesture out of page scrolling.
// Mouse and pen retain Pointer Events and capture outside the canvas.
export function attachDrawingInput(canvas:HTMLCanvasElement,cb:Callbacks){
 let active:{kind:'touch'|'pointer';id:number}|null=null;
 const options={passive:false};
 const stop=(e:Event)=>{if(e.cancelable)e.preventDefault()};
 const touchStart=(e:TouchEvent)=>{
  if(active||e.touches.length!==1)return;
  const t=e.changedTouches[0];if(!t)return;
  stop(e);if(cb.start(t))active={kind:'touch',id:t.identifier};
 };
 const touchMove=(e:TouchEvent)=>{
  if(active?.kind!=='touch')return;stop(e);
  const t=Array.from(e.changedTouches).find(t=>t.identifier===active!.id);if(t)cb.move(t);
 };
 const touchEnd=(e:TouchEvent)=>{
  if(active?.kind!=='touch')return;
  const t=Array.from(e.changedTouches).find(t=>t.identifier===active!.id);if(!t)return;
  stop(e);cb.move(t);active=null;e.type==='touchcancel'?cb.cancel():cb.end();
 };
 const pointerDown=(e:PointerEvent)=>{
  if((e.pointerType==='touch'&&'ontouchstart' in window)||active||e.isPrimary===false||(e.pointerType==='mouse'&&e.button!==0))return;
  stop(e);if(!cb.start(e))return;active={kind:'pointer',id:e.pointerId};
  try{canvas.setPointerCapture(e.pointerId)}catch{/* Window listeners still finish the stroke. */}
 };
 const pointerMove=(e:PointerEvent)=>{
  if(active?.kind!=='pointer'||e.pointerId!==active.id)return;stop(e);
  const events=e.getCoalescedEvents?.();for(const point of events?.length?events:[e])cb.move(point);
 };
 const pointerEnd=(e:PointerEvent)=>{
  if(active?.kind!=='pointer'||e.pointerId!==active.id)return;stop(e);
  const cancelled=e.type!=='pointerup';if(!cancelled)cb.move(e);active=null;
  cancelled?cb.cancel():cb.end();
 };
 const bindings:[EventTarget,string,EventListener][]=[
  [canvas,'touchstart',touchStart as EventListener],[canvas,'touchmove',touchMove as EventListener],
  [canvas,'touchend',touchEnd as EventListener],[canvas,'touchcancel',touchEnd as EventListener],
  [canvas,'pointerdown',pointerDown as EventListener],[window,'pointermove',pointerMove as EventListener],
  [window,'pointerup',pointerEnd as EventListener],[window,'pointercancel',pointerEnd as EventListener],
  [canvas,'lostpointercapture',pointerEnd as EventListener],
 ];
 for(const [target,type,fn] of bindings)target.addEventListener(type,fn,options);
 return ()=>{for(const [target,type,fn] of bindings)target.removeEventListener(type,fn);active=null};
}
