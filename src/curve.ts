export type Point = { x:number; y:number };
export type Prediction = {median:number[];q10:number[];q90:number[]};
export function resample(points:Point[]):number[]{
  if(points.length<8 || points[0].x>0.04 || points.at(-1)!.x<0.96) throw Error('왼쪽 시작 영역부터 오른쪽 끝까지 한 번에 그려주세요.');
  const p=[{x:0,y:points[0].y},...points,{x:1,y:points.at(-1)!.y}];
  for(let i=1;i<p.length;i++)if(p[i].x<p[i-1].x)throw Error('X 좌표는 왼쪽에서 오른쪽으로 진행해야 합니다.');
  let j=0;
  return Array.from({length:128},(_,i)=>{const x=i/127;while(j<p.length-2&&p[j+1].x<=x)j++;const a=p[j],b=p[j+1];return a.y+(b.y-a.y)*(b.x===a.x?0:(x-a.x)/(b.x-a.x));});
}
export function validatePrediction(p:Prediction):Prediction{
  if(!p||!['median','q10','q90'].every(k=>Array.isArray(p[k as keyof Prediction])&&p[k as keyof Prediction].length===64&&p[k as keyof Prediction].every(Number.isFinite)))throw Error('서버 응답 오류: 각 예측 배열은 유한한 숫자 64개여야 합니다.');
  if(p.median.some((v,i)=>p.q10[i]>v||v>p.q90[i]))throw Error('서버 응답 오류: 분위수 순서가 올바르지 않습니다.');
  return p;
}
export function metrics(truth:number[],p:Prediction){return {mae:truth.reduce((s,v,i)=>s+Math.abs(v-p.median[i]),0)/64,coverage:truth.filter((v,i)=>v>=p.q10[i]&&v<=p.q90[i]).length/64,exceed:truth.filter((v,i)=>v>p.q90[i]).length/64};}
export function baseline(context:number[]){const y=context.slice(-8),mean=y.reduce((s,v)=>s+v,0)/8;const slope=y.reduce((s,v,i)=>s+(i-3.5)*(v-mean),0)/42;return Array.from({length:64},(_,i)=>mean+slope*(i+8-3.5));}
