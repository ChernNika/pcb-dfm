function mountViaModel(root) {
root.innerHTML = VIA_VIEW;
"use strict";
const $=id=>root.querySelector('#'+id);
const fmt=(x,d=3)=>Number.isFinite(x)?x.toFixed(d).replace('.',','):'—';
const fmtP=x=>!Number.isFinite(x)?'—':(x>=0.99995&&x<=1.00005?'1,0000':x.toFixed(4).replace('.',','));
const SVGNS='http://www.w3.org/2000/svg';

/* ---------- maths shared ---------- */
function RCof(cls,Rv,Rd) {
  if(cls===3) return Rv-Rd-Number($('p_ring').value);
  const projection=Rd*Math.cos(Number($('b'+cls).value)*Math.PI/360);
  return Math.sqrt(Math.max(0,Rv*Rv-Rd*Rd+projection*projection))-projection;
}
function ellOf(Rv,Rd,w,wmin){ const y1=Math.sqrt(Math.max(Rv*Rv-w*w/4,0)); const a=wmin/2+Rd; const b2=((2*Rd+wmin)**2-w*w)/4; return {y1,a,b:b2>0?Math.sqrt(b2):NaN,x1:w/2}; }
function phiAt(r,E){
  if(!(E.b>0)||r<=E.y1-E.b) return Math.PI/2;
  const a2=E.a*E.a,b=E.b,s=a2-b*b; const disc=a2*(s+E.y1*E.y1)-r*r*s;
  let y2=disc>=0?(a2*E.y1-b*Math.sqrt(disc))/s:Math.min(E.y1,r);
  if(y2>E.y1) y2=Math.min(E.y1,r); if(y2>r) y2=r;
  return Math.atan2(y2,Math.sqrt(Math.max(r*r-y2*y2,0)));
}
function Pring(RC,s){ return RC<=0?0:1-Math.exp(-RC*RC/(2*s*s)); }
function Pauthor(RC,s,E){
  if(RC<=0) return 0; if(s===0) return Number(okAuthor(0,0,RC,E)); if(!(E.b>0)) return Pring(RC,s);
  const n=400,h=Math.min(RC,12*s)/n; let I=0;
  for(let i=0;i<=n;i++){ const r=i*h; const f=phiAt(r,E)*r*Math.exp(-r*r/(2*s*s)); I+=f*(i==0||i==n?1:(i%2?4:2)); }
  I*=h/3; return Math.max(0, Math.min(1,0.5*(1-Math.exp(-RC*RC/(2*s*s)))+I/(Math.PI*s*s)));
}
// seeded normals
function mulberry(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;}}
const rnd=mulberry(20251029), NMC=20000, Z=new Float64Array(2*NMC);
for(let i=0;i<NMC;i++){const u=Math.max(rnd(),1e-12),v=rnd(),R=Math.sqrt(-2*Math.log(u));Z[2*i]=R*Math.cos(2*Math.PI*v);Z[2*i+1]=R*Math.sin(2*Math.PI*v);}
function okAuthor(x,y,RC,E){ if(RC<=0||x*x+y*y>RC*RC) return false; if(!(E.b>0)) return true; if(y>=E.y1) return false; return !(x*x/(E.a*E.a)+(y-E.y1)**2/(E.b*E.b)<1); }
function breakoutDeg(d,Rv,Rd){ if(d+Rd<=Rv) return 0; if(d<1e-9) return 360; const c=(Rv*Rv-d*d-Rd*Rd)/(2*d*Rd); if(c<=-1) return 360; if(c>=1) return 0; return 2*Math.acos(c)*180/Math.PI; }
const hyp=Math.hypot;
function segDist(px,py,ax,ay,bx,by){const dx=bx-ax,dy=by-ay;let t=((px-ax)*dx+(py-ay)*dy)/(dx*dx+dy*dy);t=Math.max(0,Math.min(1,t));return hyp(px-(ax+t*dx),py-(ay+t*dy));}

/* ---------- svg helpers ---------- */
function el(tag,attrs,parent){const e=document.createElementNS(SVGNS,tag);for(const k in attrs)e.setAttribute(k,attrs[k]);if(parent)parent.appendChild(e);return e;}
function svgPoint(svg,evt){const p=svg.createSVGPoint();p.x=evt.clientX;p.y=evt.clientY;const q=p.matrixTransform(svg.getScreenCTM().inverse());return {x:q.x,y:-q.y};}
function segBind(id,cb){const box=$(id);box.addEventListener('click',e=>{const b=e.target.closest('button[data-v]');if(!b)return;box.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));cb(b.dataset.v);});}
function segSet(id,v){$(id).querySelectorAll('button[data-v]').forEach(x=>x.setAttribute('aria-pressed',String(x.dataset.v===String(v))));}
function rows(target,list){target.innerHTML=list.map(r=>r.sec?`<div class="row sec"><span>${r.sec}</span><span></span></div>`:`<div class="row"><span>${r.k}</span><span class="v">${r.v}</span></div>`).join('');}
function ellipsePath(E,yc){ // lower half, closed with chord at yc
  let d='';const n=80;for(let i=0;i<=n;i++){const t=Math.PI+i*Math.PI/n;const x=E.a*Math.cos(t),y=yc+E.b*Math.sin(t);d+=(i?'L':'M')+x.toFixed(2)+' '+(-y).toFixed(2);}return d+'Z';}

/* ============ 1. pad ============ */
const S1={cls:3,mode:'author',hx:0,hy:40,drag:false};
segBind('p_cls',v=>{S1.cls=S2.cls=+v;segSet('t_cls',v);drawAll();}); segBind('p_mode',v=>{S1.mode=v;draw1();});
['p_dvia','p_dd','p_w','p_wmin','p_s3','p_ring','b1','b2','target'].forEach(id=>$(id).addEventListener('input',drawAll));
function drawAll(){
  const message=$('via-error');
  const validNumbers=[...root.querySelectorAll('input[type=number]')].every(field=>field.checkValidity());
  const diameter=+$('p_dvia').value, width=+$('p_w').value;
  let error='';
  if (!validNumbers) error='Проверьте значения в числовых полях.';
  else if (+$('p_dd').value>=diameter) error='Диаметр отверстия должен быть меньше диаметра площадки.';
  else if (width>diameter) error='Ширина проводника превышает диаметр площадки.';
  else if (+$('p_wmin').value>width) error='Требуемая ширина соединения превышает ширину проводника.';
  message.textContent=error;
  const valid=!error;
  for (const id of ['padRead','tearRead','t5','padSvg','tearSvg','curveSvg']) $(id).toggleAttribute('hidden',!valid);
  if(valid){ S2.hy=null; draw1();draw2();draw3(); }
}
function draw1(){
  const D=+$('p_dvia').value, dd=+$('p_dd').value, w=+$('p_w').value, wm=+$('p_wmin').value, s3=+$('p_s3').value;
  const Rv=D/2,Rd=dd/2,s=s3/3,cls=S1.cls,RC=RCof(cls,Rv,Rd),E=ellOf(Rv,Rd,w,wm);
  const author=S1.mode==='author';
  const svg=$('padSvg'); svg.innerHTML='';
  const W=Rv+90, top=Rv+250, bot=Rv+70; svg.setAttribute('viewBox',`${-W} ${-top} ${2*W} ${top+bot}`);
  const fs=(2*W)/34;
  el('rect',{x:-W,y:-top,width:2*W,height:top+bot,fill:'var(--sub)'},svg);
  el('circle',{cx:0,cy:0,r:Rv,fill:'var(--copper-soft)',stroke:'var(--copper)','stroke-width':fs/7},svg);
  el('rect',{x:-E.x1,y:-top,width:w,height:top-E.y1+1,fill:'var(--copper-soft)'},svg);
  el('line',{x1:-E.x1,y1:-E.y1,x2:-E.x1,y2:-top,stroke:'var(--copper)','stroke-width':fs/7},svg);
  el('line',{x1:E.x1,y1:-E.y1,x2:E.x1,y2:-top,stroke:'var(--copper)','stroke-width':fs/7},svg);
  // regions
  if(RC>0) el('circle',{cx:0,cy:0,r:RC,fill:'none',stroke:'var(--good)','stroke-width':fs/7,'stroke-dasharray':`${fs/2} ${fs/3}`},svg);
  if(author&&E.b>0){ el('path',{d:ellipsePath(E,E.y1),fill:'var(--bad)','fill-opacity':.12,stroke:'var(--bad)','stroke-width':fs/7},svg); }
  // MC
  let good=0,dG='',dB='';const dot=fs/5;
  for(let i=0;i<NMC;i++){const x=Z[2*i]*s,y=Z[2*i+1]*s;const ok=author?okAuthor(x,y,RC,E):(RC>0&&x*x+y*y<=RC*RC);if(ok)good++;
    if(i<1500){const seg='M'+x.toFixed(1)+' '+(-y).toFixed(1)+'h0';if(ok)dG+=seg;else dB+=seg;}}
  el('path',{d:dG,stroke:'var(--good)','stroke-width':dot,'stroke-linecap':'round',fill:'none','stroke-opacity':.75},svg);
  el('path',{d:dB,stroke:'var(--bad)','stroke-width':dot*1.3,'stroke-linecap':'round',fill:'none'},svg);
  // hole
  const hx=S1.hx,hy=S1.hy;
  el('circle',{cx:hx,cy:-hy,r:Rd,fill:'var(--hole)','fill-opacity':.55,stroke:'var(--ink)','stroke-width':fs/8},svg);
  el('path',{d:`M${hx-fs/2} ${-hy}h${fs}M${hx} ${-hy-fs/2}v${fs}`,stroke:'var(--ink)','stroke-width':fs/10},svg);
  // neck lines to corners
  const neckL=hyp(hx+E.x1,hy-E.y1)-Rd, neckR=hyp(hx-E.x1,hy-E.y1)-Rd;
  [[-E.x1,neckL],[E.x1,neckR]].forEach(([cx,n])=>{const dx=cx-hx,dy=E.y1-hy,L=hyp(dx,dy);if(L<=Rd)return;const sx=hx+dx/L*Rd,sy=hy+dy/L*Rd;
    el('line',{x1:sx,y1:-sy,x2:cx,y2:-E.y1,stroke:'var(--accent)','stroke-width':fs/9,'stroke-dasharray':`${fs/4} ${fs/5}`},svg);});
  // labels
  const lab=(x,y,t,c,anchor)=>{const e=el('text',{x,y:-y,'font-size':fs,fill:c||'var(--muted)','text-anchor':anchor||'start'},svg);e.textContent=t;};
  if(RC>0) lab(RC*0.72+fs*0.3,-RC*0.72-fs*0.6,'R_C = '+Math.round(RC),'var(--good)');
  if(author&&E.b>0) lab(E.a+fs*0.4,E.y1-E.b*0.4,'эллипс','var(--bad)');
  lab(E.x1+fs*0.5,top-fs*1.6,'проводник '+w,'var(--copper)');
  lab(-W+fs*0.6,-bot+fs*1.4,'КП Ø'+D+' · отв. Ø'+dd,'var(--muted)');
  // pointer
  svg.onpointerdown=e=>{S1.drag=true;svg.setPointerCapture(e.pointerId);move1(e);};
  svg.onpointermove=e=>{if(S1.drag)move1(e);};
  svg.onpointerup=svg.onpointercancel=()=>{S1.drag=false;};
  // numbers
  const Pa=Pauthor(RC,s,E), Pr=Pring(RC,s), Pmc=good/NMC;
  const d=hyp(hx,hy), ring=Rv-d-Rd, brk=breakoutDeg(d,Rv,Rd);
  const inRC=RC>0&&d<=RC, inEll=E.b>0&&hy<E.y1&&(hx*hx/(E.a*E.a)+(hy-E.y1)**2/(E.b*E.b)<1);
  const okA=inRC&&!(E.b>0&&(hy>=E.y1||inEll)), okR=inRC;
  rows($('padRead'),[
    {sec:'Область годности'},
    {k:'R<sub>C</sub> для класса '+cls,v:fmt(RC,1)+' мкм'},
    {k:'Нижняя точка эллипса y<sub>1</sub> − b',v:E.b>0?fmt(E.y1-E.b,1)+' мкм':'нет эллипса'},
    {sec:'Вероятность годной КП'},
    {k:'Поясок и ширина соединения',v:fmtP(Pa)},
    {k:'Монте-Карло, текущий критерий',v:fmtP(Pmc)},
    {k:'Только поясок / разрыв',v:fmtP(Pr)},
    {sec:'Отверстие в текущем положении'},
    {k:'Смещение центра',v:fmt(d,1)+' мкм'},
    {k:brk>0?'Разрыв КП':'Гарантийный поясок',v:brk>0?fmt(brk,0)+'°':fmt(ring,1)+' мкм'},
    {k:'Перемычка до углов',v:fmt(Math.max(neckL,0)+Math.max(neckR,0),0)+' / '+wm+' мкм'},
    {k:'По ширине соединения',v:okA?'<span class="pill good">годна</span>':'<span class="pill bad">брак</span>'},
    {k:'По пояску / разрыву',v:okR?'<span class="pill good">годна</span>':'<span class="pill bad">брак</span>'},
  ]);
}
function move1(e){const p=svgPoint($('padSvg'),e);const Rv=+$('p_dvia').value/2;const L=hyp(p.x,p.y),M=Rv+60;const k=L>M?M/L:1;S1.hx=p.x*k;S1.hy=p.y*k;draw1();}

/* ============ 2. teardrop ============ */
const S2={cls:3,hx:0,hy:null,drag:false};
segBind('t_cls',v=>{S2.cls=S1.cls=+v;segSet('p_cls',v);drawAll();});
$('t_reset').addEventListener('click',()=>{S2.hx=0;S2.hy=null;draw2();});
function teardrop(cls,w){
  const Rv=+$('p_dvia').value/2,Rd=+$('p_dd').value/2,wm=+$('p_wmin').value,E=ellOf(Rv,Rd,w,wm),RC=RCof(cls,Rv,Rd);
  if(!(E.b>0)||RC<=E.y1-E.b) return {none:true,Rv,Rd,wm,E,RC};
  const ph=phiAt(RC,E), y1p=RC+E.b, x3=Rv*Math.cos(ph), y3=Rv*Math.sin(ph);
  return {Rv,Rd,wm,E,RC,ph,y1p,x3,y3,x1:E.x1,lp:(y1p-y3)/(2*Rv)*100,wp:Math.cos(ph)*100};
}
function draw2(){
  const w=+$('p_w').value;
  const T=teardrop(S2.cls,w), svg=$('tearSvg'); svg.innerHTML='';
  const Rv=T.Rv,Rd=T.Rd,W=Rv+90,top=Rv+330,bot=Rv+60, fs=(2*W)/34;
  svg.setAttribute('viewBox',`${-W} ${-top} ${2*W} ${top+bot}`);
  el('rect',{x:-W,y:-top,width:2*W,height:top+bot,fill:'var(--sub)'},svg);
  const lab=(x,y,t,c,anchor)=>{const e=el('text',{x,y:-y,'font-size':fs,fill:c||'var(--muted)','text-anchor':anchor||'start'},svg);e.textContent=t;};
  if(T.none){
    el('circle',{cx:0,cy:0,r:Rv,fill:'var(--copper-soft)',stroke:'var(--copper)','stroke-width':fs/7},svg);
    el('rect',{x:-w/2,y:-top,width:w,height:top-T.E.y1+1,fill:'var(--copper-soft)'},svg);
    rows($('tearRead'),[{k:'Капля',v:'не нужна: эллипс не задевает R<sub>C</sub>'}]);return;
  }
  if(S2.hy===null){S2.hx=0;S2.hy=T.RC;}
  const {x1,y1p,x3,y3,E,RC}=T;
  // copper: pad + teardrop polygon + conductor
  el('circle',{cx:0,cy:0,r:Rv,fill:'var(--copper-soft)',stroke:'var(--copper)','stroke-width':fs/7},svg);
  el('path',{d:`M${-x3} ${-y3}L${-x1} ${-y1p}L${x1} ${-y1p}L${x3} ${-y3}Z`,fill:'var(--copper-soft)'},svg);
  el('rect',{x:-x1,y:-top,width:w,height:top-y1p+1,fill:'var(--copper-soft)'},svg);
  [[-1],[1]].forEach(([sg])=>{el('line',{x1:sg*x3,y1:-y3,x2:sg*x1,y2:-y1p,stroke:'var(--copper)','stroke-width':fs/7},svg);el('line',{x1:sg*x1,y1:-y1p,x2:sg*x1,y2:-top,stroke:'var(--copper)','stroke-width':fs/7},svg);});
  el('circle',{cx:0,cy:0,r:RC,fill:'none',stroke:'var(--good)','stroke-width':fs/7,'stroke-dasharray':`${fs/2} ${fs/3}`},svg);
  const Es={a:E.a,b:E.b}; el('path',{d:ellipsePath(Es,y1p),fill:'var(--bad)','fill-opacity':.08,stroke:'var(--bad)','stroke-width':fs/9,'stroke-dasharray':`${fs/3} ${fs/4}`},svg);
  // hole
  const hx=S2.hx,hy=S2.hy;
  el('circle',{cx:hx,cy:-hy,r:Rd,fill:'var(--hole)','fill-opacity':.55,stroke:'var(--ink)','stroke-width':fs/8},svg);
  el('path',{d:`M${hx-fs/2} ${-hy}h${fs}M${hx} ${-hy-fs/2}v${fs}`,stroke:'var(--ink)','stroke-width':fs/10},svg);
  // author necks (to corners) and real necks (to flank)
  function nearestOnSeg(px,py,ax,ay,bx,by){const dx=bx-ax,dy=by-ay;let t=((px-ax)*dx+(py-ay)*dy)/(dx*dx+dy*dy);t=Math.max(0,Math.min(1,t));return [ax+t*dx,ay+t*dy];}
  const sides=[-1,1].map(sg=>{
    const cx=sg*x1, A=hyp(hx-cx,hy-y1p)-Rd;
    const [qx,qy]=nearestOnSeg(hx,hy,sg*x1,y1p,sg*x3,y3); const R=hyp(hx-qx,hy-qy)-Rd;
    return {sg,cx,A,qx,qy,R};
  });
  sides.forEach(o=>{
    const line=(tx,ty,c,dash)=>{const dx=tx-hx,dy=ty-hy,L=hyp(dx,dy);if(L<=Rd)return;el('line',{x1:hx+dx/L*Rd,y1:-(hy+dy/L*Rd),x2:tx,y2:-ty,stroke:c,'stroke-width':fs/7,'stroke-dasharray':dash||'none'},svg);};
    line(o.cx,y1p,'var(--muted)',`${fs/4} ${fs/5}`); line(o.qx,o.qy,'var(--bad)');
  });
  lab(0,top-fs*1.6,'проводник '+w,'var(--copper)','middle');
  lab(RC*0.72+fs*0.3,-RC*0.72-fs*0.6,'R_C = '+Math.round(RC),'var(--good)');
  lab(-W+fs*0.6,-bot+fs*1.4,'серый пунктир: до углов · красный: до грани','var(--muted)');
  svg.onpointerdown=e=>{S2.drag=true;svg.setPointerCapture(e.pointerId);move2(e);};
  svg.onpointermove=e=>{if(S2.drag)move2(e);};
  svg.onpointerup=svg.onpointercancel=()=>{S2.drag=false;};
  const nA=Math.max(sides[0].A,0)+Math.max(sides[1].A,0), nR=Math.max(sides[0].R,0)+Math.max(sides[1].R,0);
  rows($('tearRead'),[
    {sec:'Параметры капли'},
    {k:'Длина l′, % от D<sub>via</sub>',v:fmt(T.lp,0)},
    {k:'Ширина w′, % от D<sub>via</sub>',v:fmt(T.wp,0)},
    {sec:'Перемычка при текущем положении'},
    {k:'До углов (±x<sub>1</sub>, y<sub>1</sub>′)',v:fmt(nA,0)+' мкм'},
    {k:'До ближайшей грани меди',v:fmt(nR,0)+' мкм'},
    {k:'Требуется w<sub>min</sub>',v:T.wm+' мкм'},
    {k:'Вывод',v:nR>=T.wm-0.5?'<span class="pill good">выполняется</span>':'<span class="pill bad">не выполняется</span>'},
  ]);
}
function move2(e){
  const svg=$('tearSvg'), p=svgPoint(svg,e), box=svg.viewBox.baseVal;
  S2.hx=Math.max(box.x,Math.min(box.x+box.width,p.x));
  S2.hy=Math.max(-box.y-box.height,Math.min(-box.y,p.y));
  draw2();
}

/* ============ 3. curves ============ */
function req3s(f){let lo=0,hi=400;const goal=+$('target').value;if(f(0)<goal)return NaN;while(f(hi)>goal)hi*=2;for(let i=0;i<60;i++){const m=(lo+hi)/2;if(f(m)>+$('target').value)lo=m;else hi=m;}return 3*lo;}
function draw3(){
  const Rv=+$('p_dvia').value/2,Rd=+$('p_dd').value/2,wm=+$('p_wmin').value,E=ellOf(Rv,Rd,+$('p_w').value,wm);
  const svg=$('curveSvg');svg.innerHTML='';
  const L=62,R=18,T=18,B=48,Wd=640-L-R,Ht=400-T-B, xmax=180, ymin=-6;
  const X=v=>L+v/xmax*Wd, Y=lg=>T+(0-lg)/(0-ymin)*Ht;
  for(let k=0;k>=ymin;k--){el('line',{x1:L,x2:L+Wd,y1:Y(k),y2:Y(k),stroke:'var(--line)'},svg);const t=el('text',{x:L-8,y:Y(k)+4,'font-size':11,fill:'var(--muted)','text-anchor':'end'},svg);t.textContent=k===0?'1':'10'+['⁰','⁻¹','⁻²','⁻³','⁻⁴','⁻⁵','⁻⁶'][-k];}
  for(let v=0;v<=xmax;v+=30){el('line',{x1:X(v),x2:X(v),y1:T,y2:T+Ht,stroke:'var(--line)'},svg);const t=el('text',{x:X(v),y:T+Ht+16,'font-size':11,fill:'var(--muted)','text-anchor':'middle'},svg);t.textContent=v;}
  const xt=el('text',{x:L+Wd/2,y:400-8,'font-size':12,fill:'var(--muted)','text-anchor':'middle'},svg);xt.textContent='точность станка 3σ, мкм';
  const yt=el('text',{x:14,y:T+Ht/2,'font-size':12,fill:'var(--muted)','text-anchor':'middle',transform:`rotate(-90 14 ${T+Ht/2})`},svg);yt.textContent='доля брака 1 − P';
  const l3=Math.log10(1-+$('target').value);el('line',{x1:L,x2:L+Wd,y1:Y(l3),y2:Y(l3),stroke:'var(--ink)','stroke-dasharray':'2 4','stroke-opacity':.6},svg);
  const t97=el('text',{x:L+Wd-4,y:Y(l3)-6,'font-size':11,fill:'var(--ink)','text-anchor':'end'},svg);t97.textContent='P = '+$('target').value.replace('.',',');
  const cols={3:'var(--c3)',2:'var(--c2)',1:'var(--c1)'};
  const clipId='cclip';const cp=el('clipPath',{id:clipId},svg);el('rect',{x:L,y:T,width:Wd,height:Ht},cp);
  const res={};
  [3,2,1].forEach(cls=>{
    const RC=RCof(cls,Rv,Rd);
    ['ring','author'].forEach(m=>{
      let d='';for(let v=4;v<=xmax;v+=2){const s=v/3;const P=m==='author'?Pauthor(RC,s,E):Pring(RC,s);const q=Math.max(1-P,1e-7);d+=(d?'L':'M')+X(v).toFixed(1)+' '+Y(Math.max(Math.log10(q),ymin-0.3)).toFixed(1);}
      el('path',{d,fill:'none',stroke:cols[cls],'stroke-width':m==='author'?2.4:1.8,'stroke-dasharray':m==='ring'?'6 5':'none','clip-path':`url(#${clipId})`},svg);
    });
    res[cls]={a:req3s(s=>Pauthor(RC,s,E)),r:req3s(s=>Pring(RC,s))};
  });
  $('t5').innerHTML='<tr><th>Класс</th><th>3σ по ширине соединения, мкм</th><th>3σ по пояску, мкм</th></tr>'+
    [3,2,1].map(c=>`<tr><td>${c}</td><td class="c">${fmt(res[c].a,1)}</td><td class="c">${fmt(res[c].r,1)}</td></tr>`).join('');
}
drawAll();
function setProfile(profile, fields) {
  const cls=profile?.product_class ?? 3;
  S1.cls=S2.cls=cls;
  segSet('p_cls',cls);segSet('t_cls',cls);
  $('b1').value=profile ? (profile.breakout.class1 ?? 0) : 180;
  $('b2').value=profile ? (profile.breakout.class2 ?? 0) : 90;
  $('p_s3').value=3*(fields['drilling.sigma_drill_um'] ?? 10);
  $('p_wmin').value=fields['etch.finished_width_min_um'] ?? 150;
  $('p_ring').value=fields['annular_ring.required_um.class3.outer'] ?? 50;
  drawAll();
}

function setPad(pad, trace, criteria, err, breakout) {
  S1.cls=S2.cls=criteria.cls;
  segSet('p_cls',S1.cls); segSet('t_cls',S2.cls);
  const um=mm=>Number((1000*mm).toFixed(3));
  $('p_dvia').value=um(2*pad.padInsc);
  $('p_dd').value=um(2*pad.Rd);
  $('p_w').value=um(trace.w);
  $('p_wmin').value=um(trace.w*(1-criteria.kred));
  $('p_s3').value=um(3*err.sd);
  $('p_ring').value=um(criteria.ringReq);
  for(const cls of [1,2]) $('b'+cls).value=cls===criteria.cls ? criteria.breakout : (breakout?.['class'+cls] ?? (cls===1?180:90));
  S1.hx=S1.hy=0;
  drawAll();
}
return {setProfile, setPad};
}

if (document.getElementById('via-model')) {
  const model=mountViaModel(document.getElementById('via-model'));
  initCalculatorProfiles(model.setProfile);
}
