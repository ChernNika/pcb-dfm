(function(){
"use strict";
const report=document.getElementById('yield-panel');
const root=report ? report.attachShadow({mode:'open'}) : document;
if(report) root.append(document.getElementById('yield-template').content.cloneNode(true));
const $=id=>root.getElementById(id);
const percentFormat=new Intl.NumberFormat('ru-RU',{style:'percent',maximumFractionDigits:2});
const percent=value=>value>0 && value<0.0001 ? '< 0,01 %'
  : value<1 && value>0.9999 ? '> 99,99 %' : percentFormat.format(value);
const fmt=(x,d=3)=>!Number.isFinite(x)?'—':x.toFixed(d).replace('.',',');
const fmtP=x=>{if(!Number.isFinite(x))return'—';if(x>0&&x<1e-3)return x.toExponential(1).replace('.',',').replace('e-','·10⁻').replace(/(\d+)$/,m=>m.split('').map(c=>'⁰¹²³⁴⁵⁶⁷⁸⁹'[c]).join(''));return x.toFixed(3).replace('.',',');};
const css=n=>getComputedStyle($('status')).getPropertyValue(n).trim();
const S={files:[],source:'manual',regions:null,model:null,res:null,busy:false,dirty:true,sel:null,detail:null,layerCache:{}};
const ROLE_OPTS=[['copper:Top','Медь, верх'],['copper:Bottom','Медь, низ']].concat(Array.from({length:30},(_,i)=>['copper:In'+(i+1),'Медь, внутр. '+(i+1)])).concat([['copper:?','Медь'],['drill','Сверловка (металлиз.)'],['npth','Сверловка без металлиз.'],['ignore','Не использовать']]);
const mkCanvas=(()=>{const cache=new Map();return (w,h)=>{const k=w+'x'+h;let c=cache.get(k);if(!c){c=document.createElement('canvas');c.width=w;c.height=h;c.getContext('2d',{willReadFrequently:true});cache.set(k,c);}return c;};})();
const tick=()=>new Promise(r=>setTimeout(r,0));
function status(t){$('status').textContent=t;}
function prog(f){$('prog').style.width=(Math.max(0,Math.min(1,f))*100).toFixed(1)+'%';}

/* ---------- files ---------- */
async function addFiles(list){
  if(S.busy) return;
  setBusy(true);
  try {
    const imported=[];
    for(const file of list){
      if(!/\.zip$/i.test(file.name)){
        imported.push({name:file.name,text:await file.text()});
        continue;
      }
      const archive=await JSZip.loadAsync(file);
      for(const entry of Object.values(archive.files)){
        const name=entry.name.split('/').pop();
        if(entry.dir || !name || name.startsWith('.')) continue;
        imported.push({name,text:await entry.async('string')});
      }
    }
    if(!imported.length){status('В архиве нет файлов для загрузки.');return;}
    if(S.source!=='manual') S.files=[];
    S.source='manual';
    for(const file of imported) pushFile(file.name,file.text);
    S.regions=null;
    afterFilesChanged();
  } catch(error){
    status('Ошибка чтения файла: '+error.message);
  } finally {
    setBusy(false);
  }
}
function pushFile(name,text,sourceRole){
  if(/[\x00-\x08\x0e-\x1f]/.test(text.slice(0,2000)))return;
  const role=sourceRole || GEngine.detectRole(name,text);
  S.files.push({name,text,role,parsed:null});
}
function parseFile(f){
  if(f.parsed&&f.parsed.role===f.role)return f.parsed;
  let r={role:f.role};
  try{
    if(f.role.startsWith('copper')){const g=GEngine.parseGerber(f.text,f.name);r.prims=g.prims;r.layerNumber=Number(g.fileFunction?.match(/,L(\d+)/i)?.[1]);r.warn=g.warnings;r.info=g.prims.length+' примитивов';}
    else if(f.role==='drill'||f.role==='npth'){const d=GEngine.parseExcellon(f.text,f.name);r.holes=d.holes;r.warn=d.warnings;r.info=d.holes.length+' отверстий';}
    else r.info='—';
  }catch(e){r.error=String(e.message||e);r.info='ошибка разбора';}
  f.parsed=r;return r;
}
function renderFiles(){
  const t=$('fileTab');
  if(!S.files.length){t.innerHTML='<tr><th>Файл</th><th>Назначение</th><th>Содержимое</th></tr><tr><td colspan="3" class="empty">Файлы не загружены.</td></tr>';return;}
  t.innerHTML='<tr><th>Файл</th><th>Назначение</th><th>Содержимое</th></tr>'+S.files.map((f,i)=>{const p=parseFile(f);
    const warn=(p.warn&&p.warn.length)?` <span class="pill warn" title="${esc(p.warn.join('; '))}">${p.warn.length} замеч.</span>`:'';
    return `<tr><td class="mono" style="word-break:break-all">${esc(f.name)}</td><td><select data-i="${i}" aria-label="Назначение файла ${esc(f.name)}">${ROLE_OPTS.map(o=>`<option value="${o[0]}"${o[0]===f.role?' selected':''}>${o[1]}</option>`).join('')}</select></td><td>${p.error?'<span class="pill bad">'+esc(p.error)+'</span>':esc(p.info)}${warn}</td></tr>`;}).join('')+
    `<tr><td colspan="3"><button type="button" id="clearFiles">Очистить список</button></td></tr>`;
  t.querySelectorAll('select').forEach(s=>s.addEventListener('change',()=>{S.files[+s.dataset.i].role=s.value;S.files[+s.dataset.i].parsed=null;afterFilesChanged();}));
  $('clearFiles').onclick=()=>{S.files=[];S.regions=null;S.model=null;S.res=null;S.layerCache={};afterFilesChanged();renderResults();};
}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function afterFilesChanged(){renderFiles();markDirty();S.layerCache={};
  const cu=S.files.filter(f=>f.role.startsWith('copper')).length, dr=S.files.filter(f=>f.role==='drill').length;
  status(cu&&dr?`Слоёв меди: ${cu}, файлов сверловки: ${dr}. Нажмите «Рассчитать».`:(S.files.length?'Нужен хотя бы один слой меди и файл сверловки с металлизированными отверстиями.':'Загрузите файлы или откройте пример.'));}
const drop=$('drop');
['dragenter','dragover'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('over');}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove('over');}));
drop.addEventListener('drop',ev=>{if(ev.dataTransfer&&ev.dataTransfer.files)addFiles([...ev.dataTransfer.files]);});
$('fileIn').addEventListener('change',e=>{addFiles([...e.target.files]);e.target.value='';});
function loadExample(tear){
  const ex=GEngine.exampleFig92(tear,3);S.source='example';S.files=[];ex.files.forEach(f=>pushFile(f.name,f.text));S.regions=ex.regions;S.regions.tear=tear;
  afterFilesChanged();run();
}
$('ex1').onclick=()=>loadExample(false);$('ex2').onclick=()=>loadExample(true);

/* ---------- params ---------- */
const geomIds=['cls','ring','ringInner','breakout','junc','kred','eros'], errIds=['sl','sg','sgInner','srot','ssc','sd','mx','my'], evalIds=['kgain','wminMode','hcu','xi','F','lcorr','trials'];
let activeProfile;
function classDefaults(){
  const cls = Number($('cls').value);
  $('kgain').value = cls===3 ? 20 : 30;
  $('kred').value = 100 * (activeProfile?.width_loss['class'+cls] ?? (cls===1 ? 0.3 : 0.2));
  $('breakout').value = activeProfile ? (activeProfile.breakout['class'+cls] ?? 0) : (cls===1 ? 180 : 90);
}
$('cls').addEventListener('change',classDefaults);
geomIds.forEach(id=>$(id).addEventListener('change',markDirty));
errIds.concat(evalIds).forEach(id=>$(id).addEventListener('change',()=>{if(!validInputs()){markDirty();return;}if(!S.model || S.dirty)return;const e=readErr();if(GEngine.sigmaMax(e,S.span||0)*6.5>S.model.Rmax+1e-9)markDirty();else reevaluate();}));
function setBusy(busy){
  S.busy=busy;
  root.querySelectorAll('input, select, button').forEach(control=>control.disabled=busy);
}
function validInputs(){
  return [...root.querySelectorAll('input[type=number]')].every(field=>field.reportValidity());
}
function markDirty(){
  $('yield-value').hidden=true;
  if(report) $('runBtn').hidden=false;
  ++evalToken; S.dirty=true; S.res=null; S.model=null; S.detail=null; S.sel=null;
  renderResults(); status('Нажмите «Рассчитать» для новых параметров.');$('runBtn').classList.add('dirty');}
function num(id,def){const v=parseFloat($(id).value);return Number.isFinite(v)?v:def;}
function readCrit(){return{cls:+$('cls').value,breakout:num('breakout',90),ringReq:num('ring',50)/1000,ringInner:num('ringInner',50)/1000,junction:$('junc').value,kred:num('kred',20)/100,erosion:num('eros',0)/1000};}
function readErr(){return{sl:num('sl',0)/1000,sg:num('sg',0)/1000,sgInner:num('sgInner',0)/1000,sd:num('sd',0)/1000,srot:num('srot',0)*1e-6,ssc:num('ssc',0)*1e-6,mx:num('mx',0)/1000,my:num('my',0)/1000};}
function readRoute(){return{gain:num('kgain',20)/100,h:num('hcu',18)/1000,xi:num('xi',20)/1000,F:num('F',2),lcorr:num('lcorr',0)};}
$('runBtn').onclick=run;

/* ---------- run ---------- */
function buildModel(){
  const order=r=>r==='copper:Top'?0:r==='copper:Bottom'?999:(parseInt(r.replace(/\D/g,''),10)||500);
  const cu=S.files.filter(f=>f.role.startsWith('copper')).sort((a,b)=>order(a.role)-order(b.role));
  const layers=cu.map(f=>{
    const parsed=parseFile(f);
    const label=ROLE_OPTS.find(([role])=>role===f.role)?.[1] || 'Медь';
    const inner=f.role.startsWith('copper:In');
    const number=parsed.layerNumber || (inner ? 1+Number(f.role.slice(9)) : f.role==='copper:Top' ? 1 : null);
    return {name:label+' · '+f.name,prims:parsed.prims||[],inner,number,file:f.name};
  }).filter(layer=>layer.prims.length);
  // В многослойной плате соосные глухие отверстия могут иметь разные пролёты.
  const holes=[], seen=new Set();
  for(const file of S.files.filter(f=>f.role==='drill')){
    for(const hole of parseFile(file).holes||[]){
      const key=[hole.x,hole.y,hole.d,...(hole.span||[])].join(',');
      if(!seen.has(key)){seen.add(key);holes.push(hole);}
    }
  }
  return{layers,holes};
}
async function run(){
  if(S.busy)return;
  if (!validInputs()) return;
  const invalid=S.files.find(f=>(f.role.startsWith('copper') || f.role==='drill') && parseFile(f).error);
  if(invalid){status('Ошибка в файле '+invalid.name+': '+parseFile(invalid).error);return;}
  const m=buildModel();
  if(!m.layers.length||!m.holes.length){status('Нужен хотя бы один слой меди и файл сверловки с металлизированными отверстиями.');return;}
  setBusy(true);if(report) $('runBtn').hidden=true;S.sel=null;S.detail=null;
  const crit=readCrit(),err=readErr();
  const bb=GEngine.holeBBox(m.holes);S.span=Math.hypot(bb[2]-bb[0],bb[3]-bb[1])/2;
  const t0=performance.now();
  status(`Анализ КП: ${m.holes.length} отверстий × ${m.layers.length} слоёв…`);prog(0);
  try{
    await GEngine.analyze(mkCanvas,m,crit,err,{res:400,G:121},f=>{prog(f*0.85);status(`Анализ КП: ${Math.round(f*100)} %`);},tick);
    S.model=m;S.dirty=false;$('runBtn').classList.remove('dirty');
    await reevaluate(t0);
  }catch(e){$('runBtn').hidden=false;status('Ошибка расчёта: '+String(e.message||e));console.error(e);}
  setBusy(false);
}
let evalToken=0;
async function reevaluate(t0){
  if(!S.model)return;const my=++evalToken;
  const err=readErr(),route=readRoute(),T=Math.max(200,Math.round(num('trials',2000)));
  status('Монте-Карло…');
  const res=await GEngine.evaluate(S.model,err,route,{trials:T,wminMode:$('wminMode').value,seed:20251029,isStale:()=>my!==evalToken},async f=>{prog(0.85+0.15*(f||0));await tick();});
  if(!res||my!==evalToken)return;
  S.res=res;S.err=err;S.route=route;prog(1);
  const n=res.pads.length;status(`Готово: ${n} КП на ${S.model.layers.length} слоях, ${T} испытаний${t0?`, ${((performance.now()-t0)/1000).toFixed(1)} с`:''}.`);
  renderResults();
}

/* ---------- results ---------- */
function routeFor(lines,cls,route){ // lines: array of {w, L}
  if(!lines.length)return{pr:1,prN:1};
  const w=Math.min(...lines.map(l=>l.w)),Lt=lines.reduce((a,l)=>a+l.L,0);
  const pr=GEngine.Proute(w,route.h,cls,route.xi,route.F,S.model.crit.kred,route.gain),N=route.lcorr>0?Math.max(1,Lt/route.lcorr):1;
  return{pr,prN:Math.pow(pr,N)};
}
function linesIn(L,bb){const out=[];for(const p of L.prims){if(p.t!=='L'||p.pol!=='D'||p.ap.s!=='C')continue;const cx=(p.x1+p.x2)/2,cy=(p.y1+p.y2)/2;if(bb&&(cx<bb[0]||cx>bb[2]||cy<bb[1]||cy>bb[3]))continue;out.push({w:p.ap.d,L:Math.hypot(p.x2-p.x1,p.y2-p.y1)});}return out;}
function summary(filter,bb,wOverride){
  const r=S.res,s=GEngine.subset(r,filter),cls=S.model.crit.cls;let pr=1,prN=1;
  for(const L of S.model.layers){let lines=linesIn(L,bb);if(wOverride)lines=lines.map(l=>({w:wOverride,L:l.L}));
    if(!bb&&L.wmin)lines=lines.filter(l=>l.w>=L.wmin-1e-9);
    const q=routeFor(lines,cls,S.route);pr*=q.pr;prN*=q.prN;}
  return{s,pr,prN,text:s.authorFix*pr,ind:s.prodMarg*prN,mc:s.mc*prN,mcCi:s.ci.map(v=>v*prN)};
}
function renderResults(){
  const tiles=$('tiles'),notes=$('notes');
  const hasPads=!!S.res?.pads.length;
  $('yield-value').hidden=!hasPads;
  $('yield-caption').textContent='';
  $('method-details').hidden=!hasPads;
  $('pad-details').hidden=!hasPads;
  if(!hasPads) {
    $('pad-pick').replaceChildren();
    $('pad-count').textContent='';
  }
  if(!S.res||!S.model){$('selBox').replaceChildren();tiles.innerHTML='';notes.innerHTML='';$('regWrap').hidden=true;$('layWrap').hidden=true;drawMap();renderInspector();return;}
  const r=S.res,sm=summary(()=>true,null);
  if (!r.pads.length) {
    tiles.replaceChildren();
    notes.textContent='Нет контактных площадок для оценки. Проверьте слои меди и металлизированную сверловку.';
    $('regWrap').hidden=true; $('layWrap').hidden=true;
    S.sel=null; S.detail=null; drawMap(); renderSel(); renderInspector(); return;
  }

  const teardrops=r.pads.filter(p=>p.tear).length;
  const nominalFailures=r.pads.filter(p=>!GEngine.mapOk(p.M,0,0)).length;
  $('yield-value').textContent=percent(sm.mc);
  $('yield-caption').textContent='Оценка по площадкам и проводникам · 95 % интервал Монте-Карло '
    +percent(sm.mcCi[0])+' – '+percent(sm.mcCi[1]);
  $('pad-count').textContent='Проверено площадок по слоям: '+r.pads.length+' · обычных: '
    +(r.pads.length-teardrops)+' · с каплей: '+teardrops+'. При номинальном положении проходят '+(r.pads.length-nominalFailures)+' из '+r.pads.length+'.';
  tiles.innerHTML=[
    ['Вписанная окружность',sm.text,`КП ${fmtP(sm.s.authorFix)} · проводники ${fmtP(sm.pr)}`,''],
    ['КП независимы, точные области',sm.ind,`КП ${fmtP(sm.s.prodMarg)} · проводники ${fmtP(sm.prN)}`,''],
    ['Монте-Карло: вся плата сразу',sm.mc,`95 %: ${fmtP(sm.mcCi[0])}…${fmtP(sm.mcCi[1])}`,'mc']
  ].map(([k,v,s,c])=>`<div class="tile ${c==='mc'?'mc':''}"><div class="k">${k}</div><div class="v ${c==='over'?'over':''}">${fmtP(v)}</div><div class="s">${s}</div></div>`).join('');
  // notes
  const ns=[];
  const bad=r.pads.filter(p=>p.Raf<=0).length, planes=S.model.layers.reduce((a,L)=>a+L.pads.filter(p=>p.plane).length,0);
  if(bad)ns.push(['warn',`Для ${bad} КП расчёт по окружности даёт пустую область (R ≤ 0): нарушено условие D<sub>via</sub> ≥ D<sub>drill</sub> + w<sub>min</sub> или поясок меньше требуемого.`]);
  if(planes)ns.push(['mut',`${planes} отверстий стоят в сплошной меди (полигон) и исключены из расчёта.`]);
  notes.innerHTML=ns.map(([c,t])=>`<div class="note"><span class="pill ${c}">${c==='bad'?'ошибка':c==='warn'?'внимание':'сведения'}</span><span>${t}</span></div>`).join('');
  // regions (example)
  if(S.regions){
    $('regTab').innerHTML=`<tr><th>Область</th><th style="text-align:right">Вписанная окружность</th><th style="text-align:right">Независимые КП</th><th style="text-align:right">Монте-Карло</th></tr>`+
      S.regions.map((g,i)=>{const bb=g.bb,q=summary(p=>p.x>=bb[0]&&p.x<=bb[2]&&p.y>=bb[1]&&p.y<=bb[3],bb,g.w);
        return `<tr><td>${g.name}: w = ${fmt(g.w,1)} мм, D<sub>via</sub> = ${fmt(g.D,1)} мм</td><td class="n">${fmt(q.text)}</td><td class="n">${fmt(q.ind)}</td><td class="n">${fmt(q.mc)}</td></tr>`;}).join('');
    $('regWrap').hidden=false;
  } else $('regWrap').hidden=true;
  // layers
  $('layTab').innerHTML=`<tr><th>Слой</th><th style="text-align:right">КП</th><th style="text-align:right">в полигоне</th><th style="text-align:right">w<sub>min</sub>, мм</th><th style="text-align:right">Длина проводников, мм</th><th style="text-align:right">P<sub>route</sub> (один раз)</th><th style="text-align:right">P<sub>route</sub> по длине</th></tr>`+
    r.layerRows.map(l=>`<tr><td>${esc(l.name)}</td><td class="n">${l.pads}</td><td class="n">${l.planes}</td><td class="n">${l.wmin?fmt(l.wmin,3):'—'}</td><td class="n">${fmt(l.Ltot,0)}</td><td class="n">${fmtP(l.pr)}</td><td class="n">${fmtP(l.prN)}</td></tr>`).join('');
  $('layWrap').hidden=false;
  // map layer list
  const sel=$('mapLayer'),prev=sel.value;sel.innerHTML=S.model.layers.map((L,i)=>`<option value="${i}">${esc(L.name)}</option>`).join('');if(prev&&+prev<S.model.layers.length)sel.value=prev;
  listPads();drawMap();renderSel();renderInspector();
}

/* ---------- map ---------- */
const mapC=$('map');let view=null;
function listPads(){
  const layer=S.model.layers[+$('mapLayer').value || 0];
  const pick=$('pad-pick');
  const type=$('pad-type').value;
  const items=S.res.pads.map((pad,i)=>({pad,i})).filter(({pad})=>
    pad.layer===layer && (type==='all' || (type==='tear')===pad.tear));
  items.sort((a,b)=>a.pad.pm-b.pad.pm);
  const opts=[new Option('Выберите площадку','')];
  for(const {pad,i} of items){
    const type=pad.tear?'Капля':'КП';
    opts.push(new Option(type+' · X '+fmt(pad.x,3)+' / Y '+fmt(pad.y,3)+' мм · P '+fmtP(pad.pm),i));
  }
  pick.replaceChildren(...opts);
  if(S.detail?.pad.layer===layer) pick.value=String(S.res.pads.indexOf(S.detail.pad));
}
$('pad-type').onchange=()=>{S.detail=null;listPads();drawMap();renderInspector();};
$('pad-pick').onchange=()=>{
  if($('pad-pick').value!=='') openPad(S.res.pads[+$('pad-pick').value]);
  else { S.detail=null;drawMap();renderInspector(); }
};
$('pad-details').ontoggle=()=>{if($('pad-details').open && S.res)drawMap();};
$('mapLayer').addEventListener('change',()=>{S.detail=null;listPads();drawMap();renderInspector();});$('mapColor').addEventListener('change',drawMap);
function boardBB(L){let bb=[Infinity,Infinity,-Infinity,-Infinity];for(const p of L.prims){bb=[Math.min(bb[0],p.bb[0]),Math.min(bb[1],p.bb[1]),Math.max(bb[2],p.bb[2]),Math.max(bb[3],p.bb[3])];}for(const h of S.model.holes){bb=[Math.min(bb[0],h.x-1),Math.min(bb[1],h.y-1),Math.max(bb[2],h.x+1),Math.max(bb[3],h.y+1)];}return bb;}
function layerImage(i,W,H,bb){
  const key=i+':'+W+':'+H+':'+css('--copper');if(S.layerCache[key])return S.layerCache[key];
  const c=document.createElement('canvas');c.width=W;c.height=H;const x=c.getContext('2d');const s=view.s;
  x.setTransform(s,0,0,-s,(view.pad-bb[0]*s),(view.pad+bb[3]*s));x.fillStyle=css('--copper');x.strokeStyle=css('--copper');
  for(const p of S.model.layers[i].prims)GEngine.renderPrim(x,p,s);
  S.layerCache={};S.layerCache[key]=c;return c;
}
function riskOf(p){const mode=$('mapColor').value;if(mode==='paf')return 1-p.paf;if(mode==='mc')return p.fails.length/S.res.T;return 1-p.pm;}
function drawMap(){
  const ctx=mapC.getContext('2d');
  const Wcss=mapC.parentElement.clientWidth||900,dpr=Math.min(2,window.devicePixelRatio||1);
  if(!S.model||!S.res){mapC.width=Wcss*dpr;mapC.height=Math.round(Wcss*0.5*dpr);ctx.fillStyle=css('--sub');ctx.fillRect(0,0,mapC.width,mapC.height);ctx.fillStyle=css('--muted');ctx.font=`${14*dpr}px ${css('--f-body')}`;ctx.fillText('Карта появится после расчёта',20*dpr,30*dpr);view=null;return;}
  const li=+$('mapLayer').value||0,L=S.model.layers[li];const bb=boardBB(L);
  const W=Math.round(Wcss*dpr),pad=10*dpr,s=Math.min((W-2*pad)/(bb[2]-bb[0]),(W*1.4-2*pad)/(bb[3]-bb[1])),H=Math.round((bb[3]-bb[1])*s+2*pad);
  mapC.width=W;mapC.height=Math.min(H,W*1.4);view={bb,s,pad,li,dpr};
  ctx.fillStyle=css('--sub');ctx.fillRect(0,0,W,mapC.height);
  ctx.drawImage(layerImage(li,W,mapC.height,bb),0,0);
  const X=x=>pad+(x-bb[0])*s,Y=y=>pad+(bb[3]-y)*s;
  ctx.fillStyle=css('--hole');for(const h of S.model.holes){ctx.beginPath();ctx.arc(X(h.x),Y(h.y),Math.max(1,h.d/2*s),0,6.2832);ctx.fill();}
  const g=css('--good'),w=css('--warn'),b=css('--bad');
  for(const p of S.res.pads){if(p.layer!==L)continue;const q=riskOf(p);ctx.fillStyle=q<1e-4?g:q<1e-2?w:b;const r=Math.max(2.5*dpr,Math.min(p.padInsc*s*0.55,9*dpr));ctx.beginPath();ctx.arc(X(p.x),Y(p.y),r,0,6.2832);ctx.fill();
    if(S.detail&&S.detail.pad===p){ctx.strokeStyle=css('--ink');ctx.lineWidth=2*dpr;ctx.beginPath();ctx.arc(X(p.x),Y(p.y),r+4*dpr,0,6.2832);ctx.stroke();}}
  if(S.sel){const [x0,y0,x1,y1]=S.sel;ctx.strokeStyle=css('--ink');ctx.setLineDash([6*dpr,4*dpr]);ctx.lineWidth=1.5*dpr;ctx.strokeRect(X(x0),Y(y1),(x1-x0)*s,(y1-y0)*s);ctx.setLineDash([]);}
}
function toBoard(ev){const r=mapC.getBoundingClientRect(),k=mapC.width/r.width;const px=(ev.clientX-r.left)*k,py=(ev.clientY-r.top)*k;return[view.bb[0]+(px-view.pad)/view.s,view.bb[3]-(py-view.pad)/view.s,px,py];}
let drag=null;
mapC.addEventListener('pointerdown',e=>{if(!view)return;mapC.setPointerCapture(e.pointerId);drag={a:toBoard(e),moved:false};});
mapC.addEventListener('pointermove',e=>{if(!drag)return;const b=toBoard(e);if(Math.hypot(b[2]-drag.a[2],b[3]-drag.a[3])>6){drag.moved=true;S.sel=[Math.min(drag.a[0],b[0]),Math.min(drag.a[1],b[1]),Math.max(drag.a[0],b[0]),Math.max(drag.a[1],b[1])];drawMap();}});
mapC.addEventListener('pointerup',e=>{if(!drag)return;const d=drag;drag=null;
  if(!d.moved){const L=S.model.layers[view.li];let best=null,bd=Infinity;for(const p of S.res.pads){if(p.layer!==L)continue;const dd=Math.hypot(p.x-d.a[0],p.y-d.a[1]);if(dd<bd){bd=dd;best=p;}}
    if(best&&bd*view.s<14*view.dpr+best.padInsc*view.s){openPad(best);}else{S.sel=null;drawMap();renderSel();}return;}
  renderSel();});
function renderSel(){
  const box=$('selBox');if(!S.sel||!S.res){box.innerHTML='';return;}
  const bb=S.sel,q=summary(p=>p.x>=bb[0]&&p.x<=bb[2]&&p.y>=bb[1]&&p.y<=bb[3],bb);
  if(!q.s.n){box.innerHTML='<div class="note"><span class="pill mut">участок</span><span>В рамку не попала ни одна КП.</span></div>';return;}
  box.innerHTML=`<div class="note"><span class="pill mut">участок</span><span>${q.s.n} КП. окружность: <b class="mono">${fmtP(q.text)}</b> · независимые: <b class="mono">${fmtP(q.ind)}</b> · Монте-Карло: <b class="mono">${fmtP(q.mc)}</b></span></div>`;
}
window.addEventListener('resize',()=>{clearTimeout(window.__rt);window.__rt=setTimeout(drawMap,150);});
if(report) new MutationObserver(()=>{
  S.layerCache={};drawMap();renderInspector();
}).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
try{window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>{S.layerCache={};drawMap();renderInspector();});}catch(e){}

/* ---------- inspector ---------- */
function openPad(p){
  const L=p.layer,hole=S.model.holes[p.hole];
  const pads=GEngine.analyzeLayer(mkCanvas,L,[hole],{res:400,margin:Math.max(0.25,S.model.Rmax)});
  if($('pad-type').value!=='all' && ($('pad-type').value==='tear')!==p.tear){
    $('pad-type').value='all';listPads();
  }
  const full=pads[0];$('pad-pick').value=String(S.res.pads.indexOf(p));S.detail={pad:p,full};drawMap();renderInspector();
}
function renderInspector(){
  const box=$('insp');
  if(!S.detail||!S.res){box.className='empty';box.innerHTML='Выберите КП на карте.';return;}
  const {pad:p,full}=S.detail;box.className='';
  const T=S.res.T,cls=S.model.crit.cls;
  const trs=p.traces.length?p.traces.map(t=>`${fmt(t.w,3)} мм, ${Math.round(Math.atan2(t.uy,t.ux)*180/Math.PI)}°`).join('; '):'нет';
  box.innerHTML=`<canvas id="inspC" width="840" height="840" aria-label="Окрестность КП и область допустимых смещений"></canvas>
    <div class="legend"><span><i style="background:var(--ok-fill)"></i>допустимые положения центра отверстия</span><span>врезка справа: смещения центра отверстия в увеличенном масштабе; пунктир — допустимое смещение, сплошная — эллипс 3σ</span></div>
    <div class="kv">
      <span>Слой</span><span>${esc(p.layer.name.split(' · ')[0])}</span>
      <span>Координаты</span><span>${fmt(p.x,3)}; ${fmt(p.y,3)} мм</span>
      <span>Вписанный диаметр КП</span><span>${fmt(2*p.padInsc,3)} мм${p.flashShape?' ('+esc(p.flashShape)+')':''}</span>
      <span>Отверстие</span><span>${fmt(2*p.Rd,3)} мм</span>
      <span>Проводники</span><span style="white-space:normal">${trs}</span>
      <span>Капля</span><span>${p.tear?'есть':'нет'}</span>
      <span>Радиус допустимого смещения</span><span>${fmt(p.Raf*1000,0)} мкм</span>
      <span>P годной КП, окружность</span><span>${fmtP(p.paf)}</span>
      <span>P годной КП, точная область</span><span>${fmtP(p.pm)}</span>
      <span>Брак в Монте-Карло</span><span>${p.fails.length} из ${T}</span>
    </div>`;
  if(full && !full.plane){
    const criteria=p.layer.criteria;
    const nominal=GEngine.makeTester(full,criteria)(0,0);
    const verdict=document.createElement('p');
    const ring=(full.lookup(0,0)-p.Rd-criteria.erosion)*1000;
    verdict.textContent='При номинальном положении: '+(nominal?'годна':'не проходит критерий')
      +' · остаточный поясок '+fmt(ring,0)+' мкм.';
    box.append(verdict);
    if(p.traces.length){
      const select=document.createElement('select');
      for(const [i, trace] of p.traces.entries()){
        const angle=Math.round(Math.atan2(trace.uy,trace.ux)*180/Math.PI);
        select.add(new Option(fmt(trace.w,3)+' мм · '+angle+'°'+(trace.tear?' · с каплей':''),i));
      }
      if(p.traces.length>1){
        const label=document.createElement('label');
        label.append('Проводник для модели ',select);box.append(label);
      }
      const button=document.createElement('button');
      button.textContent='Подставить размеры в модель отверстия';
      button.onclick=()=>{
        $('via-panel').open=true;
        const model=viaModel();
        model.setPad(p,p.traces[+select.value],criteria,S.err,activeProfile?.breakout);
        $('via-source').textContent='Размеры из выбранной КП: '+p.layer.name+' · X '+fmt(p.x,3)
          +' / Y '+fmt(p.y,3)+' мм. Ниже показана модель круглой площадки и капли с этими размерами.';
        $('via-panel').scrollIntoView({block:'start'});
      };
      box.append(button);
    }
  }
  const c=$('inspC'),x=c.getContext('2d'),N=c.width;
  const F=full;if(!F||F.plane){x.fillStyle=css('--sub');x.fillRect(0,0,N,N);return;}
  const H=Math.min(F.Hw,Math.max(p.padInsc,(p.flashR||p.padInsc),p.Rd)*1.2+0.08),s=N/(2*H);
  // copper
  const m=GEngine.rasterWindow(mkCanvas,p.layer.prims,GEngine.query(p.layer.idx,p.layer.prims,F.x-H,F.y-H,F.x+H,F.y+H),F.x,F.y,H,N);
  const img=x.createImageData(N,N),cu=hex(css('--copper-soft')),bg=hex(css('--sub'));
  for(let i=0;i<N*N;i++){const c2=m[i]?cu:bg;img.data[4*i]=c2[0];img.data[4*i+1]=c2[1];img.data[4*i+2]=c2[2];img.data[4*i+3]=255;}
  x.putImageData(img,0,0);
  const X=v=>N/2+v*s,Y=v=>N/2-v*s;
  x.strokeStyle=css('--ink');x.lineWidth=2;x.globalAlpha=.6;x.beginPath();x.arc(X(0),Y(0),p.Rd*s,0,6.2832);x.stroke();x.globalAlpha=1;
  x.fillStyle=css('--ink');x.font=`22px ${css('--f-mono')}`;x.fillText(`КП: окно ±${fmt(H,2)} мм`,14,N-16);
  // inset: offsets of the hole centre, magnified
  const M=p.M;if(!M)return;
  const I=Math.round(N*0.46),ix=N-I-12,iy=12,R=M.Rmax,k=I/(2*R),IX=v=>ix+I/2+v*k,IY=v=>iy+I/2-v*k;
  x.fillStyle=css('--surface');x.fillRect(ix,iy,I,I);x.strokeStyle=css('--line');x.lineWidth=2;x.strokeRect(ix,iy,I,I);
  x.save();x.beginPath();x.rect(ix,iy,I,I);x.clip();
  x.fillStyle=css('--ok-fill');const cs=M.step*k;
  for(let j=0;j<M.G;j++)for(let i=0;i<M.G;i++){if(!M.map[j*M.G+i])continue;const ox=-R+i*M.step,oy=-R+j*M.step;x.fillRect(IX(ox)-cs/2,IY(oy)-cs/2,cs+0.6,cs+0.6);}
  x.strokeStyle=css('--muted');x.lineWidth=1;x.beginPath();x.moveTo(ix,IY(0));x.lineTo(ix+I,IY(0));x.moveTo(IX(0),iy);x.lineTo(IX(0),iy+I);x.stroke();
  x.strokeStyle=css('--bad');x.lineWidth=2.5;x.setLineDash([8,6]);if(p.Raf>0){x.beginPath();x.arc(IX(S.err.mx),IY(S.err.my),p.Raf*k,0,6.2832);x.stroke();}x.setLineDash([]);
  const ctr=S.model.center,dx=p.x-ctr[0],dy=p.y-ctr[1],e=S.err;const sg=p.layer.inner?e.sgInner:e.sg,base=sg*sg+e.sl*e.sl+e.sd*e.sd;
  const sxx=base+e.srot*e.srot*dy*dy+e.ssc*e.ssc*dx*dx,syy=base+e.srot*e.srot*dx*dx+e.ssc*e.ssc*dy*dy,sxy=-e.srot*e.srot*dx*dy;
  const tr=sxx+syy,dt=sxx*syy-sxy*sxy,l1=tr/2+Math.sqrt(Math.max(tr*tr/4-dt,0)),l2=tr/2-Math.sqrt(Math.max(tr*tr/4-dt,0)),ang=0.5*Math.atan2(2*sxy,sxx-syy);
  x.strokeStyle=css('--ink');x.lineWidth=2.5;x.beginPath();x.ellipse(IX(e.mx),IY(e.my),Math.max(1,3*Math.sqrt(Math.max(l1,0))*k),Math.max(1,3*Math.sqrt(Math.max(l2,0))*k),-ang,0,6.2832);x.stroke();
  x.restore();
  x.fillStyle=css('--ink');x.font=`20px ${css('--f-mono')}`;x.fillText(`смещение отв. ±${Math.round(R*1000)} мкм`,ix+8,iy+I-10);
}
function hex(c){c=c.trim();if(c.startsWith('#')){const v=c.length===4?c.slice(1).split('').map(h=>h+h).join(''):c.slice(1);return[parseInt(v.slice(0,2),16),parseInt(v.slice(2,4),16),parseInt(v.slice(4,6),16)];}const m=c.match(/\d+/g);return m?m.slice(0,3).map(Number):[200,200,200];}

let via, profileFields={};
function viaModel(){
  if(via) return via;
  const shadow=$('pad-model').attachShadow({mode:'open'});
  const container=document.createElement('div');
  if(report) shadow.append(document.getElementById('yield-via-style').content.cloneNode(true));
  else {
    const link=document.createElement('link');
    link.rel='stylesheet'; link.href='via.css'; shadow.append(link);
  }
  shadow.append(container);
  via=mountViaModel(container);
  via.setProfile(activeProfile,profileFields);
  return via;
}
$('via-panel').ontoggle=()=>{if($('via-panel').open) viaModel();};
renderResults();
$('current-board').hidden = location.protocol === 'file:';
$('current-board').onclick = async () => {
  if(S.busy) return;
  setBusy(true);
  let loaded=false;
  status('Экспорт слоёв и сверловки из KiCad…');
  try {
    const response = await fetch('board-gerbers', {method:'POST', body:'{}', headers:{
      'Content-Type':'application/json', 'X-DFM-Token':document.querySelector('meta[name=dfm-token]').content
    }});
    const data = await response.json();
    if (!data.files?.length) throw new Error(data.error || 'нет файлов платы');
    S.source='snapshot';S.files=[]; S.regions=null;
    for (const file of data.files) pushFile(file.name,file.text);
    afterFilesChanged(); loaded=true;
  } catch(error){ status('Ошибка экспорта: '+error.message); }
  finally { setBusy(false); }
  if(loaded) await run();
};
function applyProfile(profile, fields) {
  profileFields=fields;
  for (const id of [...geomIds,...errIds,...evalIds]) {
    const field=$(id);
    field.value = field.tagName==='SELECT' ? field.options[0].value : field.defaultValue;
  }
  if(profile){
    $('cls').value=profile.product_class;
    $('ring').value=fields['annular_ring.required_um.class3.outer'];
    $('ringInner').value=fields['annular_ring.required_um.class3.inner'];
    $('sl').value=0;
    const drill=fields['drilling.sigma_drill_um'];
    function layerSigma(side,key){
      const tolerance=fields['drilling.registration_tol_um.'+side];
      if(tolerance===undefined) return fields[key];
      const total=tolerance/(fields['drilling.registration_k'] ?? 4);
      return Math.sqrt(Math.max(0,total*total-drill*drill));
    }
    $('sg').value=layerSigma('outer','drilling.sigma_outer_pattern_um');
    $('sgInner').value=layerSigma('inner','drilling.sigma_layer_um');
    $('sd').value=drill;
    $('ssc').value=fields['drilling.scale_sigma_ppm'];
    $('F').value=fields['etch.etch_factor'];
    $('xi').value=fields['yield.litho_sigma_um']/Math.SQRT2;
    $('kred').value=100*profile.width_loss['class'+profile.product_class];
  }
  activeProfile=profile;
  if(via) via.setProfile(profile,fields);
  classDefaults();
  if(S.files.length) markDirty();
}

if(report){
  $('manual-input').hidden=true;
  $('runBtn').textContent='Пересчитать';
  const data=JSON.parse(document.getElementById('yield-data').textContent);
  const fields=Object.fromEntries(Object.entries(data.profile.fields).map(([key,field])=>[key,field.value]));
  applyProfile(data.profile,fields);
  for(const file of data.files) pushFile(file.name,file.text,file.role);
  afterFilesChanged();
  $('runBtn').hidden=true;
  if(!S.files.some(file=>file.role==='drill' && parseFile(file).holes?.length)){
    status('Для оценки выхода нужна сверловка с металлизированными отверстиями. Подходящих отверстий в этой плате нет.');
  } else {
    setTimeout(run,0);
  }
} else {
  initCalculatorProfiles(applyProfile).then(()=>{
    if(location.protocol!=='file:' && new URLSearchParams(location.search).get('source')!=='files' && !S.files.length){
      $('current-board').onclick();
    }
  });
}
})();
