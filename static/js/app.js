(function(){
const $=(s,r=document)=>r.querySelector(s), $$=(s,r=document)=>[...r.querySelectorAll(s)];
const CSRF=document.body.dataset.csrf, REQ=+document.body.dataset.required||75;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const plural=(n,w)=>`${n} ${w}${n===1?'':'s'}`;

/* theme + mobile menu */
const root=document.documentElement;
$('#theme')?.addEventListener('click',()=>{const t=root.dataset.theme==='dark'?'light':'dark';root.dataset.theme=t;localStorage.setItem('theme',t);$('#theme').textContent=t==='dark'?'☀️':'🌙'});
if($('#theme'))$('#theme').textContent=root.dataset.theme==='dark'?'☀️':'🌙';
$('#menu')?.addEventListener('click',()=>$('#links').classList.toggle('open'));

/* toasts */
function toast(msg,kind='info'){const el=document.createElement('div');el.className='toast '+kind;el.textContent=msg;el.onclick=()=>el.remove();$('#toasts').appendChild(el);setTimeout(()=>el.remove(),5000)}
$$('#toasts .toast').forEach(t=>{t.onclick=()=>t.remove();setTimeout(()=>t.remove(),5000)});

/* confirm dialog for destructive forms */
$$('form[data-confirm]').forEach(f=>f.addEventListener('submit',e=>{if(!confirm(f.dataset.confirm))e.preventDefault()}));

/* password show/hide + strength meter + match check + busy buttons */
$$('.pw button').forEach(b=>b.addEventListener('click',()=>{const i=b.previousElementSibling;i.type=i.type==='password'?'text':'password';b.textContent=i.type==='password'?'Show':'Hide'}));
const strength=p=>{let s=0;if(p.length>=8)s++;if(p.length>=12)s++;if(/[A-Za-z]/.test(p)&&/\d/.test(p))s++;if(/[^A-Za-z0-9]/.test(p))s++;if(/[a-z]/.test(p)&&/[A-Z]/.test(p))s++;return s};
$$('input[data-meter]').forEach(i=>{const m=$('#'+i.dataset.meter),bar=m.firstElementChild,lab=$('#'+i.dataset.meter+'-label');
 i.addEventListener('input',()=>{const s=strength(i.value),c=['#dc2626','#dc2626','#d97706','#65a30d','#16a34a','#16a34a'][s];bar.style.width=(i.value?Math.max(s,1)*20:0)+'%';bar.style.background=c;if(lab)lab.textContent=i.value?['Very weak','Weak','Fair','Good','Strong','Very strong'][s]:''})});
$$('input[data-match]').forEach(i=>{const o=$('#'+i.dataset.match),chk=()=>i.setCustomValidity(i.value&&i.value!==o.value?'Passwords do not match':'');i.addEventListener('input',chk);o.addEventListener('input',chk)});
$$('form[data-busy]').forEach(f=>f.addEventListener('submit',()=>{const b=$('button[type=submit]',f);if(b&&f.checkValidity()){b.disabled=true;b.textContent='Please wait…'}}));

/* sign-up: show the student or teacher fields */
const roleBoxes=()=>{const t=$('#r-t')?.checked;
 const sb=$('#box-student'),tb=$('#box-teacher');if(!sb||!tb)return;
 sb.hidden=!!t;tb.hidden=!t;
 $$('input',sb).forEach(i=>{i.required=!t;i.disabled=!!t});
 $$('input',tb).forEach(i=>{i.required=!!t;i.disabled=!t})};
$$('input[name=role]').forEach(r=>r.addEventListener('change',roleBoxes));roleBoxes();

/* ring (attendance circle) */
const C=326.73, stateOf=r=>r.shortage?'bad':(r.risk_level==='LOW'?'ok':'warn');
const ringHTML=label=>`<div class="ring"><svg viewBox="0 0 120 120" aria-hidden="true"><circle class="track" cx="60" cy="60" r="52" fill="none" stroke-width="11"/><circle class="fill s-ok" id="rg-fill" cx="60" cy="60" r="52" fill="none" stroke-width="11" transform="rotate(-90 60 60)" stroke-dasharray="0 ${C}"/><line class="tick" x1="60" y1="3" x2="60" y2="17" transform="rotate(${REQ*3.6} 60 60)"/></svg><div class="center"><div><b id="rg-val">–</b><br><span>${label}</span></div></div></div>`;
function setRing(pct,st){const f=$('#rg-fill');f.setAttribute('stroke-dasharray',`${(C*Math.min(pct,100)/100).toFixed(1)} ${C}`);f.setAttribute('class','fill s-'+st);$('#rg-val').textContent=pct.toFixed(1)+'%'}

/* trend chart (dashboard + student detail): actual % and the model's predicted % */
window.drawTrend=function(el,pts){if(!el)return;
 if(pts.length<2){el.outerHTML='<div class="empty">Save at least two updates to see a trend.</div>';return}
 const W=Math.max(300,Math.round(el.clientWidth||640)),H=230,p={l:34,r:12,t:12,b:26},xs=i=>p.l+i*(W-p.l-p.r)/(pts.length-1),ys=v=>p.t+(100-v)*(H-p.t-p.b)/100;
 let s='<defs><linearGradient id="tg" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#4f46e5" stop-opacity=".25"/><stop offset="1" stop-color="#4f46e5" stop-opacity="0"/></linearGradient></defs>';
 s+=[0,25,50,75,100].map(v=>`<line x1="${p.l}" x2="${W-p.r}" y1="${ys(v)}" y2="${ys(v)}" stroke="var(--line)"/><text x="2" y="${ys(v)+4}">${v}</text>`).join('');
 s+=`<line x1="${p.l}" x2="${W-p.r}" y1="${ys(REQ)}" y2="${ys(REQ)}" stroke="#dc2626" stroke-dasharray="5 4"/><text x="${W-p.r-86}" y="${ys(REQ)-5}" style="fill:#dc2626">required ${REQ}%</text>`;
 const line=k=>pts.map((q,i)=>`${i?'L':'M'}${xs(i).toFixed(1)} ${ys(q[k]).toFixed(1)}`).join(' ');
 s+=`<path d="${line('y')} L${xs(pts.length-1)} ${ys(0)} L${xs(0)} ${ys(0)} Z" fill="url(#tg)"/>`;
 s+=`<path d="${line('p')}" fill="none" stroke="#8b5cf6" stroke-width="2" stroke-dasharray="6 5" stroke-linejoin="round"/>`;
 s+=`<path d="${line('y')}" fill="none" stroke="#4f46e5" stroke-width="3" stroke-linejoin="round"/>`+pts.map((q,i)=>`<circle cx="${xs(i)}" cy="${ys(q.y)}" r="5" fill="#4f46e5"><title>${esc(q.x)}: ${q.y}% (predicted ${q.p}%)</title></circle>`).join('');
 s+=`<text x="${p.l}" y="${H-6}">${esc(pts[0].x.slice(0,10))}</text><text x="${W-p.r}" y="${H-6}" text-anchor="end">${esc(pts[pts.length-1].x.slice(0,10))}</text>`;
 el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" class="chart" style="height:${H}px">${s}</svg><div class="legend"><span style="--lc:#4f46e5">Attendance %</span><span style="--lc:#8b5cf6">Predicted next period</span><span style="--lc:#dc2626">Required</span></div>`};

/* update-attendance page: live preview */
const form=$('#predict-form');
if(form){
 const out=$('#result'),save=$('#save'),errBox=$('#form-err'),inputs=$$('input[type=number]',form);let timer,seq=0;
 out.innerHTML=ringHTML('attendance')+'<div id="details"></div>';
 const read=()=>Object.fromEntries(inputs.map(i=>[i.name,i.value.trim()]));
 const post=(url,data)=>fetch(url,{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':CSRF},body:JSON.stringify(data)}).then(async r=>({ok:r.ok,data:await r.json().catch(()=>({error:'Unexpected server response.'}))}));
 function check(v){const n={};for(const k in v){if(!/^\d+$/.test(v[k]))return 'Enter whole numbers (0 or more) in every box.';n[k]=+v[k]}
  if(n.Lectures_Delivered<1)return 'Lectures delivered must be at least 1.';
  const used=n.Lectures_Attended+n.Medical_Leave+n.Duty_Leave;
  if(used>n.Lectures_Delivered)return `Attended (${n.Lectures_Attended}) + medical (${n.Medical_Leave}) + duty (${n.Duty_Leave}) = ${used}, which is more than the ${n.Lectures_Delivered} lectures delivered.`;return ''}
 const stackHTML=r=>{const w=k=>(r[k]/r.delivered*100).toFixed(2);return `<div class="stack"><i class="c-att" style="width:${w('attended')}%"></i><i class="c-med" style="width:${w('medical')}%"></i><i class="c-duty" style="width:${w('duty')}%"></i><i class="c-abs" style="width:${w('absent')}%"></i></div>
  <div class="legend"><span style="--lc:var(--low)">Attended ${r.attended}</span><span style="--lc:var(--blue)">Medical ${r.medical}</span><span style="--lc:var(--violet)">Duty ${r.duty}</span><span style="--lc:var(--high)">Absent ${r.absent}</span></div>`};
 function render(r){const st=stateOf(r);setRing(r.current_attendance,st);
  const msg=r.shortage?(r.must_attend==null?`You are below ${REQ}%.`:`Attend your next <b>${r.must_attend}</b> lecture${r.must_attend===1?'':'s'} in a row to reach ${REQ}%.`):`You can still miss <b>${r.can_skip}</b> lecture${r.can_skip===1?'':'s'} and stay at or above ${REQ}%.`;
  const diff=(r.predicted_attendance-r.current_attendance).toFixed(1);
  $('#details').innerHTML=`<div style="text-align:center;margin-top:10px"><span class="pill ${r.shortage?'short':'ok'}">${r.shortage?'Below required':'Above required'}</span></div><div class="big-msg" style="text-align:center">${msg}</div>${stackHTML(r)}
  <div style="margin-top:14px"><div class="row"><span>Predicted next period</span><b class="num">${r.predicted_attendance}% <small class="muted">(${diff>0?'+':''}${diff})</small></b></div>
  <div class="row"><span>Risk level</span><span><span class="badge ${r.risk_level}">${r.risk_level}</span> <small class="muted">${r.risk_probability_pct}% chance of falling short</small></span></div>
  <div class="row"><span>Your own presence <small class="muted">(without leaves)</small></span><b class="num">${r.own_attendance}%</b></div></div>
  <div class="rec"><b>Recommendation</b><br>${esc(r.recommendation)}</div>`;save.disabled=false}
 function run(){const v=read(),bad=check(v);errBox.hidden=!bad;errBox.textContent=bad;
  if(bad){save.disabled=true;return}
  const my=++seq;post('/api/what-if',v).then(({ok,data})=>{if(my!==seq)return;if(ok)render(data);else{save.disabled=true;errBox.hidden=false;errBox.textContent=data.error}})}
 const queue=()=>{clearTimeout(timer);timer=setTimeout(run,150)};
 inputs.forEach(i=>i.addEventListener('input',queue));
 $$('.stepper button',form).forEach(b=>b.addEventListener('click',()=>{const i=$('input',b.parentElement),v=Math.max(+i.min,(parseInt(i.value,10)||0)+(+b.dataset.step));i.value=v;i.dispatchEvent(new Event('input'))}));
 form.addEventListener('submit',e=>{e.preventDefault();if(check(read()))return;save.disabled=true;save.textContent='Saving…';
  post('/api/predict',read()).then(({ok,data})=>{save.textContent='💾 Save attendance';if(ok){render(data);toast('Attendance saved. Your teacher can now see these numbers.','success')}else{save.disabled=false;toast(data.error,'error')}})});
 $('#reset')?.addEventListener('click',()=>{inputs.forEach(i=>i.value=i.dataset.default);run()});
 run();
}

/* lecture planner (dashboard) */
const pl=$('#planner');
if(pl){const counted=+pl.dataset.counted,del=+pl.dataset.delivered,nx=$('#pl-next'),at=$('#pl-att'),outp=$('#pl-out'),msg=$('#pl-msg');
 const upd=()=>{const n=+nx.value;at.max=n;if(+at.value>n)at.value=n;$('#pl-n').textContent=n;$('#pl-a').textContent=at.value;
  const pct=(counted+ +at.value)/(del+n)*100;outp.textContent=pct.toFixed(1)+'%';outp.style.color=pct>=REQ?'var(--low)':'var(--high)';
  msg.textContent=pct>=REQ?`That keeps you at or above ${REQ}% (you'd miss ${plural(n-at.value,'lecture')}).`:`That would leave you below ${REQ}%.`};
 nx.addEventListener('input',upd);at.addEventListener('input',upd);upd();}

/* table search / filter / sort (history + class pages) */
const tbl=$('table[data-table]');
if(tbl){const body=tbl.tBodies[0],search=$('#search'),filt={};
 const apply=()=>{const q=(search?.value||'').trim().toLowerCase();let n=0;
  [...body.rows].forEach(r=>{const ok=Object.entries(filt).every(([k,v])=>!v||r.dataset[k]===v)&&(!q||r.textContent.toLowerCase().includes(q));r.hidden=!ok;if(ok)n++});
  $('#count')&&($('#count').textContent=n)};
 search?.addEventListener('input',apply);
 $$('.chip[data-fkey]').forEach(c=>c.addEventListener('click',e=>{e.preventDefault();const k=c.dataset.fkey,same=$$('.chip[data-fkey="'+k+'"]');
  if(c.classList.contains('on')&&c.dataset.fval){filt[k]='';c.classList.remove('on');same.find(x=>!x.dataset.fval)?.classList.add('on')}
  else{filt[k]=c.dataset.fval;same.forEach(x=>x.classList.toggle('on',x===c))}apply()}));
 $$('select[data-fkey]').forEach(s=>s.addEventListener('change',()=>{filt[s.dataset.fkey]=s.value;apply()}));
 $$('th.sortable',tbl).forEach(th=>th.addEventListener('click',()=>{const col=th.cellIndex,dir=th.dataset.dir==='asc'?-1:1;th.dataset.dir=dir===1?'asc':'desc';
  const val=r=>{const c=r.cells[col];return c?(c.dataset.v??c.textContent.trim()):''};
  const rows=[...body.rows].sort((a,b)=>{const x=val(a),y=val(b),nx=parseFloat(x),ny=parseFloat(y);return (isNaN(nx)||isNaN(ny)?x.localeCompare(y):nx-ny)*dir});rows.forEach(r=>body.appendChild(r))}));
}
})();
