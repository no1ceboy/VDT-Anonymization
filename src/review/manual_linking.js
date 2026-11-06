const palette = ["#ffe0a3", "#bde8df", "#d6ccfb", "#f5c5cf", "#c8ddff", "#e8d5ad", "#c8e6bb", "#ffd0ab", "#c7e8ee", "#ecd0f0"];
const $ = (id) => document.getElementById(id);

let queue = [];
let saved = {};
let currentId = null;
let selectedIds = new Set();
let targetGroup = null;
let saveTimer = null;
let saving = false;
let needsSave = false;
const revisions = {};
let toastTimer = null;

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[ch]));
}
function codepoints(value) { return Array.from(value); }
function currentDoc() { return queue.find(doc => doc.doc_id === currentId) || null; }
function annotation(doc = currentDoc()) {
  if (!doc) return {doc_id:"", status:"in_progress", note:"", decisions:{}, extra_mentions:[]};
  return saved[doc.doc_id] || {doc_id:doc.doc_id,status:"in_progress",note:"",decisions:{},extra_mentions:[]};
}
function allMentions(doc = currentDoc()) {
  if (!doc) return [];
  const extras = annotation(doc).extra_mentions || [];
  return [...doc.mentions, ...extras].sort((a,b) => a.start-b.start || a.end-b.end || a.id.localeCompare(b.id));
}
function mentionDecision(id, doc = currentDoc()) {
  return annotation(doc).decisions?.[id] || {decision:"pending",cluster_id:null};
}
function clustersFor(doc = currentDoc()) {
  const clusters = new Map();
  for (const mention of allMentions(doc)) {
    const state = mentionDecision(mention.id, doc);
    if (state.decision !== "linked" || !state.cluster_id) continue;
    if (!clusters.has(state.cluster_id)) clusters.set(state.cluster_id, []);
    clusters.get(state.cluster_id).push(mention);
  }
  return clusters;
}
function resolvedCount(doc = currentDoc()) {
  return allMentions(doc).filter(m => mentionDecision(m.id, doc).decision !== "pending").length;
}
function overallDone() {
  return queue.filter(doc => ["complete","uncertain"].includes(annotation(doc).status)).length;
}
function showToast(message) {
  $("toast").textContent = message;
  $("toast").classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $("toast").classList.remove("show"), 2600);
}
function setSaveIndicator(value, error = false) {
  const node = $("saveIndicator");
  node.textContent = value;
  node.style.color = error ? "#ffb8bc" : "#b9d4dc";
}

function renderNavigation() {
  $("documentCount").textContent = queue.length;
  $("documentNav").innerHTML = queue.map((doc, index) => {
    const state = annotation(doc).status || "in_progress";
    const done = resolvedCount(doc);
    const total = allMentions(doc).length;
    const status = state === "complete" ? "complete" : state === "uncertain" ? "uncertain" : "";
    return `<button class="doc-nav-item ${doc.doc_id===currentId?"active":""}" data-doc="${escapeHtml(doc.doc_id)}">
      <span class="nav-index">${String(index+1).padStart(2,"0")}</span><span class="nav-doc"><strong>Document ${escapeHtml(doc.doc_id)}</strong><small>${done}/${total} decisions</small></span><span class="nav-status ${status}"></span>
    </button>`;
  }).join("");
  $("documentNav").querySelectorAll("[data-doc]").forEach(button => button.addEventListener("click", () => openDocument(button.dataset.doc)));
  const complete = overallDone();
  $("overallProgressText").textContent = `${complete} / ${queue.length} reviewed`;
  $("overallProgressBar").style.width = queue.length ? `${100*complete/queue.length}%` : "0%";
}

function renderDocumentText(doc) {
  const chars = codepoints(doc.text);
  const mentions = allMentions(doc);
  const boundaries = new Set([0, chars.length]);
  for (const mention of mentions) {
    if (mention.start >= 0 && mention.end <= chars.length && mention.end > mention.start) {
      boundaries.add(mention.start); boundaries.add(mention.end);
    }
  }
  const points = [...boundaries].sort((a,b)=>a-b);
  let html = "";
  for (let i=0;i<points.length-1;i++) {
    const start=points[i], end=points[i+1];
    const segment=chars.slice(start,end).join("");
    const active=mentions.filter(m=>m.start<=start && m.end>=end);
    if (!active.length) { html += escapeHtml(segment); continue; }
    active.sort((a,b)=>(a.end-a.start)-(b.end-b.start) || a.id.localeCompare(b.id));
    const mention=active.find(m=>selectedIds.has(m.id)) || active[0];
    const decision=mentionDecision(mention.id).decision;
    const group=mentionDecision(mention.id).cluster_id;
    const color=group ? palette[(parseInt(group.slice(1),10)-1)%palette.length] : (decision==="pending" ? "#fff0aa" : "#e6edf0");
    const selected=selectedIds.has(mention.id) ? " selected-mark" : "";
    const pending=decision==="pending" ? " pending-mark" : "";
    html += `<mark class="${selected}${pending}" data-mid="${escapeHtml(mention.id)}" style="background:${color}" title="${escapeHtml(mention.id)} · ${escapeHtml(decision)}">${escapeHtml(segment)}</mark>`;
  }
  $("documentText").innerHTML = html;
  $("sourceLength").textContent = `${chars.length.toLocaleString()} characters`;
  $("documentText").querySelectorAll("[data-mid]").forEach(mark => mark.addEventListener("click", () => {
    const id=mark.dataset.mid;
    selectedIds.has(id) ? selectedIds.delete(id) : selectedIds.add(id);
    renderCurrent();
  }));
}

function contextFor(doc, mention) {
  const chars=codepoints(doc.text);
  const left=Math.max(0,mention.start-72), right=Math.min(chars.length,mention.end+100);
  return `${left>0?"…":""}${chars.slice(left,mention.start).join("")}⟦${chars.slice(mention.start,mention.end).join("")}⟧${chars.slice(mention.end,right).join("")}${right<chars.length?"…":""}`;
}
function decisionLabel(state) {
  if (state.decision === "linked") return `Group ${state.cluster_id}`;
  return ({pending:"Needs review",singleton:"Separate",not_entity:"Not entity",uncertain:"Uncertain",span_error:"Bad span"})[state.decision] || "Needs review";
}
function renderMentions(doc) {
  const mentions=allMentions(doc);
  const box=$("mentionList");
  if (!mentions.length) {
    box.innerHTML='<div class="empty-state">No automatic candidates here. Select a name or place in the text above and add it as a mention.</div>';
  } else {
    box.innerHTML=mentions.map(mention=>{
      const state=mentionDecision(mention.id);
      const tag=state.decision === "linked" ? "linked" : state.decision;
      return `<div class="mention-row ${state.decision==="not_entity"||state.decision==="span_error"?"filtered":""}" data-row="${escapeHtml(mention.id)}">
        <input type="checkbox" aria-label="Select ${escapeHtml(mention.text)}" data-check="${escapeHtml(mention.id)}" ${selectedIds.has(mention.id)?"checked":""}>
        <div class="mention-body" data-jump="${escapeHtml(mention.id)}"><span class="mention-surface"><span class="mention-id">${escapeHtml(mention.id)}</span>${escapeHtml(mention.text)}</span><span class="mention-context">${escapeHtml(contextFor(doc,mention))}</span></div>
        <span class="decision-tag ${tag}">${escapeHtml(decisionLabel(state))}</span>
      </div>`;
    }).join("");
    box.querySelectorAll("[data-check]").forEach(input=>input.addEventListener("change",()=>{
      input.checked?selectedIds.add(input.dataset.check):selectedIds.delete(input.dataset.check); renderCurrent();
    }));
    box.querySelectorAll("[data-jump]").forEach(body=>body.addEventListener("click",()=>{
      const id=body.dataset.jump; selectedIds.add(id); renderCurrent();
      const mention=allMentions(doc).find(m=>m.id===id);
      const reader=$("documentText");
      const mark=reader.querySelector(`[data-mid="${CSS.escape(id)}"]`);
      if(mark) { mark.scrollIntoView({block:"center",behavior:"smooth"}); }
    }));
  }
  $("selectionCount").textContent=`${selectedIds.size} selected`;
  const enabled=selectedIds.size>0;
  for (const id of ["keepSeparate","notEntity","markUncertain","markSpanError","resetDecision"]) $(id).disabled=!enabled;
  const pending=[...selectedIds].some(id=>mentionDecision(id).decision==="pending");
  $("createGroup").disabled=selectedIds.size<2;
  $("addToGroup").disabled=!targetGroup || !enabled;
}
function renderGroups(doc) {
  const groups=clustersFor(doc);
  $("groupCount").textContent=groups.size;
  const list=$("groupList");
  if (!groups.size) list.innerHTML='<div class="empty-state">No linked groups yet. Select at least two mentions to create one.</div>';
  else list.innerHTML=[...groups.entries()].sort((a,b)=>a[0].localeCompare(b[0])).map(([id,mentions],index)=>
    `<button class="group-chip ${targetGroup===id?"active":""}" data-group="${escapeHtml(id)}" style="border-left:4px solid ${palette[index%palette.length]}"><strong>${escapeHtml(id)}</strong><small>${mentions.length} mentions</small></button>`
  ).join("");
  list.querySelectorAll("[data-group]").forEach(button=>button.addEventListener("click",()=>{targetGroup=button.dataset.group;renderCurrent();}));
}
function renderStatus(doc) {
  const total=allMentions(doc).length;
  const done=resolvedCount(doc);
  const status=annotation(doc).status || "in_progress";
  $("mentionProgress").textContent=`${done} / ${total}`;
  $("mentionProgressBar").style.width=total?`${100*done/total}%`:"0%";
  $("reviewStatusText").textContent=status==="complete"?"Review complete":status==="uncertain"?"Marked uncertain":"In progress";
  $("documentNote").value=annotation(doc).note||"";
  $("completeDocument").disabled=done<total;
}
function renderCurrent() {
  const doc=currentDoc();
  renderNavigation();
  if(!doc) return;
  $("currentDocTitle").textContent=`Document ${doc.doc_id}`;
  $("documentSubhead").textContent=`${allMentions(doc).length} candidate mentions · generated from the original masked text`;
  renderDocumentText(doc);
  renderMentions(doc);
  renderGroups(doc);
  renderStatus(doc);
  $("previousDoc").disabled=queue.findIndex(item=>item.doc_id===currentId)<=0;
  $("nextDoc").disabled=queue.findIndex(item=>item.doc_id===currentId)>=queue.length-1;
  $("addSelectedText").disabled=false;
}

function scheduleSave() {
  if(!currentDoc()) return;
  needsSave=true;
  clearTimeout(saveTimer);
  setSaveIndicator("Unsaved changes");
  saveTimer=setTimeout(saveCurrent,350);
}
async function saveCurrent() {
  const doc=currentDoc();
  if(!doc) return;
  if(saving) { needsSave=true; return; }
  saving=true; setSaveIndicator("Saving…");
  needsSave=false;
  const current=annotation(doc);
  const revision=revisions[doc.doc_id]||0;
  const body=JSON.parse(JSON.stringify({doc_id:doc.doc_id,status:current.status||"in_progress",note:current.note||"",decisions:current.decisions||{},extra_mentions:current.extra_mentions||[]}));
  try {
    const response=await fetch("/api/annotation",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    const result=await response.json();
    if(!response.ok) throw new Error(result.error||"save failed");
    if((revisions[doc.doc_id]||0)===revision) {
      saved[doc.doc_id]={...body,updated_at:result.saved_at};
      if(currentId===doc.doc_id) setSaveIndicator("Saved");
    } else needsSave=true;
    renderNavigation();
  } catch(error) {
    if(currentId===doc.doc_id) setSaveIndicator("Save failed",true);
    showToast(`Could not save: ${error.message}`);
  } finally {
    saving=false;
    if(needsSave) { clearTimeout(saveTimer); saveTimer=setTimeout(saveCurrent,180); }
  }
}
function mutateAnnotation(mutator,{save=true}={}) {
  const doc=currentDoc(); if(!doc)return;
  const value={...annotation(doc),decisions:{...(annotation(doc).decisions||{})},extra_mentions:[...(annotation(doc).extra_mentions||[])]};
  mutator(value);
  saved[doc.doc_id]=value;
  revisions[doc.doc_id]=(revisions[doc.doc_id]||0)+1;
  renderCurrent();
  if(save)scheduleSave();
}
function setDecision(id,decision,cluster_id=null) {
  mutateAnnotation(state=>{ state.decisions[id]={decision,cluster_id}; });
}
function nextGroupId() {
  const groups=clustersFor();
  let n=1; while(groups.has(`E${String(n).padStart(2,"0")}`))n++;
  return `E${String(n).padStart(2,"0")}`;
}
function applySelection(action) {
  if(!selectedIds.size)return;
  const ids=[...selectedIds];
  mutateAnnotation(state=>{
    if(action==="linked") {
      const group=targetGroup || nextGroupId();
      for(const id of ids)state.decisions[id]={decision:"linked",cluster_id:group};
    } else {
      for(const id of ids)state.decisions[id]={decision:action,cluster_id:null};
    }
  });
  if(action==="linked"&&!targetGroup)targetGroup=clustersFor().keys().next().value||null;
  selectedIds.clear();
  renderCurrent();
}

function offsetWithinDocument(range, container, endpointNode, endpointOffset) {
  const prefix=document.createRange();
  prefix.selectNodeContents(container);
  prefix.setEnd(endpointNode,endpointOffset);
  return codepoints(prefix.toString()).length;
}
function addSelectedSpan() {
  const selection=window.getSelection();
  const doc=currentDoc();
  if(!doc||!selection||selection.isCollapsed){showToast("Select a mention in the source text first.");return;}
  const root=$("documentText");
  const range=selection.getRangeAt(0);
  if(!root.contains(range.startContainer)||!root.contains(range.endContainer)){showToast("Select text inside the document panel.");return;}
  const start=offsetWithinDocument(range,root,range.startContainer,range.startOffset);
  const end=offsetWithinDocument(range,root,range.endContainer,range.endOffset);
  const text=codepoints(doc.text).slice(start,end).join("");
  if(!text.trim()||end<=start){showToast("That selection is empty.");return;}
  const duplicate=allMentions(doc).find(m=>m.start===start&&m.end===end);
  if(duplicate){selectedIds.add(duplicate.id);renderCurrent();showToast("That span is already listed.");return;}
  mutateAnnotation(state=>{
    const next=Math.max(0,...state.extra_mentions.map(m=>Number(m.id.slice(1))||0))+1;
    state.extra_mentions.push({id:`X${String(next).padStart(4,"0")}`,start,end,text});
  });
  selection.removeAllRanges();
  showToast("Added source span. Decide whether it is an entity and link it if appropriate.");
}

function openDocument(docId) {
  if(currentId===docId)return;
  clearTimeout(saveTimer);
  currentId=docId; selectedIds.clear(); targetGroup=null;
  renderCurrent();
}

function bindEvents() {
  $("previousDoc").addEventListener("click",()=>{const i=queue.findIndex(d=>d.doc_id===currentId);if(i>0)openDocument(queue[i-1].doc_id);});
  $("nextDoc").addEventListener("click",()=>{const i=queue.findIndex(d=>d.doc_id===currentId);if(i>=0&&i<queue.length-1)openDocument(queue[i+1].doc_id);});
  $("createGroup").addEventListener("click",()=>{if(selectedIds.size<2)return;targetGroup=null;applySelection("linked");targetGroup=[...clustersFor().keys()].sort().at(-1)||null;renderCurrent();});
  $("addToGroup").addEventListener("click",()=>{if(!targetGroup)return;applySelection("linked");});
  $("keepSeparate").addEventListener("click",()=>applySelection("singleton"));
  $("notEntity").addEventListener("click",()=>applySelection("not_entity"));
  $("markUncertain").addEventListener("click",()=>applySelection("uncertain"));
  $("markSpanError").addEventListener("click",()=>applySelection("span_error"));
  $("resetDecision").addEventListener("click",()=>applySelection("pending"));
  $("addSelectedText").addEventListener("click",addSelectedSpan);
  $("documentNote").addEventListener("input",()=>{
    const doc=currentDoc(); if(!doc)return;
    saved[doc.doc_id]={...annotation(doc),note:$("documentNote").value};
    revisions[doc.doc_id]=(revisions[doc.doc_id]||0)+1;
    scheduleSave();
  });
  $("completeDocument").addEventListener("click",()=>{
    if(resolvedCount()<allMentions().length)return;
    mutateAnnotation(state=>{state.status="complete";});
    showToast("Review marked complete.");
  });
  $("uncertainDocument").addEventListener("click",()=>{
    mutateAnnotation(state=>{state.status="uncertain";});
    showToast("Document marked uncertain; it will stay separate from resolved cases.");
  });
}

async function start() {
  bindEvents();
  setSaveIndicator("Loading…");
  try {
    const response=await fetch("/api/state");
    const state=await response.json();
    if(!response.ok)throw new Error(state.error||"Could not load queue");
    queue=state.queue;
    saved=state.annotations||{};
    if(!queue.length)throw new Error("The review queue is empty.");
    currentId=queue[0].doc_id;
    renderCurrent();
    setSaveIndicator("Autosave ready");
  } catch(error) {
    $("currentDocTitle").textContent="Review queue unavailable";
    $("documentSubhead").textContent=error.message;
    setSaveIndicator("Offline",true);
  }
}
start();
