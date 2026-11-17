// 双目立体深度：区段渲染、查看器（校正核对/深度/有效性/人工确认消歧）
let stereoViewer = null;          // {sid, mode}
const stereoPending = {};         // sid → {list, seed, polygon, probe, cand}
const stereoMarks = {};           // sid → 最近一次探针标记
const stereoImgCache = {};
const STEREO_REASONS = ['有效', '校正有效区外', '纹理不足', '多候选歧义', '遮挡或左右不一致', '确认后无邻近候选', '视差超出范围'];
const STEREO_COLORS = ['#222830', '#000000', '#4874d6', '#eea837', '#e25241', '#a856c4', '#3ec4d6'];

function stereoSection(doc) {
  const records = doc.stereo || [];
  const images = doc.images.map(i => `<option value="${i.id}">${esc(i.name)}</option>`).join('');
  const profiles = doc.profiles.map(p => `<option value="${p.id}">${esc(p.name)}</option>`).join('');
  return `<article><h2>4 · 双目立体深度</h2>
<p class="muted">同一时刻拍摄的左右原图，各自绑定镜头档案，录入左→右旋转（矩阵9数或旋转向量3数）与平移（毫米）。结果固定当时的原图、几何关系与人工确认；之后改档案或重算不影响已保存结果。弱纹理、重复条纹与遮挡处留空并标注原因，不做填洞或平滑猜测。</p>
<div class="toolbar"><label>左图 <select id="stereo-left">${images}</select></label><label>右图 <select id="stereo-right">${images}</select></label><label>左档案 <select id="stereo-lp">${profiles}</select></label><label>右档案 <select id="stereo-rp">${profiles}</select></label></div>
<div class="toolbar"><input id="stereo-rot" class="wide" placeholder="旋转矩阵9数（行优先，左系→右系），留空则用旋转向量"><input id="stereo-rvec" placeholder="或旋转向量3数"><input id="stereo-t" placeholder="平移毫米 tx, ty, tz"></div>
<div class="toolbar"><label>最小视差 <input id="stereo-mind" type="number" value="0"></label><label>视差数 <input id="stereo-numd" type="number" value="128"></label><label>窗口 <input id="stereo-win" type="number" value="9"></label><label>唯一性 <input id="stereo-uniq" type="number" step="0.01" value="0.10"></label><label>纹理阈值 <input id="stereo-tex" type="number" step="0.5" value="3"></label><button class="primary" id="stereo-create">计算并保存立体结果</button></div>
<div class="grid">${records.slice().reverse().map(stereoCard).join('')}</div>
<div id="stereo-viewer"></div></article>`;
}

function stereoCard(r) {
  const conflict = r.conflicts && r.conflicts.length ? `<p class="bad">矛盾确认 ${r.conflicts.length} 处（未应用）</p>` : '';
  const depth = r.stats.depth_mm ? ` · 深度 ${r.stats.depth_mm.min.toFixed(0)}–${r.stats.depth_mm.max.toFixed(0)}mm` : '';
  return `<div class="card"><div class="body"><h3>${esc(r.name)}</h3>
<p class="small">${esc(r.left.name)} ↔ ${esc(r.right.name)}<br>${esc(r.left_profile.name)} · ${esc(r.right_profile.name)}<br>基线 ${r.rig.baseline_mm.toFixed(1)}mm · 有效 ${(r.stats.valid_fraction * 100).toFixed(1)}%${depth}<br>确认 ${r.confirmations.length} 条${r.parent ? ' · 基于上一结果' : ''} · 修订 ${r.created_revision}</p>${conf}
<div class="toolbar"><button data-sview="${r.id}">查看器</button><button data-sconfirm="${r.id}">继续确认</button></div>
<p class="small"><a href="/lens/api${path('/stereo/' + r.id + '/export/depth')}">浮点深度</a> · <a href="/lens/api${path('/stereo/' + r.id + '/export/invalid')}">无效区域</a> · <a href="/lens/api${path('/stereo/' + r.id + '/export/cloud')}">点云PLY</a> · <a href="/lens/api${path('/stereo/' + r.id + '/export/manifest')}">清单</a></p></div></div>`;
}

function bindStereo() {
  const create = $('#stereo-create');
  if (!create) return;
  create.onclick = () => {
    const nums = id => $(id).value.split(/[\s,]+/).filter(Boolean).map(Number);
    const rot = nums('#stereo-rot'), rvec = nums('#stereo-rvec');
    const body = {left_image: $('#stereo-left').value, right_image: $('#stereo-right').value,
                  left_profile: $('#stereo-lp').value, right_profile: $('#stereo-rp').value,
                  translation_mm: nums('#stereo-t'),
                  min_disparity: $('#stereo-mind').value, num_disparities: $('#stereo-numd').value,
                  window: $('#stereo-win').value, uniqueness: $('#stereo-uniq').value,
                  texture: $('#stereo-tex').value, revision: doc.revision,
                  name: '立体 ' + (((doc.stereo || []).length) + 1)};
    if (rot.length) body.rotation = rot; else if (rvec.length) body.rvec = rvec;
    notice('正在计算立体匹配，大图可能需要数十秒…');
    run(() => api(path('/stereo'), body));
  };
  for (const b of document.querySelectorAll('[data-sview]')) b.onclick = () => openStereoViewer(b.dataset.sview, 'rect');
  for (const b of document.querySelectorAll('[data-sconfirm]')) b.onclick = () => openStereoViewer(b.dataset.sconfirm, 'confirm');
  if (stereoViewer && (doc.stereo || []).some(r => r.id === stereoViewer.sid)) openStereoViewer(stereoViewer.sid, stereoViewer.mode);
}

function stereoRecord() { return (doc.stereo || []).find(r => r.id === stereoViewer.sid); }

function openStereoViewer(sid, mode) {
  const record = (doc.stereo || []).find(r => r.id === sid);
  if (!record) return;
  stereoViewer = {sid, mode};
  const tabs = [['rect', '校正核对'], ['depth', '深度与距离'], ['valid', '有效性与原因'], ['confirm', '候选与人工确认']];
  $('#stereo-viewer').innerHTML = `<h3>立体结果查看器 · ${esc(record.name)}</h3>
<div class="toolbar stereo-tabs">${tabs.map(t => `<button data-stab="${t[0]}" class="${t[0] === mode ? 'primary' : ''}">${t[1]}</button>`).join('')}
<a class="button" href="/lens/api${path('/stereo/' + sid + '/export/depth')}">浮点深度.npy</a>
<a class="button" href="/lens/api${path('/stereo/' + sid + '/export/invalid')}">无效区域.png</a>
<a class="button" href="/lens/api${path('/stereo/' + sid + '/export/cloud')}">点云.ply</a>
<a class="button" href="/lens/api${path('/stereo/' + sid + '/export/manifest')}">清单.json</a></div>
<p class="muted" id="stereo-meta"></p><canvas id="stereo-canvas"></canvas><div id="stereo-panel"></div>`;
  for (const b of document.querySelectorAll('[data-stab]')) b.onclick = () => openStereoViewer(sid, b.dataset.stab);
  $('#stereo-meta').textContent = `左 ${record.left.name} · 右 ${record.right.name} · 基线 ${record.rig.baseline_mm.toFixed(1)}mm · 有效 ${(record.stats.valid_fraction * 100).toFixed(1)}% · 确认 ${record.confirmations.length} 条 · 指纹 ${record.fingerprint.slice(0, 12)}`;
  drawStereoCanvas();
}

async function stereoImg(key) {
  if (!stereoImgCache[key]) {
    stereoImgCache[key] = await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = '/lens/api/assets/' + key;
    });
  }
  return stereoImgCache[key];
}

function canvasXY(e, canvas) {
  const rect = canvas.getBoundingClientRect();
  return [Math.round((e.clientX - rect.left) * canvas.width / rect.width),
          Math.round((e.clientY - rect.top) * canvas.height / rect.height)];
}

function polyPath(ctx, pts) {
  ctx.beginPath();
  pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
  ctx.closePath();
}

async function stereoProbe(u, v) {
  try {
    return await api(path('/stereo/' + stereoViewer.sid + '/probe?u=' + u + '&v=' + v));
  } catch (e) { notice(e.message, true); return null; }
}

async function drawStereoCanvas() {
  const record = stereoRecord();
  if (!record) return;
  const mode = stereoViewer.mode, sid = record.id, A = record.assets;
  const canvas = $('#stereo-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const [w, h] = record.rectify.size;
  const mark = stereoMarks[sid];
  if (mode === 'rect') {
    canvas.width = w * 2; canvas.height = h;
    ctx.drawImage(await stereoImg(A.rect_left), 0, 0);
    ctx.drawImage(await stereoImg(A.rect_right), w, 0);
    ctx.strokeStyle = 'rgba(255,255,255,0.10)';
    ctx.lineWidth = 1;
    for (let y = 30; y < h; y += 60) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(2 * w, y); ctx.stroke(); }
    if (mark && mark.mode === 'rect') {
      ctx.strokeStyle = '#ff5340'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(0, mark.v); ctx.lineTo(2 * w, mark.v); ctx.stroke();
      ctx.fillStyle = '#37e6b0'; ctx.fillRect(mark.u - 3, mark.v - 3, 6, 6);
      if (mark.rx !== null) { ctx.fillStyle = '#ffd84d'; ctx.fillRect(w + mark.rx - 3, mark.v - 3, 6, 6); }
    }
    canvas.onclick = async e => {
      const [x, y] = canvasXY(e, canvas);
      if (x >= w) { notice('请在左半图点击'); return; }
      const p = await stereoProbe(x, y);
      if (!p) return;
      stereoMarks[sid] = {mode: 'rect', u: x, v: y, rx: p.valid ? x - p.disparity : null};
      $('#stereo-panel').innerHTML = p.valid
        ? `<p>左(${p.u}, ${p.v}) ↔ 右(${(p.u - p.disparity).toFixed(1)}, ${p.v}) · 视差 ${p.disparity}px · 深度 ${p.depth_mm.toFixed(1)}mm</p><p class="muted">红线为水平极线：校正正确时同一景物落在两图同一行。</p>`
        : `<p>左(${p.u}, ${p.v}) 无效：${esc(p.reason_text)}</p>`;
      drawStereoCanvas();
    };
    $('#stereo-panel').innerHTML = '<p class="muted">点击左图任意点：画出水平极线，并按当前视差标出右图对应位置。</p>';
  } else if (mode === 'depth' || mode === 'valid') {
    canvas.width = w; canvas.height = h;
    ctx.drawImage(await stereoImg(mode === 'depth' ? A.depth_vis : A.reason_vis), 0, 0);
    if (mark && mark.mode === mode) {
      ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(mark.u - 7, mark.v); ctx.lineTo(mark.u + 7, mark.v); ctx.moveTo(mark.u, mark.v - 7); ctx.lineTo(mark.u, mark.v + 7); ctx.stroke();
    }
    canvas.onclick = async e => {
      const [x, y] = canvasXY(e, canvas);
      const p = await stereoProbe(x, y);
      if (!p) return;
      stereoMarks[sid] = {mode, u: x, v: y};
      $('#stereo-panel').innerHTML = p.valid
        ? `<p>(${p.u}, ${p.v}) 深度 ${p.depth_mm.toFixed(1)} mm · 距离 ${p.distance_mm.toFixed(1)} mm · 视差 ${p.disparity} px<br>原始左相机坐标 [${p.point_mm.map(v => v.toFixed(1)).join(', ')}] mm</p>`
        : `<p>(${p.u}, ${p.v}) 无效：${esc(p.reason_text)} · 候选 ${p.candidates.length} 个 · 纹理std ${p.texture_std.toFixed(1)}</p>`;
      drawStereoCanvas();
    };
    if (mode === 'valid') {
      $('#stereo-panel').innerHTML = `<div class="legend">${STEREO_REASONS.map((t, i) => `<span><i style="background:${STEREO_COLORS[i]}"></i>${t}</span>`).join('')}</div><p class="muted">点击查看该点无效原因；无效即留空，不用填洞或平滑猜出的面。</p>`;
    } else {
      $('#stereo-panel').innerHTML = '<p class="muted">点击读取该点毫米深度、距离与原始左相机坐标。</p>';
    }
  } else if (mode === 'confirm') {
    canvas.width = w; canvas.height = h;
    ctx.drawImage(await stereoImg(A.rect_left), 0, 0);
    for (const c of record.confirmations) {
      polyPath(ctx, c.polygon);
      ctx.strokeStyle = c.status === 'applied' ? '#2ecc71' : (c.status === 'conflict' ? '#e25241' : '#e0b84f');
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.fillStyle = ctx.strokeStyle;
      ctx.fillRect(c.pixel[0] - 2, c.pixel[1] - 2, 5, 5);
    }
    const st = stereoPending[sid];
    if (st) {
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = '#4dd0e1';
      for (const c of st.list) { polyPath(ctx, c.polygon); ctx.stroke(); }
      ctx.setLineDash([]);
      if (st.polygon.length) {
        polyPath(ctx, st.polygon);
        ctx.strokeStyle = '#ffd84d';
        ctx.stroke();
        ctx.fillStyle = '#ffd84d';
        for (const p of st.polygon) ctx.fillRect(p[0] - 2, p[1] - 2, 4, 4);
      }
      if (st.seed) {
        ctx.strokeStyle = '#ff5340';
        ctx.beginPath(); ctx.moveTo(st.seed.u - 6, st.seed.v); ctx.lineTo(st.seed.u + 6, st.seed.v); ctx.moveTo(st.seed.u, st.seed.v - 6); ctx.lineTo(st.seed.u, st.seed.v + 6); ctx.stroke();
      }
    }
    canvas.onclick = async e => {
      const [x, y] = canvasXY(e, canvas);
      const st = stereoPending[sid] ??= {list: [], seed: null, polygon: [], probe: null, cand: -1};
      if (st.seed) {
        st.polygon.push([x, y]);
      } else {
        const p = await stereoProbe(x, y);
        if (!p) return;
        st.probe = p; st.cand = -1;
      }
      renderConfirmPanel();
      drawStereoCanvas();
    };
    renderConfirmPanel();
  }
}

function renderConfirmPanel() {
  const record = stereoRecord();
  if (!record) return;
  const sid = record.id;
  const st = stereoPending[sid] ??= {list: [], seed: null, polygon: [], probe: null, cand: -1};
  let html = '';
  if (st.probe) {
    const p = st.probe;
    html += `<p>点 (${p.u}, ${p.v})：${p.valid ? '有效 ' + p.depth_mm.toFixed(1) + 'mm' : '无效 · ' + esc(p.reason_text)} · 纹理std ${p.texture_std.toFixed(1)}</p>`;
    if (p.candidates.length) {
      html += '<p class="muted">说得通的候选对应（选一个作为该点视差）：</p>' + p.candidates.map((c, i) =>
        `<label class="cand"><input type="radio" name="stereo-cand" value="${i}" ${st.cand === i ? 'checked' : ''}> 视差 ${c.disparity.toFixed(2)}px → ${c.depth_mm ? c.depth_mm.toFixed(0) + 'mm' : '超出范围'} · 代价 ${c.cost.toFixed(0)}</label>`).join('')
        + '<button id="stereo-seed">以选中候选为确认点并开始圈选</button>';
    } else html += '<p class="muted">该点没有可用候选（可能在校正有效区外）。</p>';
  } else if (!st.seed) html += '<p class="muted">点击左校正图查看该点候选对应；重复条纹处会列出多个说得通的视差。</p>';
  if (st.seed) {
    html += `<p>确认点 (${st.seed.u}, ${st.seed.v}) · 视差 ${st.seed.d.toFixed(2)}px · 逐点点击圈选所属连续表面：${st.polygon.length} 个顶点</p>
<div class="toolbar"><button id="stereo-close" ${st.polygon.length < 3 ? 'disabled' : ''}>闭合区域</button><button id="stereo-undo">撤销顶点</button><button id="stereo-cancel">放弃</button></div>`;
  }
  if (st.list.length) {
    html += '<p>待提交确认：</p>' + st.list.map((c, i) =>
      `<p class="small">#${i + 1} 点(${c.pixel[0]}, ${c.pixel[1]}) 视差 ${c.disparity.toFixed(2)}px · ${c.polygon.length}顶点 <button data-sdel="${i}">删除</button></p>`).join('')
      + '<button class="primary" id="stereo-submit">提交确认并重新消歧</button>';
  }
  if (record.confirmations.length) {
    html += '<p class="muted">本结果已固定的确认：</p>' + record.confirmations.map(c =>
      `<p class="small">点(${c.pixel[0]}, ${c.pixel[1]}) 视差 ${c.disparity.toFixed(2)} · ${c.applied_px}px <span class="pill ${c.status === 'applied' ? 'ok' : (c.status === 'conflict' ? 'conflict' : 'empty')}">${{applied: '已应用', conflict: '矛盾未应用', empty: '无候选未应用'}[c.status] || esc(c.status)}</span></p>`).join('');
  }
  if (record.conflicts && record.conflicts.length) {
    html += `<p class="bad">互相矛盾的确认（均未应用）：${record.conflicts.map(c => c.a.slice(0, 6) + ' ↔ ' + c.b.slice(0, 6) + '（重叠' + c.overlap_px + 'px）').join('；')}</p>`;
  }
  $('#stereo-panel').innerHTML = html;
  for (const r of document.querySelectorAll('[name=stereo-cand]')) r.onchange = () => { st.cand = +r.value; };
  const seedBtn = $('#stereo-seed');
  if (seedBtn) seedBtn.onclick = () => {
    if (st.cand < 0) { notice('先选择一个候选'); return; }
    const c = st.probe.candidates[st.cand];
    st.seed = {u: st.probe.u, v: st.probe.v, d: c.disparity};
    st.polygon = []; st.probe = null; st.cand = -1;
    renderConfirmPanel(); drawStereoCanvas();
  };
  const closeBtn = $('#stereo-close');
  if (closeBtn) closeBtn.onclick = () => {
    st.list.push({pixel: [st.seed.u, st.seed.v], disparity: st.seed.d, polygon: st.polygon});
    st.seed = null; st.polygon = [];
    renderConfirmPanel(); drawStereoCanvas();
  };
  const undoBtn = $('#stereo-undo');
  if (undoBtn) undoBtn.onclick = () => { st.polygon.pop(); renderConfirmPanel(); drawStereoCanvas(); };
  const cancelBtn = $('#stereo-cancel');
  if (cancelBtn) cancelBtn.onclick = () => { st.seed = null; st.polygon = []; renderConfirmPanel(); drawStereoCanvas(); };
  for (const b of document.querySelectorAll('[data-sdel]')) b.onclick = () => { st.list.splice(+b.dataset.sdel, 1); renderConfirmPanel(); drawStereoCanvas(); };
  const submit = $('#stereo-submit');
  if (submit) submit.onclick = async () => {
    const res = await run(() => api(path('/stereo/' + sid + '/confirm'), {confirmations: st.list, revision: doc.revision}));
    if (res) {
      const records = doc.stereo || [];
      const last = records[records.length - 1];
      delete stereoPending[sid];
      stereoViewer = {sid: last.id, mode: 'confirm'};
      openStereoViewer(last.id, 'confirm');
      if (last.conflicts && last.conflicts.length) notice('存在互相矛盾的确认，未应用并已在列表标出', true);
      else notice('已按确认重新消歧，并保存为新的立体结果');
    }
  };
}
