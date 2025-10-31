// ===== Helpers =====
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);
const withBody = (obj) => ({
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(obj || {})
});

function setToggleVisual(btn, on) {
    if (!btn) return;
    if (on) btn.classList.add("active"); else btn.classList.remove("active");
}

function onOff(v) { return v ? "ON" : "off"; }

// ===== Endpoints =====
async function getStatus() {
    const r = await fetch("/status");
    return await r.json();
}
async function postAction(cmd, value = null) {
    const r = await fetch("/action", withBody({ cmd, value }));
    return await r.json();
}
async function setConf(v) {
    const r = await fetch("/set_conf", withBody({ conf: v }));
    return await r.json();
}
async function setROI(rect) { // {x1,y1,x2,y2} or null
    if (rect) {
        return await fetch("/set_roi", withBody(rect));
    } else {
        return await fetch("/clear_roi", withBody({}));
    }
}

// ===== UI binding =====
async function refreshStatus() {
    const st = await getStatus();

    // labels topo
    $("#srcLabel").textContent = `SRC: ${st.src || "—"}`;
    $("#sizeLabel").textContent = `SIZE: ${st.size || "—"}`;
    $("#fpsLabel").textContent = `FPS: ${st.fps?.toFixed ? st.fps.toFixed(1) : st.fps || "—"}`;

    // toggles: marcar botões ativos
    setToggleVisual(document.querySelector('[data-toggle="objects"]'), st.objects);
    setToggleVisual(document.querySelector('[data-toggle="faces"]'), st.faces);
    setToggleVisual(document.querySelector('[data-toggle="emotion"]'), st.emotion);
    setToggleVisual(document.querySelector('[data-toggle="hands"]'), st.hands);
    setToggleVisual(document.querySelector('[data-toggle="help"]'), st.help);
    setToggleVisual(document.querySelector('[data-toggle="face_blur"]'), st.face_blur);
    setToggleVisual(document.querySelector('[data-toggle="grid"]'), st.grid);
    setToggleVisual(document.querySelector('[data-toggle="motion_only"]'), st.motion_only);
    setToggleVisual(document.querySelector('[data-toggle="auto_rec"]'), st.auto_rec);
    setToggleVisual($("#polyModeBtn"), st.poly_mode);
    setToggleVisual($("#tripwireBtn"), st.tripwire_mode);

    // status text
    $("#st_objects").textContent = onOff(st.objects);
    $("#st_faces").textContent = onOff(st.faces);
    $("#st_emotion").textContent = onOff(st.emotion);
    $("#st_hands").textContent = onOff(st.hands);
    $("#st_face_blur").textContent = onOff(st.face_blur);
    $("#st_grid").textContent = onOff(st.grid);
    $("#st_motion_only").textContent = onOff(st.motion_only);
    $("#st_auto_rec").textContent = onOff(st.auto_rec);
    $("#st_poly_pts").textContent = (st.poly_points || 0);
    $("#st_tripwire").textContent = st.tripwire ? "ON" : "off";
    $("#st_roi").textContent = st.roi ? `${st.roi.x1},${st.roi.y1}→${st.roi.x2},${st.roi.y2}` : "—";

    // slider conf
    if (typeof st.yolo_conf === "number") {
        const percent = Math.round(st.yolo_conf * 100);
        const slider = $("#confRange");
        if (+slider.value !== percent) slider.value = percent;
        $("#confVal").textContent = (st.yolo_conf).toFixed(2);
    }
}

function bindButtons() {
    // ações simples via data-cmd
    $$(".group .btn[data-cmd]").forEach(btn => {
        btn.addEventListener("click", async () => {
            const cmd = btn.getAttribute("data-cmd");
            await postAction(cmd);
            refreshStatus();
        });
    });

    // toggles via data-toggle -> action toggle_<name>
    $$(".group .btn[data-toggle]").forEach(btn => {
        btn.addEventListener("click", async () => {
            const name = btn.getAttribute("data-toggle");
            await postAction(`toggle_${name}`);
            refreshStatus();
        });
    });

    // slider de confiança
    $("#confRange").addEventListener("input", async (e) => {
        const v = (Number(e.target.value) / 100);
        $("#confVal").textContent = v.toFixed(2);
        await setConf(v);
    });
}

// ====== ROI Overlay (canvas) ======
let isDragging = false;
let dragStart = null;
let lastRect = null;

function setupOverlay() {
    const img = $("#videoStream");
    const canvas = $("#overlay");
    const ctx = canvas.getContext("2d");

    // Resize canvas to match image rendering size
    function syncSize() {
        const r = img.getBoundingClientRect();
        canvas.width = r.width;
        canvas.height = r.height;
        canvas.style.width = `${r.width}px`;
        canvas.style.height = `${r.height}px`;
    }
    syncSize();
    window.addEventListener("resize", syncSize);

    function clear() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    function drawRect(r) {
        clear();
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#ffff00";
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(r.x, r.y, r.w, r.h);
        ctx.setLineDash([]);
    }

    function toFrameCoords(px, py) {
        // converte coordenadas do canvas (render) para coordenadas da frame (pixel real)
        // Backend deve converter com base no tamanho da frame real; aqui enviamos normalizado.
        return { x: px / canvas.width, y: py / canvas.height };
    }

    canvas.addEventListener("mousedown", (e) => {
        isDragging = true;
        const rect = canvas.getBoundingClientRect();
        dragStart = { x: e.clientX - rect.left, y: e.clientY - rect.top };
        lastRect = null;
    });

    canvas.addEventListener("mousemove", (e) => {
        if (!isDragging) return;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left, y = e.clientY - rect.top;
        const r = {
            x: Math.min(dragStart.x, x),
            y: Math.min(dragStart.y, y),
            w: Math.abs(x - dragStart.x),
            h: Math.abs(y - dragStart.y)
        };
        lastRect = r;
        drawRect(r);
    });

    canvas.addEventListener("mouseup", async () => {
        if (!isDragging) return;
        isDragging = false;
        if (lastRect && lastRect.w > 8 && lastRect.h > 8) {
            // envia ROI normalizado 0..1
            const a = toFrameCoords(lastRect.x, lastRect.y);
            const b = toFrameCoords(lastRect.x + lastRect.w, lastRect.y + lastRect.h);
            await setROI({ x1: a.x, y1: a.y, x2: b.x, y2: b.y });
        }
        clear();
        refreshStatus();
    });

    // duplo clique limpa ROI
    canvas.addEventListener("dblclick", async () => {
        await setROI(null);
        refreshStatus();
    });
}

// ===== Init =====
bindButtons();
setupOverlay();
refreshStatus();
setInterval(refreshStatus, 1500);
