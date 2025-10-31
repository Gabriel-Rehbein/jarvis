async function toggle(feature) {
    const res = await fetch("/toggle", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ feature })
    });
    const data = await res.json();
    loadStatus();
}

async function loadStatus() {
    const res = await fetch("/status");
    const data = await res.json();
    document.getElementById("statusPanel").innerText =
        `Rostos: ${data.faces ? 'Ativo' : 'Inativo'} | Objetos: ${data.objects ? 'Ativo' : 'Inativo'}`;
}

loadStatus();
setInterval(loadStatus, 2000);
