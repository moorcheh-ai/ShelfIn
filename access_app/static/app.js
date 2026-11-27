(() => {
  const banner = document.getElementById("banner");
  const gateOverlay = document.getElementById("gateOverlay");
  const decision = document.getElementById("decision");
  const identity = document.getElementById("identity");
  const scoreEl = document.getElementById("score");
  const message = document.getElementById("message");
  const edgePill = document.getElementById("edgePill");
  const camPill = document.getElementById("camPill");
  const peopleList = document.getElementById("peopleList");
  const enrollForm = document.getElementById("enrollForm");
  const enrollBtn = document.getElementById("enrollBtn");
  const enrollMsg = document.getElementById("enrollMsg");
  const photoInput = document.getElementById("photoInput");
  const fileList = document.getElementById("fileList");
  const dropZone = document.getElementById("dropZone");
  const dropText = document.getElementById("dropText");

  function setPill(el, ok, okText, badText) {
    el.textContent = ok ? okText : badText;
    el.classList.toggle("ok", !!ok);
    el.classList.toggle("bad", !ok);
  }

  function renderStatus(data) {
    const auth = !!data.authorized;
    const waiting = !data.faces;
    const state = waiting ? "waiting" : auth ? "granted" : "denied";

    banner.className = "banner " + state;
    banner.textContent = waiting ? "WAITING" : auth ? "ACCESS GRANTED" : "ACCESS DENIED";
    if (gateOverlay) gateOverlay.className = "gate-overlay " + state;

    decision.className = "decision " + state;
    decision.textContent = waiting ? "Waiting" : auth ? "Authorized" : "Denied";

    identity.textContent = data.label || "Unknown";
    scoreEl.textContent =
      data.score == null || Number.isNaN(data.score) ? "—" : Number(data.score).toFixed(3);
    // Keep errors short in the UI (full detail stays in the server log).
    if (data.error) {
      const short = String(data.error).split("\n")[0];
      message.textContent = short.startsWith("Error:") ? short : `Error: ${short}`;
    } else {
      message.textContent = data.guide || data.message || "";
    }
  }

  async function pollStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      renderStatus(data);
      setPill(edgePill, data.edge_ok, "Edge OK", "Edge down");
      setPill(
        camPill,
        data.camera_ok,
        data.camera_ok ? "Camera OK" : "Camera…",
        "Camera down"
      );
    } catch (err) {
      // This is the Visual Memory web server (/api/status), not Moorcheh Edge.
      message.textContent = "Cannot reach Visual Memory UI API - is ./start.sh running?";
      setPill(camPill, false, "Camera OK", "Camera…");
    }
  }

  async function loadHealth() {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      setPill(edgePill, data.edge_ok, "Edge OK", "Edge down");
      setPill(camPill, data.camera_ok, "Camera OK", "Camera down");
    } catch (_) {
      /* ignore */
    }
  }

  function renderPeople(people) {
    peopleList.innerHTML = "";
    if (!people || !people.length) {
      const li = document.createElement("li");
      li.className = "empty";
      li.textContent = "No enrolled people yet — add photos above.";
      peopleList.appendChild(li);
      return;
    }
    for (const person of people) {
      const li = document.createElement("li");
      li.innerHTML =
        `<span class="name">${escapeHtml(person.name)}</span>` +
        `<span class="count">${person.photo_count} photo${person.photo_count === 1 ? "" : "s"}</span>`;
      peopleList.appendChild(li);
    }
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  async function loadFaces() {
    const res = await fetch("/api/faces");
    const data = await res.json();
    renderPeople(data.people || []);
  }

  function refreshFileList() {
    fileList.innerHTML = "";
    const files = photoInput.files;
    if (!files || !files.length) {
      dropText.textContent = "Drop photos here or click to choose";
      return;
    }
    dropText.textContent = `${files.length} file${files.length === 1 ? "" : "s"} selected`;
    for (const f of files) {
      const li = document.createElement("li");
      li.textContent = `${f.name} (${Math.round(f.size / 1024)} KB)`;
      fileList.appendChild(li);
    }
  }

  photoInput.addEventListener("change", refreshFileList);

  ["dragenter", "dragover"].forEach((evt) => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.add("drag");
    });
  });
  ["dragleave", "drop"].forEach((evt) => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.remove("drag");
    });
  });
  dropZone.addEventListener("drop", (e) => {
    if (e.dataTransfer?.files?.length) {
      photoInput.files = e.dataTransfer.files;
      refreshFileList();
    }
  });

  enrollForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    enrollMsg.hidden = true;
    enrollBtn.disabled = true;
    enrollBtn.textContent = "Inserting…";

    const fd = new FormData();
    fd.append("name", document.getElementById("personName").value.trim());
    for (const f of photoInput.files) {
      fd.append("photos", f, f.name);
    }

    try {
      const res = await fetch("/api/faces/enroll", { method: "POST", body: fd });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const detail =
          typeof data.detail === "string"
            ? data.detail
            : data.detail?.message || JSON.stringify(data.detail || data);
        throw new Error(detail || `HTTP ${res.status}`);
      }
      enrollMsg.hidden = false;
      enrollMsg.className = "enroll-msg ok";
      enrollMsg.textContent = `Enrolled ${data.enrolled} face embedding(s) for ${data.name}.`;
      renderPeople(data.people || []);
      enrollForm.reset();
      refreshFileList();
    } catch (err) {
      enrollMsg.hidden = false;
      enrollMsg.className = "enroll-msg err";
      enrollMsg.textContent = err.message || String(err);
    } finally {
      enrollBtn.disabled = false;
      enrollBtn.textContent = "Insert & authorize";
    }
  });

  document.getElementById("refreshFaces").addEventListener("click", () => {
    loadFaces().catch(() => {});
  });

  const objectForm = document.getElementById("objectForm");
  const itemBtn = document.getElementById("itemBtn");
  const itemMsg = document.getElementById("itemMsg");
  const itemPhotos = document.getElementById("itemPhotos");
  const itemFileList = document.getElementById("itemFileList");
  const itemDrop = document.getElementById("itemDrop");
  const itemDropText = document.getElementById("itemDropText");
  const itemList = document.getElementById("itemList");

  function renderItems(items) {
    itemList.innerHTML = "";
    if (!items || !items.length) {
      const li = document.createElement("li");
      li.className = "empty";
      li.textContent = "No items yet — add a photo and a shelf above.";
      itemList.appendChild(li);
      return;
    }
    for (const item of items) {
      const li = document.createElement("li");
      const shelf = item.shelf ? ` · ${item.shelf}` : "";
      li.innerHTML =
        `<span class="name">${escapeHtml(item.name)}${escapeHtml(shelf)}</span>` +
        `<span class="count">${item.photo_count} photo${item.photo_count === 1 ? "" : "s"}</span>`;
      itemList.appendChild(li);
    }
  }

  function refreshItemFiles() {
    itemFileList.innerHTML = "";
    const files = itemPhotos.files;
    if (!files || !files.length) {
      itemDropText.textContent = "Drop item photos or click to choose";
      return;
    }
    itemDropText.textContent = `${files.length} file${files.length === 1 ? "" : "s"} selected`;
    for (const f of files) {
      const li = document.createElement("li");
      li.textContent = `${f.name} (${Math.round(f.size / 1024)} KB)`;
      itemFileList.appendChild(li);
    }
  }

  itemPhotos.addEventListener("change", refreshItemFiles);
  ["dragenter", "dragover"].forEach((evt) => {
    itemDrop.addEventListener(evt, (e) => {
      e.preventDefault();
      itemDrop.classList.add("drag");
    });
  });
  ["dragleave", "drop"].forEach((evt) => {
    itemDrop.addEventListener(evt, (e) => {
      e.preventDefault();
      itemDrop.classList.remove("drag");
    });
  });
  itemDrop.addEventListener("drop", (e) => {
    if (e.dataTransfer?.files?.length) {
      itemPhotos.files = e.dataTransfer.files;
      refreshItemFiles();
    }
  });

  objectForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    itemMsg.hidden = true;
    itemBtn.disabled = true;
    itemBtn.textContent = "Adding…";
    const fd = new FormData();
    fd.append("name", document.getElementById("itemName").value.trim());
    fd.append("shelf", document.getElementById("itemShelf").value.trim());
    const desc = document.getElementById("itemDescription").value.trim();
    if (desc) fd.append("description", desc);
    for (const f of itemPhotos.files) {
      fd.append("photos", f, f.name);
    }
    try {
      const res = await fetch("/api/objects/enroll", { method: "POST", body: fd });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const detail =
          typeof data.detail === "string"
            ? data.detail
            : data.detail?.message || JSON.stringify(data.detail || data);
        throw new Error(detail || `HTTP ${res.status}`);
      }
      itemMsg.hidden = false;
      itemMsg.className = "enroll-msg ok";
      itemMsg.textContent = `Added ${data.name}. It belongs on ${data.shelf}.`;
      renderItems(data.items || []);
      objectForm.reset();
      refreshItemFiles();
    } catch (err) {
      itemMsg.hidden = false;
      itemMsg.className = "enroll-msg err";
      itemMsg.textContent = err.message || String(err);
    } finally {
      itemBtn.disabled = false;
      itemBtn.textContent = "Add item";
    }
  });

  async function loadItems() {
    const res = await fetch("/api/objects");
    const data = await res.json();
    renderItems(data.items || []);
  }

  loadHealth();
  loadFaces().catch(() => {});
  loadItems().catch(() => {});
  pollStatus();
  setInterval(pollStatus, 500);
})();
