// Interfaz de SECOPEDIA: llama a la API de FastAPI (/api/...).

const $ = (id) => document.getElementById(id);
let ultimaAuditoria = null;

function mostrarError(id, mensaje) {
  const elemento = $(id);
  elemento.textContent = mensaje;
  elemento.classList.toggle("oculto", !mensaje);
}

async function llamarApi(url, opciones) {
  const respuesta = await fetch(url, opciones);
  const datos = await respuesta.json().catch(() => ({}));
  if (!respuesta.ok) {
    const detalle = typeof datos.detail === "string" ? datos.detail : JSON.stringify(datos.detail);
    throw new Error(detalle || `Error ${respuesta.status}`);
  }
  return datos;
}

function porcentaje(valor) {
  return `${(100 * valor).toFixed(1)}%`;
}

function escapar(texto) {
  const div = document.createElement("div");
  div.textContent = texto ?? "";
  return div.innerHTML;
}

async function cargarEstado() {
  const estado = await llamarApi("/api/estado");
  const selector = $("modelo");
  for (const modelo of estado.modelos) {
    const opcion = document.createElement("option");
    opcion.value = modelo.id;
    opcion.textContent = modelo.disponible ? modelo.nombre : `${modelo.nombre} (no entrenado)`;
    opcion.disabled = !modelo.disponible;
    selector.appendChild(opcion);
  }
  for (const ejemplo of estado.ejemplos) {
    const boton = document.createElement("button");
    boton.type = "button";
    boton.textContent = ejemplo;
    boton.addEventListener("click", () => { $("texto").value = ejemplo; sugerir(); });
    $("ejemplos").appendChild(boton);
  }
  if (!estado.hay_muestra_auditoria) $("usar-muestra").parentElement.classList.add("oculto");
}

async function sugerir() {
  const texto = $("texto").value.trim();
  if (texto.length < 3) return mostrarError("error-sugerir", "Escribe el objeto del contrato.");
  mostrarError("error-sugerir", "");
  $("boton-sugerir").disabled = true;
  try {
    const datos = await llamarApi("/api/predecir", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texto, modelo: $("modelo").value, k: 3 }),
    });
    $("sugerencias").innerHTML = datos.sugerencias.map((s, i) => `
      <li>
        <div class="sugerencia-cabecera">
          <span><strong>${i + 1}. Segmento ${escapar(s.segmento)}</strong>: ${escapar(s.nombre)}</span>
          <span>${porcentaje(s.probabilidad)}</span>
        </div>
        <div class="medidor ${i > 0 ? "secundaria" : ""}"><div style="width:${porcentaje(s.probabilidad)}"></div></div>
      </li>`).join("");
    $("normalizado").textContent = datos.texto_normalizado;
    $("entrada-modelo").textContent = datos.entrada_modelo;
    $("resultado-sugerir").classList.remove("oculto");
  } catch (error) {
    mostrarError("error-sugerir", error.message);
  } finally {
    $("boton-sugerir").disabled = false;
  }
}

async function auditar() {
  const formulario = new FormData();
  const archivo = $("archivo").files[0];
  if (archivo) formulario.append("archivo", archivo);
  else if ($("usar-muestra").checked) formulario.append("usar_muestra", "true");
  else return mostrarError("error-auditar", "Sube un CSV o marca la muestra de contratos.");
  formulario.append("modelo", $("modelo").value);
  formulario.append("umbral", $("umbral").value);

  mostrarError("error-auditar", "");
  $("boton-auditar").disabled = true;
  try {
    ultimaAuditoria = await llamarApi("/api/auditar", { method: "POST", body: formulario });
    const sospechosos = ultimaAuditoria.contratos.filter((c) => c.sospechoso);
    $("resumen-auditoria").textContent = `${ultimaAuditoria.sospechosos} contratos sospechosos de ${ultimaAuditoria.total}`;
    $("filas-auditoria").innerHTML = sospechosos.map((c) => `
      <tr>
        <td>${escapar(c.objeto_del_contrato)}</td>
        <td class="segmento">${escapar(c.segmento_registrado)}<small>${escapar(c.nombre_registrado)}</small></td>
        <td class="segmento">${escapar(c.segmento_sugerido)}<small>${escapar(c.nombre_sugerido)}</small></td>
        <td class="numero">${porcentaje(c.probabilidad)}</td>
        <td class="numero">${porcentaje(c.prob_registrado)}</td>
      </tr>`).join("");
    $("resultado-auditar").classList.remove("oculto");
  } catch (error) {
    mostrarError("error-auditar", error.message);
  } finally {
    $("boton-auditar").disabled = false;
  }
}

function descargarCsv() {
  if (!ultimaAuditoria) return;
  const columnas = Object.keys(ultimaAuditoria.contratos[0]);
  const celda = (valor) => `"${String(valor ?? "").replaceAll('"', '""')}"`;
  const lineas = [columnas.join(","), ...ultimaAuditoria.contratos.map((c) => columnas.map((k) => celda(c[k])).join(","))];
  const enlace = document.createElement("a");
  enlace.href = URL.createObjectURL(new Blob(["﻿" + lineas.join("\n")], { type: "text/csv" }));
  enlace.download = "auditoria_secopedia.csv";
  enlace.click();
  URL.revokeObjectURL(enlace.href);
}

document.querySelectorAll(".pestana").forEach((pestana) => {
  pestana.addEventListener("click", () => {
    document.querySelectorAll(".pestana").forEach((p) => p.classList.toggle("activa", p === pestana));
    document.querySelectorAll(".panel").forEach((panel) => panel.classList.toggle("oculto", panel.id !== pestana.dataset.panel));
  });
});
$("boton-sugerir").addEventListener("click", sugerir);
$("texto").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) sugerir(); });
$("boton-auditar").addEventListener("click", auditar);
$("boton-descargar").addEventListener("click", descargarCsv);
$("umbral").addEventListener("input", () => { $("valor-umbral").textContent = Number($("umbral").value).toFixed(2); });
cargarEstado().catch((error) => mostrarError("error-sugerir", `No se pudo conectar con la API: ${error.message}`));
