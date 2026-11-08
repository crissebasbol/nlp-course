# Despliegue en los servidores

Guía paso a paso para desplegar el taller 8 en las dos máquinas. Todo se ejecuta **desde tu computador**: los scripts entran a los servidores por SSH, copian lo necesario y levantan los contenedores.

| Máquina | Qué corre | Puerto |
|---|---|---|
| Máquina 1 (`IP_M1`) | Contenedor `llm`: llama.cpp con Qwen2.5-7B (Q4_K_M, ~5,5 GB de RAM) | 8080, solo red interna |
| Máquina 2 (`IP_M2`) | Contenedor `app`: API + interfaz web + BiomedBERT + corrector | 8000, abierto a internet |

Las IP reales de las máquinas no están en el repo; pídelas a quien te comparte esta guía.

## 0. Requisitos

**En tu computador:** `git`, `bash`, `ssh`, `tar` y conexión a la red de las máquinas. No necesitas Docker ni Python: las imágenes se construyen en los servidores.

**En cada servidor:** Docker con Compose, acceso a internet y al menos ~7 GB libres en disco (máquina 1: modelo de 4,7 GB + imagen de 1,2 GB; máquina 2: imagen de 2,7 GB + modelos de ~0,9 GB). `./deploy/check_servers.sh` lo revisa.

## 1. Código y modelo BERT

```bash
git clone git@github.com:crissebasbol/nlp-course.git
cd nlp-course && git checkout taller_8
cd talleres/taller_8
```

El modelo BiomedBERT entrenado (418 MB) **no está en el repo** porque GitHub no acepta archivos tan grandes. Te lo pasan como `bert_final.zip`; descomprímelo dentro de `talleres/taller_8`:

```bash
unzip ~/Downloads/bert_final.zip    # debe quedar outputs/models/bert/final/model.safetensors
ls outputs/models/bert/final
```

Lo demás que necesita la app ya está en el repo (programa DSPy, índice de ejemplos, splits y reportes). El modelo Qwen se descarga solo en la máquina 1.

## 2. Configuración

```bash
cp deploy/hosts.env.example deploy/hosts.env    # este archivo no se sube a git
```

Edita `deploy/hosts.env`:

| Variable | Valor |
|---|---|
| `M1_HOST` | IP de la máquina 1 (LLM) |
| `M2_HOST` | IP de la máquina 2 (app) |
| `SSH_USER` | tu usuario en las máquinas |
| `SSH_PORT` | 22, salvo que te digan otro |
| `APP_PORT` | 8000 (debe estar entre 8000 y 8025, los abiertos a internet) |
| `DOCKER` | `"docker"`, o `"sudo docker"` si tu usuario no está en el grupo `docker` |

Opcional, para no escribir la contraseña en cada paso:

```bash
ssh-copy-id <usuario>@<IP_M1>
ssh-copy-id <usuario>@<IP_M2>
```

## 3. Verificar los servidores

```bash
./deploy/check_servers.sh
```

Muestra para cada máquina: CPU (y si tiene AVX2), RAM disponible, disco libre, versión de Docker y Compose, si tu usuario tiene permisos de Docker, salida a internet (Hugging Face) y los puertos 8000 a 8090 que ya estén ocupados. Si "Permisos Docker" falla, usa `DOCKER="sudo docker"`.

## 4. Desplegar

```bash
./deploy/deploy_all.sh
```

Hace, en orden:

1. **Máquina 1** (`deploy_llm.sh`): descarga el modelo Qwen de Hugging Face (4,7 GB, retoma si se corta), levanta el contenedor `llm` y espera a que responda.
2. **Máquina 2** (`deploy_app.sh`): copia el código y los modelos (~450 MB), construye la imagen, levanta el contenedor `app` apuntando al LLM de la máquina 1 y espera a que responda.

La primera vez tarda unos 15 a 30 minutos, casi todo en la descarga y en la construcción de la imagen. Las siguientes, unos minutos.

Comprueba el estado:

```bash
./deploy/status.sh     # ambos contenedores deben salir "healthy"
```

## 5. Activar Claude Haiku (recomendado)

Sin API key la app funciona, pero usa el pipeline original con Qwen en CPU: tarda varios minutos por análisis y sus resultados son bajos (A = 0,57). Con la key se activa el **pipeline final** (BiomedBERT + Claude Haiku 5.5), que responde en segundos y llega a A = 0,83.

La key se guarda **solo en la máquina 2**, en el archivo `~/medec/.env`, que Docker Compose lee al levantar la app. Los scripts de despliegue nunca lo borran ni lo sobrescriben: solo actualizan sus propias líneas.

1. Entra a la máquina 2 y guarda la key sin que quede en el historial de la terminal:

   ```bash
   ssh <usuario>@<IP_M2>
   mkdir -p ~/medec && cd ~/medec
   read -rsp "API key de Anthropic: " KEY && echo
   printf 'ANTHROPIC_API_KEY=%s\n' "$KEY" >> .env && unset KEY
   echo 'CLAUDE_MAX_CALLS_PER_HOUR=20' >> .env     # opcional: tope de llamadas por hora
   chmod 600 .env
   exit
   ```

2. Desde tu computador, vuelve a desplegar la app para que tome la key:

   ```bash
   ./deploy/deploy_app.sh
   ```

3. Verifica que quedó activa:

   ```bash
   curl -s http://<IP_M2>:8000/api/health
   ```

   Debe mostrar `"claude": true` y `"default_verifier": "haiku"`.

**Seguridad y costo:**

- El puerto 8000 está abierto a internet: cualquiera que tenga la URL puede hacer análisis que gastan de esa key. Por eso hay un tope de llamadas por hora (`CLAUDE_MAX_CALLS_PER_HOUR`, 20 por defecto).
- Cada análisis con Haiku cuesta menos de un centavo de dólar. Aun así, lo recomendable es crear **una key solo para el servidor** en la consola de Anthropic, con un límite de gasto, y revocarla cuando termine el curso.
- La key nunca va en el repo ni en `deploy/hosts.env`.
- Para desactivar Claude: borra la línea `ANTHROPIC_API_KEY` de `~/medec/.env` en la máquina 2 y corre `./deploy/deploy_app.sh`.

## 6. Probar

Abre `http://<IP_M2>:8000`:

1. Elige un ejemplo del conjunto de prueba (o pulsa **Aleatorio**).
2. En **"¿Quién decide si hay error?"** deja *BERT elige la oración + Claude Haiku 5.5 verifica* (aparece por defecto si la key está activa).
3. Pulsa **Analizar**. En unos segundos verás la oración marcada, la corrección y la comparación con la respuesta correcta.

La pestaña **Resultados del test** muestra las métricas de todos los sistemas.

## 7. Operación

| Tarea | Comando |
|---|---|
| Estado y salud | `./deploy/status.sh` |
| Logs | `./deploy/logs.sh app` o `./deploy/logs.sh llm` |
| Detener | `./deploy/stop.sh app`, `./deploy/stop.sh llm` o `./deploy/stop.sh all` |
| Actualizar después de cambios en el repo | `git pull`, luego `./deploy/deploy_app.sh` (o `./deploy/deploy_all.sh` si cambió el LLM) |

## Problemas comunes

| Síntoma | Causa probable y solución |
|---|---|
| `permission denied ... docker.sock` | Tu usuario no está en el grupo `docker`: pon `DOCKER="sudo docker"` en `deploy/hosts.env`. |
| `Falta outputs/models/bert/final` | No descomprimiste `bert_final.zip` en `talleres/taller_8` (paso 1). |
| `no space left on device` | Libera espacio: `ssh <usuario>@<IP> "docker system prune -af"` (borra imágenes sin usar) y vuelve a desplegar. |
| La interfaz muestra "LLM (Qwen)" en rojo | El LLM de la máquina 1 no responde o aún está cargando (1 a 2 minutos tras arrancar): `./deploy/logs.sh llm`. |
| No aparece la opción de Claude Haiku | La key no está en `~/medec/.env` de la máquina 2 o no se volvió a desplegar la app (paso 5). |
| La interfaz no abre desde fuera | Comprueba que `APP_PORT` esté entre 8000 y 8025 y que el contenedor esté "healthy". |
