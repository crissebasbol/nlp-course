"""Notas del orador (una entrada por diapositiva). Se usan en el pptx y en notas_orador.md."""

NOTAS = [
    {
        "titulo": "Título",
        "tiempo": "0:20",
        "texto": [
            "Buenas. Vamos a presentar el taller 8: la implementación del pipeline que diseñamos en el taller 7 para "
            "detectar y corregir errores médicos en notas clínicas, con el dataset MEDEC del shared task MEDIQA-CORR 2024.",
            "Les vamos a contar qué diseñamos, por qué no funcionó como esperábamos, cómo lo diagnosticamos y con qué terminamos.",
        ],
    },
    {
        "titulo": "El problema",
        "tiempo": "0:45",
        "texto": [
            "Cada texto clínico es correcto o tiene exactamente un error médico: un diagnóstico, un tratamiento o un "
            "organismo causal equivocado. Hay tres subtareas. A: decir si el texto tiene error. B: decir en qué oración "
            "está. C: escribir la oración corregida. A y B se miden con accuracy; C con ROUGE-1 y BERTScore.",
            "Ejemplo real del conjunto de prueba (ms-train-90): una paciente con una herida en la pierna que pasa de roja "
            "a morada y negra, con fiebre e hipotensión, después de una picadura. La oración 7 dice «clostridium "
            "botulinum»; la respuesta correcta es «Streptococcus pyogenes».",
            "Ojo con ROUGE: una corrección que cambia «clostridium botulinum» por «clostridium perfringens» saca ROUGE-1 "
            "0,67 y está mal; una reescrita con «Streptococcus pyogenes» saca 0,50 y está bien. Por eso en la interfaz "
            "también revisamos si la corrección contiene el cambio clave.",
            "Datos: 2.189 textos en el conjunto MS, 56 % con error. Separamos train, val y test por caso clínico "
            "(1.528, 334 y 327 textos).",
        ],
    },
    {
        "titulo": "El diseño original (taller 7)",
        "tiempo": "0:50",
        "texto": [
            "En el taller 7 diseñamos este pipeline con modelos abiertos. Primero se reconstruyen las oraciones numeradas.",
            "Luego, dos detectores en paralelo. BiomedBERT, afinado con pares (oración, texto completo), da una "
            "probabilidad p_bert por oración. Qwen2.5-7B cuantizado, con un programa DSPy compilado con BootstrapFewShot, "
            "da un puntaje s_dspy.",
            "La combinación es un promedio ponderado: w · p_bert + (1 − w) · s_dspy. El peso w y el umbral se ajustan en "
            "validación. Si la oración con mayor puntaje supera el umbral, hay error ahí; un corrector con Qwen y ejemplos "
            "parecidos del entrenamiento escribe la corrección, y al final hay un control de calidad.",
            "Lo pesado (entrenar BERT y compilar DSPy) se hizo en un Mac; a los servidores solo van los modelos entrenados.",
        ],
    },
    {
        "titulo": "El giro: A = 0,57",
        "tiempo": "0:50",
        "texto": [
            "Lo implementamos tal cual y lo evaluamos en test: 327 textos que ningún paso vio.",
            "Y aquí viene el giro. En la subtarea A todos los sistemas quedan alrededor de 0,57. Decir siempre «hay error» "
            "saca 0,56, porque el 56 % de los textos de test tiene error. BiomedBERT, Qwen con DSPy y el pipeline completo "
            "quedan en 0,57: el pipeline marca con error el 92 % de los textos.",
            "La primera versión, donde la combinación también elegía la oración, quedó incluso peor: A 0,52 y B 0,51, "
            "porque el puntaje de Qwen arrastraba la elección hacia oraciones equivocadas. Por eso dejamos que BERT "
            "eligiera la oración (B sube a 0,55). Pero A no se movía.",
            "En vez de seguir agregando piezas, decidimos entender por qué.",
        ],
    },
    {
        "titulo": "Diagnóstico: dónde vs. si",
        "tiempo": "1:00",
        "texto": [
            "Primera pregunta: ¿qué aprendió BERT? Cada caso clínico aparece dos veces en el dataset: una versión con "
            "error y una correcta. Este es un par real del test: con error, el tratamiento es imatinib; en la versión "
            "correcta es ácido transretinoico. BERT elige la oración del tratamiento en los dos textos, con 0,965 y 0,968.",
            "Medido en todo el conjunto: en los textos correctos, BERT elige la oración de la respuesta en el 95 % de los "
            "casos en test y en el 98 % en validación. Pero su confianza es la misma con y sin error: el AUC para A es 0,48, "
            "es decir, azar.",
            "BERT aprendió DÓNDE está la oración clave (el diagnóstico o el tratamiento), pero no SI es correcta. Ningún "
            "umbral sobre BERT puede arreglar A.",
            "Qwen 7B tampoco lo sabe: su confianza declarada es casi siempre 0,90 o 0,95, y su AUC queda entre 0,50 y 0,60 "
            "en todas las formas que probamos.",
        ],
    },
    {
        "titulo": "Lo que probamos y la fuga de datos",
        "tiempo": "0:50",
        "texto": [
            "Probamos varias cosas para arreglar A. Primero, que Qwen solo verificara la oración que elige BERT, con una "
            "pregunta de sí o no: A quedó en 0,55; solo ganamos velocidad. Cuando le pedimos a Qwen resolver el caso a "
            "ciegas, su respuesta contiene el cambio clave en uno de cada cinco casos: le falta conocimiento médico.",
            "Luego intentamos darle ese conocimiento a BERT: lo preentrenamos con 10.150 preguntas de MedQA, quitando con "
            "TF-IDF las 1.301 que aparecen en MEDEC. La pérdida se quedó en ln 2, o sea, respondía 50/50: no aprendió nada. "
            "A siguió en 0,56.",
            "El hallazgo más importante: los textos MS se construyeron a partir de preguntas públicas de MedQA. Si "
            "buscamos la pregunta original y revisamos si la oración que elige BERT contiene la respuesta correcta, sin "
            "ningún LLM, llegamos a A = 0,84. Es una fuga de datos: no es un sistema válido, pero muestra que buena parte "
            "de la tarea se resuelve sabiendo la respuesta del examen.",
        ],
    },
    {
        "titulo": "El pipeline final",
        "tiempo": "0:50",
        "texto": [
            "Con ese diagnóstico, rediseñamos repartiendo el trabajo según lo que cada modelo sabe hacer.",
            "BiomedBERT elige la oración más sospechosa, que es lo que hace bien. Después, un LLM con conocimiento médico, "
            "Claude Haiku 5.5, responde una sola pregunta concreta: ¿esta oración es correcta? Si dice que sí, el texto "
            "queda sin error (A = 0, B = −1). Si dice que no, en la misma llamada propone la corrección, que pasa por el "
            "control de calidad.",
            "Usamos DSPy sin compilar, con una firma que recibe el texto numerado y la oración k. DSPy permite cambiar de "
            "proveedor (Haiku, Sonnet o el Qwen local) con una línea del archivo de configuración. El pipeline original "
            "sigue disponible en la interfaz y en los scripts.",
        ],
    },
    {
        "titulo": "Resultados finales",
        "tiempo": "0:55",
        "texto": [
            "Resultados en test. Pipeline original: A 0,57, B 0,55, ROUGE-1 oficial 0,34. BERT + Haiku: A 0,83, B 0,82, "
            "ROUGE-1 oficial 0,69.",
            "Para saber si el pipeline aporta o si todo es mérito de Claude, corrimos Haiku solo, con todo el texto y el "
            "mismo prompt que Opus: A 0,80, B 0,77, ROUGE-1 0,60. BERT + Haiku gana en las tres: BERT ubica mejor la "
            "oración y la pregunta concreta reduce las falsas alarmas. Los textos correctos que se reconocen como "
            "correctos pasan de 14 a 124 de 144 (Haiku solo: 96).",
            "Con honestidad: la diferencia en A con Haiku solo es de unos 8 textos y podría ser ruido; la de B (unos 17 "
            "textos) es más sólida.",
            "Claude Opus 5.5 solo, sin entrenamiento, es el techo: A 0,95, B 0,94, ROUGE-1 oficial 0,80. Costo en todo el "
            "test: Haiku solo US$0,13; Opus US$1,40 con la Batch API.",
        ],
    },
    {
        "titulo": "Demo: un texto con error",
        "tiempo": "0:40",
        "texto": [
            "Un caso con error del test (ms-train-13): un hombre joven con dolor escrotal, ardor al orinar, y una tinción "
            "de Gram con muchos leucocitos pero sin organismos.",
            "BERT puntúa cada oración y elige la 5 con 0,97. Haiku responde que no es correcta y propone «Chlamydia "
            "trachomatis» en lugar de «Neisseria gonorrhoeae», en 3,3 segundos. Coincide con la respuesta correcta.",
        ],
    },
    {
        "titulo": "Demo: un texto correcto",
        "tiempo": "0:30",
        "texto": [
            "Ahora un texto sin error (ms-train-381): el diagnóstico es pemphigus vulgaris. BERT elige esa oración, Haiku "
            "dice que es correcta y el pipeline final responde «sin error» en 1,9 segundos.",
            "El pipeline original, en cambio, marcó error y cambió el diagnóstico por «bullous pemphigoid»: una falsa "
            "alarma que además daña un diagnóstico correcto. Esto es justo lo que mejoró: reconocer los textos correctos.",
        ],
    },
    {
        "titulo": "Despliegue",
        "tiempo": "0:45",
        "texto": [
            "Todo va en Docker, en dos servidores sin GPU (4 vCPU y unos 7,7 GB de RAM libre cada uno). Qwen cuantizado "
            "ocupa unos 5,5 GB y no cabe junto a la app, así que la máquina 1 solo sirve el LLM con llama.cpp en el puerto "
            "8080, que no está abierto a internet. La máquina 2 sirve la app: FastAPI con la interfaz web y BiomedBERT en "
            "CPU, en el puerto 8000.",
            "BiomedBERT afinado está publicado en Hugging Face (crissebasbol/medec-biomedbert-detector), con su ficha; la "
            "app lo descarga al arrancar, así que no hay que reentrenarlo ni copiarlo a mano.",
            "Con la API key, la app usa el pipeline final con Haiku: en los servidores el análisis completo responde en "
            "aproximadamente 30 segundos. Como el puerto 8000 es público, hay un tope de llamadas a Claude por hora. Todo "
            "está en DESPLIEGUE.md, con scripts para revisar los servidores, desplegar, ver logs y detener.",
        ],
    },
    {
        "titulo": "Limitaciones y lecciones",
        "tiempo": "0:45",
        "texto": [
            "Memorización: los casos son preguntas públicas de MedQA, así que Haiku y Opus pudieron verlas en su "
            "entrenamiento. Parte del resultado puede ser memoria y no razonamiento clínico.",
            "Privacidad: el pipeline final envía el texto a una API externa. Con el subset MS, que es público, está bien; "
            "con notas clínicas reales, como el subset UW, no sería aceptable.",
            "Compilar DSPy no ayudó: con Qwen, B bajó en test de 0,370 a 0,355 y lo sesgó hacia «hay error» (los 3 ejemplos "
            "que eligió eran todos con error); con Haiku, en validación, A bajó de 0,835 a 0,826.",
            "La lección principal: el pipeline original tenía más piezas y no funcionaba. Lo que nos sacó del estancamiento "
            "no fue agregar complejidad, sino medir qué sabía cada componente y diagnosticar dónde estaba el cuello de botella.",
        ],
    },
    {
        "titulo": "Demo en vivo de la interfaz",
        "tiempo": "opcional",
        "texto": [
            "Si hay tiempo y conexión: abrir la interfaz en http://<IP máquina 2>:8000, elegir un ejemplo del test, dejar "
            "«BERT elige la oración + Claude Haiku 5.5 verifica» y pulsar Analizar.",
            "Mostrar el progreso de cada paso, la oración marcada, la corrección con las palabras cambiadas y la comparación "
            "con la respuesta correcta. Si no hay conexión, reproducir el GIF o video de la interfaz.",
        ],
    },
    {
        "titulo": "Cierre",
        "tiempo": "0:20",
        "texto": [
            "En resumen: BERT sabe dónde mirar, pero saber si algo está bien requiere conocimiento médico. Medir y "
            "diagnosticar nos llevó de A = 0,57 a 0,83.",
            "Gracias. ¿Preguntas?",
        ],
    },
]

PREGUNTAS = [
    ("¿Por qué no usar solo Claude Opus, si es el mejor?",
     "Opus solo marca el techo (A 0,95), pero cuesta más (US$1,40 en test frente a centavos) y no aprovecha nada del "
     "pipeline. Además, el objetivo del taller era un pipeline con BERT y un LLM; BERT + Haiku supera a Haiku solo."),
    ("¿El resultado de Claude es razonamiento o memoria?",
     "No lo sabemos. Los textos vienen de MedQA, que es público, y una «hoja de respuestas» sin LLM ya llega a A 0,84. "
     "Por eso lo presentamos como limitación."),
    ("¿Por qué no un LLM médico más grande en los servidores?",
     "Los servidores no tienen GPU (4 vCPU, ~7,7 GB de RAM libre). Qwen 7B cuantizado ya ocupa ~5,5 GB y tarda minutos "
     "por análisis en CPU."),
    ("¿Cómo se eligieron w y el umbral?",
     "En validación. El pipeline final del diseño original usa w = 0,4 y umbral 0,4; test no se usó para ajustar nada."),
    ("¿Qué es el «cambio clave»?",
     "Las palabras que la referencia agrega a la oración original (el organismo o el fármaco correcto). Haiku lo incluye "
     "en el 60 % de sus correcciones; Qwen en el 17 %."),
    ("¿Cuántos errores deja pasar BERT + Haiku?",
     "Detecta 146 de 183 textos con error (Haiku solo: 166). De los 37 que deja pasar, 32 son por conocimiento de Haiku "
     "y 5 porque BERT eligió otra oración."),
]
