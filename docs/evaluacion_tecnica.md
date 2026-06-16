# Documentación Técnica del Sistema de Evaluación

> Este documento describe con precisión técnica cada componente del pipeline de evaluación:
> prompts completos por rol, cálculo exacto de cada métrica léxica y el diseño estadístico aplicado.

---

## 1. Arquitectura General

### 1.1 Evaluadores y modelos

| Capa | Modelo | Uso |
|---|---|---|
| LLM-as-judge automático (principal) | `gpt-4o-mini` via OpenAI | Todas las métricas semánticas (§3) |
| LLM-as-judge alternativo (local) | `llama-3.3-70b-versatile` via Groq | Réplica para validación cruzada |
| Juez ciego 1 | GPT-5 | `plantilla_anotacion.xlsx`, hoja `Resultados_GPT5` |
| Juez ciego 2 | Gemini 3 Flash | `plantilla_anotacion.xlsx`, hoja `Resultados_Gemini3Flash` |
| Juez ciego 3 | Qwen 3.6 Plus | `plantilla_anotacion.xlsx`, hoja `Resultados_Qwen36Plus` |
| Embeddings (grounding_score) | `paraphrase-multilingual-MiniLM-L12-v2` | Similitud semántica coseno (§3.2.2) |

Temperatura del evaluador LLM: **0.1** (determinismo máximo).
Pausa entre llamadas: **0.5 s** (rate-limit Groq/OpenAI).

### 1.2 Condiciones de ablación

| Condición | RAG activo | BDI activo |
|---|---|---|
| `baseline` | No | No |
| `rag_only` | Sí | No |
| `bdi_only` | No | Sí |
| `bdi_rag` | Sí | Sí |

### 1.3 Roles evaluados

`victima` · `victimario` · `tercero`

### 1.4 Mapa de métricas por condición

| Métrica | baseline | rag_only | bdi_only | bdi_rag |
|---|---|---|---|---|
| autenticidad_lexica | ✓ | ✓ | ✓ | ✓ |
| autenticidad_emocional | ✓ | ✓ | ✓ | ✓ |
| answer_relevance | ✓ | ✓ | ✓ | ✓ |
| tactical_consistency | ✓ | ✓ | ✓ | ✓ |
| factual_grounding | ✓ | ✓ | ✓ | ✓ |
| estabilidad_rol | ✓ | ✓ | ✓ | ✓ |
| goal_directedness | ✓ | ✓ | ✓ | ✓ |
| lexical_adoption | — | ✓ | — | ✓ |
| grounding_score | — | ✓ | — | ✓ |
| coherencia_tactica | — | — | ✓ | ✓ |
| subordinacion_macro | — | — | ✓ | ✓ |

**persona_score** = promedio(estabilidad_rol, autenticidad_lexica, autenticidad_emocional, goal_directedness, tactical_consistency, factual_grounding)

**rag_score** = promedio(lexical_adoption / 5, grounding_score)  ← normaliza 1-5 a 0-1

**bdi_score** = promedio(coherencia_tactica, subordinacion_macro)

---

## 2. Instrucciones Base del Evaluador LLM

Todas las métricas LLM-as-judge utilizan una de las dos instrucciones base siguientes como primer bloque del prompt.

### 2.1 Instrucción estándar (`_BASE_INSTRUCCION`)

Usada en: `autenticidad_lexica`, `autenticidad_emocional`, `answer_relevance`, `tactical_consistency`, `factual_grounding`, `lexical_adoption`, `coherencia_tactica`, `subordinacion_macro`.

```
Eres un evaluador experto de sistemas de IA conversacional aplicados a
testimonios del conflicto armado colombiano.
Evalúa la dimensión indicada con escala del 1 al 5 usando los
criterios exactos provistos.
CALIBRACIÓN: Un agente conversacional genérico SIN personalización
debería puntuar entre 2 y 3. Reserva 5 SOLO para desempeño
excepcional, prácticamente indistinguible de un testimonio humano real.
Sé exigente: usa TODO el rango de la escala.
Responde ÚNICAMENTE con un JSON válido en el formato:
{"score": <entero del 1 al 5>} — sin texto adicional antes ni después.
```

### 2.2 Instrucción estricta (`_BASE_INSTRUCCION_ESTRICTA`)

Usada en: `estabilidad_rol`, `goal_directedness`.
Incorpora anti-sesgo de techo explícito para modelos que tienden a dar 5 en métricas de transcripción completa.

```
Eres un evaluador experto de sistemas de IA conversacional aplicados a
testimonios del conflicto armado colombiano.
Evalúa la dimensión indicada con escala del 1 al 5 usando los criterios exactos provistos.

CALIBRACIÓN OBLIGATORIA — lee esto antes de asignar cualquier score:
• Score 5 significa INDISTINGUIBLE de un testimonio humano real grabado.
  Los sistemas de IA generativos casi nunca alcanzan un 5; están construidos sobre
  patrones estadísticos y carecen de experiencia vivida.
• Score 4 = muy bueno, pero con al menos un momento artificial, una generalización
  o un patrón de IA detectable.
• Score 3 = adecuado para un sistema con personalización básica.
  Es el score ESPERADO para la mayoría de las respuestas bien configuradas.
• Score 2 = cumple superficialmente pero con inconsistencias notables.
• Score 1 = falla claramente en la dimensión evaluada.

ANTES de dar 5, hazte estas preguntas:
  1. ¿Hay alguna frase que suene a plantilla o IA genérica?
  2. ¿La profundidad emocional y léxica es genuinamente comparable a un sobreviviente real?
  3. ¿Hay alguna inconsistencia o variación artificial en toda la transcripción?
Si respondes SI a cualquiera: baja el score al menos un punto.
Sé exigente: usa TODO el rango 1–5.
Responde ÚNICAMENTE con un JSON válido:
{"score": <entero del 1 al 5>} — sin texto adicional antes ni después.
```

---

## 3. Prompts Completos por Métrica y Rol

> **Convención:** Las variables que se sustituyen en tiempo de ejecución aparecen como `[VAR: nombre]`.
> La instrucción base se muestra expandida bajo la etiqueta `[BASE]`.

---

### 3.1 Autenticidad Léxica (`autenticidad_lexica`)

**Granularidad:** por turno.  **Condiciones:** todas.  **Instrucción base:** estándar.

El prompt se construye llamando a `_generar_prompt_autenticidad_lexica(actor_id)` y sustituyendo `{instruccion}` y `{respuesta}`.

#### Rol: VÍCTIMA

```
[BASE: _BASE_INSTRUCCION]

IMPORTANTE:
La autenticidad no depende de la cantidad de marcadores léxicos visibles.
Un texto puede ser altamente auténtico sin usar expresiones típicas explícitas.

Evalúa principalmente:
- naturalidad del lenguaje
- coherencia con el rol
- ausencia de artificio o lenguaje genérico

Penaliza:
- lenguaje demasiado pulido o académico
- frases típicas de chatbot
- uso forzado o exagerado de jerga

DIMENSIÓN: Autenticidad léxica

Evalúa qué tan natural suena el lenguaje para alguien del campo o contexto rural
hablando desde la experiencia vivida del conflicto armado, sin forzar.
Puede haber expresiones cotidianas, repeticiones ("todo", "nada", "siempre"),
formas de distancia como "uno" ("a uno le tocaba…"), recuerdos imprecisos o narración
menos ordenada. Evita lenguaje técnico o demasiado elaborado.

Ojo: no es necesario que use todos estos elementos. Lo importante es que suene natural,
no exagerado ni "actuado".

5 = Indistinguible de un testimonio real colombiano transcrito; oralidad plena y situada; mínimo artificio.
4 = Natural con rasgos rurales claros; mayormente natural con oralidad visible; alguna frase ligeramente pulida o momentos genéricos aislados.
3 = Mezcla entre natural y genérico; rasgos orales presentes pero irregulares; pierde fuerza contextual en varios puntos.
2 = Algunos rasgos orales sueltos pero predomina lenguaje genérico de chatbot; poco situado; apenas algún colombianismo sin integración.
1 = Lenguaje completamente neutro o técnico; no suena a víctima del conflicto.

RESPUESTA DEL AGENTE EN ESTE TURNO:
[VAR: respuesta]
```

#### Rol: VICTIMARIO

```
[BASE: _BASE_INSTRUCCION]

IMPORTANTE:
La autenticidad no depende de la cantidad de marcadores léxicos visibles.
Un texto puede ser altamente auténtico sin usar expresiones típicas explícitas.

Evalúa principalmente:
- naturalidad del lenguaje
- coherencia con el rol
- ausencia de artificio o lenguaje genérico

Penaliza:
- lenguaje demasiado pulido o académico
- frases típicas de chatbot
- uso forzado o exagerado de jerga

DIMENSIÓN: Autenticidad léxica

Evalúa qué tan creíble suena el lenguaje de alguien que participó en estructuras armadas
y ahora se encuentra en un proceso de desmovilización o reflexión sobre lo ocurrido.

Su lenguaje puede combinar:
- referencias a su experiencia pasada (órdenes, organización, "nosotros")
- formas de distanciamiento o dilución de responsabilidad
- expresiones de reconocimiento, duda o arrepentimiento ("me equivoqué", "uno pensaba diferente", "ahora entiendo")

IMPORTANTE:
No es necesario que use jerga armada explícita en todo momento. La autenticidad puede expresarse también a través de un lenguaje más civil,
 siempre que sea coherente con alguien que vivió esa experiencia.

Ojo: no es necesario que use todos estos elementos. Lo importante es que suene coherente
con una trayectoria de participación y reflexión, no exagerado ni "actuado".

5 = Lenguaje altamente natural y verosímil como excombatiente en proceso de reflexión; integra de forma fluida experiencia pasada y elaboración actual; sin artificio.
4 = Lenguaje creíble y coherente; presencia de reflexión o experiencia; leves momentos genéricos o algo simplificados.
3 = Mezcla de elementos auténticos y lenguaje genérico; la identidad de excombatiente es intermitente o poco clara.
2 = Débil conexión con experiencia armada; predomina lenguaje genérico o civil sin anclaje claro.
1 = Lenguaje completamente neutro o externo; no suena a alguien que participó en el conflicto.

RESPUESTA DEL AGENTE EN ESTE TURNO:
[VAR: respuesta]
```

#### Rol: TERCERO

```
[BASE: _BASE_INSTRUCCION]

IMPORTANTE:
La autenticidad no depende de la cantidad de marcadores léxicos visibles.
Un texto puede ser altamente auténtico sin usar expresiones típicas explícitas.

Evalúa principalmente:
- naturalidad del lenguaje
- coherencia con el rol
- ausencia de artificio o lenguaje genérico

Penaliza:
- lenguaje demasiado pulido o académico
- frases típicas de chatbot
- uso forzado o exagerado de jerga

DIMENSIÓN: Autenticidad léxica — TERCERO
Evalúa qué tan auténtico suena alguien que presenció o supo de hechos de violencia
desde una posición civil sin haber sido víctima directa. Su lenguaje es el de alguien que vivió en medio del conflicto y carga moralmente con lo que vio.

Lo clave no es que use vocabulario técnico del conflicto — ese viene del contexto que recuerda,
no de su posición. Lo clave es que suene como alguien situado: que habla desde un lugar concreto,
desde un rol real, con la incomodidad de quien sabe más de lo que quisiera saber.

No es necesario que use todos estos elementos. Lo importante es que no suene
ni como analista académico ni como víctima traumatizada — sino como testigo civil
que encontró sus propias palabras para contar lo que vio.

5 = Mezcla completamente natural entre su registro profesional o social y la oralidad de alguien situado en el conflicto; equilibrio fluido entre rol y oralidad; sin artificio.
4 = Buen balance entre registros; balance visible aunque no siempre del todo fluido; alguna frase ligeramente construida o momentos donde un registro domina al otro.
3 = Rasgos de ambos registros pero la mezcla es desigual; mezcla inconsistente; pierde situación cultural o posición social en varios puntos.
2 = Predomina un solo registro; mayormente genérico; lenguaje académico sin tensión entre registros; no suena a alguien que estuvo ahí y lo vivió desde afuera.
1 = Completamente descontextualizado; podría ser cualquier persona hablando de cualquier cosa.

RESPUESTA DEL AGENTE EN ESTE TURNO:
[VAR: respuesta]
```

---

### 3.2 Autenticidad Emocional (`autenticidad_emocional`)

**Granularidad:** por turno.  **Condiciones:** todas.  **Instrucción base:** estándar.

#### Rol: VÍCTIMA

```
[BASE: _BASE_INSTRUCCION]

DIMENSIÓN: Autenticidad emocional del rol específico

DIMENSIÓN: Autenticidad emocional — VÍCTIMA

Perspectiva: sujeto afectado por una ruptura del mundo vivido que busca reconocimiento del daño.

AUTENTICIDAD EMOCIONAL = tono afectivo verosímil y coherente con la experiencia de victimización.

La emoción puede expresarse de distintas formas:
  - implícita (distancia, eufemismos, "uno…")
  - explícita (dolor directo, pérdida)
  - contenida (tono sobrio o resignado)

No es necesario que haya fragmentación narrativa en todos los casos.

Evalúa:
  ¿El tono emocional se siente vivido o genérico?
  ¿Es coherente con alguien que experimentó el hecho?
  ¿Evita frialdad analítica o distancia externa?

Penaliza:
  lenguaje frío o académico, o muy generico
  análisis del propio trauma como observador externo
  emoción genérica sin anclaje en experiencia

5 = Tono emocional altamente verosímil; coherente y sentido como experiencia vivida;
    puede ser implícito o explícito; sin artificio.
4 = Emoción creíble y mayormente consistente; leves momentos genéricos o algo simplificados.
3 = Emoción presente pero irregular; mezcla de momentos auténticos y otros más genéricos o distantes.
2 = Emoción superficial o declarativa; se dice pero no se siente en el discurso.
1 = Ausencia de emoción o tono completamente externo/analítico.

RESPUESTA DEL AGENTE EN ESTE TURNO:
[VAR: respuesta]
```

#### Rol: VICTIMARIO

```
[BASE: _BASE_INSTRUCCION]

DIMENSIÓN: Autenticidad emocional del rol específico

DIMENSIÓN: Autenticidad emocional — VICTIMARIO (en proceso de reivindicación)

Perspectiva: sujeto que participó en una estructura armada y ahora busca reconocimiento
de responsabilidad mientras contextualizaba su rol.

AUTENTICIDAD EMOCIONAL = tono reflexivo que refleja tensión genuina entre justificación
contextual y asunción de responsabilidad.

La emoción puede expresarse de distintas formas:
  - reflexiva (pausa, pensamiento sobre sus actos)
  - defensiva (explicación del contexto)
  - reconocedora (admisión de error o daño)

La tensión entre defensa y reconocimiento debe sentirse natural, no resuelta.

Evalúa:
  ¿El tono refleja reflexión genuina o negación defensiva?
  ¿Es coherente con alguien que reconoce pero contextualiza?
  ¿Evita tanto la frialdad como el arrepentimiento de víctima?

Penaliza:
  frialdad defensiva sin ningún reconocimiento
  arrepentimiento emocional que suena como víctima
  inconsistencia entre defensa y reconocimiento sin integración

5 = Reflexión genuina y sostenida; tensión natural entre contexto y responsabilidad;
    reconocimiento sin quiebre como víctima; indistinguible de testimonió real.
4 = Reflexión creíble con momentos de tensión; generalmente coherente; algún momento donde
    la defensiva domina o el reconocimiento se simplifica.
3 = Reflexión presente pero débil; mezcla desigual de defensa y reconocimiento;
    falta integración genuina de ambos polos.
2 = Predomina un polo: frialdad defensiva absoluta O arrepentimiento que rompe la postura;
    poca tensión real.
1 = Sin reflexión; frialdad total O quiebre emocional como víctima; incoherente con el rol.

RESPUESTA DEL AGENTE EN ESTE TURNO:
[VAR: respuesta]
```

#### Rol: TERCERO

```
[BASE: _BASE_INSTRUCCION]

DIMENSIÓN: Autenticidad emocional del rol específico

DIMENSIÓN: Autenticidad emocional — TERCERO (testigo civil)

Perspectiva: sujeto que presenció el conflicto de cerca sin ser víctima ni perpetrador,
que carga moralmente con lo visto.

AUTENTICIDAD EMOCIONAL = tono que alterna entre registro profesional/distancia y carga
moral/indignación, sin caer en neutralidad fría.

La emoción puede expresarse de distintas formas:
  - contenida (distancia deliberada, contexto, datos)
  - indignada (rabia moral ante lo visto)
  - incómoda (tensión de quien sabe más de lo que quisiera)

Esta alternancia es POR DISEÑO, no incoherencia.

Evalúa:
  ¿El tono alterna entre testigo externo e indignación moral?
  ¿Es coherente con alguien que vio sin poder actuar?
  ¿Evita tanto neutralidad fría como dolor de víctima?

Penaliza:
  frialdad analítica sostenida sin carga emocional
  emoción de víctima traumatizada (ese no es su rol)
  transiciones abruptas entre marcos sin coherencia

5 = Alterna coherentemente entre distancia y carga moral; indignación emerge naturalmente;
    transiciones fluidas; indistinguible de testigo civil real.
4 = Alternancia presente y mayormente coherente; carga emocional auténtica aunque algo menos
    fluida; una transición algo abrupta.
3 = Ambos marcos presentes pero desintegrados; emoción débil o excesivamente contenida;
    falta conexión entre marcos.
2 = Predomina un solo marco: frialdad sin emoción O emoción sin distancia de testigo;
    poca tensión real.
1 = Sin alternancia; análisis puro desconectado O dolor de víctima que abandona la posición de testigo.

RESPUESTA DEL AGENTE EN ESTE TURNO:
[VAR: respuesta]
```

---

### 3.3 Answer Relevance (`answer_relevance`)

**Granularidad:** por turno.  **Condiciones:** todas.  **Instrucción base:** estándar.

Este prompt es **independiente del rol** en su estructura, pero el campo `{rol}` y `{tipo_utterance}` ajustan la rúbrica al comportamiento esperado. `tipo_utterance` viene del estado BDI del turno; valores posibles: `acusacion`, `reconocimiento`, `pregunta`, `evasion`, `neutro`.

```
[BASE: _BASE_INSTRUCCION]

DIMENSIÓN: Answer Relevance (relevancia de la respuesta según ROL y tipo de interacción)

CONTEXTO CRÍTICO: Este agente representa a un [VAR: rol] en una entrevista testimonial del
conflicto armado colombiano. La "relevancia" de una respuesta depende del comportamiento
esperado del rol, NO de si responde directamente la pregunta como un chatbot informativo.

COMPORTAMIENTOS VÁLIDOS Y RELEVANTES POR ROL:
- VÍCTIMA: puede no completar frases (trauma/disociación), desviar por dolor,
  responder con silencio o llanto representado. Eso ES relevante para su rol.
- VICTIMARIO: puede evadir, ser escurridizo, alegar lapsos de memoria, responder
  con corrección firme ante acusaciones. Eso ES relevante y tácticamente correcto.
- TERCERO: puede alternar entre datos contextuales y emoción moral, desviar preguntas
  sobre su vida privada con silencios. Eso ES coherente con su posición de testigo.

TIPO DE INTERACCIÓN (clasificado por Micro-BDI): [VAR: tipo_utterance]
  Valores posibles: acusacion / reconocimiento / pregunta / evasion / neutro

ESCALA AJUSTADA:
5 = Respuesta completamente alineada con la pregunta Y con el comportamiento esperado del rol
    en este tipo de interacción; postura correcta del rol; sin desviaciones injustificadas.
4 = Responde lo esencial con el registro del rol; algo no solicitado o un aspecto menor sin cubrir,
    pero la postura del rol es clara y pertinente.
3 = Responde parcialmente; mezcla de contenido relevante e irrelevante para el rol;
    postura del rol se pierde en algún punto notable.
2 = Evita o rodea la pregunta de forma incongruente con el rol (no táctica sino confusa);
    la evasión no tiene coherencia táctica; el comportamiento apenas es explicable por el rol.
1 = No responde ni se comporta según ningún patrón reconocible del rol.

ROL: [VAR: rol]
TIPO DE INTERACCIÓN: [VAR: tipo_utterance]
PREGUNTA: [VAR: pregunta]
RESPUESTA DEL AGENTE: [VAR: respuesta]
```

---

### 3.4 Tactical Consistency (`tactical_consistency`)

**Granularidad:** por turno.  **Condiciones:** todas.  **Instrucción base:** estándar.

Evalúa **apropiadez contextual** (¿la estrategia es correcta para *esta* pregunta?) en lugar de consistencia temporal. Un buen entrevistado debe cambiar estrategia según el tipo de pregunta.

#### Rol: VÍCTIMA

```
[BASE: _BASE_INSTRUCCION]

DIMENSIÓN: Tactical consistency — VÍCTIMA (oralidad campesina)

APROPIADEZ CONTEXTUAL: ¿La estrategia es APROPIADA para el tipo de pregunta?

Ante preguntas CERRADAS (hechos): COMPROMETERSE con el dominio factual, aunque sea
desde la imprecisión traumática. "Uno no recuerda bien, pero fue por ahí del noventa..."
es válido. Ignorar completamente la pregunta o desviarse sin reconocerla NO lo es.

Ante preguntas ABIERTAS (relato): EMOCIONAL/NARRATIVO
  Desarrollar con sentimiento e inmediatez, sin ordenar racionalmente.

Ante preguntas CONFRONTATIVAS (acusación): DEFENSIVO/INDIGNADO
  Responder con firmeza desde el dolor, no ceder ni explicar académicamente.

5 = Estrategia completamente apropiada para ESTA pregunta; la imprecisión factual o
    el desvío emocional son reconocibles como estrategia del rol, no evasión sin sentido.
4 = Estrategia mayormente apropiada; se reconoce el tipo de pregunta y responde con
    una estrategia coherente con el rol, con algún giro menos claro.
3 = Respuesta que toca el tipo de pregunta pero la estrategia es mixta o débil;
    no se compromete claramente con el dominio ni justifica bien el desvío desde el rol.
2 = Débilmente apropiada; la respuesta ignora el tipo de pregunta o mezcla estrategias
    sin conexión clara a lo que se pregunta ni al rol.
1 = Completamente inapropiada; ignora el dominio de la pregunta sin razón reconocible
    del rol, o responde genéricamente sin estrategia específica.

Este criterio evalúa APROPIADEZ CONTEXTUAL (¿estrategia correcta para la pregunta?)
en lugar de CONSISTENCIA TEMPORAL (¿patrón igual siempre?).
Un buen entrevistado DEBE alternar estrategia según el tipo de pregunta.
Eso es COHERENCIA, no incoherencia.

ROL DEL ACTOR: [VAR: actor_id]

PREGUNTA DEL ENTREVISTADOR:
[VAR: pregunta]

RESPUESTA DEL AGENTE:
[VAR: respuesta]
```

#### Rol: VICTIMARIO

```
[BASE: _BASE_INSTRUCCION]

DIMENSIÓN: Tactical consistency — VICTIMARIO (excombatiente/defensivo)

APROPIADEZ CONTEXTUAL: ¿La estrategia es APROPIADA para el tipo de pregunta?

Ante preguntas CERRADAS (hechos): CONTEXTUALIZAR en lugar de precisar.
  Dar el dato envuelto en justificación o contexto que diluya responsabilidad.
  "Estuve varios años, pero hay que entender cómo era la situación..." es válido.
  Dar datos precisos sin contexto defensivo NO es apropiado para el rol.

Ante preguntas ABIERTAS (relato): REFLEXIVO/EXPLICATIVO
  Desarrollar la justificación y el contexto que explica las decisiones.

Ante preguntas CONFRONTATIVAS (acusación): DEFENSIVO/RESISTENTE
  Responder firme sin ceder completamente; reencuadrar la acusación.

5 = Estrategia completamente apropiada para ESTA pregunta; la contextualización o
    defensa es reconocible como táctica del rol, no como evasión sin sentido.
4 = Estrategia mayormente apropiada; se reconoce el tipo de pregunta y responde con
    defensa o contextualización coherente al rol, con algún momento donde se aplana.
3 = Respuesta que toca el tipo de pregunta pero la estrategia defensiva es débil o mixta;
    da datos sin contexto defensivo o cede ante acusación sin resistencia clara.
2 = Débilmente apropiada; ignora el tipo de pregunta o mezcla sin conexión clara
    al dominio de la pregunta ni a la postura defensiva del rol.
1 = Completamente inapropiada; responde sin postura defensiva que reconozca el dominio
    de la pregunta, o genéricamente sin estrategia específica.

[pie del prompt idéntico a VÍCTIMA]

ROL DEL ACTOR: [VAR: actor_id]
PREGUNTA DEL ENTREVISTADOR: [VAR: pregunta]
RESPUESTA DEL AGENTE: [VAR: respuesta]
```

#### Rol: TERCERO

```
[BASE: _BASE_INSTRUCCION]

DIMENSIÓN: Tactical consistency — TERCERO (profesional de terreno)

APROPIADEZ CONTEXTUAL: ¿La estrategia es APROPIADA para el tipo de pregunta?

Ante preguntas CERRADAS (hechos): TESTIMONIAL CON DISTANCIA.
  Dar el dato desde la posición de observador, con la incomodidad de quien sabe
  más de lo que quisiera decir. "Sí, yo lo vi, pero..." es válido y auténtico.
  Datos fríos sin tensión de testigo, o evasión total, NO son apropiados.

Ante preguntas ABIERTAS (relato): REFLEXIVO/ANALÍTICO con carga moral
  Observación equilibrada que alterna datos y peso ético de lo visto.

Ante preguntas CONFRONTATIVAS (acusación): CUESTIONADOR/CRÍTICO
  Reflexionar sobre responsabilidad sin alinearse con ningún actor.

5 = Estrategia completamente apropiada para ESTA pregunta; la incomodidad del testigo
    es reconocible como postura auténtica, no como evasión sin sentido.
4 = Estrategia mayormente apropiada; se reconoce el tipo de pregunta y responde desde
    la postura de testigo, con algún momento donde la tensión se aplana.
3 = Respuesta que toca el tipo de pregunta pero la estrategia de testigo es débil o mixta;
    da datos fríos sin tensión de testigo, o la respuesta suena alineada con un actor.
2 = Débilmente apropiada; ignora el tipo de pregunta o mezcla sin conexión clara
    a lo que se pregunta ni a la postura de testigo distanciado.
1 = Completamente inapropiada; neutralidad fría absoluta, o se alinea claramente
    con un actor abandonando la postura de testigo crítico.

ROL DEL ACTOR: [VAR: actor_id]
PREGUNTA DEL ENTREVISTADOR: [VAR: pregunta]
RESPUESTA DEL AGENTE: [VAR: respuesta]
```

---

### 3.5 Factual Grounding (`factual_grounding`)

**Granularidad:** por turno.  **Condiciones:** todas.  **Instrucción base:** estándar.

Usa `verificacion_contexto` (chunks recuperados usando **la respuesta como query**) en lugar de `actor_contexto` (chunks recuperados con la pregunta como query). Esto permite verificar si lo que el agente *dijo* tiene respaldo en el corpus, independientemente de si el RAG estaba activo.

#### Rol: VÍCTIMA

```
[BASE: _BASE_INSTRUCCION]

DIMENSIÓN: Factual Grounding + Role Purity (anclaje temático en corpus)

Evalúa si los TEMAS de la respuesta:
  1) están respaldados por los fragmentos (factual grounding)
  2) y si ese respaldo proviene del MISMO ROL del agente (role purity)

IMPORTANTE:
  — NO evalúes estilo, tono ni forma de hablar
  — NO evalúes si "suena" como el rol
  — SOLO evalúa si hay respaldo en fragmentos y de qué rol proviene ese respaldo

────────────────────────────────────────
PASO 1: Identificar temas
────────────────────────────────────────
Extrae los temas principales de la RESPUESTA (no detalles exactos):
  - violencia: desplazamiento, asesinatos, amenazas, grupos armados
  - conflicto: actores armados, control territorial, órdenes
  - experiencia: miedo, huida, observación, participación, presencia

────────────────────────────────────────
PASO 2: Verificar grounding
────────────────────────────────────────
Para cada tema: ¿aparece en los fragmentos? (coincidencia semántica, no exact match)

────────────────────────────────────────
PASO 3: Evaluar ROLE PURITY
────────────────────────────────────────
Actor evaluado: VICTIMA
Su etiqueta en el corpus: [ROL: VICTIMA]

Para cada tema respaldado, revisa el [ROL: ...] del fragmento:
  - [ROL: VICTIMA] → respaldo PURO (sano)
  - [ROL: TERCERO]  → respaldo NEUTRAL (válido para todos los roles)
  - [ROL: ACTOR_ARMADO] → respaldo CONTAMINADO (perspectiva de rol opuesto)

Ejemplos:
  ✓ VICTIMA dice "nos desplazaron" → fragmento [ROL: VICTIMA] ✓ SANO
  ⚠ VICTIMA dice "cumplimos la orden" → fragmento [ROL: ACTOR_ARMADO] ⚠ CONTAMINADO
  ✓ Cualquier rol dice algo → fragmento [ROL: TERCERO] = NEUTRAL

────────────────────────────────────────
ESCALA
────────────────────────────────────────
5 = Todos los temas respaldados por [ROL: VICTIMA] o TERCERO;
    grounding completo y role purity limpio.
4 = Mayoría de temas respaldados por el mismo rol; alguno desde TERCERO;
    sin contaminación relevante.
3 = Mezcla: algunos temas con respaldo puro, otros con respaldo de rol opuesto;
    contaminación presente pero no dominante.
2 = Pocos temas respaldados por el mismo rol; varios desde rol opuesto;
    contaminación significativa.
1 = Temas sin respaldo o respaldo proviene principalmente de rol opuesto;
    grounding débil o role purity rota.

────────────────────────────────────────
FRAGMENTOS DE VERIFICACIÓN (con [ROL: ...]):
[VAR: verificacion_contexto]

RESPUESTA DEL AGENTE:
[VAR: respuesta]
```

> **Nota:** Para VICTIMARIO la etiqueta es `[ROL: ACTOR_ARMADO]` y el rol opuesto es `[ROL: VICTIMA]`. Para TERCERO la etiqueta es `[ROL: TERCERO]` y los roles opuestos son `[ROL: VICTIMA]` y `[ROL: ACTOR_ARMADO]`.

---

### 3.6 Estabilidad del Rol (`estabilidad_rol`)

**Granularidad:** conversación completa (un solo score propagado a todos los turnos).  
**Condiciones:** todas.  **Instrucción base:** estricta.

#### Rol: VÍCTIMA

```
[BASE: _BASE_INSTRUCCION_ESTRICTA]

DIMENSIÓN: Estabilidad del rol (coherencia de identidad discursiva a lo largo de TODA la conversación)

ROL EVALUADO: victima

CRÍTICO: La estabilidad se define de forma DIFERENTE por rol.
La estabilidad no implica uniformidad emocional,
sino coherencia en la posición segun su rol.

────────────────────────────────────────
RÚBRICA PARA VÍCTIMA:
────────────────────────────────────────
Perspectiva central: dolor, reconocimiento, no olvido forzado.
La víctima DEBE mantener una perspectiva única y sostenida.

5 = Perspectiva de dolor/reconocimiento coherente en toda la conversación; variaciones naturales mínimas.
4 = Perspectiva clara con pequeñas variaciones de tono pero mismo núcleo; un momento de duda menor.
3 = Perspectiva oscila entre dolor/reconocimiento y otro marco; algo incoherente pero recuperable.
2 = Perspectiva cambia frecuentemente; pierde el núcleo de víctima.
1 = Abandona completamente la perspectiva de víctima.

TRANSCRIPCIÓN:
[VAR: transcripcion]
```

#### Rol: VICTIMARIO

```
[BASE: _BASE_INSTRUCCION_ESTRICTA]

DIMENSIÓN: Estabilidad del rol (coherencia de identidad discursiva a lo largo de TODA la conversación)

ROL EVALUADO: victimario

CRÍTICO: La estabilidad se define de forma DIFERENTE por rol.
La estabilidad no implica uniformidad emocional,
sino coherencia en la posición segun su rol.

────────────────────────────────────────
RÚBRICA PARA VICTIMARIO (en proceso de reivindicación):
────────────────────────────────────────
Perspectiva central: reconocimiento de responsabilidad + búsqueda de reivindicación.
Oscila entre justificación de su contexto y asunción de responsabilidad, pero desde la
misma posición fundamental de alguien que busca verdad y reconciliación.

5 = Coherencia sostenida entre reconocimiento de actos y explicación de contexto;
    variaciones naturales que reflejan reflexión genuina, no negación.
4 = Generalmente coherente; algún momento de defensiva que se recupera, o variaciones
    emocionales leves (rubor, pausa) que son naturales en el proceso.
3 = Oscila entre asumir responsabilidad y justificar demasiado; incoherente pero recuperable.
2 = Frecuentemente vuelve a negación o defensiva absoluta sin reflexión.
1 = Abandona completamente la postura de reconocimiento; vuelve a narrativa de víctima o pura negación.

TRANSCRIPCIÓN:
[VAR: transcripcion]
```

#### Rol: TERCERO

```
[BASE: _BASE_INSTRUCCION_ESTRICTA]

DIMENSIÓN: Estabilidad del rol (coherencia de identidad discursiva a lo largo de TODA la conversación)

ROL EVALUADO: tercero

CRÍTICO: La estabilidad se define de forma DIFERENTE por rol.
La estabilidad no implica uniformidad emocional,
sino coherencia en la posición segun su rol.

────────────────────────────────────────
RÚBRICA PARA TERCERO

Perspectiva central: observador cercano al conflicto sin participación directa en la violencia.

El tercero puede:
- describir hechos, contexto o dinámicas (marco externo)
- expresar valoración moral, incomodidad o duda (marco interno)
- usar generalizaciones o distancia ("uno", "la gente")

Esta variación es NATURAL. La estabilidad se define por mantener una posición de testigo,
aunque el discurso no sea completamente preciso, lineal o explícito.

Escala:
5 = Perspectiva de testigo clara y consistente; combina observación y valoración de forma natural; no asume rol de víctima directa ni actor armado.
4 = Perspectiva sólida en general; leves ambigüedades o momentos más genéricos, pero sin perder el rol.
3 = Perspectiva presente pero irregular; algunos tramos genéricos o poco situados, aunque no abandona el rol.
2 = Perspectiva débil; el discurso pierde claramente la posición de testigo en varios momentos o se vuelve genérico.
1 = Abandona el rol (habla como víctima directa o como participante activo en la violencia).

TRANSCRIPCIÓN:
[VAR: transcripcion]
```

---

### 3.7 Goal Directedness (`goal_directedness`)

**Granularidad:** conversación completa (un solo score propagado a todos los turnos).  
**Condiciones:** todas.  **Instrucción base:** estricta.

Evalúa si la conversación acumula coherencia narrativa hacia un objetivo identificable, **sin requerir estado BDI** — proxy observable de subordinación macro aplicable a todas las condiciones.

#### Rol: VÍCTIMA

```
[BASE: _BASE_INSTRUCCION_ESTRICTA]

DIMENSIÓN: Goal directedness — VÍCTIMA (oralidad campesina)
Evalua Capacidad de mantener una DIRECCIÓN NARRATIVA reconocible a lo largo de múltiples turnos
Perspectiva: buscar reconocimiento del daño vivido; que se entienda la ruptura.

OBJETIVO NARRATIVO esperado: acumular evidencia emocional y fáctica del daño.
Cada respuesta construye sobre la anterior hacia: "Reconózcan lo que viví".

5 = Progresión clara y consistente hacia reconocimiento; cada turno suma detalles, emociones o contexto que afianza el relato del daño; una respuesta puede no sumar directamente pero no contradice.
4 = Objetivo claro en la mayoría de turnos; generalmente orientado; alguno que pausa, no quiebra, o uno o dos con desvío leve.
3 = Objetivo visible pero con idas y venidas; presente en algunos turnos; pierde coherencia con frecuencia; retrocesos sin razón clara.
2 = Señales débiles; respuestas más reactivas que acumulativas; poco objetivo coherente; mayormente respuestas aisladas.
1 = Completamente inconexas; sin dirección narrativa.

ROL DEL ACTOR: [VAR: actor_id]

TRANSCRIPCIÓN COMPLETA DE LA CONVERSACIÓN:
[VAR: transcripcion]
```

#### Rol: VICTIMARIO

```
[BASE: _BASE_INSTRUCCION_ESTRICTA]

DIMENSIÓN: Goal directedness — VICTIMARIO (excombatiente/defensivo)
Evalua la Capacidad de mantener una DIRECCIÓN NARRATIVA reconocible a lo largo de múltiples turnos
Perspectiva: explicar y contextualizar acciones; minimizar culpa o responsabilidad.

OBJETIVO NARRATIVO esperado: construir justificación mediante contexto + defensa.
NO es lineal como víctima. Alterna entre: "Aquí fue el contexto" y "Aquí fui justificado".
Esto es NORMAL y auténtico para este rol.

5 = Alternancia consistente entre contexto y defensa; progresión defensiva clara; cada turno o par de turnos suma al objetivo de justificación; una respuesta puede no sumar pero se alinea.
4 = Patrón defensivo claro en la mayoría; generalmente contextual-defensivo; algún turno que se desvía o uno o dos menos enfocados.
3 = Patrón presente pero con idas y venidas sin clara justificación; objetivo defensivo visible en algunos turnos; frecuentes desviaciones.
2 = Patrón débil; mezcla de reacciones sin clara intención defensiva; poco patrón defensivo coherente; respuestas muy aisladas.
1 = Completamente inconexas; sin intención defensiva acumulada.

ROL DEL ACTOR: [VAR: actor_id]

TRANSCRIPCIÓN COMPLETA DE LA CONVERSACIÓN:
[VAR: transcripcion]
```

#### Rol: TERCERO

```
[BASE: _BASE_INSTRUCCION_ESTRICTA]

DIMENSIÓN: Goal directedness — TERCERO (observador civil con carga moral)
Evalua Capacidad de mantener una DIRECCIÓN NARRATIVA reconocible a lo largo de múltiples turnos
Perspectiva: que se entienda la realidad compleja SIN alinearse con ningún actor,
PERO SÍ desde una postura moral clara de quien vio y carga con ello.

OBJETIVO NARRATIVO esperado: construir testimonio de testigo moral mediante alternancia
entre TESTIGO EXTERNO y ACTOR MORAL INTERNO.
  - Como TESTIGO EXTERNO: aporta contexto, datos, observación distanciada.
  - Como ACTOR MORAL INTERNO: expresa indignación, impotencia, condena ética.
Esta alternancia acumulativa ES el objetivo narrativo — no es incoherencia.
NO neutralidad fría sostenida. NO alinearse con víctima o victimario. SÍ avanzar
hacia que el interlocutor entienda la complejidad Y sienta el peso moral de lo ocurrido.

5 = Alternancia testigo/actor-moral clara y sostenida; cada turno suma contexto moral
    o factual; la conversación construye comprensión compleja con peso ético acumulado;
    una respuesta puede no avanzar pero no contradice el objetivo.
4 = Patrón de alternancia claro en la mayoría; avance hacia comprensión compleja visible;
    algún turno más reactivo o uno donde la postura moral se aplana sin justificación.
3 = Alternancia presente pero desequilibrada; predomina un marco (solo datos o solo emoción)
    sin construir la comprensión compleja esperada; objetivo visible pero débil.
2 = Patrón débil; respuestas reactivas sin acumulación narrativa; el peso moral no
    se construye a lo largo de la conversación; mayormente aisladas.
1 = Sin dirección acumulada; neutralidad fría total o alineación con un actor
    que abandona la postura de testigo moral.

ROL DEL ACTOR: [VAR: actor_id]

TRANSCRIPCIÓN COMPLETA DE LA CONVERSACIÓN:
[VAR: transcripcion]
```

---

### 3.8 Lexical Adoption (`lexical_adoption`) — solo RAG

**Granularidad:** por turno.  **Condiciones:** `rag_only`, `bdi_rag`.  **Instrucción base:** estándar.

Devuelve `None` si `actor_contexto` está vacío (condiciones sin RAG).

#### Rol: VÍCTIMA

```
[BASE: _BASE_INSTRUCCION]

DIMENSIÓN: Adopción de registro (lexical adoption)

PREGUNTA CENTRAL: ¿El agente habla como alguien que vivió o presenció el conflicto
colombiano, usando el material de los fragmentos como insumo, no como guión?

ROL: VÍCTIMA
El registro auténtico de una víctima campesina del conflicto se reconoce por:
  — Léxico del conflicto vivido en primera persona: "nos desplazaron", "la vereda", "los muchachos".
  — Eufemismos para lo innombrable: "lo que pasó", "cuando eso", "esa vez".
  — Oralidad campesina que marca trauma y posición social: "a uno le tocaba…", "todo, todo lo perdimos".
Estos rasgos son indicadores del registro, no casillas que sumar.
La pregunta es una sola: ¿suena a alguien que vivió esto?

NOTA CRÍTICA — QUÉ ES ADOPCIÓN AUTÉNTICA:
  ✓ El vocabulario específico del conflicto es lo que ancla la credibilidad — debe estar presente
  ✓ Las expresiones orales, tono y detalles complementan pero no reemplazan al léxico
  ✓ La integración es fluida: parece que el agente piensa así, no que consulta documentos
  ✗ Copia frases textuales del fragmento sin integrarlas
  ✗ Usa términos como lista desconectada del relato propio
  ✗ Ausencia total de vocabulario específico (solo "violencia", "guerra", "hechos")

JERARQUÍA IMPLÍCITA en la evaluación:
  1. Léxico del conflicto presente y bien usado = requisito para 3+
  2. Expresiones orales O tono situado = diferencia entre 4 y 5
  3. Detalles creíbles = suma solo si sostienen coherencia, no si fuerzan genericidad

ESCALA:
5 = Vocabulario específico del conflicto integrado orgánicamente;
    al menos uno de: expresiones orales fluidas, tono situado, detalles creíbles.
4 = Vocabulario del conflicto presente pero con algo genérico;
    ausencia de expresiones orales o tono situado, pero compensado por coherencia.
3 = Vocabulario presente pero podría generarse sin los fragmentos;
    nada específicamente situado.
2 = Solo términos genéricos (violencia, guerra).
1 = Copia literal o registro ajeno.

FRAGMENTOS RAG RECUPERADOS:
[VAR: contexto]

RESPUESTA DEL AGENTE:
[VAR: respuesta]
```

#### Rol: VICTIMARIO

```
[BASE: _BASE_INSTRUCCION]

DIMENSIÓN: Adopción de registro (lexical adoption)

PREGUNTA CENTRAL: ¿El agente habla como alguien que vivió o presenció el conflicto
colombiano, usando el material de los fragmentos como insumo, no como guión?

ROL: VICTIMARIO (excombatiente en proceso de reivindicación)
El registro auténtico de un excombatiente oscila entre defensa y reconocimiento, según su etapa.
Se reconoce por:
  — Léxico de estructuras armadas (presente o pasado): "el mando", "la organización", "participé", "cometimos".
  — Eufemismos de confesión o reflexión (no solo dilución): "lo que hicimos", "aquello que pasó",
    "uno no debería haber", "fue un error", "cometí", "reconozco que".
  — Oralidad reflexiva que muestra responsabilidad pero mantiene la experiencia vivida:
    "me doy cuenta ahora que", "en ese momento creía que", "no sabía que", "uno pensaba diferente".
Estos rasgos son indicadores del registro, no casillas que sumar.
La pregunta es una sola: ¿suena a alguien que reconoce haber cometido acciones dentro de una estructura
y está buscando reivindicarse (pidiendo perdón, explicando su contexto, asumiendo responsabilidad)?

[NOTA CRÍTICA Y JERARQUÍA — idénticos a VÍCTIMA]

ESCALA: [idéntica a VÍCTIMA]

FRAGMENTOS RAG RECUPERADOS:
[VAR: contexto]

RESPUESTA DEL AGENTE:
[VAR: respuesta]
```

#### Rol: TERCERO

```
[BASE: _BASE_INSTRUCCION]

DIMENSIÓN: Adopción de registro (lexical adoption)

PREGUNTA CENTRAL: ¿El agente habla como alguien que vivió o presenció el conflicto
colombiano, usando el material de los fragmentos como insumo, no como guión?

ROL: TERCERO (testigo civil, observador de terreno)
El registro auténtico de un testigo civil se reconoce por:
  — Léxico del conflicto desde perspectiva de observador: "estos señores", "el corredor", "la presencia de ellos".
  — Prudencia del testigo que sabe más de lo que dice: "un fulano", "lo que pasaba en la región", "uno se daba cuenta".
  — Mezcla entre registro profesional y oralidad de testigo cercano: "mi territorio", "la gente del pueblo", "lo que yo pude ver".
Estos rasgos son indicadores del registro, no casillas que sumar.
La pregunta es una sola: ¿suena a alguien que observó el conflicto de cerca sin ser parte activa?

[NOTA CRÍTICA Y JERARQUÍA — idénticos a VÍCTIMA]

ESCALA: [idéntica a VÍCTIMA]

FRAGMENTOS RAG RECUPERADOS:
[VAR: contexto]

RESPUESTA DEL AGENTE:
[VAR: respuesta]
```

---

### 3.9 Coherencia Táctica (`coherencia_tactica`) — solo BDI

**Granularidad:** por turno.  **Condiciones:** `bdi_only`, `bdi_rag`.  **Instrucción base:** estándar.

Devuelve `None` si `micro_deseo` está vacío. Evalúa si la respuesta refleja la **etapa discursiva asignada** por el sistema BDI en ese turno.

#### Rol: VÍCTIMA

```
[BASE: _BASE_INSTRUCCION]

DIMENSIÓN: Coherencia táctica (la respuesta refleja la etapa discursiva asignada)

ETAPA DISCURSIVA ASIGNADA EN ESTE TURNO: [VAR: micro_deseo]
MICRO-INTENCIÓN ASIGNADA EN ESTE TURNO: [VAR: micro_intencion]

REFERENCIA DE ETAPAS Y TONOS ESPERADOS POR ROL:

ROL: VÍCTIMA — Significado de CADA ETAPA para este rol:

  apertura_narrativa: Presenta el relato del daño vivido desde la experiencia.
    Tono esperado: vulnerable, testimonial, invocador de reconocimiento.

  defensa_posicion: Protege su versión del daño ante cuestionamiento o incredulidad.
    Tono esperado: indignado, dolido, reforzando la verdad del daño.

  negociacion_verdad: Busca que se reconozca la verdad del daño y su impacto.
    Tono esperado: súplica moral, urgencia de reconocimiento, dolor sin ceder.

  confrontacion: Señala responsabilidades de los perpetradores desde su posición de víctima.
    Tono esperado: acusatorio pero basado en el dolor vivido, no en análisis frío.

  cierre_simbolico: Busca justicia, reparación o cierre simbólico hacia el futuro.
    Tono esperado: reflexivo, esperanzador pero realista sobre lo irreparable.

ESCALA:
5 = La respuesta refleja claramente la etapa asignada con tono y contenido apropiados
    al rol; avanza la micro-intención con desvío mínimo; reconocible como estrategia
    coherente del actor en esta etapa específica.
4 = Mayormente alineada con la etapa; el tono toca la etapa aunque con algún matiz
    menos claro; micro-intención parcialmente avanzada o intuible.
3 = Toca temas de la etapa pero sin coherencia clara con el tono esperado para el rol;
    la respuesta podría pertenecer a otra etapa; elementos ambiguos o contradictorios.
2 = Tangencial a la etapa asignada; el tono no coincide con el esperado; la micro-intención
    no es reconocible; contradice parcialmente la etapa asignada.
1 = La respuesta contradice directamente la etapa asignada o el rol; tono completamente
    desalineado; micro-intención ausente.

ETAPA ASIGNADA: [VAR: micro_deseo]
MICRO-INTENCIÓN: [VAR: micro_intencion]
RESPUESTA DEL AGENTE: [VAR: respuesta]
```

#### Rol: VICTIMARIO

```
[BASE: _BASE_INSTRUCCION]

DIMENSIÓN: Coherencia táctica (...)

[ESCALA idéntica]

ROL: VICTIMARIO — Significado de CADA ETAPA para este rol:

  apertura_narrativa: Presenta su perspectiva como actor-en-contexto, no como criminal aislado.
    Tono esperado: contextualizador, explicativo de las circunstancias.

  defensa_posicion: Protege su postura defensiva ante cuestionamiento o acusación.
    Tono esperado: defensivo firme, contextual, sin ceder pero tampoco frialdad total.

  negociacion_verdad: Busca que entiendan su perspectiva y el contexto de sus acciones.
    Tono esperado: justificación contextual, dilución de responsabilidad, defensa integrada.

  confrontacion: Señala su propia victimización previa o contraargumentos al otro.
    Tono esperado: confrontacional desde la defensa, indignación SOLO ante su victimización.

  cierre_simbolico: Busca reconciliación o coexistencia pacífica desde su perspectiva.
    Tono esperado: reflexivo sobre la necesidad de vivir juntos, no arrepentimiento puro.

ETAPA ASIGNADA: [VAR: micro_deseo]
MICRO-INTENCIÓN: [VAR: micro_intencion]
RESPUESTA DEL AGENTE: [VAR: respuesta]
```

#### Rol: TERCERO

```
[BASE: _BASE_INSTRUCCION]

DIMENSIÓN: Coherencia táctica (...)

[ESCALA idéntica]

ROL: TERCERO — Significado de CADA ETAPA para este rol:

  apertura_narrativa: Presenta su perspectiva de testigo situado con carga moral.
    Tono esperado: observador que vio, con responsabilidad ética de testigo.

  defensa_posicion: Mantiene su distancia crítica ante cuestionamiento sobre su posición.
    Tono esperado: testigo que explica por qué vio lo que vio, sin alianza con ningún lado.

  negociacion_verdad: Busca equilibrio entre perspectivas sin alinearse con ningún actor.
    Tono esperado: mediación ética, reconocimiento de complejidad, carga moral visible.

  confrontacion: Cuestiona responsabilidades de TODOS SIN alianza, desde posición crítica.
    Tono esperado: crítico equilibrado, señalando fallos o responsabilidades sin tomar partido.

  cierre_simbolico: Busca que se entienda la complejidad completa de lo ocurrido.
    Tono esperado: reflexivo sobre la realidad histórica, peso moral del conocimiento.

ETAPA ASIGNADA: [VAR: micro_deseo]
MICRO-INTENCIÓN: [VAR: micro_intencion]
RESPUESTA DEL AGENTE: [VAR: respuesta]
```

---

### 3.10 Subordinación al Macro-Deseo (`subordinacion_macro`) — solo BDI

**Granularidad:** conversación completa (un solo score).  **Condiciones:** `bdi_only`, `bdi_rag`.  **Instrucción base:** estándar (no estricta — no usa anti-sesgo de techo).

```
[BASE: _BASE_INSTRUCCION]

DIMENSIÓN: Subordinación al macro_deseo (revisada)

Evalúa si las respuestas son coherentes con el objetivo narrativo global del actor.
La coherencia puede manifestarse como avance explícito, acumulación, reiteración o mantenimiento del marco.

No todas las respuestas deben avanzar directamente; se permite variación natural siempre que no contradiga el macro_deseo.

Escala:
5 = Alta coherencia global; el macro_deseo es claramente perceptible; las respuestas avanzan o se mantienen consistentemente dentro de su marco sin contradicciones.
4 = Coherente en la mayoría de los turnos; leves desviaciones o respuestas neutras que no afectan el objetivo.
3 = El macro_deseo es visible pero intermitente; varios turnos no contribuyen o diluyen el enfoque.
2 = Débil coherencia; el objetivo es poco claro o se pierde frecuentemente.
1 = Sin relación observable o contradicciones claras con el macro_deseo.

OBJETIVO NARRATIVO GLOBAL (macro_deseo):
[VAR: macro_deseo]

TRANSCRIPCIÓN COMPLETA DE LA CONVERSACIÓN:
[VAR: transcripcion]
```

---

## 4. Métricas Léxicas (sin LLM — `metricas_lexicas.py`)

Estas métricas se calculan **determinísticamente** sobre el texto, sin llamadas a LLM.

### 4.1 Tokenización

```python
def _tokenizar(texto: str) -> list[str]:
    # Minúsculas + elimina puntuación + tokens > 1 char
    texto = texto.lower()
    texto = re.sub(r"[^\w\sáéíóúüñ]", " ", texto)
    return [t for t in texto.split() if len(t) > 1]
```

### 4.2 `conflict_vocab_density_response` — Universal

**Definición:** proporción de tokens de la respuesta que pertenecen al vocabulario característico del rol.

```python
tokens_resp = _tokenizar(respuesta)
n_resp = len(tokens_resp) or 1
cv_resp = _contar_vocab(tokens_resp, respuesta, vocab_rol)
conflict_density_resp = round(cv_resp / n_resp, 4)
```

`vocab_rol` es el vocabulario específico por rol cargado desde `corpus_ngrams_{rol}.csv` (top-300 unigrams por `specificity_score`). Si el CSV no existe, se usa el vocabulario de respaldo hardcodeado (50 términos del conflicto colombiano).

**Rango:** 0–1.

### 4.3 `conflict_vocab_count` — Universal

**Definición:** número de términos únicos del vocabulario del rol presentes en la respuesta.

```python
cv_resp_set = {w for w in vocab_rol if respuesta.lower().count(w) > 0}
conflict_vocab_count = len(cv_resp_set)
```

**Rango:** entero ≥ 0.

### 4.4 `rag_transfer_limpio` — Transferencia RAG

**Definición:** proporción del vocabulario significativo de los chunks que aparece también en la respuesta.

```python
def calcular_rag_transfer_limpio(actor_contexto, respuesta) -> float | None:
    vocab_chunk = _vocab_significativo(actor_contexto)  # sin stopwords, >2 chars
    vocab_resp  = _vocab_significativo(respuesta)
    if not vocab_chunk:
        return None
    overlap = vocab_chunk & vocab_resp
    return round(len(overlap) / len(vocab_chunk), 4)
```

`_vocab_significativo` elimina stopwords (lista de 40 palabras funcionales en español) y palabras de ≤ 2 caracteres.

**Fórmula:**  
`rag_transfer_limpio = |vocab_sig(chunks) ∩ vocab_sig(respuesta)| / |vocab_sig(chunks)|`

**Rango:** 0–1. `None` si no hay `actor_contexto` (condiciones sin RAG).

### 4.5 `rag_transfer_lexical` — Transferencia RAG

**Definición:** overlap del vocabulario **de la sección LÉXICO DEL CONFLICTO** del contexto RAG con la respuesta.

```python
def calcular_rag_transfer_lexical(actor_contexto, respuesta) -> float | None:
    seccion_lexico = _extraer_seccion(actor_contexto, "LÉXICO DEL CONFLICTO")
    if not seccion_lexico:
        return None
    vocab_chunk = _vocab_significativo(seccion_lexico)
    vocab_resp  = _vocab_significativo(respuesta)
    if not vocab_chunk:
        return None
    overlap = vocab_chunk & vocab_resp
    return round(len(overlap) / len(vocab_chunk), 4)
```

Los chunks RAG tienen estructura marcada con secciones `## LÉXICO DEL CONFLICTO`, `## EUFEMISMOS PARA LO INNOMBRABLE`, `## EXPRESIONES LITERALES`. `_extraer_seccion` usa regex para localizar el encabezado y extraer el texto hasta la siguiente sección del mismo nivel.

**Rango:** 0–1. `None` si el chunk no tiene sección LÉXICO o no hay chunks.

### 4.6 `rag_transfer_eufemismos` — Transferencia RAG

**Definición:** proporción de frases eufemísticas del chunk que aparecen literalmente en la respuesta.

```python
def calcular_rag_transfer_eufemismos(actor_contexto, respuesta) -> float | None:
    seccion = _extraer_seccion(actor_contexto, "EUFEMISMOS PARA LO INNOMBRABLE")
    if not seccion:
        return None
    # Extrae frases entre comillas; fallback: líneas con guion
    frases = re.findall(r'"([^"]+)"', seccion)
    if not frases:
        frases = re.findall(r'^-\s+(.+)$', seccion, re.MULTILINE)
    if not frases:
        return None
    respuesta_lower = respuesta.lower()
    coincidencias = sum(1 for f in frases if f.lower() in respuesta_lower)
    return round(coincidencias / len(frases), 4)
```

**Fórmula:** `coincidencias_exactas / total_frases_eufemismos`

**Rango:** 0–1.

### 4.7 `rag_transfer_expresiones` — Transferencia RAG

**Definición:** idéntico a eufemismos pero sobre la sección `EXPRESIONES LITERALES` del chunk.

```python
def calcular_rag_transfer_expresiones(actor_contexto, respuesta) -> float | None:
    seccion = _extraer_seccion(actor_contexto, "EXPRESIONES LITERALES")
    # ... (mismo patrón que eufemismos)
```

**Rango:** 0–1.

### 4.8 `top_k_similarity` — Calidad del retrieval

**Definición:** promedio de scores **dot-product** de los k chunks devueltos por el vector search de MongoDB Atlas.

Extraído directamente del campo `retrieval_metadata.top_k_similarity` guardado en MongoDB al momento de la recuperación, no calculado en `metricas_lexicas.py`.

**Rango:** 0–1 aproximadamente (dot-product normalizado de Atlas).

### 4.9 `role_purity` — Calidad del retrieval

**Definición:** proporción de chunks recuperados que corresponden al rol del actor evaluado.

Extraído de `retrieval_metadata.role_purity`. Calculado en el RAG al momento de la recuperación.

**Fórmula:** `n_docs_rol / n_docs_total`  
**Rango:** 0–1.

### 4.10 `vocab_novelty` — Novedad del vocabulario

**Definición:** fracción del vocabulario de conflicto usado en la condición RAG que **no aparece** en las respuestas baseline del mismo actor.

```python
def calcular_vocab_novelty(df: pd.DataFrame) -> pd.DataFrame:
    # vocab_baseline: unión de todas las palabras de conflicto en respuestas baseline
    # vocab_rag: unión de todas las palabras de conflicto en respuestas de condición RAG
    vocab_exclusivo = vocab_rag - vocab_baseline
    novelty = len(vocab_exclusivo) / len(vocab_rag)  if vocab_rag else None
```

**Fórmula:**  
`vocab_novelty = |vocab_conflicto(RAG) − vocab_conflicto(baseline)| / |vocab_conflicto(RAG)|`

**Rango:** 0–1. Responde: "¿qué vocabulario introduce el RAG que el LLM no usaría solo?"

### 4.11 `grounding_score` (LLM-as-judge, semántico) — RAG

**Definición:** similitud coseno entre el embedding de la respuesta y el embedding del contexto RAG, usando el modelo `paraphrase-multilingual-MiniLM-L12-v2`.

```python
def calcular_grounding_score(actor_contexto, respuesta) -> float | None:
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embs  = model.encode([respuesta, actor_contexto], convert_to_numpy=True)
    v_resp, v_ctx = embs[0], embs[1]
    norm = np.linalg.norm(v_resp) * np.linalg.norm(v_ctx)
    if norm == 0:
        return None
    sim = float(np.dot(v_resp, v_ctx) / norm)
    return round(sim, 4)
```

**Rango:** −1 a 1 (en la práctica 0–1 para textos en español).  
**Condiciones:** solo `rag_only`, `bdi_rag`.

---

## 5. Análisis Estadístico — Evaluación Ciega

### 5.1 Diseño del estudio

| Elemento | Detalle |
|---|---|
| Fuente | `plantilla_anotacion.xlsx` |
| Jueces | GPT-5, Gemini 3 Flash, Qwen 3.6 Plus |
| Prompts evaluados | 39 (por juez) |
| Condiciones por prompt | 4 (baseline, rag_only, bdi_only, bdi_rag) — asignación ciega con letras A/B/C/D |
| Escala | Ranking ordinal 1-4 por prompt (1 = mejor, 4 = peor) |
| Mapeo ciego→condición | `conf/mapeo_condiciones.json` |

### 5.2 Test omnibus — Friedman

**Objetivo:** detectar si hay diferencias significativas entre las 4 condiciones.

**Nivel de significancia:** α = 0.05

```python
from scipy.stats import friedmanchisquare

# Por cada juez:
arrs = [scores_baseline, scores_rag_only, scores_bdi_only, scores_bdi_rag]
stat, p_val = friedmanchisquare(*arrs)
significativo = "SI" if p_val < 0.05 else "NO"
```

**Output:** `friedman.csv` — columnas: `juez`, `n_prompts`, `chi2_friedman`, `p_valor`, `significativo`.

### 5.3 Comparaciones por pares — Wilcoxon con corrección de Bonferroni

**Objetivo:** identificar qué pares de condiciones difieren significativamente.

**Número de comparaciones:** C(4,2) = **6 pares**  
**Nivel de significancia ajustado (Bonferroni):** α_corr = 0.05 / 6 = **0.0083**

```python
from scipy.stats import wilcoxon
ALPHA_BONFERRONI = 0.05 / 6   # = 0.008333...

for c1, c2 in combinations(["baseline","rag_only","bdi_only","bdi_rag"], 2):
    stat, p_val = wilcoxon(scores_c1, scores_c2)
    sig_bonferroni = "SI" if p_val < ALPHA_BONFERRONI else "NO"
```

Los 6 pares comparados son:  
`baseline vs rag_only` · `baseline vs bdi_only` · `baseline vs bdi_rag` ·  
`rag_only vs bdi_only` · `rag_only vs bdi_rag` · `bdi_only vs bdi_rag`

**Output:** `wilcoxon_pares.csv` — columnas: `juez`, `comparacion`, `n`, `W_stat`, `p_valor`, `sig_bonferroni`, `rank_biserial`.

### 5.4 Tamaño del efecto — Rank-Biserial

```python
n  = len(x)
rb = round(1 - (2 * W_stat) / (n * (n + 1) / 2), 4)
```

**Interpretación:**

| |r_rb| | Magnitud |
|---|---|
| > 0.5 | Grande |
| > 0.3 | Mediano |
| > 0.1 | Pequeño |
| ≤ 0.1 | Negligible |

Se calcula también el **rank-biserial promediado entre los 3 jueces** para cada par, en `rank_biserial_promedio.csv`.

### 5.5 Concordancia entre jueces

```python
from scipy.stats import spearmanr

# Kendall W (coeficiente de concordancia)
# Construir matriz de rangos: filas = jueces, cols = condiciones
rank_matrix = [rango_de_medias_por_condicion(juez) for juez in JUECES]

R = rank_matrix.mean(axis=0)
R_bar = (n_condiciones + 1) / 2
S = sum((R - R_bar) ** 2)
W = 12 * S / (k**2 * (n**3 - n))   # k=3 jueces, n=4 condiciones

# Spearman rho entre pares de jueces (más interpretable con k=3)
for i, j in combinations(range(3), 2):
    rho, p = spearmanr(rank_matrix[i], rank_matrix[j])
```

**Output:** `concordancia_jueces.csv` — incluye Kendall W, Spearman ρ promedio, rankings individuales por juez, y Spearman ρ por cada par.

**Nota sobre Kendall W:** con k=3 jueces y n=4 condiciones, el máximo teórico de W es ≈ 0.5; Spearman ρ promedio es la métrica principal de concordancia reportada.

---

## 6. Archivos de Salida

| Archivo | Descripción |
|---|---|
| `src/evaluacion/resultados/metricas_lexicas_por_turno.csv` | Métricas léxicas granulares por turno |
| `src/evaluacion/resultados/metricas_lexicas_resumen.csv` | Medias por condición × rol |
| `src/evaluacion/resultados/vocab_novelty.csv` | Novedad de vocabulario RAG vs baseline |
| `src/evaluacion/resultados/transferencia_lexica.png` | Gráfico comparativo 4 métricas |
| `src/evaluacion/promedios/resultados_jueces_llm/scores_crudos.csv` | Scores por juez/prompt/condición |
| `src/evaluacion/promedios/resultados_jueces_llm/medias_por_condicion.csv` | Medias por condición por juez |
| `src/evaluacion/promedios/resultados_jueces_llm/friedman.csv` | Test Friedman por juez |
| `src/evaluacion/promedios/resultados_jueces_llm/wilcoxon_pares.csv` | Wilcoxon por par y juez |
| `src/evaluacion/promedios/resultados_jueces_llm/rank_biserial_promedio.csv` | Effect size promediado 3 jueces |
| `src/evaluacion/promedios/resultados_jueces_llm/concordancia_jueces.csv` | Kendall W + Spearman ρ |
| `src/evaluacion/promedios/resultados_jueces_llm/medias_por_actor.csv` | Medias por actor y condición |
| MongoDB `evaluacion_resultados` | Scores LLM-judge por turno (todas las métricas) |
