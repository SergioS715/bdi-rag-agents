.PHONY: run install eval-full eval-local promedios lint

# ── Servidor web ───────────────────────────────────────────────────────────────
run:
	cd src/app && python api.py

run-dev:
	uvicorn src.app.api:app --host 0.0.0.0 --port 8000 --reload

# ── Dependencias ───────────────────────────────────────────────────────────────
install:
	uv sync

# ── Pipeline de evaluación ─────────────────────────────────────────────────────
eval-full:
	cd src/evaluacion && python correr_evaluacion_completa.py

eval-local:
	cd src/evaluacion && python correr_evaluacion_local.py

trazas:
	cd src/evaluacion && python generador_trazas.py --condicion all

promedios:
	cd src/evaluacion && python generar_promedios.py
	cd src/evaluacion && python generar_promedios_condicion_rol.py

exportar-jueces:
	cd src/evaluacion && python exportar_resultados_jueces.py

# ── Linting ────────────────────────────────────────────────────────────────────
lint:
	python -m py_compile src/app/api.py \
		src/servicio_conversacion/generar_respuesta.py \
		src/servicio_conversacion/reiniciar_conversacion.py \
		src/evaluacion/generador_trazas.py \
		src/evaluacion/correr_evaluacion_completa.py
	@echo "Sintaxis OK"
