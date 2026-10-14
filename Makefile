.PHONY: test test-fast reproduce build-demo build-web serve lint typecheck

test:
	pytest

test-fast:
	pytest -q -m "not slow"

reproduce:
	python -m speechlens reproduce

build-demo:
	python scripts/build_demo.py

build-web:
	cd web && npm ci && npm run build

serve:
	python -m speechlens serve --host 0.0.0.0 --port 7860

lint:
	cd web && npm run lint

typecheck:
	cd web && npx tsc --noEmit