## ClinicalRiskPredictor

Проект реализует базовую модель **ChronoFormer** для предсказания риска (baseline: бинарная смертность) по последовательности клинических событий (condition codes) и **промежуткам времени между событиями**. Поверх базовой модели добавлен модуль **XAI**:

- **Attention Rollout** (агрегация attention → важность токенов для последней позиции)
- **Saliency** (градиентная важность по входным embeddings)
- **Fidelity** (маскирование top-k важных токенов → падение вероятности)

Данные в baseline соответствуют ноутбуку `baseline.ipynb`: Hugging Face `richardyoung/synthea-575k-patients` (parquet).

### Установка

Рекомендуется Python **3.10–3.12** (PyTorch может не иметь готовых wheels для Python 3.13 на вашей платформе).

```bash
pip install -r requirements.txt
pip install -e .
```

### Обучение

- **Dry-run (без скачивания датасета, быстрый smoke-test):**

```bash
python scripts/train.py --dry-run --epochs 2 --batch-size 64 --max-len 64
```

- **Полный запуск на Synthea (долго и требует RAM):**

```bash
python scripts/train.py --epochs 50 --batch-size 128 --max-len 256 --out-dir artifacts
```

Чекпоинт сохраняется в `artifacts/chronoformer_best.pt`.

### Объяснения (heatmaps)

Генерирует 2 картинки: rollout и saliency (по позициям токенов) для одного сэмпла из test.

```bash
python scripts/explain.py --dry-run --checkpoint artifacts/chronoformer_best.pt --out-dir artifacts/xai --sample-index 0
```

Результат: `artifacts/xai/rollout_sample0.png`, `artifacts/xai/saliency_sample0.png`.

### Fidelity

Сравнивает, насколько падает вероятность при маскировании top-k важных токенов.

```bash
python scripts/fidelity.py --dry-run --checkpoint artifacts/chronoformer_best.pt --k 10
```

