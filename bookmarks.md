# Цікаві знахідки та корисні репозиторії

Колекція цікавих проєктів, інструментів та ресурсів, знайдених під час навчання.

---

## AI / Deep Learning / Inference

### [TensorRT-Model-Connect](https://github.com/NVIDIA/TensorRT-Model-Connect)

Інструмент від **NVIDIA** (public preview, серпень 2026), який дозволяє взяти PyTorch-модель з Hugging Face і перетворити її на оптимізований TensorRT inference буквально двома командами:

```bash
trtmc build Qwen/Qwen3-0.6B --output qwen3-0.6b.bundle
trtmc run ./qwen3-0.6b.bundle --prompt "What is the capital of France?"
```

- Будує TensorRT-движок **без проміжного ONNX-експорту**
- Створює `.bundle` артефакт, який працює і з Python, і з C++
- Підтримує генерацію тексту, транскрипцію, генерацію зображень/відео, сегментацію, ембедінги тощо
- Рівень **deployment/MLOps** — оптимізація вже навчених моделей для швидкого inference на NVIDIA GPU

---

## QA / Testing

### [TPI-Next](https://github.com/qamania/TPI-Next)

Веб-інструмент від **QAMania** для оцінки зрілості процесів тестування за методологією TPI NEXT (Test Process Improvement). Оригінальний TPI Next — це Excel-файл; цей проєкт — зручна веб-версія.

- Відповідаєш на запитання анкети — отримуєш оцінку зрілості процесу тестування
- Усі дані зберігаються локально у браузері (IndexedDB), без серверу
- Додана категорія **AI**, якої немає в оригінальному інструменті
- Результати можна завантажити як CSV або PDF-звіт
- [Спробувати онлайн](https://qamania.github.io/TPI-Next/en/index.html)

---

## Data Science / APIs

### [TPI Assessment API](https://github.com/lse-ds205/tpi-apis)

**FastAPI**-додаток від студентів LSE Data Science Institute для роботи з даними Transition Pathway Initiative Centre (TPI) — організації, що оцінює готовність країн та компаній до кліматичного переходу.

- RESTful API для фреймворків оцінки **ASCOR** (країни) та **Carbon Performance / Management Quality** (компанії)
- Автоматичне завантаження останніх датасетів, пагінація, фільтрація за секторами
- Порівняння результатів оцінки між різними циклами
- Побудований на FastAPI + Pydantic + pandas
- Гарний приклад навчального проєкту: структура коду, валідація, документація
