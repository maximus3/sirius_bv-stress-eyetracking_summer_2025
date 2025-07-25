# 🎨 Презентационные графики: Значимые показатели айтрекинга при стрессе

## 📋 Назначение скрипта

Скрипт `presentation_stress_eyetracking_graphs.py` создан специально для генерации **высококачественных презентационных графиков** на основе результатов интегрированного анализа стресса и айтрекинга.

### 🎯 Основная задача
Создание красивых, информативных графиков для презентаций, отчетов и публикаций, которые фокусируются **только на статистически значимых результатах**.

---

## 🏆 Что анализируется

### ✅ **ТОП-4 ЗНАЧИМЫХ ПОКАЗАТЕЛЯ** (p < 0.05):
1. **AVERAGE_BLINK_DURATION** 🥇 - Длительность моргания (-38.8%, d = -0.891)
2. **AVERAGE_FIXATION_DURATION** 🥈 - Длительность фиксации (-7.0%, d = -0.795)  
3. **FIXATIONS_PER_SECOND** 🥉 - Частота фиксаций (+7.2%, d = 0.754)
4. **SACCADES_PER_SECOND** 🏅 - Частота саккад (+7.2%, d = 0.752)

### 🤔 **ИНТЕРЕСНЫЕ ТЕНДЕНЦИИ** (p > 0.05, но близко к значимости):
- **PUPIL_SIZE_MEAN** - Размер зрачка (+14.5%, d = 0.473)
- **TEXT_COVERAGE_PERCENT** - Покрытие текста (+4.9%, d = 0.355)
- **BLINK_COUNT** - Количество морганий (-24.7%, d = -0.311)

---

## 🎨 Создаваемые графики

### 1. **top_4_significant_measures.png**
- **Назначение**: Основной график для презентаций
- **Содержание**: Boxplot сравнение респондеров vs нон-респондеров по 4 значимым показателям
- **Особенности**: Статистические аннотации, ранжирование по значимости, красивое оформление

### 2. **effect_sizes_chart.png**
- **Назначение**: Визуализация размеров эффектов
- **Содержание**: Горизонтальный bar chart с размерами эффектов всех показателей
- **Особенности**: Цветовое кодирование значимости, интерпретационные линии (малый/средний/большой эффект)

### 3. **key_dynamics_timeline.png**  
- **Назначение**: Временная динамика ключевых показателей
- **Содержание**: Изменения по текстам для длительности фиксации и частоты фиксаций
- **Особенности**: Разделение на базовую линию и стрессовую фазу, доверительные интервалы

### 4. **summary_infographic.png**
- **Назначение**: Итоговая инфографика для презентаций
- **Содержание**: Ключевые результаты, статистика, рекомендации в едином дизайне
- **Особенности**: Структурированная подача информации, готовый слайд для презентации

---

## 🔬 Научная обоснованность

### 📊 **Статистическая база**:
- Данные 14 участников 
- 16 анализируемых показателей айтрекинга
- Интеграция с результатами полиграфа для классификации стресса
- Использование Cohen's d для оценки размера эффекта
- t-тесты и Mann-Whitney U для статистических сравнений

### 👥 **Классификация участников**:
- **7 респондеров**: участники, которые реально испытывали стресс (по данным полиграфа)
- **7 нон-респондеров**: участники, которые не реагировали на стрессовые стимулы

---

## 🚀 Использование

### 📋 **Требования**:
```python
pandas, numpy, matplotlib, seaborn, scipy, pathlib
```

### ▶️ **Запуск**:
```bash
cd eyetracking/
python presentation_stress_eyetracking_graphs.py
```

### 📁 **Результаты**:
Все графики сохраняются в директории `eyetracking/presentation_results/`

---

## 🎯 Отличия от основного анализа

### ✂️ **Упрощения**:
- **Фокус только на значимых результатах** (не все 16 показателей)
- **Презентационное качество** графиков (высокое разрешение, красивое оформление)
- **Готовые материалы** для использования в отчетах и презентациях
- **Краткая сводка** вместо подробного анализа

### 🎨 **Улучшения дизайна**:
- Высокое разрешение (300 DPI)
- Цветовое кодирование по смыслу
- Информативные аннотации на графиках
- Готовая инфографика
- Профессиональные шрифты и стиль

---

## 💡 Когда использовать

### ✅ **Рекомендуется для**:
- **Презентаций** на конференциях и семинарах
- **Отчетов** для заказчиков и руководства  
- **Публикаций** в научных журналах
- **Быстрой демонстрации** ключевых результатов

### ❌ **Не подходит для**:
- Подробного исследовательского анализа (используйте `integrated_stress_eyetracking_analysis.py`)
- Анализа всех показателей (только значимые)
- Детальной статистической интерпретации (акцент на визуальном представлении)

---

## 📈 Ключевые выводы для презентаций

### 🏆 **Главный результат**:
**Айтрекинг показывает ОГРАНИЧЕННУЮ эффективность для детекции стресса** - только 25% показателей (4 из 16) статистически значимо различаются.

### 💡 **Практические рекомендации**:
1. **Фокус на временных характеристиках**: длительность морганий и фиксаций
2. **Комбинирование с физиологическими данными** для повышения точности
3. **Учет индивидуальных различий** участников
4. **Создание составных индексов** стресса

### 🔬 **Научная значимость**:
- Первое исследование, объединившее данные полиграфа и айтрекинга
- Выявлены специфические паттерны чтения при стрессе
- Обоснована необходимость мультимодального подхода к детекции стресса

---

## 👥 Авторы и использование

Скрипт создан в рамках летней школы "Сириус БВ - Стресс и айтрекинг 2025".

**Для цитирования результатов** обязательно ссылайтесь на полный интегрированный анализ (`INTEGRATED_RESULTS_SUMMARY.md`).

---

*Создано для быстрой визуализации ключевых научных результатов исследования стресса через айтрекинг* 🎯 