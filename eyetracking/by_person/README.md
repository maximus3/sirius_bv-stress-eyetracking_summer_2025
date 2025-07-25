# Запуск

```
uv run eyetracking/by_person/person_level_analysis.py --exclude-participants 1707KAV 1807HEE 1807CAA 1607LVA 1907ZSI 1707DMA 1707SAA 1807SAV --show-plots
```

# Описание файла trial.xls

Файл `trial.xls` содержит данные айтрекинга (отслеживания движений глаз) для различных участников эксперимента. Каждая строка представляет собой отдельный трайл (попытку) для конкретного участника.

## Описание колонок

### Основная информация
- **RECORDING_SESSION_LABEL** - Метка сессии записи данных (идентификатор участника), всего 14 участников
- **INDEX** - Порядковый номер трайла (текста) в записи, всего 6 текстов
- 1-3 индексы - тексты без стресса
- 4-6 индексы - тексты со стрессом

### Параметры моргания
- **AVERAGE_BLINK_DURATION** - Средняя продолжительность морганий в миллисекундах в текущем трайле (если морганий не было - просто '.')
- **BLINK_COUNT** - Общее количество морганий в трайле

### Параметры фиксаций
- **AVERAGE_FIXATION_DURATION** - Средняя продолжительность фиксаций в миллисекундах (время, которое глаз останавливается на одном месте)
- **MEDIAN_FIXATION_DURATION** - Медианная продолжительность фиксаций в миллисекундах (серединное значение всех фиксаций)
- **SD_FIXATION_DURATION** - Стандартное отклонение продолжительности фиксаций в миллисекундах (показатель разброса длительности фиксаций)
- **FIXATION_DURATION_MAX** - Максимальная продолжительность фиксации в миллисекундах (самая длинная остановка взгляда)
- **FIXATION_DURATION_MIN** - Минимальная продолжительность фиксации в миллисекундах (самая короткая остановка взгляда)
- **FIXATION_COUNT** - Общее количество фиксаций в трайле (сколько раз глаз останавливался)

### Параметры саккад
- **AVERAGE_SACCADE_AMPLITUDE** - Средняя амплитуда саккад в градусах визуального угла (среднее расстояние, на которое "прыгает" взгляд между фиксациями)
- **MEDIAN_SACCADE_AMPLITUDE** - Медианная амплитуда саккад в градусах визуального угла (серединное значение расстояний между фиксациями)
- **SD_SACCADE_AMPLITUDE** - Стандартное отклонение амплитуды саккад в градусах визуального угла (показатель разброса расстояний между фиксациями)
- **SACCADE_COUNT** - Общее количество саккад в трайле (сколько раз взгляд "прыгал" между точками)

### Временные параметры
- **DURATION** - Продолжительность записи айтрекера для текущего трайла (между событиями "START" и "END")

### Параметры размера зрачка
- **PUPIL_SIZE_MAX** - Максимальный размер зрачка в произвольных единицах
- **PUPIL_SIZE_MEAN** - Средний размер зрачка в произвольных единицах
- **PUPIL_SIZE_MIN** - Минимальный размер зрачка в произвольных единицах

### Параметры областей интереса (AOI)
- **VISITED_INTEREST_AREA_COUNT** - Общее количество уникальных областей интереса, посещенных в трайле (1 область интереса = 1 слово, то есть сколько разных слов участник посмотрел)
- **IA_COUNT** - Общее количество уникальных областей интереса (слов) в трайле (сколько всего слов было в тексте)
- **RUN_COUNT** - Общее количество серий фиксаций в выбранном периоде интереса в трайле (две последовательные фиксации в одной области интереса принадлежат к одной серии)
  - Пример: если последовательность фиксаций [1, 2, 3, 3, 3, 4, 5, 5, 6, 7, 3, 3, 4], то RUN_COUNT = 8 (серии: [1], [2], [3,3,3], [4], [5,5], [6], [7], [3,3], [4])
- **SAMPLE_COUNT** - Общее количество сэмплов в трайле в выбранном периоде интереса (каждое измерение положения взгляда айтрекером - это один сэмпл, обычно 60-1000 сэмплов в секунду)

### Последовательности фиксаций
- **INTEREST_AREA_FIXATION_SEQUENCE** - Список порядка, в котором были зафиксированы области интереса в текущем трайле (последовательность номеров слов, на которые смотрел участник)
То есть если последовательность [1, 2, 3, 4, 2, 3, 4, 5, 6, 7, 3, 4, 7], то тут у нас есть две возвратные саккады. Это очень важно, так как важно посмотреть на количество возвратных саккад
- **INTEREST_AREA_FIXATION_SEQUENCE_DWELL_TIMES** - Список времени пребывания для каждой серии фиксаций в порядке, указанном в INTEREST_AREA_FIXATION_SEQUENCE (сколько времени участник смотрел на каждое слово)

### Информация о глазах
- **EYE_REPORTED** - Записывает, данные какого глаза (ЛЕВЫЙ или ПРАВЫЙ) использовались для создания этого отчета (какой глаз лучше отслеживался)
- **VALIDATION_RESULT_LEFT_EYE** - Результат валидации левого глаза перед записью текущего трайла (качество калибровки левого глаза: GOOD/FAIR/POOR)
- **VALIDATION_RESULT_RIGHT_EYE** - Результат валидации правого глаза перед записью текущего трайла (качество калибровки правого глаза: GOOD/FAIR/POOR)

## Примечания

- **Фиксация** - это остановка взгляда на одном месте (обычно на слове или части текста)
- **Саккада** - это быстрое движение глаза между фиксациями (как "прыжок" взгляда)
- Значения "0" в последовательностях фиксаций представляют случаи, когда не была зафиксирована ни одна область интереса (взгляд был между словами)
- Размеры зрачка измеряются в произвольных единицах и могут отсутствовать, если сэмплы не загружены в сессию просмотра
- Результаты валидации глаз могут отсутствовать, если валидация не проводилась или соответствующий глаз не отслеживался
- В файле представлены данные для различных участников с уникальными идентификаторами сессий (например, 1607KYA, 1607LVA, 1707KAV и т.д.)
- **Область интереса (AOI)** - это заранее определенная зона на экране (в данном случае - отдельные слова)
- **Серия фиксаций (Run)** - это группа последовательных фиксаций на одном и том же слове (если участник смотрит на слово "дом" несколько раз подряд, не переходя к другим словам - это одна серия)
- **Сэмпл** - это одно измерение положения взгляда айтрекером (например, если айтрекер работает с частотой 60 Гц, то за секунду будет 60 сэмплов) 
