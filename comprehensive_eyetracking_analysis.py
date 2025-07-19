import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import (
    ttest_ind,
    mannwhitneyu,
)
import os
import warnings
import argparse

# =============================================================================
# КОНФИГУРАЦИЯ И КОНСТАНТЫ
# =============================================================================


# Функция для парсинга аргументов командной строки
def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="Комплексный анализ данных айтрекинга для детекции стресса",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--word-file",
        type=str,
        default="data/ia_avg.xls",
        help="Путь к файлу с данными на уровне слов",
    )

    parser.add_argument(
        "--trial-file",
        type=str,
        default="data/events.xls",
        help="Путь к файлу с данными на уровне трайлов",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Путь к папке для сохранения результатов",
    )

    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Отображать созданные графики (по умолчанию только сохранять)",
    )

    return parser.parse_args()


# Глобальные переменные для конфигурации (будут установлены из аргументов)
WORD_DATA_FILE = None
TRIAL_DATA_FILE = None
RESULTS_DIR = None
SHOW_PLOTS = False

# Параметры данных
WORD_NUMERIC_COLUMNS_POSITIONS = [3, 4, 7, 8, 9, 10, 11]
WORD_COLUMN_FOR_FILTERING = 6  # Колонка с текстом слова для фильтрации

# Пороги для групп стресса
STRESS_THRESHOLD = 3  # Трайлы 1-STRESS_THRESHOLD без стресса, остальные со стрессом

# Параметры трайлов (будут определены динамически из данных)
TOTAL_TRIALS = None  # Будет определено автоматически при загрузке данных
MAX_TRIAL_NUMBER = None  # Максимальный номер трайла

# Статистические параметры
ALPHA_LEVEL = 0.05
MIN_SAMPLE_SIZE_WARNING = 30  # Минимальный размер выборки для надежных выводов

# Пороги размеров эффектов Cohen's d
EFFECT_SIZE_SMALL = 0.2
EFFECT_SIZE_MEDIUM = 0.5
EFFECT_SIZE_LARGE = 0.8

# Параметры визуализации
FIGURE_DPI = 300
COLORS_NO_STRESS = "#2E86C1"  # Синий
COLORS_STRESS = "#E74C3C"  # Красный

# Настройки для предупреждений
SHOW_STATISTICAL_WARNINGS = False

# =============================================================================

# Настройка warnings
if not SHOW_STATISTICAL_WARNINGS:
    warnings.filterwarnings("ignore")
    # Подавляем специфичные matplotlib warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    warnings.filterwarnings("ignore", message=".*labels.*parameter.*boxplot.*")
    warnings.filterwarnings("ignore", message=".*Glyph.*missing from font.*")
    plt.set_loglevel("WARNING")

# Настройка для красивых графиков
plt.style.use("seaborn-v0_8")
plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12


def ensure_results_directory():
    """Создает папку для результатов если она не существует"""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"📁 Создана папка '{RESULTS_DIR}' для сохранения графиков")
    else:
        print(f"📁 Папка '{RESULTS_DIR}' уже существует")


def show_plot_conditionally():
    """Показывает график только если установлен флаг SHOW_PLOTS"""
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()  # Закрываем фигуру для экономии памяти


def determine_trial_parameters(trial_data):
    """Определяет параметры трайлов динамически из данных"""
    global TOTAL_TRIALS, MAX_TRIAL_NUMBER

    TOTAL_TRIALS = len(trial_data)
    MAX_TRIAL_NUMBER = trial_data["trial"].max()

    print(f"📊 ОПРЕДЕЛЕНЫ ПАРАМЕТРЫ ТРАЙЛОВ:")
    print(f"   • Общее количество трайлов: {TOTAL_TRIALS}")
    print(f"   • Максимальный номер трайла: {MAX_TRIAL_NUMBER}")
    print(
        f"   • Порог стресса: трайлы 1-{STRESS_THRESHOLD} (без стресса), {STRESS_THRESHOLD + 1}-{MAX_TRIAL_NUMBER} (стресс)"
    )

    return TOTAL_TRIALS, MAX_TRIAL_NUMBER


def create_dynamic_phase_mapping(max_trial):
    """Создает динамический маппинг фаз в зависимости от количества трайлов"""
    phase_mapping = {}

    # Базовая линия: трайлы 1 до STRESS_THRESHOLD
    for i in range(1, STRESS_THRESHOLD + 1):
        if i <= max_trial:
            phase_mapping[i] = f"baseline_{i}"

    # Стрессовые трайлы: от STRESS_THRESHOLD+1 до max_trial
    stress_trials = list(range(STRESS_THRESHOLD + 1, max_trial + 1))

    if len(stress_trials) == 0:
        # Нет стрессовых трайлов
        pass
    elif len(stress_trials) == 1:
        # Только один стрессовый трайл
        phase_mapping[stress_trials[0]] = "stress_peak"
    elif len(stress_trials) == 2:
        # Два стрессовых трайла: пик и восстановление
        phase_mapping[stress_trials[0]] = "stress_peak"
        phase_mapping[stress_trials[1]] = "stress_recovery"
    else:
        # Три и более стрессовых трайлов: пик, адаптация(и), восстановление
        phase_mapping[stress_trials[0]] = "stress_peak"
        phase_mapping[stress_trials[-1]] = "stress_recovery"

        # Промежуточные трайлы - адаптация
        for i, trial_num in enumerate(stress_trials[1:-1], 1):
            phase_mapping[trial_num] = f"stress_adapt_{i}"

    return phase_mapping


def validate_sample_size(n, analysis_name="анализ"):
    """Проверяет достаточность размера выборки и выводит предупреждения"""
    if n < MIN_SAMPLE_SIZE_WARNING:
        print(f"🚨 КРИТИЧЕСКОЕ ПРЕДУПРЕЖДЕНИЕ ({analysis_name}):")
        print(f"   Размер выборки N={n} КРИТИЧЕСКИ МАЛ!")
        print(
            f"   Для надежных статистических выводов требуется N ≥ {MIN_SAMPLE_SIZE_WARNING}"
        )
        print(f"   ⚠️ ВСЕ РЕЗУЛЬТАТЫ МОГУТ БЫТЬ НЕНАДЕЖНЫМИ!")
        return False
    return True


def apply_bonferroni_correction(p_values, alpha=ALPHA_LEVEL):
    """Применяет поправку Бонферрони на множественные сравнения"""
    if not p_values or len(p_values) == 0:
        return []

    corrected_alpha = alpha / len(p_values)
    corrected_p_values = [min(1.0, p * len(p_values)) for p in p_values]

    print(f"🔧 ПОПРАВКА БОНФЕРРОНИ:")
    print(f"   Количество тестов: {len(p_values)}")
    print(f"   Скорректированный α: {corrected_alpha:.4f}")

    return corrected_p_values, corrected_alpha


def get_phase_color(phase_name):
    """Возвращает цвет для любой фазы, включая динамические"""
    if phase_name.startswith("baseline"):
        baseline_colors = ["#3498DB", "#5DADE2", "#85C1E9"]
        try:
            baseline_num = int(phase_name.split("_")[1]) - 1
            return (
                baseline_colors[baseline_num]
                if baseline_num < len(baseline_colors)
                else "#85C1E9"
            )
        except:
            return "#3498DB"
    elif phase_name == "stress_peak":
        return COLORS_STRESS
    elif phase_name.startswith("stress_adapt"):
        return "#F1948A"
    elif phase_name == "stress_recovery":
        return "#F8C471"
    else:
        return "#95A5A6"  # Серый для неизвестных фаз


def interpret_effect_size_with_warnings(cohens_d, p_value, n_total, variable_name):
    """
    Интерпретирует размер эффекта с критическими предупреждениями о надежности
    """
    abs_d = abs(cohens_d)

    if abs_d >= EFFECT_SIZE_LARGE:
        size_text = "большой"
    elif abs_d >= EFFECT_SIZE_MEDIUM:
        size_text = "средний"
    elif abs_d >= EFFECT_SIZE_SMALL:
        size_text = "малый"
    else:
        size_text = "пренебрежимо малый"

    # КРИТИЧЕСКИ ВАЖНО: Предупреждения о надежности
    warnings_list = []

    # Предупреждение о малой выборке
    if n_total < MIN_SAMPLE_SIZE_WARNING:
        warnings_list.append(f"Критически малый размер выборки (N={n_total})")

    # Предупреждение о незначимых результатах
    if p_value >= ALPHA_LEVEL:
        warnings_list.append(
            "Размер эффекта ненадежен при незначимом статистическом результате"
        )
        if n_total < 20:
            warnings_list.append(
                "При малой выборке и p≥0.05 Cohen's d может вводить в заблуждение"
            )

    # Предупреждение об экстремальных значениях
    if abs_d > 2.0 and p_value >= ALPHA_LEVEL:
        warnings_list.append(
            "Подозрительно большой размер эффекта при незначимом результате"
        )

    result_text = f"📏 Размер эффекта: {size_text} (d = {cohens_d:.3f})"

    if warnings_list:
        result_text += "\n   ⚠️ ПРЕДУПРЕЖДЕНИЯ О НАДЕЖНОСТИ:"
        for warning in warnings_list:
            result_text += f"\n     - {warning}"

        # Особо критическое предупреждение
        if p_value >= ALPHA_LEVEL and n_total < MIN_SAMPLE_SIZE_WARNING:
            result_text += f"\n   🚨 НЕ ИНТЕРПРЕТИРОВАТЬ как доказательство различий!"

    return result_text, warnings_list


def load_comprehensive_data():
    """
    Загружает и подготавливает данные айтрекинга из двух источников:
    1. ia_avg.xls - данные на уровне отдельных слов
    2. events.xls - агрегированные данные на уровне трайлов
    """
    print("🔄 КОМПЛЕКСНАЯ ЗАГРУЗКА ДАННЫХ АЙТРЕКИНГА")
    print("=" * 70)

    # 1. ДАННЫЕ НА УРОВНЕ СЛОВ
    print("📖 Загрузка данных на уровне слов...")
    word_data = pd.read_csv(WORD_DATA_FILE, sep="\t", encoding="utf-16", skiprows=1)
    col_names = list(word_data.columns)

    # Создаем группировку по условиям для данных слов
    trial_col = col_names[0]
    word_data["trial"] = word_data[trial_col].astype(int)
    word_data["condition"] = word_data["trial"].apply(
        lambda x: "no_stress" if x <= STRESS_THRESHOLD else "stress"
    )

    # Определяем максимальный номер трайла для создания динамического маппинга фаз
    max_trial_in_words = word_data["trial"].max()
    phase_mapping = create_dynamic_phase_mapping(max_trial_in_words)

    # Создаем детальную группировку по фазам для данных слов
    word_data["stress_phase"] = word_data["trial"].map(phase_mapping)

    # Русские названия для показателей слов
    word_measure_names = {
        "IA_FIRST_FIXATION_DURATION": "Длительность первой фиксации (мс)",
        "IA_FIXATION_COUNT": "Количество фиксаций на слово",
        "IA_DWELL_TIME": "Общее время пребывания (мс)",
        "IA_DWELL_TIME_%": "Процент времени пребывания (%)",
        "IA_VISITED_TRIAL_%": "Процент визитов к слову (%)",
        "IA_REVISIT_TRIAL_%": "Процент повторных визитов (%)",
        "IA_RUN_COUNT": "Количество забеганий взгляда",
    }

    # Находим числовые колонки для слов
    word_numeric_cols = [
        col_names[i] for i in WORD_NUMERIC_COLUMNS_POSITIONS if i < len(col_names)
    ]
    word_short_names = [
        "IA_FIRST_FIXATION_DURATION",
        "IA_FIXATION_COUNT",
        "IA_DWELL_TIME_%",
        "IA_DWELL_TIME",
        "IA_VISITED_TRIAL_%",
        "IA_REVISIT_TRIAL_%",
        "IA_RUN_COUNT",
    ]

    # Преобразование данных слов
    for col in word_numeric_cols:
        if col in word_data.columns:
            word_data[col] = pd.to_numeric(
                word_data[col].astype(str).str.replace(",", "."), errors="coerce"
            )

    # Создаем короткие названия для слов
    for i, col in enumerate(word_numeric_cols):
        if col in word_data.columns and i < len(word_short_names):
            word_data[word_short_names[i]] = word_data[col]

    # Фильтрация валидных данных слов
    word_col = col_names[WORD_COLUMN_FOR_FILTERING]
    word_data = word_data.dropna(subset=[word_col])
    word_data = word_data[word_data[word_col] != "."]
    word_data = word_data[word_data[word_col] != "–"]
    word_data = word_data[word_data[word_col] != "—"]

    print(f"✅ Загружено {len(word_data)} наблюдений на уровне слов")

    # Критическая проверка размера выборки для слов
    validate_sample_size(len(word_data), "данные на уровне слов")

    # 2. ДАННЫЕ НА УРОВНЕ ТРАЙЛОВ
    print("📊 Загрузка данных на уровне трайлов...")
    trial_data = pd.read_csv(TRIAL_DATA_FILE, sep="\t", encoding="utf-16", header=0)

    # Удаляем строку с описаниями и берем только данные
    trial_data = trial_data.iloc[1:].reset_index(drop=True)

    # Преобразуем запятые в точки и конвертируем в числа
    for col in trial_data.columns:
        trial_data[col] = pd.to_numeric(
            trial_data[col].astype(str).str.replace(",", "."), errors="coerce"
        )

    # Добавляем информацию о трайлах
    trial_data["trial"] = range(1, len(trial_data) + 1)
    trial_data["condition"] = trial_data["trial"].apply(
        lambda x: "no_stress" if x <= STRESS_THRESHOLD else "stress"
    )

    # Определяем параметры трайлов динамически
    determine_trial_parameters(trial_data)

    # Создаем динамическое маппинг фаз
    phase_mapping = create_dynamic_phase_mapping(MAX_TRIAL_NUMBER)
    trial_data["phase"] = trial_data["trial"].map(phase_mapping)

    # Вычисляем долю посещенных зон (в процентах) перед переименованием
    # INTEREST_AREA_COUNT - предпоследняя колонка (индекс -2)
    # VISITED_INTEREST_AREA_COUNT - последняя колонка (индекс -1)
    interest_areas_count = pd.to_numeric(
        trial_data.iloc[:, 13], errors="coerce"
    )  # INTEREST_AREA_COUNT
    visited_areas_count = pd.to_numeric(
        trial_data.iloc[:, 14], errors="coerce"
    )  # VISITED_INTEREST_AREA_COUNT

    # Создаем долю посещенных зон в процентах
    visited_areas_percent = (visited_areas_count / interest_areas_count * 100).fillna(0)

    # Переименовываем колонки для удобства
    trial_data.columns = [
        "blinks",
        "fixations",
        "fixation_duration_mean",
        "fixation_duration_median",
        "fixation_duration_sd",
        "pupil_size",
        "runs",
        "saccade_amplitude_mean",
        "saccade_amplitude_median",
        "saccade_amplitude_sd",
        "saccades",
        "samples",
        "trial_duration",
        "interest_areas_total",
        "visited_areas_absolute",
        "trial",
        "condition",
        "phase",
    ]

    # Заменяем абсолютное количество на долю в процентах
    trial_data["visited_areas"] = visited_areas_percent

    print(f"✅ Загружено {len(trial_data)} трайлов")
    print(
        f"   • Без стресса (трайлы 1-{STRESS_THRESHOLD}): {len(trial_data[trial_data['condition'] == 'no_stress'])} трайла"
    )
    print(
        f"   • Со стрессом (трайлы {STRESS_THRESHOLD + 1}-{MAX_TRIAL_NUMBER}): {len(trial_data[trial_data['condition'] == 'stress'])} трайла"
    )

    # КРИТИЧЕСКАЯ проверка размера выборки для трайлов
    validate_sample_size(len(trial_data), "данные на уровне трайлов")

    return word_data, trial_data, word_measure_names


def analyze_word_level_differences(word_data, word_measure_names):
    """
    Анализирует различия на уровне отдельных слов между условиями без стресса и со стрессом
    """
    print(f"\n📝 АНАЛИЗ РАЗЛИЧИЙ НА УРОВНЕ СЛОВ")
    print("=" * 70)

    measures = [
        "IA_FIRST_FIXATION_DURATION",
        "IA_FIXATION_COUNT",
        "IA_DWELL_TIME",
        "IA_DWELL_TIME_%",
        "IA_VISITED_TRIAL_%",
        "IA_REVISIT_TRIAL_%",
        "IA_RUN_COUNT",
    ]

    results = []
    p_values_for_correction = []

    # Данные по условиям
    no_stress = word_data[word_data["condition"] == "no_stress"]
    stress = word_data[word_data["condition"] == "stress"]

    print(f"📊 СРАВНЕНИЕ ГРУПП НА УРОВНЕ СЛОВ:")
    print(f"   Без стресса: {len(no_stress)} наблюдений")
    print(f"   Со стрессом: {len(stress)} наблюдений")

    # Проверяем размер выборки
    total_n = len(no_stress) + len(stress)
    validate_sample_size(total_n, "анализ на уровне слов")

    for measure in measures:
        if measure not in word_data.columns:
            continue

        print(f"\n📈 {word_measure_names.get(measure, measure)}")
        print("-" * 50)

        no_stress_vals = no_stress[measure].dropna()
        stress_vals = stress[measure].dropna()

        if len(no_stress_vals) == 0 or len(stress_vals) == 0:
            continue

        # Описательная статистика
        ns_mean, ns_std = no_stress_vals.mean(), no_stress_vals.std()
        s_mean, s_std = stress_vals.mean(), stress_vals.std()

        print(f"Без стресса: M = {ns_mean:.2f} ± {ns_std:.2f}")
        print(f"Со стрессом:  M = {s_mean:.2f} ± {s_std:.2f}")

        # Изменение
        change = s_mean - ns_mean
        change_pct = (change / ns_mean) * 100 if ns_mean != 0 else 0
        change_symbol = "📈" if change > 0 else "📉"
        change_direction = "увеличение" if change > 0 else "уменьшение"
        print(f"{change_symbol} Изменение: {change:+.2f} ({change_pct:+.1f}%)")

        # Статистический тест (проверка H0: μ₁ = μ₂ против H1: μ₁ ≠ μ₂)
        try:
            stat, p_value = mannwhitneyu(
                no_stress_vals, stress_vals, alternative="two-sided"
            )
            test_name = "Критерий Манна-Уитни"
        except:
            try:
                stat, p_value = ttest_ind(no_stress_vals, stress_vals)
                test_name = "t-критерий"
            except:
                p_value = np.nan
                test_name = "Недостаточно данных"

        if not np.isnan(p_value):
            if p_value < ALPHA_LEVEL:
                significance = "✅ H0 ОТКЛОНЯЕТСЯ (p < 0.05)"
                hypothesis_result = "Различия СТАТИСТИЧЕСКИ ЗНАЧИМЫ"
            else:
                significance = "🔸 H0 НЕ ОТКЛОНЯЕТСЯ (p ≥ 0.05)"
                hypothesis_result = "Различия статистически НЕ ЗНАЧИМЫ"
            print(f"🧪 {test_name}: p = {p_value:.4f} - {significance}")
            print(f"⚖️ Заключение: {hypothesis_result}")
            p_values_for_correction.append(p_value)
        else:
            print(f"🧪 {test_name}: недостаточно данных для проверки гипотез")
            p_values_for_correction.append(np.nan)

        # Размер эффекта (Cohen's d) с критическими предупреждениями
        pooled_std = np.sqrt(
            ((len(no_stress_vals) - 1) * ns_std**2 + (len(stress_vals) - 1) * s_std**2)
            / (len(no_stress_vals) + len(stress_vals) - 2)
        )
        cohens_d = (s_mean - ns_mean) / pooled_std if pooled_std > 0 else 0

        # Интерпретация с предупреждениями
        effect_interpretation, warnings = interpret_effect_size_with_warnings(
            cohens_d, p_value, len(no_stress_vals) + len(stress_vals), measure
        )
        print(effect_interpretation)

        results.append(
            {
                "measure": measure,
                "measure_name": word_measure_names.get(measure, measure),
                "mean_no_stress": ns_mean,
                "std_no_stress": ns_std,
                "mean_stress": s_mean,
                "std_stress": s_std,
                "change_absolute": change,
                "change_percent": change_pct,
                "change_direction": change_direction,
                "p_value": p_value,
                "cohens_d": cohens_d,
                "significant": p_value < ALPHA_LEVEL
                if not np.isnan(p_value)
                else False,
                "reliability_warnings": warnings,
            }
        )

    # Применяем поправку на множественные сравнения
    if len(p_values_for_correction) > 1:
        print(f"\n🔧 ПОПРАВКА НА МНОЖЕСТВЕННЫЕ СРАВНЕНИЯ:")
        valid_p_values = [p for p in p_values_for_correction if not np.isnan(p)]
        if valid_p_values:
            corrected_p_values, corrected_alpha = apply_bonferroni_correction(
                valid_p_values
            )
            significant_after_correction = sum(
                1 for p in corrected_p_values if p < ALPHA_LEVEL
            )
            print(
                f"   После поправки Бонферрони: {significant_after_correction}/{len(valid_p_values)} значимых результатов"
            )

    return results


def analyze_trial_level_differences(trial_data):
    """
    Анализирует различия на уровне трайлов между условиями без стресса и со стрессом
    """
    print(f"\n🧮 АНАЛИЗ РАЗЛИЧИЙ НА УРОВНЕ ТРАЙЛОВ")
    print("=" * 70)

    # Основные показатели для анализа
    measures = [
        "blinks",
        "fixations",
        "fixation_duration_mean",
        "fixation_duration_median",
        "pupil_size",
        "runs",
        "saccade_amplitude_mean",
        "saccades",
        "trial_duration",
        "visited_areas",
    ]

    # Русские названия
    measure_names = {
        "blinks": "Количество морганий",
        "fixations": "Общее количество фиксаций",
        "fixation_duration_mean": "Средняя длительность фиксаций (мс)",
        "fixation_duration_median": "Медианная длительность фиксаций (мс)",
        "pupil_size": "Размер зрачка (средний)",
        "runs": "Количество забеганий взгляда",
        "saccade_amplitude_mean": "Средняя амплитуда саккад (°)",
        "saccades": "Общее количество саккад",
        "trial_duration": "Длительность трайла (мс)",
        "visited_areas": "Количество посещенных зон",
    }

    results = []
    p_values_for_correction = []

    # Данные по условиям
    no_stress = trial_data[trial_data["condition"] == "no_stress"]
    stress = trial_data[trial_data["condition"] == "stress"]

    print(f"📊 СРАВНЕНИЕ ГРУПП:")
    print(f"   Без стресса: {len(no_stress)} трайла")
    print(f"   Со стрессом: {len(stress)} трайла")

    # КРИТИЧЕСКАЯ проверка размера выборки
    total_n = len(no_stress) + len(stress)
    validate_sample_size(total_n, "анализ на уровне трайлов")

    for measure in measures:
        if measure not in trial_data.columns:
            continue

        print(f"\n📈 {measure_names.get(measure, measure)}")
        print("-" * 50)

        no_stress_vals = no_stress[measure].dropna()
        stress_vals = stress[measure].dropna()

        # Описательная статистика
        ns_mean, ns_std = no_stress_vals.mean(), no_stress_vals.std()
        s_mean, s_std = stress_vals.mean(), stress_vals.std()

        print(f"Без стресса: M = {ns_mean:.2f} ± {ns_std:.2f}")
        print(f"Со стрессом:  M = {s_mean:.2f} ± {s_std:.2f}")

        # Изменение
        change = s_mean - ns_mean
        change_pct = (change / ns_mean) * 100 if ns_mean != 0 else 0
        change_symbol = "📈" if change > 0 else "📉"
        print(f"{change_symbol} Изменение: {change:+.2f} ({change_pct:+.1f}%)")

        # Статистический тест с предупреждением о малой выборке
        if len(no_stress_vals) >= 3 and len(stress_vals) >= 3:
            try:
                stat, p_value = mannwhitneyu(
                    no_stress_vals, stress_vals, alternative="two-sided"
                )
                test_name = "Критерий Манна-Уитни"
            except:
                stat, p_value = ttest_ind(no_stress_vals, stress_vals)
                test_name = "t-критерий"

            if p_value < ALPHA_LEVEL:
                significance = "✅ H0 ОТКЛОНЯЕТСЯ (p < 0.05)"
                hypothesis_result = "Различия СТАТИСТИЧЕСКИ ЗНАЧИМЫ"
            else:
                significance = "🔸 H0 НЕ ОТКЛОНЯЕТСЯ (p ≥ 0.05)"
                hypothesis_result = "Различия статистически НЕ ЗНАЧИМЫ"

            print(f"🧪 {test_name}: p = {p_value:.4f} - {significance}")
            print(f"⚖️ Заключение: {hypothesis_result}")

            # КРИТИЧЕСКОЕ предупреждение о малой выборке
            if p_value >= ALPHA_LEVEL:
                print(f"🚨 ВНИМАНИЕ: Критически малый размер выборки (N = {total_n})")
                print(f"   При такой малой выборке НЕВОЗМОЖНО делать выводы:")
                print(f"   • НИ о наличии различий (если p ≥ 0.05)")
                print(f"   • НИ об отсутствии различий")

            p_values_for_correction.append(p_value)

            # Размер эффекта с критическими предупреждениями
            pooled_std = np.sqrt(
                (
                    (len(no_stress_vals) - 1) * ns_std**2
                    + (len(stress_vals) - 1) * s_std**2
                )
                / (len(no_stress_vals) + len(stress_vals) - 2)
            )
            cohens_d = (s_mean - ns_mean) / pooled_std if pooled_std > 0 else 0

            # Интерпретация с предупреждениями
            effect_interpretation, warnings = interpret_effect_size_with_warnings(
                cohens_d, p_value, len(no_stress_vals) + len(stress_vals), measure
            )
            print(effect_interpretation)

        else:
            p_value = np.nan
            test_name = "Недостаточно данных"
            cohens_d = 0
            warnings = ["Критически малая выборка"]
            p_values_for_correction.append(np.nan)

        results.append(
            {
                "measure": measure,
                "measure_name": measure_names.get(measure, measure),
                "no_stress_mean": ns_mean,
                "no_stress_std": ns_std,
                "stress_mean": s_mean,
                "stress_std": s_std,
                "change_absolute": change,
                "change_percent": change_pct,
                "p_value": p_value,
                "cohens_d": cohens_d,
                "significant": p_value < ALPHA_LEVEL
                if not np.isnan(p_value)
                else False,
                "reliability_warnings": warnings if "warnings" in locals() else [],
            }
        )

    # Применяем поправку на множественные сравнения
    if len(p_values_for_correction) > 1:
        print(f"\n🔧 ПОПРАВКА НА МНОЖЕСТВЕННЫЕ СРАВНЕНИЯ:")
        valid_p_values = [p for p in p_values_for_correction if not np.isnan(p)]
        if valid_p_values:
            corrected_p_values, corrected_alpha = apply_bonferroni_correction(
                valid_p_values
            )
            significant_after_correction = sum(
                1 for p in corrected_p_values if p < ALPHA_LEVEL
            )
            print(
                f"   После поправки Бонферрони: {significant_after_correction}/{len(valid_p_values)} значимых результатов"
            )

    return results


def analyze_trial_dynamics(trial_data):
    """
    Анализирует динамику изменений по отдельным трайлам
    """
    print(f"\n📈 АНАЛИЗ ДИНАМИКИ ПО ТРАЙЛАМ")
    print("=" * 70)

    measures = [
        "blinks",
        "fixations",
        "fixation_duration_mean",
        "pupil_size",
        "runs",
        "saccade_amplitude_mean",
        "saccades",
        "visited_areas",
    ]

    measure_names = {
        "blinks": "Количество морганий",
        "fixations": "Общее количество фиксаций",
        "fixation_duration_mean": "Средняя длительность фиксаций (мс)",
        "pupil_size": "Размер зрачка",
        "runs": "Количество забеганий",
        "saccade_amplitude_mean": "Амплитуда саккад (°)",
        "saccades": "Количество саккад",
        "visited_areas": "Покрытие текста (%)",
    }

    dynamics = {}
    trial_stats = {}

    for measure in measures:
        if measure not in trial_data.columns:
            continue

        print(f"\n📊 {measure_names.get(measure, measure)}")
        print("-" * 40)

        # Значения по трайлам
        values = []
        phases = []
        trial_stats[measure] = {}

        for trial in range(1, MAX_TRIAL_NUMBER + 1):
            val = trial_data[trial_data["trial"] == trial][measure].iloc[0]
            phase = trial_data[trial_data["trial"] == trial]["phase"].iloc[0]
            values.append(val)
            phases.append(phase)

            # Сохраняем статистику для каждого трайла
            trial_stats[measure][trial] = {
                "mean": val,
                "std": 0,  # У нас только одно значение на трайл
                "phase": phase,
            }

            phase_emoji = {
                "baseline_1": "📊",
                "baseline_2": "📊",
                "baseline_3": "📊",
                "stress_peak": "🔥",
                "stress_adapt": "📉",
                "stress_recovery": "😌",
            }

            print(f"   {phase_emoji.get(phase, '📊')} Трайл {trial}: {val:.2f}")

        # Базовая линия
        baseline = np.mean(values[:3])

        # Изменения относительно базовой линии
        changes = [(v - baseline) / baseline * 100 for v in values]

        # Определяем паттерн
        peak_trial = np.argmax(np.abs(changes)) + 1
        expected_peak_trial = STRESS_THRESHOLD + 1  # Первый стрессовый трайл

        if (
            peak_trial == expected_peak_trial
        ):  # Пик в ожидаемом первом стрессовом трайле
            if len(changes) > expected_peak_trial and abs(changes[-1]) < abs(
                changes[expected_peak_trial - 1]
            ):  # Восстановление к последнему трайлу
                pattern = f"🎯 ПИК в Т{expected_peak_trial} → ВОССТАНОВЛЕНИЕ"
            else:
                pattern = f"🔥 ПИК в Т{expected_peak_trial} → БЕЗ ВОССТАНОВЛЕНИЯ"
        else:
            pattern = "❓ НЕСТАНДАРТНЫЙ ПАТТЕРН"

        print(f"   🎯 Паттерн: {pattern}")
        print(f"   📊 Базовая линия: {baseline:.2f}")
        print(
            f"   📈 Макс. изменение: {max(changes, key=abs):.1f}% (Трайл {peak_trial})"
        )

        dynamics[measure] = {
            "values": values,
            "phases": phases,
            "changes": changes,
            "baseline": baseline,
            "pattern": pattern,
        }

    return dynamics, trial_stats


def analyze_word_dynamics(word_data, word_measure_names):
    """
    Анализирует динамику изменений на уровне слов по трайлам
    """
    print(f"\n📝 АНАЛИЗ ДИНАМИКИ СЛОВ ПО ТРАЙЛАМ")
    print("=" * 70)

    measures = [
        "IA_FIRST_FIXATION_DURATION",
        "IA_FIXATION_COUNT",
        "IA_DWELL_TIME",
        "IA_DWELL_TIME_%",
        "IA_VISITED_TRIAL_%",
        "IA_REVISIT_TRIAL_%",
        "IA_RUN_COUNT",
    ]

    word_trial_stats = {}
    word_dynamics_results = {}

    for measure in measures:
        if measure not in word_data.columns:
            continue

        print(f"\n📊 {word_measure_names.get(measure, measure)}")
        print("-" * 50)

        word_trial_stats[measure] = {}

        # Вычисляем статистику по каждому трайлу
        trial_values = []
        for trial in range(1, MAX_TRIAL_NUMBER + 1):
            trial_data = word_data[word_data["trial"] == trial][measure].dropna()
            if len(trial_data) > 0:
                mean_val = trial_data.mean()
                std_val = trial_data.std()
                phase = word_data[word_data["trial"] == trial]["stress_phase"].iloc[0]
            else:
                mean_val = 0
                std_val = 0
                phase = f"trial_{trial}"

            word_trial_stats[measure][trial] = {
                "mean": mean_val,
                "std": std_val,
                "phase": phase,
                "count": len(trial_data),
            }
            trial_values.append(mean_val)

            print(
                f"   Трайл {trial}: M = {mean_val:.2f} ± {std_val:.2f} (n = {len(trial_data)})"
            )

        # Статистические тесты
        baseline_vs_peak_p = np.nan
        peak_vs_recovery_p = np.nan

        try:
            # Сравнение базовой линии с пиком стресса (первый стрессовый трайл)
            baseline_data = []
            for t in range(1, STRESS_THRESHOLD + 1):
                baseline_data.extend(
                    word_data[word_data["trial"] == t][measure].dropna().tolist()
                )
            peak_trial = STRESS_THRESHOLD + 1  # Первый стрессовый трайл
            peak_data = (
                word_data[word_data["trial"] == peak_trial][measure].dropna().tolist()
            )

            if len(baseline_data) > 0 and len(peak_data) > 0:
                _, baseline_vs_peak_p = mannwhitneyu(
                    baseline_data, peak_data, alternative="two-sided"
                )

            # Сравнение пика стресса с восстановлением (последний трайл)
            recovery_data = (
                word_data[word_data["trial"] == MAX_TRIAL_NUMBER][measure]
                .dropna()
                .tolist()
            )
            if len(peak_data) > 0 and len(recovery_data) > 0:
                _, peak_vs_recovery_p = mannwhitneyu(
                    peak_data, recovery_data, alternative="two-sided"
                )

        except Exception as e:
            print(f"   ⚠️ Ошибка в статистическом тесте: {e}")

        word_dynamics_results[measure] = {
            "baseline_vs_peak_p": baseline_vs_peak_p,
            "peak_vs_recovery_p": peak_vs_recovery_p,
            "measure": measure,
        }

        if not np.isnan(baseline_vs_peak_p):
            print(f"   🧪 База vs Т{peak_trial}: p = {baseline_vs_peak_p:.4f}")
        if not np.isnan(peak_vs_recovery_p):
            print(
                f"   🧪 Т{peak_trial} vs Т{MAX_TRIAL_NUMBER}: p = {peak_vs_recovery_p:.4f}"
            )

    return word_trial_stats, word_dynamics_results


def create_enhanced_word_visualizations(
    word_data, word_test_results, word_measure_names
):
    """Создает улучшенные графики анализа данных на уровне слов"""
    print(f"\n🎨 СОЗДАНИЕ ГРАФИКОВ АНАЛИЗА СЛОВ")
    print("=" * 50)

    measures = [r["measure"] for r in word_test_results]

    # Создаем фигуру
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()

    colors_no_stress = "#2E86C1"  # Синий
    colors_stress = "#E74C3C"  # Красный

    for i, result in enumerate(word_test_results):
        if i >= len(axes):
            break

        ax = axes[i]
        measure = result["measure"]

        # Данные для boxplot
        no_stress_data = word_data[word_data["condition"] == "no_stress"][
            measure
        ].dropna()
        stress_data = word_data[word_data["condition"] == "stress"][measure].dropna()

        if len(no_stress_data) == 0 or len(stress_data) == 0:
            ax.text(
                0.5,
                0.5,
                "Недостаточно\nданных",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{result['measure_name']}", fontweight="bold", fontsize=11)
            continue

        # Создаем boxplot
        bp = ax.boxplot(
            [no_stress_data, stress_data],
            labels=["Без стресса", "Со стрессом"],
            patch_artist=True,
            notch=True,
            widths=0.6,
        )

        # Раскрашиваем боксы
        bp["boxes"][0].set_facecolor(colors_no_stress)
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor(colors_stress)
        bp["boxes"][1].set_alpha(0.7)

        # Заголовок с названием показателя
        ax.set_title(f"{result['measure_name']}", fontweight="bold", fontsize=11)

        # Добавляем стрелку показывающую направление изменения
        y_max = max(no_stress_data.max(), stress_data.max())
        y_min = min(no_stress_data.min(), stress_data.min())
        y_range = y_max - y_min

        # Позиция для стрелки и текста
        arrow_y = y_max + y_range * 0.1
        text_y = y_max + y_range * 0.15

        if result["change_direction"] == "увеличение":
            ax.annotate(
                "",
                xy=(2, arrow_y),
                xytext=(1, arrow_y),
                arrowprops=dict(arrowstyle="->", lw=2, color="red"),
            )
            direction_symbol = "↗️"
        else:
            ax.annotate(
                "",
                xy=(1, arrow_y),
                xytext=(2, arrow_y),
                arrowprops=dict(arrowstyle="->", lw=2, color="blue"),
            )
            direction_symbol = "↘️"

        # Информация об изменении
        change_text = f"{direction_symbol} {abs(result['change_percent']):.1f}%"
        ax.text(
            1.5,
            text_y,
            change_text,
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

        # Информация о значимости
        if result["significant"]:
            significance_text = f"p = {result['p_value']:.3f} ✅"
            bbox_color = "lightgreen"
        else:
            significance_text = f"p = {result['p_value']:.3f} ❌"
            bbox_color = "lightcoral"

        ax.text(
            0.5,
            0.02,
            significance_text,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=bbox_color, alpha=0.7),
        )

        # Добавляем средние значения как точки
        ax.scatter(
            [1],
            [result["mean_no_stress"]],
            color="darkblue",
            s=50,
            zorder=10,
            marker="D",
        )
        ax.scatter(
            [2], [result["mean_stress"]], color="darkred", s=50, zorder=10, marker="D"
        )

        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Значение")

    # Удаляем лишние подграфики
    for i in range(len(word_test_results), len(axes)):
        fig.delaxes(axes[i])

    # Общий заголовок
    fig.suptitle(
        "📝 ВЛИЯНИЕ СТРЕССА НА ПАРАМЕТРЫ АЙТРЕКИНГА ПРИ ЧТЕНИИ (УРОВЕНЬ СЛОВ)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(
        f"{RESULTS_DIR}/word_level_stress_analysis.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    show_plot_conditionally()


def create_dynamics_visualizations(
    trial_data,
    trial_stats,
    word_data,
    word_trial_stats,
    word_dynamics_results,
    word_measure_names,
):
    """Создает графики динамики стресса по трайлам для обоих типов данных"""
    print(f"\n🎨 СОЗДАНИЕ ГРАФИКОВ ДИНАМИКИ ПО ТРАЙЛАМ")
    print("=" * 50)

    # 1. ДИНАМИКА НА УРОВНЕ ТРАЙЛОВ
    print("📊 Графики динамики на уровне трайлов...")

    trial_measures = list(trial_stats.keys())

    # Создаем большую фигуру для данных трайлов
    fig1, axes1 = plt.subplots(3, 3, figsize=(24, 18))
    axes1 = axes1.flatten()

    # Цвета для фаз
    phase_colors = {
        "baseline_1": "#3498DB",  # Синий
        "baseline_2": "#5DADE2",  # Светло-синий
        "baseline_3": "#85C1E9",  # Еще светлее синий
        "stress_peak": "#E74C3C",  # Красный (пик)
        "stress_adapt": "#F1948A",  # Розовый (адаптация)
        "stress_recovery": "#F8C471",  # Желтый (восстановление)
    }

    for i, measure in enumerate(trial_measures):
        if i >= len(axes1):
            break

        ax = axes1[i]

        # Данные для графика
        trials = sorted(trial_stats[measure].keys())
        means = [trial_stats[measure][t]["mean"] for t in trials]
        stds = [trial_stats[measure][t]["std"] for t in trials]
        phases = [trial_stats[measure][t]["phase"] for t in trials]
        colors = [get_phase_color(phase) for phase in phases]

        # Создаем bar plot с разными цветами для фаз
        bars = ax.bar(
            trials,
            means,
            yerr=stds,
            capsize=5,
            color=colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
        )

        # Добавляем линию тренда
        ax.plot(trials, means, "k--", alpha=0.7, linewidth=2, marker="o", markersize=6)

        # Вертикальная линия разделяющая фазы
        ax.axvline(x=3.5, color="red", linestyle=":", linewidth=2, alpha=0.7)
        ax.text(
            3.5,
            max(means) * 0.9,
            "НАЧАЛО\nСТРЕССА",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3),
        )

        # Заголовок
        trial_measure_names = {
            "blinks": "Моргания",
            "fixations": "Фиксации",
            "fixation_duration_mean": "Длит. фиксаций",
            "pupil_size": "Размер зрачка",
            "runs": "Забегания",
            "saccade_amplitude_mean": "Амплитуда саккад",
            "saccades": "Саккады",
            "visited_areas": "Посещ. зоны",
        }
        ax.set_title(
            f"{trial_measure_names.get(measure, measure)}",
            fontweight="bold",
            fontsize=13,
        )

        # Подписи процентных изменений на барах
        baseline_mean = np.mean([trial_stats[measure][t]["mean"] for t in [1, 2, 3]])
        for j, (trial, mean_val) in enumerate(zip(trials, means)):
            if trial > 3:  # Только для стрессовых трайлов
                change_pct = ((mean_val - baseline_mean) / baseline_mean) * 100
                symbol = "↗" if change_pct > 0 else "↘"
                ax.text(
                    trial,
                    mean_val + stds[j] + max(means) * 0.02,
                    f"{symbol}{abs(change_pct):.1f}%",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=10,
                )

        # Настройка осей
        ax.set_xlabel("Номер трайла")
        ax.set_ylabel("Значение")
        ax.set_xticks(trials)
        ax.grid(True, alpha=0.3)

        # Добавляем легенду фаз (только на первом графике)
        if i == 0:
            legend_elements = []
            legend_labels = {
                "baseline_1": "Базовая линия",
                "stress_peak": "Пик стресса",
                "stress_adapt": "Адаптация",
                "stress_recovery": "Восстановление",
            }
            for phase, color in phase_colors.items():
                if phase in legend_labels:
                    legend_elements.append(
                        plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.8)
                    )

            ax.legend(
                legend_elements,
                list(legend_labels.values()),
                loc="upper right",
                fontsize=9,
            )

    # Удаляем лишние подграфики
    for i in range(len(trial_measures), len(axes1)):
        fig1.delaxes(axes1[i])

    # Общий заголовок
    fig1.suptitle(
        "📊 ДИНАМИКА СТРЕССА ПО ТРАЙЛАМ (УРОВЕНЬ ТРАЙЛОВ)",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(
        f"{RESULTS_DIR}/trial_level_dynamics.png", dpi=FIGURE_DPI, bbox_inches="tight"
    )
    show_plot_conditionally()

    # 2. ДИНАМИКА НА УРОВНЕ СЛОВ
    print("📝 Графики динамики на уровне слов...")

    word_measures = list(word_trial_stats.keys())[:8]  # Берем первые 8 для красоты

    # Создаем большую фигуру для данных слов
    fig2, axes2 = plt.subplots(3, 3, figsize=(24, 18))
    axes2 = axes2.flatten()

    for i, measure in enumerate(word_measures):
        if i >= len(axes2):
            break

        ax = axes2[i]

        # Данные для графика
        trials = sorted(word_trial_stats[measure].keys())
        means = [word_trial_stats[measure][t]["mean"] for t in trials]
        stds = [word_trial_stats[measure][t]["std"] for t in trials]
        phases = [word_trial_stats[measure][t]["phase"] for t in trials]
        colors = [get_phase_color(phase) for phase in phases]

        # Создаем bar plot с error bars
        bars = ax.bar(
            trials,
            means,
            yerr=stds,
            capsize=5,
            color=colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
        )

        # Добавляем линию тренда
        ax.plot(trials, means, "k--", alpha=0.7, linewidth=2, marker="o", markersize=6)

        # Вертикальная линия разделяющая фазы
        ax.axvline(x=3.5, color="red", linestyle=":", linewidth=2, alpha=0.7)
        ax.text(
            3.5,
            max(means) * 0.9,
            "НАЧАЛО\nСТРЕССА",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3),
        )

        # Заголовок
        ax.set_title(
            f"{word_measure_names.get(measure, measure)}",
            fontweight="bold",
            fontsize=11,
        )

        # Показываем значимость основных сравнений
        result = word_dynamics_results.get(measure)
        if result:
            significance_text = ""
            peak_trial_num = STRESS_THRESHOLD + 1
            if (
                not np.isnan(result["baseline_vs_peak_p"])
                and result["baseline_vs_peak_p"] < ALPHA_LEVEL
            ):
                significance_text += f"База↔{peak_trial_num}: ✅ "
            if (
                not np.isnan(result["peak_vs_recovery_p"])
                and result["peak_vs_recovery_p"] < ALPHA_LEVEL
            ):
                significance_text += f"{peak_trial_num}↔{MAX_TRIAL_NUMBER}: ✅"

            if significance_text:
                ax.text(
                    0.02,
                    0.98,
                    significance_text,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=9,
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7
                    ),
                )

        # Настройка осей
        ax.set_xlabel("Номер трайла")
        ax.set_ylabel("Значение")
        ax.set_xticks(trials)
        ax.grid(True, alpha=0.3)

    # Удаляем лишние подграфики
    for i in range(len(word_measures), len(axes2)):
        fig2.delaxes(axes2[i])

    # Общий заголовок
    fig2.suptitle(
        "📝 ДИНАМИКА СТРЕССА ПО ТРАЙЛАМ (УРОВЕНЬ СЛОВ)",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(
        f"{RESULTS_DIR}/word_level_dynamics.png", dpi=FIGURE_DPI, bbox_inches="tight"
    )
    show_plot_conditionally()


def create_comprehensive_visualizations(trial_data, comparison_results, dynamics):
    """
    Создает комплексные визуализации данных айтрекинга
    """
    print(f"\n🎨 СОЗДАНИЕ КОМПЛЕКСНЫХ ГРАФИКОВ")
    print("=" * 50)

    # 1. СРАВНЕНИЕ УСЛОВИЙ (БЕЗ СТРЕССА VS СО СТРЕССОМ)
    fig, axes = plt.subplots(2, 5, figsize=(25, 12))
    axes = axes.flatten()

    for i, result in enumerate(comparison_results[:10]):
        if i >= len(axes):
            break

        ax = axes[i]
        measure = result["measure"]

        # Данные для boxplot
        no_stress_data = trial_data[trial_data["condition"] == "no_stress"][measure]
        stress_data = trial_data[trial_data["condition"] == "stress"][measure]

        # Создаем boxplot
        bp = ax.boxplot(
            [no_stress_data, stress_data],
            labels=[
                f"Без стресса\n(Т1-{STRESS_THRESHOLD})",
                f"Со стрессом\n(Т{STRESS_THRESHOLD + 1}-{MAX_TRIAL_NUMBER})",
            ],
            patch_artist=True,
            widths=0.6,
        )

        # Раскрашиваем
        bp["boxes"][0].set_facecolor(COLORS_NO_STRESS)
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor(COLORS_STRESS)
        bp["boxes"][1].set_alpha(0.7)

        # Заголовок
        ax.set_title(f"{result['measure_name']}", fontweight="bold", fontsize=11)

        # Показываем изменение
        change_pct = result["change_percent"]
        symbol = "↗" if change_pct > 0 else "↘"
        ax.text(
            0.5,
            0.95,
            f"{symbol}{abs(change_pct):.1f}%",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontweight="bold",
            fontsize=12,
        )

        # Значимость
        if result["significant"]:
            significance_text = f"p = {result['p_value']:.3f} ✅"
            bbox_color = "lightgreen"
        else:
            significance_text = (
                f"p = {result['p_value']:.3f}"
                if not np.isnan(result["p_value"])
                else "n.s."
            )
            bbox_color = "lightgray"

        ax.text(
            0.5,
            0.02,
            significance_text,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=bbox_color, alpha=0.7),
        )

        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "🧠 СРАВНЕНИЕ ПОКАЗАТЕЛЕЙ АЙТРЕКИНГА: БЕЗ СТРЕССА vs СО СТРЕССОМ",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(
        f"{RESULTS_DIR}/comprehensive_stress_comparison.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    show_plot_conditionally()

    # 2. ДИНАМИКА ПО ТРАЙЛАМ
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()

    measures_for_dynamics = list(dynamics.keys())[:8]

    # Используем динамическую функцию получения цветов фаз
    for i, measure in enumerate(measures_for_dynamics):
        ax = axes[i]

        values = dynamics[measure]["values"]
        phases = dynamics[measure]["phases"]

        # Создаем линейный график
        trials = list(range(1, 7))
        ax.plot(trials, values, "o-", linewidth=3, markersize=8, color="darkblue")

        # Раскрашиваем точки по фазам (используем динамическую функцию)
        for j, (trial, value, phase) in enumerate(zip(trials, values, phases)):
            ax.scatter(
                trial,
                value,
                s=150,
                c=get_phase_color(phase),
                edgecolor="black",
                linewidth=2,
                zorder=5,
            )

        # Вертикальная линия разделяющая фазы
        ax.axvline(x=3.5, color="red", linestyle="--", alpha=0.7, linewidth=2)
        ax.text(
            3.5,
            max(values) * 0.9,
            "НАЧАЛО\nСТРЕССА",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3),
        )

        # Заголовок с паттерном
        measure_name = {
            "blinks": "Моргания",
            "fixations": "Фиксации",
            "fixation_duration_mean": "Длит. фиксаций",
            "pupil_size": "Размер зрачка",
            "runs": "Забегания",
            "saccade_amplitude_mean": "Амплитуда саккад",
            "saccades": "Саккады",
            "visited_areas": "Посещ. зоны",
        }.get(measure, measure)

        ax.set_title(
            f"{measure_name}\n{dynamics[measure]['pattern']}",
            fontweight="bold",
            fontsize=10,
        )

        ax.set_xlabel("Номер трайла")
        ax.set_ylabel("Значение")
        ax.set_xticks(trials)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "📊 ДИНАМИКА ПОКАЗАТЕЛЕЙ АЙТРЕКИНГА ПО ТРАЙЛАМ", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(
        f"{RESULTS_DIR}/comprehensive_trial_dynamics.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    show_plot_conditionally()

    # 3. СПЕЦИАЛЬНЫЙ ГРАФИК: КЛЮЧЕВЫЕ МАРКЕРЫ СТРЕССА
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Самые информативные показатели
    key_measures = ["pupil_size", "blinks", "saccade_amplitude_mean"]
    key_names = [
        "Размер зрачка (стресс-маркер)",
        "Количество морганий",
        "Амплитуда саккад",
    ]

    for i, (measure, name) in enumerate(zip(key_measures, key_names)):
        ax = axes[i]

        no_stress = trial_data[trial_data["condition"] == "no_stress"][measure]
        stress = trial_data[trial_data["condition"] == "stress"][measure]

        # Violin plot для красоты
        violin = ax.violinplot([no_stress, stress], positions=[1, 2], widths=0.5)
        ax.scatter(
            [1] * len(no_stress),
            no_stress,
            alpha=0.7,
            s=100,
            color="blue",
            label="Без стресса",
        )
        ax.scatter(
            [2] * len(stress),
            stress,
            alpha=0.7,
            s=100,
            color="red",
            label="Со стрессом",
        )

        # Средние линии
        ax.hlines(
            no_stress.mean(), 0.7, 1.3, colors="blue", linestyle="--", linewidth=2
        )
        ax.hlines(stress.mean(), 1.7, 2.3, colors="red", linestyle="--", linewidth=2)

        ax.set_title(name, fontweight="bold")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Без стресса", "Со стрессом"])
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend()

    plt.suptitle(
        "🎯 КЛЮЧЕВЫЕ МАРКЕРЫ СТРЕССА В ДАННЫХ АЙТРЕКИНГА",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(
        f"{RESULTS_DIR}/key_stress_markers.png", dpi=FIGURE_DPI, bbox_inches="tight"
    )
    show_plot_conditionally()


def test_formal_hypotheses(word_comparison_results, trial_comparison_results):
    """
    Проводит формальное тестирование научных гипотез с выводами
    """
    print(f"\n🔬 ФОРМАЛЬНОЕ ТЕСТИРОВАНИЕ ГИПОТЕЗ")
    print("=" * 80)

    # Подсчет значимых результатов
    word_significant = [r for r in word_comparison_results if r["significant"]]
    trial_significant = [r for r in trial_comparison_results if r["significant"]]
    total_significant = len(word_significant) + len(trial_significant)
    total_tests = len(word_comparison_results) + len(trial_comparison_results)

    print(f"📊 РЕЗУЛЬТАТЫ СТАТИСТИЧЕСКОГО ТЕСТИРОВАНИЯ:")
    print(f"   • Общее количество проведенных тестов: {total_tests}")
    print(
        f"   • Количество значимых результатов (p < {ALPHA_LEVEL}): {total_significant}"
    )
    print(
        f"   • Процент значимых результатов: {total_significant / total_tests * 100:.1f}%"
    )

    # КРИТИЧЕСКАЯ проверка общего размера выборки
    all_sample_sizes = []
    for result in trial_comparison_results:
        if "no_stress_mean" in result:  # trial results
            all_sample_sizes.append(TOTAL_TRIALS)  # Dynamic sample size for trials
    for result in word_comparison_results:
        if "mean_no_stress" in result:  # word results
            all_sample_sizes.append(
                548
            )  # Word level sample size (could be dynamic too)

    min_trial_sample = TOTAL_TRIALS  # Dynamic from loaded data
    if min_trial_sample < MIN_SAMPLE_SIZE_WARNING:
        print(f"\n🚨 КРИТИЧЕСКОЕ МЕТОДОЛОГИЧЕСКОЕ ОГРАНИЧЕНИЕ:")
        print(f"   Размер выборки для трайлов (N={min_trial_sample}) КРИТИЧЕСКИ МАЛ!")
        print(f"   При таком размере выборки НЕВОЗМОЖНО делать надежные выводы")

    print(f"\n⚖️ ПРОВЕРКА ГИПОТЕЗ ПО УРОВНЯМ АНАЛИЗА:")

    # Анализ уровня слов
    print(f"\n   📝 УРОВЕНЬ СЛОВ (N = 548 наблюдений):")
    print(f"   • Проведено тестов: {len(word_comparison_results)}")
    print(f"   • Значимых результатов: {len(word_significant)}")

    if len(word_significant) > 0:
        print(f"   🔹 ВЫВОД: H0 ОТКЛОНЯЕТСЯ по {len(word_significant)} показателю(ям)")
        for result in word_significant:
            print(
                f"      ✅ {result['measure_name']}: p = {result['p_value']:.4f} < 0.05"
            )
    else:
        print(f"   🔸 ВЫВОД: H0 НЕ ОТКЛОНЯЕТСЯ (нет значимых различий)")

    # Анализ уровня трайлов
    print(f"\n   📊 УРОВЕНЬ ТРАЙЛОВ (N = {TOTAL_TRIALS} трайлов):")
    print(f"   • Проведено тестов: {len(trial_comparison_results)}")
    print(f"   • Значимых результатов: {len(trial_significant)}")

    if len(trial_significant) > 0:
        print(f"   🔹 ВЫВОД: H0 ОТКЛОНЯЕТСЯ по {len(trial_significant)} показателю(ям)")
        for result in trial_significant:
            print(
                f"      ✅ {result['measure_name']}: p = {result['p_value']:.4f} < {ALPHA_LEVEL}"
            )
    else:
        print(f"   🔸 ВЫВОД: H0 НЕ ОТКЛОНЯЕТСЯ (нет значимых различий)")
        print(f"   ⚠️  ПРИЧИНА: Малый размер выборки (N = {TOTAL_TRIALS})")

        # Но анализируем размеры эффектов
        large_effects = [
            r
            for r in trial_comparison_results
            if abs(r["cohens_d"]) >= EFFECT_SIZE_LARGE
        ]
        print(
            f"   📈 ОДНАКО: Обнаружено {len(large_effects)} показателей с БОЛЬШИМИ размерами эффекта"
        )

    # ОБЩЕЕ ЗАКЛЮЧЕНИЕ ПО ГИПОТЕЗАМ
    print(f"\n🏛️ ИТОГОВОЕ ЗАКЛЮЧЕНИЕ ПО ГИПОТЕЗАМ:")

    if total_significant > 0:
        print(f"   ✅ НУЛЕВАЯ ГИПОТЕЗА H0 ЧАСТИЧНО ОТКЛОНЯЕТСЯ")
        print(f"   ✅ АЛЬТЕРНАТИВНАЯ ГИПОТЕЗА H1 ЧАСТИЧНО ПОДТВЕРЖДАЕТСЯ")
        print(f"   🎯 ЗАКЛЮЧЕНИЕ: Существуют статистически значимые различия")
        print(f"      в параметрах айтрекинга между условиями стресса")
    else:
        print(f"   🔸 НУЛЕВАЯ ГИПОТЕЗА H0 НЕ ОТКЛОНЯЕТСЯ на уровне p < {ALPHA_LEVEL}")

        # КРИТИЧЕСКИ ВАЖНО: Честная интерпретация при малой выборке
        min_sample = min(TOTAL_TRIALS, 548)  # Минимальная выборка среди анализов
        if min_sample < MIN_SAMPLE_SIZE_WARNING:
            print(f"\n   🚨 КРИТИЧЕСКОЕ ОГРАНИЧЕНИЕ ИССЛЕДОВАНИЯ:")
            print(
                f"   • Размер выборки для трайлов (N={TOTAL_TRIALS}) НЕДОСТАТОЧЕН для выводов"
            )
            print(f"   • НЕВОЗМОЖНО утверждать наличие ИЛИ отсутствие эффекта")
            print(
                f"   • Cohen's d при незначимых результатах и малой выборке НЕНАДЕЖЕН"
            )
            print(f"   • Требуется увеличение выборки до N ≥ {MIN_SAMPLE_SIZE_WARNING}")

            print(f"\n   📋 НАУЧНО ОБОСНОВАННОЕ ЗАКЛЮЧЕНИЕ:")
            print(f"   • Данное исследование является ПИЛОТНЫМ")
            print(f"   • Основной результат: отработка методологии")
            print(f"   • НЕ предоставляет доказательств детекции стресса")
            print(f"   • НЕ опровергает возможность детекции стресса")
        else:
            # Подсчет больших эффектов только при адекватной выборке
            all_large_effects = [
                r
                for r in word_comparison_results + trial_comparison_results
                if abs(r["cohens_d"]) >= EFFECT_SIZE_LARGE
            ]
            print(
                f"   📊 Показателей с большим размером эффекта: {len(all_large_effects)}"
            )
            print(f"   🔬 При адекватной выборке различий не обнаружено")

    # Рекомендации по дальнейшим исследованиям
    print(f"\n📋 РЕКОМЕНДАЦИИ ДЛЯ ДАЛЬНЕЙШИХ ИССЛЕДОВАНИЙ:")

    if total_significant == 0:
        print(f"   1. Увеличить размер выборки до N ≥ 30 участников")
        print(f"   2. Использовать более контрастные стрессовые стимулы")
        print(f"   3. Провести power-анализ для определения необходимой выборки")

    print(f"   4. Применить коррекцию на множественные сравнения (Bonferroni)")
    print(f"   5. Использовать байесовский подход для малых выборок")
    print(f"   6. Комбинировать показатели в композитный индекс стресса")

    return total_significant, total_tests


def generate_comprehensive_report(
    word_comparison_results, trial_comparison_results, dynamics
):
    """
    Создает итоговый отчет о возможности детекции стресса через айтрекинг
    """
    print(f"\n🏆 ИТОГОВЫЙ ОТЧЕТ: ПИЛОТНОЕ ИССЛЕДОВАНИЕ АЙТРЕКИНГА И СТРЕССА")
    print("=" * 80)
    print(f"⚠️ ВАЖНО: Данные результаты получены на критически малой выборке")
    print(f"   и НЕ могут служить доказательством детекции стресса!")

    # Анализ результатов на уровне слов
    word_significant_results = [r for r in word_comparison_results if r["significant"]]
    word_large_effects = [
        r for r in word_comparison_results if abs(r["cohens_d"]) >= 0.8
    ]
    word_medium_effects = [
        r for r in word_comparison_results if 0.5 <= abs(r["cohens_d"]) < 0.8
    ]

    # Анализ результатов на уровне трайлов
    trial_significant_results = [
        r for r in trial_comparison_results if r["significant"]
    ]
    trial_large_effects = [
        r for r in trial_comparison_results if abs(r["cohens_d"]) >= 0.8
    ]
    trial_medium_effects = [
        r for r in trial_comparison_results if 0.5 <= abs(r["cohens_d"]) < 0.8
    ]

    print(f"📊 СТАТИСТИЧЕСКИЙ АНАЛИЗ:")
    print(f"\n   📝 УРОВЕНЬ СЛОВ (N = 548 наблюдений):")
    print(f"   • Всего проанализировано показателей: {len(word_comparison_results)}")
    print(f"   • Статистически значимых различий: {len(word_significant_results)}")
    print(f"   • Эффекты большого размера: {len(word_large_effects)}")
    print(f"   • Эффекты среднего размера: {len(word_medium_effects)}")

    print(f"\n   📊 УРОВЕНЬ ТРАЙЛОВ (N = 6 трайлов):")
    print(f"   • Всего проанализировано показателей: {len(trial_comparison_results)}")
    print(f"   • Статистически значимых различий: {len(trial_significant_results)}")
    print(f"   • Эффекты большого размера: {len(trial_large_effects)}")
    print(f"   • Эффекты среднего размера: {len(trial_medium_effects)}")

    # Объединенный анализ самых значимых результатов
    all_significant = word_significant_results + trial_significant_results
    all_large_effects = word_large_effects + trial_large_effects

    if all_significant:
        print(f"\n✅ СТАТИСТИЧЕСКИ ЗНАЧИМЫЕ РАЗЛИЧИЯ:")
        for result in all_significant:
            if "change_absolute" in result:  # Трайлы
                direction = "выше" if result["change_absolute"] > 0 else "ниже"
                print(
                    f"   🎯 {result['measure_name']}: {direction} на {abs(result['change_percent']):.1f}%"
                )
                print(
                    f"      p = {result['p_value']:.3f}, Cohen's d = {result['cohens_d']:.3f}"
                )
            else:  # Слова
                direction = result["change_direction"]
                print(
                    f"   🎯 {result['measure_name']}: {direction} на {abs(result['change_percent']):.1f}%"
                )
                print(
                    f"      p = {result['p_value']:.3f}, Cohen's d = {result['cohens_d']:.3f}"
                )

    print(f"\n📈 НАБЛЮДАЕМЫЕ ИЗМЕНЕНИЯ ПОД ВЛИЯНИЕМ СТРЕССА:")

    # Объединяем все изменения
    all_increases = []
    all_decreases = []

    for result in word_comparison_results + trial_comparison_results:
        if "change_absolute" in result:  # Трайлы
            if result["change_absolute"] > 0:
                all_increases.append(result)
            else:
                all_decreases.append(result)
        else:  # Слова
            if result["change_direction"] == "увеличение":
                all_increases.append(result)
            else:
                all_decreases.append(result)

    if all_increases:
        print(f"\n   ⬆️ УВЕЛИЧЕНИЕ:")
        for result in sorted(
            all_increases, key=lambda x: abs(x["change_percent"]), reverse=True
        )[:10]:
            significance = " ✅" if result["significant"] else ""
            print(
                f"      • {result['measure_name']}: +{abs(result['change_percent']):.1f}%{significance}"
            )

    if all_decreases:
        print(f"\n   ⬇️ УМЕНЬШЕНИЕ:")
        for result in sorted(
            all_decreases, key=lambda x: abs(x["change_percent"]), reverse=True
        )[:10]:
            significance = " ✅" if result["significant"] else ""
            print(
                f"      • {result['measure_name']}: {result['change_percent']:.1f}%{significance}"
            )

    # Анализ паттернов динамики
    print(f"\n🔄 АНАЛИЗ ДИНАМИЧЕСКИХ ПАТТЕРНОВ:")

    peak_patterns = 0
    recovery_patterns = 0

    for measure, data in dynamics.items():
        if "ПИК в Т4" in data["pattern"]:
            peak_patterns += 1
            if "ВОССТАНОВЛЕНИЕ" in data["pattern"]:
                recovery_patterns += 1

    print(f"   • Показателей с пиком в трайле 4: {peak_patterns}/{len(dynamics)}")
    print(f"   • Показателей с восстановлением: {recovery_patterns}/{len(dynamics)}")

    # ЧЕСТНЫЕ КЛЮЧЕВЫЕ ВЫВОДЫ
    print(f"\n🧠 КЛЮЧЕВЫЕ ВЫВОДЫ ПИЛОТНОГО ИССЛЕДОВАНИЯ:")

    total_significant = len(all_significant)
    total_large_effects = len(all_large_effects)

    # Проверяем размер выборки для честного вывода
    min_sample_size = TOTAL_TRIALS  # Размер выборки на уровне трайлов

    if min_sample_size < MIN_SAMPLE_SIZE_WARNING:
        print(f"   🚨 КРИТИЧЕСКОЕ ОГРАНИЧЕНИЕ:")
        print(f"   • Размер выборки (N={min_sample_size}) КРИТИЧЕСКИ МАЛ")
        print(f"   • НЕВОЗМОЖНО сделать выводы о детекции стресса")
        print(f"   • Данное исследование является ПИЛОТНЫМ")

        print(f"\n   📋 НАУЧНО ОБОСНОВАННЫЕ ВЫВОДЫ:")
        if total_significant > 0:
            print(
                f"   • Обнаружено {total_significant} значимых различий (требует подтверждения)"
            )
        else:
            print(f"   • Значимых различий не обнаружено")
            print(f"   • НО это НЕ означает отсутствие эффекта при малой выборке")

        print(f"\n   🔬 МЕТОДОЛОГИЧЕСКИЕ ПРОБЛЕМЫ:")
        print(f"   • Высокий риск ошибок I и II типа")
        print(f"   • Размеры эффектов ненадежны при p ≥ {ALPHA_LEVEL}")
        print(f"   • Отсутствие поправок на множественные сравнения")
        print(f"   • Необходимость репликации на большей выборке")

    else:
        # Этот блок выполнится только при достаточной выборке
        if total_significant > 0 or total_large_effects > 5:
            print("   ✅ НАЙДЕНЫ доказательства возможности детекции стресса")
            print("   🎯 Перспективные маркеры:")

            # Объединяем лучшие результаты
            best_results = all_significant + all_large_effects
            unique_results = {r["measure_name"]: r for r in best_results}.values()
            sorted_results = sorted(
                unique_results, key=lambda x: abs(x["cohens_d"]), reverse=True
            )

            for result in sorted_results[:5]:
                level = "слова" if "change_direction" in result else "трайлы"
                print(f"      • {result['measure_name']} (уровень: {level})")
        else:
            print("   📊 При достаточной выборке различий не обнаружено")
            print("   🎯 Возможные причины:")
            print("      • Айтрекинг не чувствителен к данному виду стресса")
            print("      • Необходимы более контрастные условия")
            print("      • Быстрая адаптация к стрессовым стимулам")

    print(f"\n🔍 НАИБОЛЕЕ ПЕРСПЕКТИВНЫЕ МАРКЕРЫ (по размеру эффекта):")

    all_effects = sorted(
        word_comparison_results + trial_comparison_results,
        key=lambda x: abs(x["cohens_d"]),
        reverse=True,
    )

    for i, result in enumerate(all_effects[:8], 1):
        effect_description = (
            "большой"
            if abs(result["cohens_d"]) >= 0.8
            else "средний"
            if abs(result["cohens_d"]) >= 0.5
            else "малый"
        )
        level = "слова" if "change_direction" in result else "трайлы"
        print(
            f"   {i}. {result['measure_name']} ({level}): {effect_description} эффект (d = {result['cohens_d']:.3f})"
        )

    # ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ
    print(f"\n💡 ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ:")

    # Проверяем размер выборки для честных рекомендаций
    if min_sample_size < MIN_SAMPLE_SIZE_WARNING:
        print("   🚨 НА ОСНОВЕ ДАННОГО ИССЛЕДОВАНИЯ:")
        print("   • НЕЛЬЗЯ использовать айтрекинг для детекции стресса")
        print("   • НЕЛЬЗЯ утверждать, что метод не работает")
        print("   • Исследование носит ПИЛОТНЫЙ характер")

        print(f"\n   📋 НЕОБХОДИМЫЕ СЛЕДУЮЩИЕ ШАГИ:")
        print(
            f"   1. КРИТИЧЕСКИ ВАЖНО: Увеличить выборку до N ≥ {MIN_SAMPLE_SIZE_WARNING}"
        )
        print(f"   2. Применить поправки на множественные сравнения")
        print(f"   3. Провести power-анализ для расчета необходимого N")
        print(f"   4. Использовать контрольные группы и рандомизацию")
        print(f"   5. Валидировать на независимой выборке")

        print(f"\n   🔬 МЕТОДОЛОГИЧЕСКИЕ УЛУЧШЕНИЯ:")
        print(f"   • Проверка предпосылок статистических тестов")
        print(f"   • Использование смешанных моделей (mixed-effects)")
        print(f"   • Байесовский анализ для малых выборок")
        print(f"   • Индивидуальные базовые линии участников")

        if total_significant == 0:
            print(f"\n   📊 ИНТЕРПРЕТАЦИЯ ОТСУТСТВИЯ ЗНАЧИМОСТИ:")
            print(f"   • НЕ означает отсутствие эффекта")
            print(f"   • НЕ означает наличие эффекта")
            print(f"   • Указывает на недостаточную мощность исследования")
            print(f"   • Требует репликации с большей выборкой")

    else:
        # Этот блок выполнится только при достаточной выборке
        if total_significant > 0 or total_large_effects > 3:
            print("   ✅ АЙТРЕКИНГ показывает потенциал для детекции стресса")
            print("   🎯 Перспективные показатели:")
            key_measures = (all_significant + all_large_effects)[:5]
            for result in key_measures:
                level = "слова" if "change_direction" in result else "трайлы"
                print(f"      • {result['measure_name']} (уровень: {level})")
            print("   📊 Необходимы дополнительные исследования:")
            print("      • Валидация на независимых данных")
            print("      • Индивидуальные базовые линии")
            print("      • Комбинирование с физиологическими маркерами")
        else:
            print("   📊 АЙТРЕКИНГ НЕ показал эффективность для данного вида стресса")
            print("   🎯 Возможные направления:")
            print("      • Тестирование других типов стрессовых стимулов")
            print("      • Комбинирование с другими методами")
            print("      • Фокус на индивидуальных различиях")

    print(f"\n🚀 ЧЕСТНОЕ РЕЗЮМЕ ПИЛОТНОГО ИССЛЕДОВАНИЯ:")
    print(
        f"   • Анализировано {len(word_comparison_results) + len(trial_comparison_results)} показателей айтрекинга"
    )
    print(
        f"   • Статистически значимых результатов: {total_significant} из {len(word_comparison_results) + len(trial_comparison_results)}"
    )
    print(
        f"   • Размер выборки на уровне трайлов: N = {min_sample_size} (критически мал)"
    )
    print(f"   • Паттернов восстановления: {recovery_patterns} из {len(dynamics)}")

    if min_sample_size < MIN_SAMPLE_SIZE_WARNING:
        print(
            f"   • 🚨 ОСНОВНОЙ ВЫВОД: Исследование ПИЛОТНОЕ, доказательств детекции стресса НЕТ"
        )
        print(f"   • 📋 СТАТУС: Методология отработана, требуется увеличение выборки")
        print(f"   • 🎯 ПОТЕНЦИАЛ: Невозможно оценить без адекватного размера выборки")
    else:
        potential = (
            "высокий"
            if total_large_effects > 5
            else "средний"
            if total_large_effects > 2
            else "низкий"
        )
        print(f"   • 🎯 ПОТЕНЦИАЛ для детекции стресса: {potential}")
        if total_significant > 0:
            print(f"   • ✅ НАЙДЕНЫ статистически значимые различия")
        else:
            print(f"   • ❌ Статистически значимых различий не обнаружено")


def print_research_hypotheses():
    """
    Печатает формальные научные гипотезы исследования
    """
    print("🔬 НАУЧНЫЕ ГИПОТЕЗЫ ИССЛЕДОВАНИЯ")
    print("=" * 80)

    print("\n📋 ФОРМУЛИРОВКА ГИПОТЕЗ:")

    print("\n🔸 НУЛЕВАЯ ГИПОТЕЗА (H0):")
    print("   Отсутствуют статистически значимые различия в параметрах айтрекинга")
    print(
        "   между условиями БЕЗ СТРЕССА (трайлы 1-{}) и СО СТРЕССОМ (трайлы {}-{})".format(
            STRESS_THRESHOLD, STRESS_THRESHOLD + 1, MAX_TRIAL_NUMBER
        )
    )
    print("   H0: μ₁ = μ₂ (средние значения равны)")

    print("\n🔹 АЛЬТЕРНАТИВНАЯ ГИПОТЕЗА (H1):")
    print("   Существуют статистически значимые различия в параметрах айтрекинга")
    print("   между условиями БЕЗ СТРЕССА и СО СТРЕССОМ")
    print("   H1: μ₁ ≠ μ₂ (средние значения различаются)")

    print("\n⚖️ КРИТЕРИИ ПРИНЯТИЯ РЕШЕНИЯ:")
    print(f"   • Уровень значимости: α = {ALPHA_LEVEL}")
    print(f"   • Если p-value < {ALPHA_LEVEL} → ОТКЛОНЯЕМ H0, ПРИНИМАЕМ H1")
    print(f"   • Если p-value ≥ {ALPHA_LEVEL} → НЕ ОТКЛОНЯЕМ H0")

    print("\n🧪 СТАТИСТИЧЕСКИЕ МЕТОДЫ:")
    print("   • Критерий Манна-Уитни (непараметрический)")
    print("   • t-критерий Стьюдента (при нормальном распределении)")
    print("   • Размер эффекта: Cohen's d")

    print("\n📊 ОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ:")
    print("   Если H1 подтверждается → айтрекинг может детектировать стресс")
    print("   Если H0 не отклоняется → необходимы дополнительные исследования")


def main():
    """
    Главная функция - комплексный анализ данных айтрекинга
    """
    # Парсим аргументы командной строки
    args = parse_arguments()

    # Устанавливаем глобальные переменные из аргументов
    global WORD_DATA_FILE, TRIAL_DATA_FILE, RESULTS_DIR, SHOW_PLOTS
    WORD_DATA_FILE = args.word_file
    TRIAL_DATA_FILE = args.trial_file
    RESULTS_DIR = args.results_dir

    # Определяем флаг показа графиков
    if args.show_plots:
        SHOW_PLOTS = True
    else:
        # По умолчанию не показывать графики
        SHOW_PLOTS = False

    print("🚀 КОМПЛЕКСНЫЙ АНАЛИЗ ДАННЫХ АЙТРЕКИНГА")
    print("🎯 Цель: Доказать возможность детекции стресса через движения глаз")
    print("📊 Анализ на двух уровнях: отдельные слова + целые трайлы")
    print(f"📁 Данные слов: {WORD_DATA_FILE}")
    print(f"📁 Данные трайлов: {TRIAL_DATA_FILE}")
    print(f"📊 Папка результатов: {RESULTS_DIR}")
    print(f"📺 Показ графиков: {'Да' if SHOW_PLOTS else 'Нет (только сохранение)'}")
    if not SHOW_STATISTICAL_WARNINGS:
        print("🔇 Режим: Warnings подавлены")
    print("=" * 80)

    # Формулировка научных гипотез
    print_research_hypotheses()

    try:
        # 0. Создание папки для результатов
        ensure_results_directory()

        # 1. Загрузка данных
        word_data, trial_data, word_measure_names = load_comprehensive_data()

        # 2. Анализ различий на уровне слов
        word_comparison_results = analyze_word_level_differences(
            word_data, word_measure_names
        )

        # 3. Анализ различий на уровне трайлов
        trial_comparison_results = analyze_trial_level_differences(trial_data)

        # 4. Анализ динамики на уровне трайлов
        dynamics, trial_stats = analyze_trial_dynamics(trial_data)

        # 5. Анализ динамики на уровне слов
        word_trial_stats, word_dynamics_results = analyze_word_dynamics(
            word_data, word_measure_names
        )

        # 6. Создание всех графиков
        # 6a. Графики сравнения условий для слов
        create_enhanced_word_visualizations(
            word_data, word_comparison_results, word_measure_names
        )

        # 6b. Графики динамики для обоих уровней
        create_dynamics_visualizations(
            trial_data,
            trial_stats,
            word_data,
            word_trial_stats,
            word_dynamics_results,
            word_measure_names,
        )

        # 6c. Комплексные графики на уровне трайлов
        create_comprehensive_visualizations(
            trial_data, trial_comparison_results, dynamics
        )

        # 7. Формальное тестирование гипотез
        total_significant, total_tests = test_formal_hypotheses(
            word_comparison_results, trial_comparison_results
        )

        # 8. Итоговый отчет
        generate_comprehensive_report(
            word_comparison_results, trial_comparison_results, dynamics
        )

        print(f"\n🎉 КОМПЛЕКСНЫЙ АНАЛИЗ ЗАВЕРШЕН!")
        print(f"📊 Создано 6 наборов графиков в папке '{RESULTS_DIR}/':")
        print(
            f"   • {RESULTS_DIR}/word_level_stress_analysis.png - сравнение условий (слова)"
        )
        print(f"   • {RESULTS_DIR}/trial_level_dynamics.png - динамика по трайлам")
        print(f"   • {RESULTS_DIR}/word_level_dynamics.png - динамика по словам")
        print(
            f"   • {RESULTS_DIR}/comprehensive_stress_comparison.png - сравнение условий"
        )
        print(
            f"   • {RESULTS_DIR}/comprehensive_trial_dynamics.png - динамика по трайлам"
        )
        print(f"   • {RESULTS_DIR}/key_stress_markers.png - ключевые маркеры стресса")

        # Результаты тестирования гипотез
        print(f"\n🔬 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ГИПОТЕЗ:")
        print(f"   • Всего проведено тестов: {total_tests}")
        print(f"   • Статистически значимых результатов: {total_significant}")

        # Проверяем размер выборки для честных выводов
        trial_sample_size = TOTAL_TRIALS  # Размер выборки на уровне трайлов
        validate_sample_size(trial_sample_size, "итоговые выводы")

        if total_significant > 0:
            print(f"   🔸 H0 ЧАСТИЧНО ОТКЛОНЯЕТСЯ → H1 частично подтверждается")
            if trial_sample_size < MIN_SAMPLE_SIZE_WARNING:
                print(f"   ⚠️ ОДНАКО: Критически малая выборка (N={trial_sample_size})")
                print(
                    f"   🚨 ЗАКЛЮЧЕНИЕ: Результаты требуют подтверждения на большей выборке"
                )
                print(
                    f"   📋 СТАТУС: ПИЛОТНОЕ исследование, НЕ доказательство детекции стресса"
                )
            else:
                print(
                    f"   ✅ ЗАКЛЮЧЕНИЕ: Найдены доказательства различий в айтрекинге при стрессе"
                )
        else:
            print(f"   🔸 H0 НЕ ОТКЛОНЯЕТСЯ на уровне p < {ALPHA_LEVEL}")

            if trial_sample_size < MIN_SAMPLE_SIZE_WARNING:
                print(
                    f"   🚨 КРИТИЧЕСКОЕ ОГРАНИЧЕНИЕ: Размер выборки N={trial_sample_size} недостаточен"
                )
                print(f"   📋 ЧЕСТНОЕ ЗАКЛЮЧЕНИЕ:")
                print(
                    f"   • НЕВОЗМОЖНО утверждать, что айтрекинг НЕ может детектировать стресс"
                )
                print(
                    f"   • НЕВОЗМОЖНО утверждать, что айтрекинг МОЖЕТ детектировать стресс"
                )
                print(f"   • Исследование является ПИЛОТНЫМ")
                print(f"   • Основной результат: отработка методологии")
                print(
                    f"   • Требуется увеличение выборки до N ≥ {MIN_SAMPLE_SIZE_WARNING}"
                )
            else:
                large_effects = len(
                    [
                        r
                        for r in word_comparison_results + trial_comparison_results
                        if abs(r["cohens_d"]) >= EFFECT_SIZE_LARGE
                    ]
                )
                if large_effects > 5:
                    print(
                        f"   📊 Обнаружено {large_effects} показателей с большим размером эффекта"
                    )
                    print(
                        f"   🔬 ЗАКЛЮЧЕНИЕ: Потенциал есть, но статистически не подтвержден"
                    )
                else:
                    print(
                        f"   📊 ЗАКЛЮЧЕНИЕ: При адекватной выборке различий не обнаружено"
                    )

    except Exception as e:
        print(f"❌ ОШИБКА: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
