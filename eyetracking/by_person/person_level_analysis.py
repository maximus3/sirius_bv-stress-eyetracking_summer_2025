import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import (
    ttest_ind,
    mannwhitneyu,
)
import pathlib
import warnings
import argparse

# =============================================================================
# КОНФИГУРАЦИЯ И КОНСТАНТЫ
# =============================================================================

# Функция для парсинга аргументов командной строки
def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="Анализ данных айтрекинга на уровне отдельных людей для детекции стресса",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-file",
        type=str,
        default="eyetracking/by_person/data/trial.xls",
        help="Путь к файлу с данными по людям",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="eyetracking/by_person/results",
        help="Путь к папке для сохранения результатов",
    )

    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Отображать созданные графики (по умолчанию только сохранять)",
    )

    parser.add_argument(
        "--exclude-participants",
        type=str,
        nargs="+",
        default=[],
        help="Список участников для исключения из анализа (например: 1807AAA 1907TRE 1607OOO)",
    )

    return parser.parse_args()

# Глобальные переменные для конфигурации (будут установлены из аргументов)
DATA_FILE = None
RESULTS_DIR = None
SHOW_PLOTS = False
EXCLUDED_PARTICIPANTS = []

# Пороги для групп стресса
STRESS_THRESHOLD = 3  # Трайлы 1-STRESS_THRESHOLD без стресса, остальные со стрессом

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

# Словарь с единицами измерения для графиков
UNITS = {
    # Базовые параметры фиксаций
    "AVERAGE_FIXATION_DURATION": "мс",
    "MEDIAN_FIXATION_DURATION": "мс",
    "SD_FIXATION_DURATION": "мс",
    "FIXATION_DURATION_MAX": "мс",
    "FIXATION_DURATION_MIN": "мс",
    
    # Базовые параметры саккад
    "AVERAGE_SACCADE_AMPLITUDE": "°",
    "MEDIAN_SACCADE_AMPLITUDE": "°",
    "SD_SACCADE_AMPLITUDE": "°",
    
    # Базовые параметры морганий
    "AVERAGE_BLINK_DURATION": "мс",
    
    # Параметры зрачка
    "PUPIL_SIZE_MAX": "у.е.",
    "PUPIL_SIZE_MEAN": "у.е.",
    "PUPIL_SIZE_MIN": "у.е.",
    
    # Нормализованные метрики (на слово)
    "DURATION_PER_WORD": "мс/слово",
    "FIXATIONS_PER_WORD": "ед/слово",
    "SACCADES_PER_WORD": "ед/слово",
    "BLINKS_PER_WORD": "ед/слово",
    
    # Нормализованные метрики (в секунду)
    "FIXATIONS_PER_SECOND": "ед/с",
    "SACCADES_PER_SECOND": "ед/с",
    "BLINKS_PER_SECOND": "ед/с",
    
    # Покрытие текста
    "TEXT_COVERAGE_PERCENT": "%",
    "REVISITED_WORDS_PERCENT": "%",
    
    # Возвратные саккады
    "REGRESSIVE_SACCADES": "ед.",
    "REGRESSIVE_SACCADES_PER_WORD": "ед/слово",
    "REGRESSIVE_SACCADES_PER_SECOND": "ед/с",
    "REGRESSIVE_SACCADES_PERCENT": "%",
}

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
    if not pathlib.Path(RESULTS_DIR).exists():
        pathlib.Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        print(f"📁 Создана папка '{RESULTS_DIR}' для сохранения графиков")
    else:
        print(f"📁 Папка '{RESULTS_DIR}' уже существует")


def show_plot_conditionally():
    """Показывает график только если установлен флаг SHOW_PLOTS"""
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()  # Закрываем фигуру для экономии памяти


def validate_sample_size(n, analysis_name="анализ"):
    """Проверяет размер выборки и выдает предупреждения"""
    if n < MIN_SAMPLE_SIZE_WARNING:
        print(f"⚠️  ПРЕДУПРЕЖДЕНИЕ: Размер выборки ({n}) для {analysis_name} меньше рекомендуемого ({MIN_SAMPLE_SIZE_WARNING})")
        print(f"   Результаты могут быть ненадежными")
        return False
    return True


def apply_bonferroni_correction(p_values, alpha=ALPHA_LEVEL):
    """Применяет поправку Bonferroni для множественных сравнений"""
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests
    significant_count = sum(1 for p in p_values if p < corrected_alpha)
    
    print(f"📊 ПОПРАВКА BONFERRONI:")
    print(f"   • Количество тестов: {n_tests}")
    print(f"   • Исходный α: {alpha}")
    print(f"   • Скорректированный α: {corrected_alpha:.4f}")
    print(f"   • Значимых результатов: {significant_count}/{n_tests}")
    
    return corrected_alpha, significant_count


def get_phase_color(phase_name):
    """Возвращает цвет для фазы"""
    if "baseline" in phase_name:
        return COLORS_NO_STRESS
    elif "stress" in phase_name:
        return COLORS_STRESS
    else:
        return "#95A5A6"  # Серый для неизвестных фаз


def interpret_effect_size_with_warnings(cohens_d, p_value, n_total, variable_name):
    """Интерпретирует размер эффекта с предупреждениями о надежности"""
    # Определяем размер эффекта
    if abs(cohens_d) < EFFECT_SIZE_SMALL:
        effect_size = "малый"
    elif abs(cohens_d) < EFFECT_SIZE_MEDIUM:
        effect_size = "средний"
    elif abs(cohens_d) < EFFECT_SIZE_LARGE:
        effect_size = "большой"
    else:
        effect_size = "очень большой"
    
    # Проверяем надежность
    is_significant = p_value < ALPHA_LEVEL
    is_reliable_sample = n_total >= MIN_SAMPLE_SIZE_WARNING
    
    interpretation = f"d = {cohens_d:.3f} ({effect_size})"
    
    if not is_significant:
        interpretation += " ⚠️ НЕЗНАЧИМО"
        if not is_reliable_sample:
            interpretation += " + НЕНАДЕЖНО (малая выборка)"
    elif not is_reliable_sample:
        interpretation += " ⚠️ НЕНАДЕЖНО (малая выборка)"
    
    return interpretation


def count_regressive_saccades(sequence_data):
    """
    Подсчитывает количество возвратных саккад в последовательности фиксаций.
    
    Возвратная саккада - это когда участник возвращается к ранее посещенной области интереса.
    Например, в последовательности [1, 2, 3, 2, 4, 5, 3] есть 2 возвратные саккады:
    - возврат к 2 (после посещения 3)
    - возврат к 3 (после посещения 5)
    
    Args:
        sequence_data: Данные последовательности (может быть строкой, списком или NaN)
        
    Returns:
        int: Количество возвратных саккад
    """
    if pd.isna(sequence_data) or sequence_data == '.' or sequence_data == '':
        return 0
    
    try:
        # Если это уже список, используем его
        if isinstance(sequence_data, list):
            sequence = sequence_data
        else:
            # Если это строка, пытаемся преобразовать в список
            # Убираем квадратные скобки и разделяем по запятым
            sequence_str = str(sequence_data).strip('[]')
            sequence = [int(x.strip()) for x in sequence_str.split(',') if x.strip().isdigit()]
        
        if len(sequence) < 2:
            return 0
        
        regressive_count = 0
        visited_positions = set()
        
        for i, current_word in enumerate(sequence):
            # Пропускаем 0 (означает, что взгляд был между словами)
            if current_word == 0:
                continue
                
            # Если текущее слово уже было посещено ранее, это возвратная саккада
            if current_word in visited_positions:
                regressive_count += 1
            else:
                # Добавляем текущее слово в множество посещенных
                visited_positions.add(current_word)
        
        return regressive_count
        
    except (ValueError, AttributeError, TypeError):
        # Если не удается обработать последовательность, возвращаем 0
        return 0


def load_person_data():
    """Загружает данные по людям"""
    print("📂 ЗАГРУЗКА ДАННЫХ ПО ЛЮДЯМ...")
    
    # Читаем данные как в существующем файле - header=0 и пропускаем вторую строку
    data = pd.read_csv(DATA_FILE, sep="\t", encoding="utf-16", header=0)
    
    # Удаляем строку с описаниями и берем только данные (как в существующем файле)
    data = data.iloc[1:].reset_index(drop=True)
    
    print(f"   • Загружено строк: {len(data)}")
    print(f"   • Количество колонок: {len(data.columns)}")
    
    # Преобразуем запятые в точки и конвертируем в числа только для числовых колонок
    numeric_columns = [
        'AVERAGE_BLINK_DURATION', 'BLINK_COUNT', 'AVERAGE_FIXATION_DURATION',
        'MEDIAN_FIXATION_DURATION', 'SD_FIXATION_DURATION', 'FIXATION_DURATION_MAX',
        'FIXATION_DURATION_MIN', 'FIXATION_COUNT', 'AVERAGE_SACCADE_AMPLITUDE',
        'MEDIAN_SACCADE_AMPLITUDE', 'SD_SACCADE_AMPLITUDE', 'SACCADE_COUNT',
        'DURATION', 'PUPIL_SIZE_MAX', 'PUPIL_SIZE_MEAN', 'PUPIL_SIZE_MIN',
        'VISITED_INTEREST_AREA_COUNT', 'IA_COUNT', 'RUN_COUNT', 'SAMPLE_COUNT'
    ]
    
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(
                data[col].astype(str).str.replace(",", "."), errors="coerce"
            )
    
    # Преобразуем INDEX в числовой
    data['INDEX'] = pd.to_numeric(data['INDEX'], errors='coerce')
    
    # Создаем нормализованные метрики (как в существующем файле)
    print("   • Создание нормализованных метрик...")
    
    # 1. Длительность на одно слово (мс/слово)
    data['DURATION_PER_WORD'] = data['DURATION'] / data['IA_COUNT']
    
    # 2. Количество фиксаций на одно слово
    data['FIXATIONS_PER_WORD'] = data['FIXATION_COUNT'] / data['IA_COUNT']
    
    # 3. Количество саккад на одно слово
    data['SACCADES_PER_WORD'] = data['SACCADE_COUNT'] / data['IA_COUNT']
    
    # 4. Количество морганий на одно слово
    data['BLINKS_PER_WORD'] = data['BLINK_COUNT'] / data['IA_COUNT']
    
    # 5. Покрытие текста (% посещенных слов от общего количества)
    data['TEXT_COVERAGE_PERCENT'] = (data['VISITED_INTEREST_AREA_COUNT'] / data['IA_COUNT']) * 100
    
    # 6. Количество повторных посещений слов
    data['REVISITED_WORDS'] = data['RUN_COUNT'] - data['VISITED_INTEREST_AREA_COUNT']
    
    # 7. Процент повторных посещений от общего количества слов
    data['REVISITED_WORDS_PERCENT'] = (data['REVISITED_WORDS'] / data['IA_COUNT']) * 100
    
    # 8. Количество фиксаций в секунду
    data['FIXATIONS_PER_SECOND'] = data['FIXATION_COUNT'] / (data['DURATION'] / 1000)
    
    # 9. Количество саккад в секунду
    data['SACCADES_PER_SECOND'] = data['SACCADE_COUNT'] / (data['DURATION'] / 1000)
    
    # 10. Количество морганий в секунду
    data['BLINKS_PER_SECOND'] = data['BLINK_COUNT'] / (data['DURATION'] / 1000)
    
    # 11. Анализ возвратных саккад
    print("   • Анализ возвратных саккад...")
    data['REGRESSIVE_SACCADES'] = data['INTEREST_AREA_FIXATION_SEQUENCE'].apply(count_regressive_saccades)
    
    # 12. Количество возвратных саккад на слово
    data['REGRESSIVE_SACCADES_PER_WORD'] = data['REGRESSIVE_SACCADES'] / data['IA_COUNT']
    
    # 13. Количество возвратных саккад в секунду
    data['REGRESSIVE_SACCADES_PER_SECOND'] = data['REGRESSIVE_SACCADES'] / (data['DURATION'] / 1000)
    
    # 14. Процент возвратных саккад от общего количества саккад
    data['REGRESSIVE_SACCADES_PERCENT'] = (data['REGRESSIVE_SACCADES'] / data['SACCADE_COUNT']) * 100
    
    # Добавляем колонку с условием (стресс/без стресса)
    data['condition'] = data['INDEX'].apply(lambda x: 'stress' if x > STRESS_THRESHOLD else 'no_stress')
    
    # Добавляем колонку с фазой
    data['phase'] = data['INDEX'].apply(lambda x: f"trial_{x}")
    
    # Исключаем указанных участников
    if EXCLUDED_PARTICIPANTS:
        initial_participants = data['RECORDING_SESSION_LABEL'].nunique()
        data = data[~data['RECORDING_SESSION_LABEL'].isin(EXCLUDED_PARTICIPANTS)]
        remaining_participants = data['RECORDING_SESSION_LABEL'].nunique()
        excluded_count = initial_participants - remaining_participants
        print(f"   • Исключено участников: {excluded_count} ({', '.join(EXCLUDED_PARTICIPANTS)})")
    
    print(f"   • Уникальных участников: {data['RECORDING_SESSION_LABEL'].nunique()}")
    print(f"   • Трайлов на участника: {data.groupby('RECORDING_SESSION_LABEL').size().iloc[0]}")
    print(f"   • Условия: {data['condition'].value_counts().to_dict()}")
    
    return data


def analyze_person_level_differences(data):
    """Анализирует различия между условиями на уровне людей"""
    print("\n🔬 АНАЛИЗ РАЗЛИЧИЙ МЕЖДУ УСЛОВИЯМИ (УРОВЕНЬ ЛЮДЕЙ)...")
    
    # Группируем данные по участникам и условиям
    person_conditions = data.groupby(['RECORDING_SESSION_LABEL', 'condition']).agg({
        # Базовые параметры фиксаций
        'AVERAGE_FIXATION_DURATION': 'mean',
        'MEDIAN_FIXATION_DURATION': 'mean',
        'SD_FIXATION_DURATION': 'mean',
        'FIXATION_DURATION_MAX': 'mean',
        'FIXATION_DURATION_MIN': 'mean',
        
        # Базовые параметры саккад
        'AVERAGE_SACCADE_AMPLITUDE': 'mean',
        'MEDIAN_SACCADE_AMPLITUDE': 'mean',
        'SD_SACCADE_AMPLITUDE': 'mean',
        
        # Базовые параметры морганий
        'AVERAGE_BLINK_DURATION': 'mean',
        
        # Параметры зрачка
        'PUPIL_SIZE_MAX': 'mean',
        'PUPIL_SIZE_MEAN': 'mean',
        'PUPIL_SIZE_MIN': 'mean',
        
        # Нормализованные метрики (на слово)
        'DURATION_PER_WORD': 'mean',
        'FIXATIONS_PER_WORD': 'mean',
        'SACCADES_PER_WORD': 'mean',
        'BLINKS_PER_WORD': 'mean',
        
        # Нормализованные метрики (в секунду)
        'FIXATIONS_PER_SECOND': 'mean',
        'SACCADES_PER_SECOND': 'mean',
        'BLINKS_PER_SECOND': 'mean',
        
        # Покрытие текста
        'TEXT_COVERAGE_PERCENT': 'mean',
        'REVISITED_WORDS_PERCENT': 'mean',
        
        # Возвратные саккады
        'REGRESSIVE_SACCADES': 'mean',
        'REGRESSIVE_SACCADES_PER_WORD': 'mean',
        'REGRESSIVE_SACCADES_PER_SECOND': 'mean',
        'REGRESSIVE_SACCADES_PERCENT': 'mean'
    }).reset_index()
    
    # Разделяем на группы
    no_stress_data = person_conditions[person_conditions['condition'] == 'no_stress']
    stress_data = person_conditions[person_conditions['condition'] == 'stress']
    
    print(f"   • Участников в группе без стресса: {len(no_stress_data)}")
    print(f"   • Участников в группе со стрессом: {len(stress_data)}")
    
    # Проверяем размер выборки
    validate_sample_size(len(no_stress_data), "группа без стресса")
    validate_sample_size(len(stress_data), "группа со стрессом")
    
    # Анализируем каждый показатель
    results = []
    measure_names = [
        # Базовые параметры фиксаций
        'AVERAGE_FIXATION_DURATION', 'MEDIAN_FIXATION_DURATION', 'SD_FIXATION_DURATION',
        'FIXATION_DURATION_MAX', 'FIXATION_DURATION_MIN',
        
        # Базовые параметры саккад
        'AVERAGE_SACCADE_AMPLITUDE', 'MEDIAN_SACCADE_AMPLITUDE', 'SD_SACCADE_AMPLITUDE',
        
        # Базовые параметры морганий
        'AVERAGE_BLINK_DURATION',
        
        # Параметры зрачка
        'PUPIL_SIZE_MAX', 'PUPIL_SIZE_MEAN', 'PUPIL_SIZE_MIN',
        
        # Нормализованные метрики (на слово)
        'DURATION_PER_WORD', 'FIXATIONS_PER_WORD', 'SACCADES_PER_WORD', 'BLINKS_PER_WORD',
        
        # Нормализованные метрики (в секунду)
        'FIXATIONS_PER_SECOND', 'SACCADES_PER_SECOND', 'BLINKS_PER_SECOND',
        
        # Покрытие текста
        'TEXT_COVERAGE_PERCENT', 'REVISITED_WORDS_PERCENT',
        
        # Возвратные саккады
        'REGRESSIVE_SACCADES', 'REGRESSIVE_SACCADES_PER_WORD', 
        'REGRESSIVE_SACCADES_PER_SECOND', 'REGRESSIVE_SACCADES_PERCENT'
    ]
    
    for measure in measure_names:
        # Удаляем NaN значения
        no_stress_clean = no_stress_data[measure].dropna()
        stress_clean = stress_data[measure].dropna()
        
        if len(no_stress_clean) == 0 or len(stress_clean) == 0:
            print(f"   ⚠️  Пропущен {measure}: недостаточно данных")
            continue
        
        # Проверяем нормальность
        try:
            from scipy.stats import shapiro
            _, p_no_stress = shapiro(no_stress_clean)
            _, p_stress = shapiro(stress_clean)
            normal_distribution = p_no_stress > 0.05 and p_stress > 0.05
        except:
            normal_distribution = False
        
        # Выбираем тест
        if normal_distribution:
            statistic, p_value = ttest_ind(no_stress_clean, stress_clean)
            test_name = "t-критерий Стьюдента"
        else:
            statistic, p_value = mannwhitneyu(no_stress_clean, stress_clean, alternative='two-sided')
            test_name = "критерий Манна-Уитни"
        
        # Рассчитываем размер эффекта
        pooled_std = np.sqrt(((len(no_stress_clean) - 1) * no_stress_clean.var() + 
                             (len(stress_clean) - 1) * stress_clean.var()) / 
                            (len(no_stress_clean) + len(stress_clean) - 2))
        cohens_d = (no_stress_clean.mean() - stress_clean.mean()) / pooled_std
        
        # Рассчитываем процентное изменение
        if no_stress_clean.mean() != 0:
            percent_change = ((stress_clean.mean() - no_stress_clean.mean()) / no_stress_clean.mean()) * 100
        else:
            percent_change = 0
        
        result = {
            'measure': measure,
            'test_name': test_name,
            'no_stress_mean': no_stress_clean.mean(),
            'no_stress_std': no_stress_clean.std(),
            'stress_mean': stress_clean.mean(),
            'stress_std': stress_clean.std(),
            'statistic': statistic,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'percent_change': percent_change,
            'n_no_stress': len(no_stress_clean),
            'n_stress': len(stress_clean)
        }
        
        results.append(result)
        
        # Выводим результат
        interpretation = interpret_effect_size_with_warnings(
            cohens_d, p_value, len(no_stress_clean) + len(stress_clean), measure
        )
        
        print(f"   • {measure}:")
        print(f"     Без стресса: {no_stress_clean.mean():.2f}±{no_stress_clean.std():.2f}")
        print(f"     Со стрессом: {stress_clean.mean():.2f}±{stress_clean.std():.2f}")
        print(f"     Изменение: {percent_change:+.1f}% | {interpretation}")
        print(f"     Тест: {test_name}, p = {p_value:.4f}")
    
    return results


def analyze_person_dynamics(data):
    """Анализирует динамику показателей по трайлам для каждого человека"""
    print("\n📈 АНАЛИЗ ДИНАМИКИ ПО ТРАЙЛАМ...")
    
    # Группируем по трайлам
    trial_stats = data.groupby('INDEX').agg({
        'AVERAGE_BLINK_DURATION': ['mean', 'std'],
        'BLINK_COUNT': ['mean', 'std'],
        'AVERAGE_FIXATION_DURATION': ['mean', 'std'],
        'MEDIAN_FIXATION_DURATION': ['mean', 'std'],
        'SD_FIXATION_DURATION': ['mean', 'std'],
        'FIXATION_DURATION_MAX': ['mean', 'std'],
        'FIXATION_DURATION_MIN': ['mean', 'std'],
        'FIXATION_COUNT': ['mean', 'std'],
        'AVERAGE_SACCADE_AMPLITUDE': ['mean', 'std'],
        'MEDIAN_SACCADE_AMPLITUDE': ['mean', 'std'],
        'SD_SACCADE_AMPLITUDE': ['mean', 'std'],
        'SACCADE_COUNT': ['mean', 'std'],
        'DURATION': ['mean', 'std'],
        'PUPIL_SIZE_MAX': ['mean', 'std'],
        'PUPIL_SIZE_MEAN': ['mean', 'std'],
        'PUPIL_SIZE_MIN': ['mean', 'std'],
        'VISITED_INTEREST_AREA_COUNT': ['mean', 'std'],
        'IA_COUNT': ['mean', 'std'],
        'RUN_COUNT': ['mean', 'std'],
        'SAMPLE_COUNT': ['mean', 'std']
    }).reset_index()
    
    # Упрощаем названия колонок
    trial_stats.columns = ['INDEX'] + [f"{col[0]}_{col[1]}" for col in trial_stats.columns[1:]]
    
    print(f"   • Анализировано трайлов: {len(trial_stats)}")
    
    return trial_stats


def create_enhanced_trial_dynamics_visualization(data, key_measures):
    """Создает улучшенный график динамики по трайлам (аналогично comprehensive_eyetracking_analysis.py)"""
    print(f"\n🎨 СОЗДАНИЕ УЛУЧШЕННОГО ГРАФИКА ДИНАМИКИ ПО ТРАЙЛАМ")
    print("=" * 50)
    
    # Создаем большую фигуру для данных трайлов
    fig, axes = plt.subplots(3, 3, figsize=(24, 18))
    axes = axes.flatten()
    
    # Цвета для фаз (аналогично comprehensive_eyetracking_analysis.py)
    phase_colors = {
        "baseline_1": "#3498DB",  # Синий
        "baseline_2": "#5DADE2",  # Светло-синий
        "baseline_3": "#85C1E9",  # Еще светлее синий
        "stress_peak": "#E74C3C",  # Красный (пик)
        "stress_adapt": "#F1948A",  # Розовый (адаптация)
        "stress_recovery": "#F8C471",  # Желтый (восстановление)
    }
    
    # Функция для определения фазы трайла
    def get_trial_phase(trial_num):
        if trial_num <= 3:
            return f"baseline_{trial_num}"
        elif trial_num == 4:
            return "stress_peak"
        elif trial_num == 5:
            return "stress_adapt"
        else:
            return "stress_recovery"
    
    # Словарь с русскими названиями мер
    measure_names_map = {
        "AVERAGE_FIXATION_DURATION": "Длительность фиксаций (средняя)",
        "MEDIAN_FIXATION_DURATION": "Длительность фиксаций (медиана)",
        "FIXATION_DURATION_MAX": "Длительность фиксаций (макс)",
        "AVERAGE_SACCADE_AMPLITUDE": "Амплитуда саккад (средняя)",
        "MEDIAN_SACCADE_AMPLITUDE": "Амплитуда саккад (медиана)",
        "AVERAGE_BLINK_DURATION": "Длительность морганий",
        "PUPIL_SIZE_MEAN": "Размер зрачка (средний)",
        "PUPIL_SIZE_MAX": "Размер зрачка (макс)",
        "DURATION_PER_WORD": "Длительность на слово",
        "FIXATIONS_PER_WORD": "Фиксации на слово",
        "SACCADES_PER_WORD": "Саккады на слово",
        "FIXATIONS_PER_SECOND": "Фиксации в секунду",
        "SACCADES_PER_SECOND": "Саккады в секунду",
        "TEXT_COVERAGE_PERCENT": "Покрытие текста (%)",
        "REVISITED_WORDS_PERCENT": "Повторные слова (%)",
        "REGRESSIVE_SACCADES": "Возвратные саккады",
        "REGRESSIVE_SACCADES_PERCENT": "Возвратные саккады (%)"
    }
    
    for i, measure in enumerate(key_measures[:9]):  # Берем первые 9 для красивого расположения
        ax = axes[i]
        
        # Группируем по трайлам
        trial_means = data.groupby('INDEX')[measure].mean()
        trial_stds = data.groupby('INDEX')[measure].std()
        
        if len(trial_means) > 0:
            trials = sorted(trial_means.index)
            means = [trial_means[t] for t in trials]
            stds = [trial_stds[t] for t in trials]
            phases = [get_trial_phase(t) for t in trials]
            colors = [phase_colors.get(phase, "#95A5A6") for phase in phases]
            
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
            measure_name = measure_names_map.get(measure, measure)
            ax.set_title(
                f"{measure_name}",
                fontweight="bold",
                fontsize=13,
            )
            
            # Подписи процентных изменений на барах
            baseline_mean = np.mean([trial_means[t] for t in [1, 2, 3]])
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
            unit = UNITS.get(measure, "")
            ax.set_ylabel(f"Значение, {unit}" if unit else "Значение")
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
    for i in range(9, len(axes)):
        fig.delaxes(axes[i])
    
    # Общий заголовок
    fig.suptitle(
        "📊 ДИНАМИКА СТРЕССА ПО ТРАЙЛАМ (УРОВЕНЬ ЛЮДЕЙ)",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(
        f"{RESULTS_DIR}/person_level_dynamics.png", dpi=FIGURE_DPI, bbox_inches="tight"
    )
    show_plot_conditionally()


def create_key_dynamics_visualization(data):
    """
    Создает 3 отдельных графика с динамикой ключевых показателей по трайлам.
    """
    print(f"\n🎨 СОЗДАНИЕ ГРАФИКОВ ДИНАМИКИ КЛЮЧЕВЫХ ПОКАЗАТЕЛЕЙ")
    print("=" * 50)
    
    # Группы показателей
    measure_groups = [
        # График 1: Частота и длительность фиксаций
        {
            "measures": ["FIXATIONS_PER_SECOND", "AVERAGE_FIXATION_DURATION"],
            "title": "Фиксации: частота и длительность",
            "colors": ["#2E86C1", "#E74C3C"],
            "names": ["Частота фиксаций (ед/с)", "Длительность фиксаций (мс)"],
            "filename": "key_trial_dynamics_fixations.png"
        },
        # График 2: Частота и амплитуда саккад
        {
            "measures": ["SACCADES_PER_SECOND", "AVERAGE_SACCADE_AMPLITUDE"], 
            "title": "Саккады: частота и амплитуда",
            "colors": ["#28B463", "#F39C12"],
            "names": ["Частота саккад (ед/с)", "Амплитуда саккад (°)"],
            "filename": "key_trial_dynamics_saccades.png"
        },
        # График 3: Возвратные саккады, длительность на слово, размер зрачка
        {
            "measures": ["REGRESSIVE_SACCADES", "DURATION_PER_WORD", "PUPIL_SIZE_MEAN"],
            "title": "Комплексные показатели",
            "colors": ["#8E44AD", "#D35400", "#17A2B8"],
            "names": ["Возвратные саккады (ед)", "Длительность на слово (мс)", "Размер зрачка (у.е.)"],
            "filename": "key_trial_dynamics_complex.png"
        }
    ]
    
    # Цвета для фаз
    phase_colors = {
        "baseline_1": "#3498DB",  # Синий
        "baseline_2": "#5DADE2",  # Светло-синий
        "baseline_3": "#85C1E9",  # Еще светлее синий
        "stress_peak": "#E74C3C",  # Красный (пик)
        "stress_adapt": "#F1948A",  # Розовый (адаптация)
        "stress_recovery": "#F8C471",  # Желтый (восстановление)
    }
    
    # Функция для определения фазы трайла
    def get_trial_phase(trial_num):
        if trial_num <= 3:
            return f"baseline_{trial_num}"
        elif trial_num == 4:
            return "stress_peak"
        elif trial_num == 5:
            return "stress_adapt"
        else:
            return "stress_recovery"
    
    # Создаем отдельный график для каждой группы показателей
    for group_idx, group_info in enumerate(measure_groups):
        print(f"   📊 Создание графика {group_idx + 1}/3: {group_info['title']}")
        
        # Определяем количество подграфиков
        n_measures = len(group_info["measures"])
        if n_measures == 2:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        else:  # n_measures == 3
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Если только один подграфик, делаем axes списком
        if n_measures == 1:
            axes = [axes]
        
        # Для каждого показателя создаем отдельный подграфик
        for measure_idx, measure in enumerate(group_info["measures"]):
            ax = axes[measure_idx]
            
            # Группируем по трайлам
            trial_means = data.groupby('INDEX')[measure].mean()
            
            if len(trial_means) > 0:
                trials = sorted(trial_means.index)
                values = [trial_means[t] for t in trials]
                phases = [get_trial_phase(t) for t in trials]
                
                # Создаем линейный график
                line_color = group_info["colors"][measure_idx]
                ax.plot(trials, values, "o-", linewidth=3, markersize=8, 
                       color=line_color, label=group_info["names"][measure_idx])
                
                # Раскрашиваем точки по фазам
                for j, (trial, value, phase) in enumerate(zip(trials, values, phases)):
                    ax.scatter(
                        trial,
                        value,
                        s=150,
                        c=phase_colors.get(phase, "#95A5A6"),
                        edgecolor=line_color,
                        linewidth=3,
                        zorder=5,
                    )
                
                # Подписи процентных изменений под точками (относительно трайла 3)
                trial3_mean = trial_means[3] if 3 in trial_means else np.mean([trial_means[t] for t in [1, 2, 3]])
                values_range = max(values) - min(values)
                for j, (trial, mean_val) in enumerate(zip(trials, values)):
                    if trial >= 4:  # Только для стрессовых трайлов (4-6)
                        change_pct = ((mean_val - trial3_mean) / trial3_mean) * 100
                        symbol = "↗" if change_pct > 0 else "↘"
                        
                        ax.text(
                            trial,
                            mean_val - values_range * 0.08,
                            f"{symbol}{abs(change_pct):.1f}%",
                            ha="center",
                            va="top",
                            fontweight="bold",
                            fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                        )
                
                # Настройки подграфика
                ax.set_xlabel("Номер трайла", fontsize=11)
                unit = UNITS.get(measure, "")
                ax.set_ylabel(f"Значение, {unit}" if unit else "Значение", fontsize=11)
                ax.set_title(group_info["names"][measure_idx], fontweight="bold", fontsize=13)
                ax.set_xticks(trials)
                ax.grid(True, alpha=0.3)
                
                # Вертикальная линия разделяющая фазы
                ax.axvline(
                    x=STRESS_THRESHOLD + 0.5, color="red", linestyle="--", alpha=0.7, linewidth=2
                )
        
        # Общий заголовок
        fig.suptitle(group_info["title"], fontweight="bold", fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        # Сохраняем каждый график в отдельный файл
        filename = f"{RESULTS_DIR}/{group_info['filename']}"
        plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"   ✅ Сохранен: {filename}")
        
        show_plot_conditionally()
    
    print(f"📁 Создано 3 отдельных файла с графиками динамики")


def create_clean_stress_dynamics_visualization(data):
    """
    Создает график динамики с исключением адаптационных трайлов (1 и 6).
    Анализирует только трайлы 2-5 для более чистого сравнения стрессового эффекта.
    """
    print(f"\n🎨 СОЗДАНИЕ ГРАФИКА ДИНАМИКИ БЕЗ АДАПТАЦИОННЫХ ТРАЙЛОВ")
    print("=" * 50)
    
    # Фильтруем данные: исключаем трайлы 1 и 6
    clean_data = data[data['INDEX'].isin([2, 3, 4, 5])].copy()
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    axes = axes.flatten()
    
    # Ключевые показатели для анализа
    measures_for_clean_dynamics = [
        "FIXATIONS_PER_SECOND",
        "AVERAGE_FIXATION_DURATION", 
        "PUPIL_SIZE_MEAN",
        "SACCADES_PER_SECOND",
        "DURATION_PER_WORD",
        "REGRESSIVE_SACCADES",
    ]
    
    measure_names_map = {
        "FIXATIONS_PER_SECOND": "Частота фиксаций (ед/с)",
        "AVERAGE_FIXATION_DURATION": "Длительность фиксаций (мс)",
        "PUPIL_SIZE_MEAN": "Размер зрачка (у.е.)",
        "SACCADES_PER_SECOND": "Частота саккад (ед/с)",
        "DURATION_PER_WORD": "Длительность на слово (мс)",
        "REGRESSIVE_SACCADES": "Возвратные саккады (ед)",
    }
    
    # Цвета для фаз (только базовые и стрессовые)
    phase_colors = {
        "baseline_2": "#3498DB",  # Синий
        "baseline_3": "#5DADE2",  # Светло-синий
        "stress_peak": "#E74C3C",  # Красный (пик)
        "stress_adapt": "#F1948A",  # Розовый (адаптация)
    }
    
    # Функция для определения фазы трайла
    def get_trial_phase(trial_num):
        if trial_num == 2:
            return "baseline_2"
        elif trial_num == 3:
            return "baseline_3"
        elif trial_num == 4:
            return "stress_peak"
        else:  # trial_num == 5
            return "stress_adapt"
    
    for i, measure in enumerate(measures_for_clean_dynamics):
        ax = axes[i]
        
        # Группируем по трайлам
        trial_means = clean_data.groupby('INDEX')[measure].mean()
        trial_stds = clean_data.groupby('INDEX')[measure].std()
        
        if len(trial_means) > 0:
            trials = sorted(trial_means.index)
            means = [trial_means[t] for t in trials]
            stds = [trial_stds[t] for t in trials]
            phases = [get_trial_phase(t) for t in trials]
            colors = [phase_colors.get(phase, "#95A5A6") for phase in phases]
            
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
            measure_name = measure_names_map.get(measure, measure)
            ax.set_title(
                f"{measure_name}\n(Без трайлов 1 и 6)",
                fontweight="bold",
                fontsize=12,
            )
            
            # Подписи процентных изменений на барах
            baseline_mean = np.mean([trial_means[t] for t in [2, 3]])
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
            unit = UNITS.get(measure, "")
            ax.set_ylabel(f"Значение, {unit}" if unit else "Значение")
            ax.set_xticks(trials)
            ax.grid(True, alpha=0.3)
            
            # Добавляем легенду фаз (только на первом графике)
            if i == 0:
                legend_elements = []
                legend_labels = {
                    "baseline_2": "Базовая линия (Т2)",
                    "baseline_3": "Базовая линия (Т3)",
                    "stress_peak": "Пик стресса (Т4)",
                    "stress_adapt": "Адаптация (Т5)",
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
    
    # Общий заголовок
    fig.suptitle(
        "📊 ДИНАМИКА СТРЕССА БЕЗ АДАПТАЦИОННЫХ ТРАЙЛОВ (Т2-Т5)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(
        f"{RESULTS_DIR}/clean_stress_dynamics.png", dpi=FIGURE_DPI, bbox_inches="tight"
    )
    show_plot_conditionally()
    
    # Дополнительно создаем статистический анализ для чистых данных
    print(f"\n📊 СТАТИСТИЧЕСКИЙ АНАЛИЗ БЕЗ АДАПТАЦИОННЫХ ТРАЙЛОВ:")
    print("=" * 50)
    
    # Сравниваем базовые трайлы (2-3) со стрессовыми (4-5)
    baseline_data = clean_data[clean_data['INDEX'].isin([2, 3])]
    stress_data = clean_data[clean_data['INDEX'].isin([4, 5])]
    
    print(f"   • Базовые трайлы (2-3): {len(baseline_data)} наблюдений")
    print(f"   • Стрессовые трайлы (4-5): {len(stress_data)} наблюдений")
    
    for measure in measures_for_clean_dynamics:
        baseline_values = baseline_data[measure].dropna()
        stress_values = stress_data[measure].dropna()
        
        if len(baseline_values) > 0 and len(stress_values) > 0:
            # t-тест
            from scipy.stats import ttest_ind
            t_stat, p_value = ttest_ind(baseline_values, stress_values)
            
            # Cohen's d
            pooled_std = np.sqrt(((len(baseline_values) - 1) * baseline_values.var() + 
                                 (len(stress_values) - 1) * stress_values.var()) / 
                                (len(baseline_values) + len(stress_values) - 2))
            cohens_d = (stress_values.mean() - baseline_values.mean()) / pooled_std
            
            # Процентное изменение
            change_pct = ((stress_values.mean() - baseline_values.mean()) / baseline_values.mean()) * 100
            
            print(f"\n   📊 {measure_names_map.get(measure, measure)}:")
            print(f"      Базовые: {baseline_values.mean():.2f}±{baseline_values.std():.2f}")
            print(f"      Стрессовые: {stress_values.mean():.2f}±{stress_values.std():.2f}")
            print(f"      Изменение: {change_pct:+.1f}% | d = {cohens_d:.3f} | p = {p_value:.4f}")


def create_person_visualizations(data, comparison_results, trial_stats):
    """Создает визуализации для анализа на уровне людей"""
    print("\n🎨 СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ...")
    
    # 1. Сравнение условий (boxplot)
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    
    key_measures = [
        # Базовые параметры фиксаций
        'AVERAGE_FIXATION_DURATION', 'MEDIAN_FIXATION_DURATION', 'FIXATION_DURATION_MAX',
        
        # Базовые параметры саккад
        'AVERAGE_SACCADE_AMPLITUDE', 'MEDIAN_SACCADE_AMPLITUDE',
        
        # Базовые параметры морганий
        'AVERAGE_BLINK_DURATION',
        
        # Параметры зрачка
        'PUPIL_SIZE_MEAN', 'PUPIL_SIZE_MAX',
        
        # Нормализованные метрики (на слово)
        'DURATION_PER_WORD', 'FIXATIONS_PER_WORD', 'SACCADES_PER_WORD',
        
        # Нормализованные метрики (в секунду)
        'FIXATIONS_PER_SECOND', 'SACCADES_PER_SECOND',
        
        # Покрытие текста
        'TEXT_COVERAGE_PERCENT', 'REVISITED_WORDS_PERCENT',
        
        # Возвратные саккады
        'REGRESSIVE_SACCADES', 'REGRESSIVE_SACCADES_PERCENT'
    ]
    
    for i, measure in enumerate(key_measures):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Подготавливаем данные
        no_stress_data = data[data['condition'] == 'no_stress'][measure].dropna()
        stress_data = data[data['condition'] == 'stress'][measure].dropna()
        
        if len(no_stress_data) > 0 and len(stress_data) > 0:
            # Создаем boxplot
            bp = ax.boxplot([no_stress_data, stress_data], 
                          labels=['Без стресса', 'Со стрессом'],
                          patch_artist=True)
            
            # Раскрашиваем
            bp['boxes'][0].set_facecolor(COLORS_NO_STRESS)
            bp['boxes'][1].set_facecolor(COLORS_STRESS)
            
            ax.set_title(f"{measure}")
            if measure in UNITS:
                ax.set_ylabel(f"Значение ({UNITS[measure]})")
            
            # Добавляем статистику
            result = next((r for r in comparison_results if r['measure'] == measure), None)
            if result:
                p_text = f"p = {result['p_value']:.3f}"
                d_text = f"d = {result['cohens_d']:.2f}"
                ax.text(0.02, 0.98, f"{p_text}\n{d_text}", 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Удаляем лишние оси
    for i in range(len(key_measures), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/person_level_comparison.png", dpi=FIGURE_DPI, bbox_inches='tight')
    show_plot_conditionally()
    
    # 2. Улучшенная динамика по трайлам (аналогично comprehensive_eyetracking_analysis.py)
    create_enhanced_trial_dynamics_visualization(data, key_measures)
    
    # 3. График ключевых показателей динамики
    create_key_dynamics_visualization(data)
    
    # 4. График с исключением адаптационных трайлов (1 и 6)
    create_clean_stress_dynamics_visualization(data)
    
    # 3. Индивидуальные траектории
    fig, axes = plt.subplots(5, 4, figsize=(20, 20))
    axes = axes.flatten()
    
    # Получаем уникальных участников
    participants = data['RECORDING_SESSION_LABEL'].unique()
    
    # Создаем цветовую палитру для участников
    import matplotlib.cm as cm
    colors = cm.tab20(np.linspace(0, 1, len(participants)))
    
    for i, measure in enumerate(key_measures):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        for j, participant in enumerate(participants):
            participant_data = data[data['RECORDING_SESSION_LABEL'] == participant]
            participant_data = participant_data.sort_values('INDEX')
            
            if len(participant_data) > 0:
                trials = participant_data['INDEX'].values
                values = participant_data[measure].values
                
                # Используем уникальный цвет для каждого участника
                participant_color = colors[j]
                
                # Рисуем полную траекторию участника
                ax.plot(trials, values, 'o-', 
                       color=participant_color, alpha=0.7, linewidth=1.5, markersize=4,
                       label=f'Участник {participant}')
        
        ax.set_title(f"{measure}")
        ax.set_xlabel("Номер трайла")
        if measure in UNITS:
            ax.set_ylabel(f"Значение ({UNITS[measure]})")
        
        # Добавляем разделительную линию
        ax.axvline(x=STRESS_THRESHOLD + 0.5, color='black', linestyle='--', alpha=0.5)
        
        # Добавляем аннотацию о разделении условий
        ax.text(0.02, 0.98, f'Трайлы 1-{STRESS_THRESHOLD}: базовые\nТрайлы {STRESS_THRESHOLD+1}-6: стресс', 
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Добавляем компактную легенду со всеми участниками
        ax.legend(loc='upper right', fontsize=6, ncol=2, frameon=True, 
                 title='Участники', title_fontsize=7)
    
    # Удаляем лишние оси
    for i in range(len(key_measures), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/person_individual_trajectories.png", dpi=FIGURE_DPI, bbox_inches='tight')
    show_plot_conditionally()


def test_formal_hypotheses(comparison_results):
    """Тестирует формальные научные гипотезы"""
    print("\n🔬 ТЕСТИРОВАНИЕ ФОРМАЛЬНЫХ ГИПОТЕЗ...")
    
    # Извлекаем p-значения
    p_values = [r['p_value'] for r in comparison_results]
    
    # Применяем поправку Bonferroni
    corrected_alpha, significant_count = apply_bonferroni_correction(p_values)
    
    # Подсчитываем статистику
    total_tests = len(comparison_results)
    significant_tests = sum(1 for r in comparison_results if r['p_value'] < ALPHA_LEVEL)
    significant_bonferroni = sum(1 for r in comparison_results if r['p_value'] < corrected_alpha)
    
    print(f"\n📊 ИТОГОВАЯ СТАТИСТИКА:")
    print(f"   • Всего проведено тестов: {total_tests}")
    print(f"   • Значимых результатов (p < {ALPHA_LEVEL}): {significant_tests}")
    print(f"   • Значимых после поправки Bonferroni: {significant_bonferroni}")
    print(f"   • Процент значимых результатов: {(significant_tests/total_tests)*100:.1f}%")
    
    # Формулируем гипотезы
    print(f"\n🔸 НУЛЕВАЯ ГИПОТЕЗА (H0):")
    print(f"   Отсутствуют статистически значимые различия в параметрах айтрекинга")
    print(f"   между условиями без стресса и со стрессом на уровне отдельных людей")
    
    print(f"\n🔹 АЛЬТЕРНАТИВНАЯ ГИПОТЕЗА (H1):")
    print(f"   Существуют статистически значимые различия в параметрах айтрекинга")
    print(f"   между условиями без стресса и со стрессом на уровне отдельных людей")
    
    # Результат тестирования
    if significant_bonferroni > 0:
        print(f"\n✅ РЕЗУЛЬТАТ: H0 ОТКЛОНЯЕТСЯ")
        print(f"   Обнаружены статистически значимые различия после поправки на множественные сравнения")
    else:
        print(f"\n❌ РЕЗУЛЬТАТ: H0 НЕ ОТКЛОНЯЕТСЯ")
        print(f"   Статистически значимых различий не обнаружено")
    
    return {
        'total_tests': total_tests,
        'significant_tests': significant_tests,
        'significant_bonferroni': significant_bonferroni,
        'percent_significant': (significant_tests/total_tests)*100
    }


def generate_comprehensive_report(comparison_results, hypothesis_results):
    """Генерирует комплексный отчет"""
    print("\n📋 ГЕНЕРАЦИЯ КОМПЛЕКСНОГО ОТЧЕТА...")
    
    # Сортируем результаты по размеру эффекта
    sorted_results = sorted(comparison_results, key=lambda x: abs(x['cohens_d']), reverse=True)
    
    print(f"\n🏆 ТОП-10 ПОКАЗАТЕЛЕЙ ПО РАЗМЕРУ ЭФФЕКТА:")
    for i, result in enumerate(sorted_results[:10], 1):
        interpretation = interpret_effect_size_with_warnings(
            result['cohens_d'], result['p_value'], 
            result['n_no_stress'] + result['n_stress'], 
            result['measure']
        )
        print(f"   {i:2d}. {result['measure']}: {interpretation}")
        print(f"       Изменение: {result['percent_change']:+.1f}% | p = {result['p_value']:.4f}")
    
    # Статистика по размерам эффектов
    large_effects = sum(1 for r in comparison_results if abs(r['cohens_d']) >= EFFECT_SIZE_LARGE)
    medium_effects = sum(1 for r in comparison_results if EFFECT_SIZE_MEDIUM <= abs(r['cohens_d']) < EFFECT_SIZE_LARGE)
    small_effects = sum(1 for r in comparison_results if abs(r['cohens_d']) < EFFECT_SIZE_MEDIUM)
    
    print(f"\n📊 РАСПРЕДЕЛЕНИЕ РАЗМЕРОВ ЭФФЕКТОВ:")
    print(f"   • Большие эффекты (|d| ≥ {EFFECT_SIZE_LARGE}): {large_effects}")
    print(f"   • Средние эффекты ({EFFECT_SIZE_MEDIUM} ≤ |d| < {EFFECT_SIZE_LARGE}): {medium_effects}")
    print(f"   • Малые эффекты (|d| < {EFFECT_SIZE_MEDIUM}): {small_effects}")
    
    # Рекомендации
    print(f"\n💡 РЕКОМЕНДАЦИИ:")
    if hypothesis_results['significant_bonferroni'] > 0:
        print(f"   ✅ Обнаружены статистически значимые различия")
        print(f"   ✅ Айтрекинг может быть эффективен для детекции стресса")
    else:
        print(f"   ⚠️  Статистически значимых различий не обнаружено")
        print(f"   ⚠️  Требуется увеличение размера выборки или улучшение методологии")
    
    print(f"   📈 Рекомендуется фокус на показателях с большими размерами эффектов")
    print(f"   🔬 Необходима валидация на независимой выборке")


def print_research_hypotheses():
    """Выводит научные гипотезы исследования"""
    print("\n" + "="*80)
    print("🔬 НАУЧНЫЕ ГИПОТЕЗЫ ИССЛЕДОВАНИЯ")
    print("="*80)
    
    print("\n🔸 НУЛЕВАЯ ГИПОТЕЗА (H0):")
    print("   Отсутствуют статистически значимые различия в параметрах айтрекинга")
    print("   между условиями без стресса и со стрессом на уровне отдельных людей")
    print("   Формально: H0: μ₁ = μ₂")
    
    print("\n🔹 АЛЬТЕРНАТИВНАЯ ГИПОТЕЗА (H1):")
    print("   Существуют статистически значимые различия в параметрах айтрекинга")
    print("   между условиями без стресса и со стрессом на уровне отдельных людей")
    print("   Формально: H1: μ₁ ≠ μ₂")
    
    print("\n📊 КРИТЕРИИ ПРИНЯТИЯ РЕШЕНИЯ:")
    print(f"   • Уровень значимости: α = {ALPHA_LEVEL}")
    print(f"   • Критерий отклонения H0: p-value < {ALPHA_LEVEL}")
    print(f"   • Критерий практической значимости: |Cohen's d| ≥ {EFFECT_SIZE_LARGE}")
    
    print("\n🎯 ОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ:")
    print("   При подтверждении H1 ожидается:")
    print("   • Уменьшение длительности фиксаций (поверхностная обработка)")
    print("   • Увеличение количества движений глаз (потеря концентрации)")
    print("   • Расширение зрачков (активация симпатической нервной системы)")
    print("   • Замедление выполнения задач (когнитивный дефицит)")
    
    print("="*80)


def main():
    """Основная функция"""
    global DATA_FILE, RESULTS_DIR, SHOW_PLOTS, EXCLUDED_PARTICIPANTS
    
    # Парсим аргументы
    args = parse_arguments()
    DATA_FILE = args.data_file
    RESULTS_DIR = args.results_dir
    SHOW_PLOTS = args.show_plots
    EXCLUDED_PARTICIPANTS = args.exclude_participants
    
    print("👁️  АНАЛИЗ ДАННЫХ АЙТРЕКИНГА НА УРОВНЕ ЛЮДЕЙ")
    print("="*60)
    
    # Выводим гипотезы
    print_research_hypotheses()
    
    # Создаем папку для результатов
    ensure_results_directory()
    
    # Загружаем данные
    data = load_person_data()
    
    # Анализируем различия между условиями
    comparison_results = analyze_person_level_differences(data)
    
    # Анализируем динамику
    trial_stats = analyze_person_dynamics(data)
    
    # Создаем визуализации
    create_person_visualizations(data, comparison_results, trial_stats)
    
    # Тестируем гипотезы
    hypothesis_results = test_formal_hypotheses(comparison_results)
    
    # Генерируем отчет
    generate_comprehensive_report(comparison_results, hypothesis_results)
    
    print(f"\n✅ АНАЛИЗ ЗАВЕРШЕН!")
    print(f"📁 Результаты сохранены в папке: {RESULTS_DIR}")
    print(f"📊 Создано графиков: 3")
    print(f"🔬 Проведено статистических тестов: {len(comparison_results)}")


if __name__ == "__main__":
    main() 