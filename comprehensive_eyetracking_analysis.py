import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon, shapiro, levene, kruskal, spearmanr
import os
import warnings
warnings.filterwarnings('ignore')

# Настройка для красивых графиков
plt.style.use('seaborn-v0_8')
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def ensure_results_directory():
    """Создает папку results если она не существует"""
    if not os.path.exists('results'):
        os.makedirs('results')
        print("📁 Создана папка 'results' для сохранения графиков")
    else:
        print("📁 Папка 'results' уже существует")

def load_comprehensive_data():
    """
    Загружает и подготавливает данные айтрекинга из двух источников:
    1. ia_avg.xls - данные на уровне отдельных слов
    2. events.xls - агрегированные данные на уровне трайлов
    """
    print("🔄 КОМПЛЕКСНАЯ ЗАГРУЗКА ДАННЫХ АЙТРЕКИНГА")
    print("=" * 70)
    
    # 1. ДАННЫЕ НА УРОВНЕ СЛОВ (ia_avg.xls)
    print("📖 Загрузка данных на уровне слов...")
    word_data = pd.read_csv('ia_avg.xls', sep='\t', encoding='utf-16', skiprows=1)
    col_names = list(word_data.columns)
    
    # Создаем группировку по условиям для данных слов
    trial_col = col_names[0]
    word_data['trial'] = word_data[trial_col].astype(int)
    word_data['condition'] = word_data['trial'].apply(lambda x: 'no_stress' if x <= 3 else 'stress')
    
    # Создаем детальную группировку по фазам для данных слов
    word_data['stress_phase'] = word_data['trial'].map({
        1: 'baseline_1', 2: 'baseline_2', 3: 'baseline_3',
        4: 'stress_peak', 5: 'stress_adapt', 6: 'stress_recovery'
    })
    
    # Русские названия для показателей слов
    word_measure_names = {
        'IA_FIRST_FIXATION_DURATION': 'Длительность первой фиксации (мс)',
        'IA_FIXATION_COUNT': 'Количество фиксаций на слово',
        'IA_DWELL_TIME': 'Общее время пребывания (мс)',
        'IA_DWELL_TIME_%': 'Процент времени пребывания (%)',
        'IA_VISITED_TRIAL_%': 'Процент визитов к слову (%)',
        'IA_REVISIT_TRIAL_%': 'Процент повторных визитов (%)',
        'IA_RUN_COUNT': 'Количество забеганий взгляда'
    }
    
    # Находим числовые колонки для слов
    numeric_cols_positions = [3, 4, 7, 8, 9, 10, 11]
    word_numeric_cols = [col_names[i] for i in numeric_cols_positions if i < len(col_names)]
    word_short_names = [
        'IA_FIRST_FIXATION_DURATION', 'IA_FIXATION_COUNT', 'IA_DWELL_TIME_%', 
        'IA_DWELL_TIME', 'IA_VISITED_TRIAL_%', 'IA_REVISIT_TRIAL_%', 'IA_RUN_COUNT'
    ]
    
    # Преобразование данных слов
    for col in word_numeric_cols:
        if col in word_data.columns:
            word_data[col] = pd.to_numeric(word_data[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    # Создаем короткие названия для слов
    for i, col in enumerate(word_numeric_cols):
        if col in word_data.columns and i < len(word_short_names):
            word_data[word_short_names[i]] = word_data[col]
    
    # Фильтрация валидных данных слов
    word_col = col_names[6]
    word_data = word_data.dropna(subset=[word_col])
    word_data = word_data[word_data[word_col] != '.']
    word_data = word_data[word_data[word_col] != '–']
    word_data = word_data[word_data[word_col] != '—']
    
    print(f"✅ Загружено {len(word_data)} наблюдений на уровне слов")
    
    # 2. ДАННЫЕ НА УРОВНЕ ТРАЙЛОВ (events.xls)
    print("📊 Загрузка данных на уровне трайлов...")
    trial_data = pd.read_csv('events.xls', sep='\t', encoding='utf-16', header=0)
    
    # Удаляем строку с описаниями и берем только данные
    trial_data = trial_data.iloc[1:].reset_index(drop=True)
    
    # Преобразуем запятые в точки и конвертируем в числа
    for col in trial_data.columns:
        trial_data[col] = pd.to_numeric(trial_data[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    # Добавляем информацию о трайлах
    trial_data['trial'] = range(1, len(trial_data) + 1)
    trial_data['condition'] = trial_data['trial'].apply(lambda x: 'no_stress' if x <= 3 else 'stress')
    trial_data['phase'] = trial_data['trial'].map({
        1: 'baseline_1', 2: 'baseline_2', 3: 'baseline_3',
        4: 'stress_peak', 5: 'stress_adapt', 6: 'stress_recovery'
    })
    
    # Переименовываем колонки для удобства
    trial_data.columns = [
        'blinks', 'fixations', 'fixation_duration_mean', 'fixation_duration_median',
        'fixation_duration_sd', 'pupil_size', 'runs', 'saccade_amplitude_mean',
        'saccade_amplitude_median', 'saccade_amplitude_sd', 'saccades',
        'samples', 'trial_duration', 'interest_areas', 'visited_areas',
        'trial', 'condition', 'phase'
    ]
    
    print(f"✅ Загружено {len(trial_data)} трайлов")
    print(f"   • Без стресса (трайлы 1-3): {len(trial_data[trial_data['condition'] == 'no_stress'])} трайла")
    print(f"   • Со стрессом (трайлы 4-6): {len(trial_data[trial_data['condition'] == 'stress'])} трайла")
    
    return word_data, trial_data, word_measure_names

def analyze_word_level_differences(word_data, word_measure_names):
    """
    Анализирует различия на уровне отдельных слов между условиями без стресса и со стрессом
    """
    print(f"\n📝 АНАЛИЗ РАЗЛИЧИЙ НА УРОВНЕ СЛОВ")
    print("=" * 70)
    
    measures = ['IA_FIRST_FIXATION_DURATION', 'IA_FIXATION_COUNT', 'IA_DWELL_TIME',
                'IA_DWELL_TIME_%', 'IA_VISITED_TRIAL_%', 'IA_REVISIT_TRIAL_%', 'IA_RUN_COUNT']
    
    results = []
    
    # Данные по условиям
    no_stress = word_data[word_data['condition'] == 'no_stress']
    stress = word_data[word_data['condition'] == 'stress']
    
    print(f"📊 СРАВНЕНИЕ ГРУПП НА УРОВНЕ СЛОВ:")
    print(f"   Без стресса: {len(no_stress)} наблюдений")  
    print(f"   Со стрессом: {len(stress)} наблюдений")
    
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
            stat, p_value = mannwhitneyu(no_stress_vals, stress_vals, alternative='two-sided')
            test_name = "Критерий Манна-Уитни"
        except:
            try:
                stat, p_value = ttest_ind(no_stress_vals, stress_vals)
                test_name = "t-критерий"
            except:
                p_value = np.nan
                test_name = "Недостаточно данных"
                
        if not np.isnan(p_value):
            if p_value < 0.05:
                significance = "✅ H0 ОТКЛОНЯЕТСЯ (p < 0.05)"
                hypothesis_result = "Различия СТАТИСТИЧЕСКИ ЗНАЧИМЫ"
            else:
                significance = "🔸 H0 НЕ ОТКЛОНЯЕТСЯ (p ≥ 0.05)"  
                hypothesis_result = "Различия статистически НЕ ЗНАЧИМЫ"
            print(f"🧪 {test_name}: p = {p_value:.4f} - {significance}")
            print(f"⚖️ Заключение: {hypothesis_result}")
        else:
            print(f"🧪 {test_name}: недостаточно данных для проверки гипотез")
        
        # Размер эффекта (Cohen's d)
        pooled_std = np.sqrt(((len(no_stress_vals) - 1) * ns_std**2 + 
                            (len(stress_vals) - 1) * s_std**2) / 
                           (len(no_stress_vals) + len(stress_vals) - 2))
        cohens_d = (s_mean - ns_mean) / pooled_std if pooled_std > 0 else 0
        
        effect_size = "большой" if abs(cohens_d) >= 0.8 else "средний" if abs(cohens_d) >= 0.5 else "малый"
        print(f"📏 Размер эффекта: {effect_size} (d = {cohens_d:.3f})")
        
        results.append({
            'measure': measure,
            'measure_name': word_measure_names.get(measure, measure),
            'mean_no_stress': ns_mean,
            'std_no_stress': ns_std,
            'mean_stress': s_mean,
            'std_stress': s_std,
            'change_absolute': change,
            'change_percent': change_pct,
            'change_direction': change_direction,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05 if not np.isnan(p_value) else False
        })
    
    return results

def analyze_trial_level_differences(trial_data):
    """
    Анализирует различия на уровне трайлов между условиями без стресса и со стрессом
    """
    print(f"\n🧮 АНАЛИЗ РАЗЛИЧИЙ НА УРОВНЕ ТРАЙЛОВ")
    print("=" * 70)
    
    # Основные показатели для анализа
    measures = [
        'blinks', 'fixations', 'fixation_duration_mean', 'fixation_duration_median',
        'pupil_size', 'runs', 'saccade_amplitude_mean', 'saccades', 
        'trial_duration', 'visited_areas'
    ]
    
    # Русские названия
    measure_names = {
        'blinks': 'Количество морганий',
        'fixations': 'Общее количество фиксаций', 
        'fixation_duration_mean': 'Средняя длительность фиксаций (мс)',
        'fixation_duration_median': 'Медианная длительность фиксаций (мс)',
        'pupil_size': 'Размер зрачка (средний)',
        'runs': 'Количество забеганий взгляда',
        'saccade_amplitude_mean': 'Средняя амплитуда саккад (°)',
        'saccades': 'Общее количество саккад',
        'trial_duration': 'Длительность трайла (мс)',
        'visited_areas': 'Количество посещенных зон'
    }
    
    results = []
    
    # Данные по условиям
    no_stress = trial_data[trial_data['condition'] == 'no_stress']
    stress = trial_data[trial_data['condition'] == 'stress']
    
    print(f"📊 СРАВНЕНИЕ ГРУПП:")
    print(f"   Без стресса: {len(no_stress)} трайла")  
    print(f"   Со стрессом: {len(stress)} трайла")
    
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
        
        # Статистический тест (проверка H0: μ₁ = μ₂ против H1: μ₁ ≠ μ₂)
        # С такой маленькой выборкой (3 vs 3) используем точные тесты
        if len(no_stress_vals) >= 3 and len(stress_vals) >= 3:
            # Wilcoxon rank-sum test (аналог Mann-Whitney для маленьких выборок)
            try:
                stat, p_value = mannwhitneyu(no_stress_vals, stress_vals, alternative='two-sided')
                test_name = "Критерий Манна-Уитни"
            except:
                # Если не хватает данных, используем t-test
                stat, p_value = ttest_ind(no_stress_vals, stress_vals)
                test_name = "t-критерий"
            
            if p_value < 0.05:
                significance = "✅ H0 ОТКЛОНЯЕТСЯ (p < 0.05)"
                hypothesis_result = "Различия СТАТИСТИЧЕСКИ ЗНАЧИМЫ"
            else:
                significance = "🔸 H0 НЕ ОТКЛОНЯЕТСЯ (p ≥ 0.05)"
                hypothesis_result = "Различия статистически НЕ ЗНАЧИМЫ"
                
            print(f"🧪 {test_name}: p = {p_value:.4f} - {significance}")
            print(f"⚖️ Заключение: {hypothesis_result}")
            
            if p_value >= 0.05:
                print(f"💡 Примечание: Малый размер выборки (N = 6) может влиять на мощность теста")
            
            # Размер эффекта (Cohen's d)
            pooled_std = np.sqrt(((len(no_stress_vals) - 1) * ns_std**2 + 
                                (len(stress_vals) - 1) * s_std**2) / 
                               (len(no_stress_vals) + len(stress_vals) - 2))
            cohens_d = (s_mean - ns_mean) / pooled_std if pooled_std > 0 else 0
            
            effect_size = "большой" if abs(cohens_d) >= 0.8 else "средний" if abs(cohens_d) >= 0.5 else "малый"
            print(f"📏 Размер эффекта: {effect_size} (d = {cohens_d:.3f})")
            
        else:
            p_value = np.nan
            test_name = "Недостаточно данных"
            cohens_d = 0
            
        results.append({
            'measure': measure,
            'measure_name': measure_names.get(measure, measure),
            'no_stress_mean': ns_mean,
            'no_stress_std': ns_std,
            'stress_mean': s_mean,
            'stress_std': s_std,
            'change_absolute': change,
            'change_percent': change_pct,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05 if not np.isnan(p_value) else False
        })
    
    return results

def analyze_trial_dynamics(trial_data):
    """
    Анализирует динамику изменений по отдельным трайлам
    """
    print(f"\n📈 АНАЛИЗ ДИНАМИКИ ПО ТРАЙЛАМ")
    print("=" * 70)
    
    measures = [
        'blinks', 'fixations', 'fixation_duration_mean', 'pupil_size', 
        'runs', 'saccade_amplitude_mean', 'saccades', 'visited_areas'
    ]
    
    measure_names = {
        'blinks': 'Количество морганий',
        'fixations': 'Общее количество фиксаций', 
        'fixation_duration_mean': 'Средняя длительность фиксаций (мс)',
        'pupil_size': 'Размер зрачка',
        'runs': 'Количество забеганий',
        'saccade_amplitude_mean': 'Амплитуда саккад (°)',
        'saccades': 'Количество саккад',
        'visited_areas': 'Посещенные зоны'
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
        
        for trial in range(1, 7):
            val = trial_data[trial_data['trial'] == trial][measure].iloc[0]
            phase = trial_data[trial_data['trial'] == trial]['phase'].iloc[0]
            values.append(val)
            phases.append(phase)
            
            # Сохраняем статистику для каждого трайла
            trial_stats[measure][trial] = {
                'mean': val,
                'std': 0,  # У нас только одно значение на трайл
                'phase': phase
            }
            
            phase_emoji = {"baseline_1": "📊", "baseline_2": "📊", "baseline_3": "📊",
                          "stress_peak": "🔥", "stress_adapt": "📉", "stress_recovery": "😌"}
            
            print(f"   {phase_emoji.get(phase, '📊')} Трайл {trial}: {val:.2f}")
        
        # Базовая линия
        baseline = np.mean(values[:3])
        
        # Изменения относительно базовой линии
        changes = [(v - baseline) / baseline * 100 for v in values]
        
        # Определяем паттерн
        peak_trial = np.argmax(np.abs(changes)) + 1
        if peak_trial == 4:  # Пик в трайле 4
            if abs(changes[5]) < abs(changes[3]):  # Восстановление к трайлу 6
                pattern = "🎯 ПИК в Т4 → ВОССТАНОВЛЕНИЕ"
            else:
                pattern = "🔥 ПИК в Т4 → БЕЗ ВОССТАНОВЛЕНИЯ"
        else:
            pattern = "❓ НЕСТАНДАРТНЫЙ ПАТТЕРН"
            
        print(f"   🎯 Паттерн: {pattern}")
        print(f"   📊 Базовая линия: {baseline:.2f}")
        print(f"   📈 Макс. изменение: {max(changes, key=abs):.1f}% (Трайл {peak_trial})")
        
        dynamics[measure] = {
            'values': values,
            'phases': phases,
            'changes': changes,
            'baseline': baseline,
            'pattern': pattern
        }
    
    return dynamics, trial_stats

def analyze_word_dynamics(word_data, word_measure_names):
    """
    Анализирует динамику изменений на уровне слов по трайлам
    """
    print(f"\n📝 АНАЛИЗ ДИНАМИКИ СЛОВ ПО ТРАЙЛАМ")
    print("=" * 70)
    
    measures = ['IA_FIRST_FIXATION_DURATION', 'IA_FIXATION_COUNT', 'IA_DWELL_TIME',
                'IA_DWELL_TIME_%', 'IA_VISITED_TRIAL_%', 'IA_REVISIT_TRIAL_%', 'IA_RUN_COUNT']
    
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
        for trial in range(1, 7):
            trial_data = word_data[word_data['trial'] == trial][measure].dropna()
            if len(trial_data) > 0:
                mean_val = trial_data.mean()
                std_val = trial_data.std()
                phase = word_data[word_data['trial'] == trial]['stress_phase'].iloc[0]
            else:
                mean_val = 0
                std_val = 0
                phase = f'trial_{trial}'
            
            word_trial_stats[measure][trial] = {
                'mean': mean_val,
                'std': std_val,
                'phase': phase,
                'count': len(trial_data)
            }
            trial_values.append(mean_val)
            
            print(f"   Трайл {trial}: M = {mean_val:.2f} ± {std_val:.2f} (n = {len(trial_data)})")
        
        # Статистические тесты
        baseline_vs_peak_p = np.nan
        peak_vs_recovery_p = np.nan
        
        try:
            # Сравнение базовой линии (1-3) с пиком стресса (4)
            baseline_data = []
            for t in [1, 2, 3]:
                baseline_data.extend(word_data[word_data['trial'] == t][measure].dropna().tolist())
            peak_data = word_data[word_data['trial'] == 4][measure].dropna().tolist()
            
            if len(baseline_data) > 0 and len(peak_data) > 0:
                _, baseline_vs_peak_p = mannwhitneyu(baseline_data, peak_data, alternative='two-sided')
            
            # Сравнение пика стресса (4) с восстановлением (6)
            recovery_data = word_data[word_data['trial'] == 6][measure].dropna().tolist()
            if len(peak_data) > 0 and len(recovery_data) > 0:
                _, peak_vs_recovery_p = mannwhitneyu(peak_data, recovery_data, alternative='two-sided')
                
        except Exception as e:
            print(f"   ⚠️ Ошибка в статистическом тесте: {e}")
        
        word_dynamics_results[measure] = {
            'baseline_vs_peak_p': baseline_vs_peak_p,
            'peak_vs_recovery_p': peak_vs_recovery_p,
            'measure': measure
        }
        
        if not np.isnan(baseline_vs_peak_p):
            print(f"   🧪 База vs Пик: p = {baseline_vs_peak_p:.4f}")
        if not np.isnan(peak_vs_recovery_p):
            print(f"   🧪 Пик vs Восст.: p = {peak_vs_recovery_p:.4f}")
    
    return word_trial_stats, word_dynamics_results

def create_enhanced_word_visualizations(word_data, word_test_results, word_measure_names):
    """Создает улучшенные графики анализа данных на уровне слов"""
    print(f"\n🎨 СОЗДАНИЕ ГРАФИКОВ АНАЛИЗА СЛОВ")
    print("=" * 50)
    
    measures = [r['measure'] for r in word_test_results]
    
    # Создаем фигуру
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()
    
    colors_no_stress = '#2E86C1'  # Синий
    colors_stress = '#E74C3C'     # Красный
    
    for i, result in enumerate(word_test_results):
        if i >= len(axes):
            break
            
        ax = axes[i]
        measure = result['measure']
        
        # Данные для boxplot
        no_stress_data = word_data[word_data['condition'] == 'no_stress'][measure].dropna()
        stress_data = word_data[word_data['condition'] == 'stress'][measure].dropna()
        
        if len(no_stress_data) == 0 or len(stress_data) == 0:
            ax.text(0.5, 0.5, 'Недостаточно\nданных', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{result['measure_name']}", fontweight='bold', fontsize=11)
            continue
        
        # Создаем boxplot
        bp = ax.boxplot([no_stress_data, stress_data], 
                       labels=['Без стресса', 'Со стрессом'],
                       patch_artist=True,
                       notch=True,
                       widths=0.6)
        
        # Раскрашиваем боксы
        bp['boxes'][0].set_facecolor(colors_no_stress)
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor(colors_stress)
        bp['boxes'][1].set_alpha(0.7)
        
        # Заголовок с названием показателя
        ax.set_title(f"{result['measure_name']}", fontweight='bold', fontsize=11)
        
        # Добавляем стрелку показывающую направление изменения
        y_max = max(no_stress_data.max(), stress_data.max())
        y_min = min(no_stress_data.min(), stress_data.min())
        y_range = y_max - y_min
        
        # Позиция для стрелки и текста
        arrow_y = y_max + y_range * 0.1
        text_y = y_max + y_range * 0.15
        
        if result['change_direction'] == 'увеличение':
            ax.annotate('', xy=(2, arrow_y), xytext=(1, arrow_y),
                       arrowprops=dict(arrowstyle='->', lw=2, color='red'))
            direction_symbol = "↗️"
        else:
            ax.annotate('', xy=(1, arrow_y), xytext=(2, arrow_y),
                       arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
            direction_symbol = "↘️"
        
        # Информация об изменении
        change_text = f"{direction_symbol} {abs(result['change_percent']):.1f}%"
        ax.text(1.5, text_y, change_text, ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
        
        # Информация о значимости
        if result['significant']:
            significance_text = f"p = {result['p_value']:.3f} ✅"
            bbox_color = 'lightgreen'
        else:
            significance_text = f"p = {result['p_value']:.3f} ❌"
            bbox_color = 'lightcoral'
        
        ax.text(0.5, 0.02, significance_text, transform=ax.transAxes, 
                ha='center', va='bottom',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=bbox_color, alpha=0.7))
        
        # Добавляем средние значения как точки
        ax.scatter([1], [result['mean_no_stress']], color='darkblue', s=50, zorder=10, marker='D')
        ax.scatter([2], [result['mean_stress']], color='darkred', s=50, zorder=10, marker='D')
        
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Значение')
    
    # Удаляем лишние подграфики
    for i in range(len(word_test_results), len(axes)):
        fig.delaxes(axes[i])
    
    # Общий заголовок
    fig.suptitle('📝 ВЛИЯНИЕ СТРЕССА НА ПАРАМЕТРЫ АЙТРЕКИНГА ПРИ ЧТЕНИИ (УРОВЕНЬ СЛОВ)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig('results/word_level_stress_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_dynamics_visualizations(trial_data, trial_stats, word_data, word_trial_stats, 
                                 word_dynamics_results, word_measure_names):
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
        'baseline_1': '#3498DB',   # Синий
        'baseline_2': '#5DADE2',   # Светло-синий  
        'baseline_3': '#85C1E9',   # Еще светлее синий
        'stress_peak': '#E74C3C',  # Красный (пик)
        'stress_adapt': '#F1948A', # Розовый (адаптация)
        'stress_recovery': '#F8C471' # Желтый (восстановление)
    }
    
    for i, measure in enumerate(trial_measures):
        if i >= len(axes1):
            break
        
        ax = axes1[i]
        
        # Данные для графика
        trials = sorted(trial_stats[measure].keys())
        means = [trial_stats[measure][t]['mean'] for t in trials]
        stds = [trial_stats[measure][t]['std'] for t in trials]
        phases = [trial_stats[measure][t]['phase'] for t in trials]
        colors = [phase_colors[phase] for phase in phases]
        
        # Создаем bar plot с разными цветами для фаз
        bars = ax.bar(trials, means, yerr=stds, capsize=5, 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Добавляем линию тренда
        ax.plot(trials, means, 'k--', alpha=0.7, linewidth=2, marker='o', markersize=6)
        
        # Вертикальная линия разделяющая фазы
        ax.axvline(x=3.5, color='red', linestyle=':', linewidth=2, alpha=0.7)
        ax.text(3.5, max(means) * 0.9, 'НАЧАЛО\nСТРЕССА', ha='center', va='center', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
        
        # Заголовок
        trial_measure_names = {
            'blinks': 'Моргания', 'fixations': 'Фиксации', 
            'fixation_duration_mean': 'Длит. фиксаций', 'pupil_size': 'Размер зрачка',
            'runs': 'Забегания', 'saccade_amplitude_mean': 'Амплитуда саккад',
            'saccades': 'Саккады', 'visited_areas': 'Посещ. зоны'
        }
        ax.set_title(f"{trial_measure_names.get(measure, measure)}", 
                    fontweight='bold', fontsize=13)
        
        # Подписи процентных изменений на барах
        baseline_mean = np.mean([trial_stats[measure][t]['mean'] for t in [1, 2, 3]])
        for j, (trial, mean_val) in enumerate(zip(trials, means)):
            if trial > 3:  # Только для стрессовых трайлов
                change_pct = ((mean_val - baseline_mean) / baseline_mean) * 100
                symbol = "↗" if change_pct > 0 else "↘"
                ax.text(trial, mean_val + stds[j] + max(means) * 0.02, 
                       f'{symbol}{abs(change_pct):.1f}%', 
                       ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Настройка осей
        ax.set_xlabel('Номер трайла')
        ax.set_ylabel('Значение')
        ax.set_xticks(trials)
        ax.grid(True, alpha=0.3)
        
        # Добавляем легенду фаз (только на первом графике)
        if i == 0:
            legend_elements = []
            legend_labels = {
                'baseline_1': 'Базовая линия', 
                'stress_peak': 'Пик стресса',
                'stress_adapt': 'Адаптация', 
                'stress_recovery': 'Восстановление'
            }
            for phase, color in phase_colors.items():
                if phase in legend_labels:
                    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8))
            
            ax.legend(legend_elements, list(legend_labels.values()), 
                     loc='upper right', fontsize=9)
    
    # Удаляем лишние подграфики
    for i in range(len(trial_measures), len(axes1)):
        fig1.delaxes(axes1[i])
    
    # Общий заголовок
    fig1.suptitle('📊 ДИНАМИКА СТРЕССА ПО ТРАЙЛАМ (УРОВЕНЬ ТРАЙЛОВ)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('results/trial_level_dynamics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
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
        means = [word_trial_stats[measure][t]['mean'] for t in trials]
        stds = [word_trial_stats[measure][t]['std'] for t in trials]
        phases = [word_trial_stats[measure][t]['phase'] for t in trials]
        colors = [phase_colors[phase] for phase in phases]
        
        # Создаем bar plot с error bars
        bars = ax.bar(trials, means, yerr=stds, capsize=5, 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Добавляем линию тренда
        ax.plot(trials, means, 'k--', alpha=0.7, linewidth=2, marker='o', markersize=6)
        
        # Вертикальная линия разделяющая фазы
        ax.axvline(x=3.5, color='red', linestyle=':', linewidth=2, alpha=0.7)
        ax.text(3.5, max(means) * 0.9, 'НАЧАЛО\nСТРЕССА', ha='center', va='center', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
        
        # Заголовок
        ax.set_title(f"{word_measure_names.get(measure, measure)}", 
                    fontweight='bold', fontsize=11)
        
        # Показываем значимость основных сравнений
        result = word_dynamics_results.get(measure)
        if result:
            significance_text = ""
            if not np.isnan(result['baseline_vs_peak_p']) and result['baseline_vs_peak_p'] < 0.05:
                significance_text += "База↔4: ✅ "
            if not np.isnan(result['peak_vs_recovery_p']) and result['peak_vs_recovery_p'] < 0.05:
                significance_text += "4↔6: ✅"
            
            if significance_text:
                ax.text(0.02, 0.98, significance_text, transform=ax.transAxes,
                       va='top', ha='left', fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        # Настройка осей
        ax.set_xlabel('Номер трайла')
        ax.set_ylabel('Значение')
        ax.set_xticks(trials)
        ax.grid(True, alpha=0.3)
    
    # Удаляем лишние подграфики
    for i in range(len(word_measures), len(axes2)):
        fig2.delaxes(axes2[i])
    
    # Общий заголовок
    fig2.suptitle('📝 ДИНАМИКА СТРЕССА ПО ТРАЙЛАМ (УРОВЕНЬ СЛОВ)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('results/word_level_dynamics.png', dpi=300, bbox_inches='tight')
    plt.show()

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
        measure = result['measure']
        
        # Данные для boxplot
        no_stress_data = trial_data[trial_data['condition'] == 'no_stress'][measure]
        stress_data = trial_data[trial_data['condition'] == 'stress'][measure]
        
        # Создаем boxplot
        bp = ax.boxplot([no_stress_data, stress_data], 
                       labels=['Без стресса\n(Т1-3)', 'Со стрессом\n(Т4-6)'],
                       patch_artist=True, widths=0.6)
        
        # Раскрашиваем
        bp['boxes'][0].set_facecolor('#2E86C1')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor('#E74C3C')
        bp['boxes'][1].set_alpha(0.7)
        
        # Заголовок
        ax.set_title(f"{result['measure_name']}", fontweight='bold', fontsize=11)
        
        # Показываем изменение
        change_pct = result['change_percent']
        symbol = "↗" if change_pct > 0 else "↘"
        ax.text(0.5, 0.95, f'{symbol}{abs(change_pct):.1f}%', 
               transform=ax.transAxes, ha='center', va='top',
               fontweight='bold', fontsize=12)
        
        # Значимость
        if result['significant']:
            significance_text = f"p = {result['p_value']:.3f} ✅"
            bbox_color = 'lightgreen'
        else:
            significance_text = f"p = {result['p_value']:.3f}" if not np.isnan(result['p_value']) else "n.s."
            bbox_color = 'lightgray'
        
        ax.text(0.5, 0.02, significance_text, transform=ax.transAxes,
               ha='center', va='bottom', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor=bbox_color, alpha=0.7))
        
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('🧠 СРАВНЕНИЕ ПОКАЗАТЕЛЕЙ АЙТРЕКИНГА: БЕЗ СТРЕССА vs СО СТРЕССОМ', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('results/comprehensive_stress_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. ДИНАМИКА ПО ТРАЙЛАМ
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()
    
    measures_for_dynamics = list(dynamics.keys())[:8]
    
    colors = ['#3498DB', '#5DADE2', '#85C1E9', '#E74C3C', '#F1948A', '#F8C471']
    
    for i, measure in enumerate(measures_for_dynamics):
        ax = axes[i]
        
        values = dynamics[measure]['values']
        phases = dynamics[measure]['phases']
        
        # Создаем линейный график
        trials = list(range(1, 7))
        ax.plot(trials, values, 'o-', linewidth=3, markersize=8, color='darkblue')
        
        # Раскрашиваем точки по фазам
        phase_colors = {
            'baseline_1': '#3498DB', 'baseline_2': '#5DADE2', 'baseline_3': '#85C1E9',
            'stress_peak': '#E74C3C', 'stress_adapt': '#F1948A', 'stress_recovery': '#F8C471'
        }
        
        for j, (trial, value, phase) in enumerate(zip(trials, values, phases)):
            ax.scatter(trial, value, s=150, c=phase_colors.get(phase, 'gray'), 
                      edgecolor='black', linewidth=2, zorder=5)
        
        # Вертикальная линия разделяющая фазы
        ax.axvline(x=3.5, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.text(3.5, max(values) * 0.9, 'НАЧАЛО\nСТРЕССА', ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
        
        # Заголовок с паттерном
        measure_name = {
            'blinks': 'Моргания', 'fixations': 'Фиксации', 
            'fixation_duration_mean': 'Длит. фиксаций', 'pupil_size': 'Размер зрачка',
            'runs': 'Забегания', 'saccade_amplitude_mean': 'Амплитуда саккад',
            'saccades': 'Саккады', 'visited_areas': 'Посещ. зоны'
        }.get(measure, measure)
        
        ax.set_title(f"{measure_name}\n{dynamics[measure]['pattern']}", 
                    fontweight='bold', fontsize=10)
        
        ax.set_xlabel('Номер трайла')
        ax.set_ylabel('Значение')
        ax.set_xticks(trials)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('📊 ДИНАМИКА ПОКАЗАТЕЛЕЙ АЙТРЕКИНГА ПО ТРАЙЛАМ', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('results/comprehensive_trial_dynamics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. СПЕЦИАЛЬНЫЙ ГРАФИК: КЛЮЧЕВЫЕ МАРКЕРЫ СТРЕССА
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Самые информативные показатели
    key_measures = ['pupil_size', 'blinks', 'saccade_amplitude_mean']
    key_names = ['Размер зрачка (стресс-маркер)', 'Количество морганий', 'Амплитуда саккад']
    
    for i, (measure, name) in enumerate(zip(key_measures, key_names)):
        ax = axes[i]
        
        no_stress = trial_data[trial_data['condition'] == 'no_stress'][measure]
        stress = trial_data[trial_data['condition'] == 'stress'][measure]
        
        # Violin plot для красоты
        violin = ax.violinplot([no_stress, stress], positions=[1, 2], widths=0.5)
        ax.scatter([1] * len(no_stress), no_stress, alpha=0.7, s=100, color='blue', label='Без стресса')
        ax.scatter([2] * len(stress), stress, alpha=0.7, s=100, color='red', label='Со стрессом')
        
        # Средние линии
        ax.hlines(no_stress.mean(), 0.7, 1.3, colors='blue', linestyle='--', linewidth=2)
        ax.hlines(stress.mean(), 1.7, 2.3, colors='red', linestyle='--', linewidth=2)
        
        ax.set_title(name, fontweight='bold')
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Без стресса', 'Со стрессом'])
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend()
    
    plt.suptitle('🎯 КЛЮЧЕВЫЕ МАРКЕРЫ СТРЕССА В ДАННЫХ АЙТРЕКИНГА', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig('results/key_stress_markers.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_formal_hypotheses(word_comparison_results, trial_comparison_results):
    """
    Проводит формальное тестирование научных гипотез с выводами
    """
    print(f"\n🔬 ФОРМАЛЬНОЕ ТЕСТИРОВАНИЕ ГИПОТЕЗ")
    print("=" * 80)
    
    # Подсчет значимых результатов
    word_significant = [r for r in word_comparison_results if r['significant']]
    trial_significant = [r for r in trial_comparison_results if r['significant']]
    total_significant = len(word_significant) + len(trial_significant)
    total_tests = len(word_comparison_results) + len(trial_comparison_results)
    
    print(f"📊 РЕЗУЛЬТАТЫ СТАТИСТИЧЕСКОГО ТЕСТИРОВАНИЯ:")
    print(f"   • Общее количество проведенных тестов: {total_tests}")
    print(f"   • Количество значимых результатов (p < 0.05): {total_significant}")
    print(f"   • Процент значимых результатов: {total_significant/total_tests*100:.1f}%")
    
    print(f"\n⚖️ ПРОВЕРКА ГИПОТЕЗ ПО УРОВНЯМ АНАЛИЗА:")
    
    # Анализ уровня слов
    print(f"\n   📝 УРОВЕНЬ СЛОВ (N = 548 наблюдений):")
    print(f"   • Проведено тестов: {len(word_comparison_results)}")
    print(f"   • Значимых результатов: {len(word_significant)}")
    
    if len(word_significant) > 0:
        print(f"   🔹 ВЫВОД: H0 ОТКЛОНЯЕТСЯ по {len(word_significant)} показателю(ям)")
        for result in word_significant:
            print(f"      ✅ {result['measure_name']}: p = {result['p_value']:.4f} < 0.05")
    else:
        print(f"   🔸 ВЫВОД: H0 НЕ ОТКЛОНЯЕТСЯ (нет значимых различий)")
    
    # Анализ уровня трайлов
    print(f"\n   📊 УРОВЕНЬ ТРАЙЛОВ (N = 6 трайлов):")
    print(f"   • Проведено тестов: {len(trial_comparison_results)}")
    print(f"   • Значимых результатов: {len(trial_significant)}")
    
    if len(trial_significant) > 0:
        print(f"   🔹 ВЫВОД: H0 ОТКЛОНЯЕТСЯ по {len(trial_significant)} показателю(ям)")
        for result in trial_significant:
            print(f"      ✅ {result['measure_name']}: p = {result['p_value']:.4f} < 0.05")
    else:
        print(f"   🔸 ВЫВОД: H0 НЕ ОТКЛОНЯЕТСЯ (нет значимых различий)")
        print(f"   ⚠️  ПРИЧИНА: Малый размер выборки (N = 6)")
        
        # Но анализируем размеры эффектов
        large_effects = [r for r in trial_comparison_results if abs(r['cohens_d']) >= 0.8]
        print(f"   📈 ОДНАКО: Обнаружено {len(large_effects)} показателей с БОЛЬШИМИ размерами эффекта")
    
    # ОБЩЕЕ ЗАКЛЮЧЕНИЕ ПО ГИПОТЕЗАМ
    print(f"\n🏛️ ИТОГОВОЕ ЗАКЛЮЧЕНИЕ ПО ГИПОТЕЗАМ:")
    
    if total_significant > 0:
        print(f"   ✅ НУЛЕВАЯ ГИПОТЕЗА H0 ЧАСТИЧНО ОТКЛОНЯЕТСЯ")
        print(f"   ✅ АЛЬТЕРНАТИВНАЯ ГИПОТЕЗА H1 ЧАСТИЧНО ПОДТВЕРЖДАЕТСЯ") 
        print(f"   🎯 ЗАКЛЮЧЕНИЕ: Существуют статистически значимые различия")
        print(f"      в параметрах айтрекинга между условиями стресса")
    else:
        print(f"   🔸 НУЛЕВАЯ ГИПОТЕЗА H0 НЕ ОТКЛОНЯЕТСЯ на уровне p < 0.05")
        print(f"   ⚠️  ОДНАКО: Обнаружены БОЛЬШИЕ размеры эффектов")
        
        # Подсчет больших эффектов
        all_large_effects = [r for r in word_comparison_results + trial_comparison_results 
                           if abs(r['cohens_d']) >= 0.8]
        
        print(f"   📊 ПРАКТИЧЕСКАЯ ЗНАЧИМОСТЬ: {len(all_large_effects)} показателей с d > 0.8")
        print(f"   🔬 РЕКОМЕНДАЦИЯ: Увеличить размер выборки для подтверждения")
    
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

def generate_comprehensive_report(word_comparison_results, trial_comparison_results, dynamics):
    """
    Создает итоговый отчет о возможности детекции стресса через айтрекинг
    """
    print(f"\n🏆 ИТОГОВЫЙ ОТЧЕТ: ДЕТЕКЦИЯ СТРЕССА ЧЕРЕЗ АЙТРЕКИНГ")
    print("=" * 80)
    
    # Анализ результатов на уровне слов
    word_significant_results = [r for r in word_comparison_results if r['significant']]
    word_large_effects = [r for r in word_comparison_results if abs(r['cohens_d']) >= 0.8]
    word_medium_effects = [r for r in word_comparison_results if 0.5 <= abs(r['cohens_d']) < 0.8]
    
    # Анализ результатов на уровне трайлов
    trial_significant_results = [r for r in trial_comparison_results if r['significant']]
    trial_large_effects = [r for r in trial_comparison_results if abs(r['cohens_d']) >= 0.8]
    trial_medium_effects = [r for r in trial_comparison_results if 0.5 <= abs(r['cohens_d']) < 0.8]
    
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
            if 'change_absolute' in result:  # Трайлы
                direction = "выше" if result['change_absolute'] > 0 else "ниже"
                print(f"   🎯 {result['measure_name']}: {direction} на {abs(result['change_percent']):.1f}%")
                print(f"      p = {result['p_value']:.3f}, Cohen's d = {result['cohens_d']:.3f}")
            else:  # Слова
                direction = result['change_direction']
                print(f"   🎯 {result['measure_name']}: {direction} на {abs(result['change_percent']):.1f}%")
                print(f"      p = {result['p_value']:.3f}, Cohen's d = {result['cohens_d']:.3f}")
    
    print(f"\n📈 НАБЛЮДАЕМЫЕ ИЗМЕНЕНИЯ ПОД ВЛИЯНИЕМ СТРЕССА:")
    
    # Объединяем все изменения
    all_increases = []
    all_decreases = []
    
    for result in word_comparison_results + trial_comparison_results:
        if 'change_absolute' in result:  # Трайлы
            if result['change_absolute'] > 0:
                all_increases.append(result)
            else:
                all_decreases.append(result)
        else:  # Слова
            if result['change_direction'] == 'увеличение':
                all_increases.append(result)
            else:
                all_decreases.append(result)
    
    if all_increases:
        print(f"\n   ⬆️ УВЕЛИЧЕНИЕ:")
        for result in sorted(all_increases, key=lambda x: abs(x['change_percent']), reverse=True)[:10]:
            significance = " ✅" if result['significant'] else ""
            print(f"      • {result['measure_name']}: +{abs(result['change_percent']):.1f}%{significance}")
    
    if all_decreases:
        print(f"\n   ⬇️ УМЕНЬШЕНИЕ:")
        for result in sorted(all_decreases, key=lambda x: abs(x['change_percent']), reverse=True)[:10]:
            significance = " ✅" if result['significant'] else ""
            print(f"      • {result['measure_name']}: {result['change_percent']:.1f}%{significance}")
    
    # Анализ паттернов динамики
    print(f"\n🔄 АНАЛИЗ ДИНАМИЧЕСКИХ ПАТТЕРНОВ:")
    
    peak_patterns = 0
    recovery_patterns = 0
    
    for measure, data in dynamics.items():
        if "ПИК в Т4" in data['pattern']:
            peak_patterns += 1
            if "ВОССТАНОВЛЕНИЕ" in data['pattern']:
                recovery_patterns += 1
    
    print(f"   • Показателей с пиком в трайле 4: {peak_patterns}/{len(dynamics)}")
    print(f"   • Показателей с восстановлением: {recovery_patterns}/{len(dynamics)}")
    
    # КЛЮЧЕВЫЕ ВЫВОДЫ
    print(f"\n🧠 КЛЮЧЕВЫЕ ВЫВОДЫ О ДЕТЕКЦИИ СТРЕССА:")
    
    total_significant = len(all_significant)
    total_large_effects = len(all_large_effects)
    
    if total_significant > 0 or total_large_effects > 5:
        print("   ✅ СТРЕСС ДЕТЕКТИРУЕТСЯ через айтрекинг!")
        print("   🎯 Лучшие маркеры для детекции:")
        
        # Объединяем лучшие результаты
        best_results = all_significant + all_large_effects
        # Убираем дубликаты и сортируем по размеру эффекта
        unique_results = {r['measure_name']: r for r in best_results}.values()
        sorted_results = sorted(unique_results, key=lambda x: abs(x['cohens_d']), reverse=True)
        
        for result in sorted_results[:5]:
            level = "слова" if 'change_direction' in result else "трайлы"
            print(f"      • {result['measure_name']} (уровень: {level})")
        
    else:
        print("   ⚠️  Статистически значимых различий мало")
        print("   📊 Возможные причины:")
        print("      • Малый размер выборки")
        print("      • Высокая индивидуальная вариабельность")
        print("      • Быстрая адаптация к стрессу")
    
    print(f"\n🔍 НАИБОЛЕЕ ПЕРСПЕКТИВНЫЕ МАРКЕРЫ (по размеру эффекта):")
    
    all_effects = sorted(word_comparison_results + trial_comparison_results, 
                        key=lambda x: abs(x['cohens_d']), reverse=True)
    
    for i, result in enumerate(all_effects[:8], 1):
        effect_description = "большой" if abs(result['cohens_d']) >= 0.8 else "средний" if abs(result['cohens_d']) >= 0.5 else "малый"
        level = "слова" if 'change_direction' in result else "трайлы"
        print(f"   {i}. {result['measure_name']} ({level}): {effect_description} эффект (d = {result['cohens_d']:.3f})")
    
    # ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ
    print(f"\n💡 ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ:")
    
    if total_significant > 0 or total_large_effects > 3:
        print("   ✅ АЙТРЕКИНГ МОЖЕТ ИСПОЛЬЗОВАТЬСЯ для детекции стресса")
        print("   🎯 Рекомендуемые показатели:")
        key_measures = (all_significant + all_large_effects)[:5]
        for result in key_measures:
            level = "слова" if 'change_direction' in result else "трайлы"
            print(f"      • {result['measure_name']} (уровень: {level})")
        print("   📊 Необходимы:")
        print("      • Больший размер выборки для подтверждения")
        print("      • Индивидуальные базовые линии")
        print("      • Мультимодальная валидация (физиология + айтрекинг)")
        print("      • Комбинирование показателей разных уровней")
    else:
        print("   📊 ДЛЯ НАДЕЖНОЙ ДЕТЕКЦИИ СТРЕССА необходимо:")
        print("      • Увеличить размер выборки (>10 участников)")
        print("      • Использовать более контрастные стрессовые условия")
        print("      • Добавить физиологические маркеры")
        print("      • Применить машинное обучение для комбинации показателей")
    
    print(f"\n🚀 РЕЗЮМЕ:")
    print(f"   • Анализировано {len(word_comparison_results) + len(trial_comparison_results)} показателей айтрекинга")
    print(f"   • Обнаружено {total_large_effects} показателей с большими размерами эффекта")
    print(f"   • {recovery_patterns} показателей демонстрируют адаптацию к стрессу")
    print(f"   • Айтрекинг показывает {'высокий' if total_large_effects > 5 else 'средний' if total_large_effects > 2 else 'низкий'} потенциал для детекции стресса")

def print_research_hypotheses():
    """
    Печатает формальные научные гипотезы исследования
    """
    print("🔬 НАУЧНЫЕ ГИПОТЕЗЫ ИССЛЕДОВАНИЯ")
    print("=" * 80)
    
    print("\n📋 ФОРМУЛИРОВКА ГИПОТЕЗ:")
    
    print("\n🔸 НУЛЕВАЯ ГИПОТЕЗА (H0):")
    print("   Отсутствуют статистически значимые различия в параметрах айтрекинга")
    print("   между условиями БЕЗ СТРЕССА (трайлы 1-3) и СО СТРЕССОМ (трайлы 4-6)")
    print("   H0: μ₁ = μ₂ (средние значения равны)")
    
    print("\n🔹 АЛЬТЕРНАТИВНАЯ ГИПОТЕЗА (H1):")
    print("   Существуют статистически значимые различия в параметрах айтрекинга")
    print("   между условиями БЕЗ СТРЕССА и СО СТРЕССОМ")
    print("   H1: μ₁ ≠ μ₂ (средние значения различаются)")
    
    print("\n⚖️ КРИТЕРИИ ПРИНЯТИЯ РЕШЕНИЯ:")
    print("   • Уровень значимости: α = 0.05")
    print("   • Если p-value < 0.05 → ОТКЛОНЯЕМ H0, ПРИНИМАЕМ H1")
    print("   • Если p-value ≥ 0.05 → НЕ ОТКЛОНЯЕМ H0")
    
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
    print("🚀 КОМПЛЕКСНЫЙ АНАЛИЗ ДАННЫХ АЙТРЕКИНГА")
    print("🎯 Цель: Доказать возможность детекции стресса через движения глаз")
    print("📊 Анализ на двух уровнях: отдельные слова + целые трайлы")
    print("=" * 80)
    
    # Формулировка научных гипотез
    print_research_hypotheses()
    
    try:
        # 0. Создание папки для результатов
        ensure_results_directory()
        
        # 1. Загрузка данных
        word_data, trial_data, word_measure_names = load_comprehensive_data()
        
        # 2. Анализ различий на уровне слов
        word_comparison_results = analyze_word_level_differences(word_data, word_measure_names)
        
        # 3. Анализ различий на уровне трайлов
        trial_comparison_results = analyze_trial_level_differences(trial_data)
        
        # 4. Анализ динамики на уровне трайлов
        dynamics, trial_stats = analyze_trial_dynamics(trial_data)
        
        # 5. Анализ динамики на уровне слов
        word_trial_stats, word_dynamics_results = analyze_word_dynamics(word_data, word_measure_names)
        
        # 6. Создание всех графиков
        # 6a. Графики сравнения условий для слов
        create_enhanced_word_visualizations(word_data, word_comparison_results, word_measure_names)
        
        # 6b. Графики динамики для обоих уровней
        create_dynamics_visualizations(trial_data, trial_stats, word_data, word_trial_stats, 
                                     word_dynamics_results, word_measure_names)
        
        # 6c. Комплексные графики на уровне трайлов
        create_comprehensive_visualizations(trial_data, trial_comparison_results, dynamics)
        
        # 7. Формальное тестирование гипотез
        total_significant, total_tests = test_formal_hypotheses(word_comparison_results, trial_comparison_results)
        
        # 8. Итоговый отчет
        generate_comprehensive_report(word_comparison_results, trial_comparison_results, dynamics)
        
        print(f"\n🎉 КОМПЛЕКСНЫЙ АНАЛИЗ ЗАВЕРШЕН!")
        print(f"📊 Создано 6 наборов графиков в папке 'results/':")
        print(f"   • results/word_level_stress_analysis.png - сравнение условий (слова)")
        print(f"   • results/trial_level_dynamics.png - динамика по трайлам")
        print(f"   • results/word_level_dynamics.png - динамика по словам")
        print(f"   • results/comprehensive_stress_comparison.png - сравнение условий")
        print(f"   • results/comprehensive_trial_dynamics.png - динамика по трайлам")
        print(f"   • results/key_stress_markers.png - ключевые маркеры стресса")
        
        # Результаты тестирования гипотез
        print(f"\n🔬 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ГИПОТЕЗ:")
        print(f"   • Всего проведено тестов: {total_tests}")
        print(f"   • Статистически значимых результатов: {total_significant}")
        
        if total_significant > 0:
            print(f"   ✅ ЗАКЛЮЧЕНИЕ: H0 частично отклоняется → H1 подтверждается")
            print(f"   🎯 АЙТРЕКИНГ МОЖЕТ ДЕТЕКТИРОВАТЬ СТРЕСС!")
        else:
            large_effects = len([r for r in word_comparison_results + trial_comparison_results if abs(r['cohens_d']) >= 0.8])
            if large_effects > 5:
                print(f"   🔸 ЗАКЛЮЧЕНИЕ: H0 не отклоняется, НО обнаружены большие эффекты")
                print(f"   📊 ПОТЕНЦИАЛ ДЛЯ ДЕТЕКЦИИ СТРЕССА ВЫСОКИЙ (при увеличении выборки)")
            else:
                print(f"   🔸 ЗАКЛЮЧЕНИЕ: H0 не отклоняется")
                print(f"   📊 ПОТЕНЦИАЛ ДЛЯ ДЕТЕКЦИИ СТРЕССА СРЕДНИЙ")
        
    except Exception as e:
        print(f"❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 