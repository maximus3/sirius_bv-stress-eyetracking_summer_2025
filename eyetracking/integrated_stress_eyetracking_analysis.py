#!/usr/bin/env python3
"""
Интегрированный анализ данных айтрекинга с учетом реального стресса участников.

Использует результаты анализа стресса из полиграфа для более точной классификации
участников на стрессовых и нестрессовых, а затем анализирует различия в паттернах чтения.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import warnings
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu

# Настройка matplotlib для русского языка
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Tahoma', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')

# Подавляем warnings
warnings.filterwarnings('ignore')

class IntegratedStressEyetrackingAnalyzer:
    """Класс для интегрированного анализа стресса и айтрекинга"""
    
    def __init__(self, eyetracking_data_path="eyetracking/by_person/data/trial.xls"):
        self.eyetracking_data_path = pathlib.Path(eyetracking_data_path)
        self.results_dir = pathlib.Path("eyetracking/stress_integrated_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Классификация участников на основе реального анализа стресса из полиграфа
        self.stress_responders = {
            # Участники, которые реагировали на стресс (из полиграфа)
            '1707LTA': {'type': 'strong_responder', 'peak_text': 4, 'description': 'Самая яркая реакция (+31.8%)'},
            '1807KNV': {'type': 'strong_responder', 'peak_text': 4, 'description': 'Резкий пик в 4-м тексте (+150%)'},
            '1607KYA': {'type': 'moderate_responder', 'peak_text': 4, 'description': 'Умеренная реакция'},
            '1807OVA': {'type': 'moderate_responder', 'peak_text': 4, 'description': 'Умеренная реакция'},
            '1807ZUG': {'type': 'weak_responder', 'peak_text': 5, 'description': 'Слабая реакция'},
            # Дополнительные участники без данных полиграфа
            '1807SAV': {'type': 'assumed_responder', 'peak_text': 4, 'description': 'Предполагаемый стресс в 4-м тексте'},
            '1807KAN': {'type': 'assumed_responder', 'peak_text': 4, 'description': 'Предполагаемый стресс в 4-м тексте'},
        }
        
        self.stress_non_responders = {
            # Участники, которые НЕ реагировали на стресс (из полиграфа)
            '1707KAV': {'type': 'strong_non_responder', 'description': 'Сильное снижение стресса (-60%)'},
            '1807HEE': {'type': 'strong_non_responder', 'description': 'Значительное снижение (-66.7%)'},
            '1807CAA': {'type': 'paradoxical_non_responder', 'description': 'Неожиданное снижение (-26.1%)'},
            '1607LVA': {'type': 'stable_non_responder', 'description': 'Стабильный уровень (0%)'},
            '1907ZSI': {'type': 'stable_non_responder', 'description': 'Стабильный уровень (0%)'},
            '1707DMA': {'type': 'weak_non_responder', 'description': 'Легкое снижение (-10%)'},
            '1707SAA': {'type': 'weak_non_responder', 'description': 'Легкое снижение (-9.1%)'},
        }
        
        # Объединяем все классификации
        self.all_participants_stress_data = {**self.stress_responders, **self.stress_non_responders}
        
        print(f"📊 Классификация участников по стрессу:")
        print(f"   • Респондеры: {len(self.stress_responders)}")
        print(f"   • Нон-респондеры: {len(self.stress_non_responders)}")
        print(f"   • Всего участников с данными о стрессе: {len(self.all_participants_stress_data)}")
    
    def load_eyetracking_data(self):
        """Загружает данные айтрекинга"""
        print("\n📂 ЗАГРУЗКА ДАННЫХ АЙТРЕКИНГА...")
        
        # Читаем данные аналогично person_level_analysis.py
        data = pd.read_csv(self.eyetracking_data_path, sep="\t", encoding="utf-16", header=0)
        data = data.iloc[1:].reset_index(drop=True)  # Удаляем строку с описаниями
        
        print(f"   • Загружено строк: {len(data)}")
        print(f"   • Количество колонок: {len(data.columns)}")
        
        # Преобразуем запятые в точки для числовых колонок
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
        
        data['INDEX'] = pd.to_numeric(data['INDEX'], errors='coerce')
        
        # Создаем нормализованные метрики
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
        data['REVISITED_WORDS_PERCENT'] = (data['REVISITED_WORDS'] / data['IA_COUNT']) * 100
        
        # 7. Частотные метрики (в секунду)
        data['FIXATIONS_PER_SECOND'] = data['FIXATION_COUNT'] / (data['DURATION'] / 1000)
        data['SACCADES_PER_SECOND'] = data['SACCADE_COUNT'] / (data['DURATION'] / 1000)
        data['BLINKS_PER_SECOND'] = data['BLINK_COUNT'] / (data['DURATION'] / 1000)
        
        # 8. Анализ возвратных саккад
        data['REGRESSIVE_SACCADES'] = data['INTEREST_AREA_FIXATION_SEQUENCE'].apply(self._count_regressive_saccades)
        data['REGRESSIVE_SACCADES_PER_WORD'] = data['REGRESSIVE_SACCADES'] / data['IA_COUNT']
        data['REGRESSIVE_SACCADES_PER_SECOND'] = data['REGRESSIVE_SACCADES'] / (data['DURATION'] / 1000)
        data['REGRESSIVE_SACCADES_PERCENT'] = (data['REGRESSIVE_SACCADES'] / data['SACCADE_COUNT']) * 100
        
        # Добавляем классификацию стресса
        print("   • Применение классификации стресса...")
        data = self._add_stress_classification(data)
        
        print(f"   • Уникальных участников: {data['RECORDING_SESSION_LABEL'].nunique()}")
        print(f"   • Участников с данными о стрессе: {len(data[data['stress_group'].notna()])}")
        print(f"   • Распределение по группам стресса:")
        print(data['stress_group'].value_counts().to_string())
        
        return data
    
    def _count_regressive_saccades(self, sequence_data):
        """Подсчитывает количество возвратных саккад"""
        if pd.isna(sequence_data) or sequence_data == '.' or sequence_data == '':
            return 0
        
        try:
            if isinstance(sequence_data, list):
                sequence = sequence_data
            else:
                sequence_str = str(sequence_data).strip('[]')
                sequence = [int(x.strip()) for x in sequence_str.split(',') if x.strip().isdigit()]
            
            if len(sequence) < 2:
                return 0
            
            regressive_count = 0
            visited_positions = set()
            
            for current_word in sequence:
                if current_word == 0:
                    continue
                    
                if current_word in visited_positions:
                    regressive_count += 1
                else:
                    visited_positions.add(current_word)
            
            return regressive_count
            
        except (ValueError, AttributeError, TypeError):
            return 0
    
    def _add_stress_classification(self, data):
        """Добавляет классификацию стресса к данным айтрекинга"""
        
        def classify_participant_text(row):
            participant_id = row['RECORDING_SESSION_LABEL']
            text_index = row['INDEX']
            
            if participant_id in self.stress_responders:
                # Для респондеров: тексты 4-6 считаем стрессовыми
                if text_index >= 4:
                    return 'actual_stress'
                else:
                    return 'baseline'
            elif participant_id in self.stress_non_responders:
                # Для нон-респондеров: все тексты считаем нестрессовыми
                if text_index >= 4:
                    return 'intended_stress_no_response'
                else:
                    return 'baseline'
            else:
                # Для участников без данных о стрессе используем старую классификацию
                return 'unknown'
        
        # Применяем классификацию
        data['stress_classification'] = data.apply(classify_participant_text, axis=1)
        
        # Создаем упрощенную группировку
        def get_stress_group(row):
            participant_id = row['RECORDING_SESSION_LABEL']
            if participant_id in self.stress_responders:
                return 'stress_responders'
            elif participant_id in self.stress_non_responders:
                return 'stress_non_responders'
            else:
                return 'unknown'
        
        data['stress_group'] = data.apply(get_stress_group, axis=1)
        
        # Создаем детальную классификацию типов респондеров
        def get_responder_type(row):
            participant_id = row['RECORDING_SESSION_LABEL']
            if participant_id in self.all_participants_stress_data:
                return self.all_participants_stress_data[participant_id]['type']
            else:
                return 'unknown'
        
        data['responder_type'] = data.apply(get_responder_type, axis=1)
        
        return data
    
    def analyze_stress_vs_eyetracking(self, data):
        """Анализирует различия в айтрекинге между стрессовыми и нестрессовыми участниками"""
        print("\n🔬 АНАЛИЗ РАЗЛИЧИЙ В АЙТРЕКИНГЕ: СТРЕСС VS НЕ СТРЕСС...")
        
        # Фильтруем данные для анализа (только тексты 4-6, где должен проявляться стресс)
        stress_phase_data = data[data['INDEX'] >= 4].copy()
        
        # Разделяем на группы
        actual_stress_data = stress_phase_data[stress_phase_data['stress_classification'] == 'actual_stress']
        no_stress_response_data = stress_phase_data[stress_phase_data['stress_classification'] == 'intended_stress_no_response']
        
        print(f"   • Участники с реальным стрессом (тексты 4-6): {len(actual_stress_data)} наблюдений")
        print(f"   • Участники без стрессовой реакции (тексты 4-6): {len(no_stress_response_data)} наблюдений")
        
        # Ключевые показатели для анализа
        key_measures = [
            'AVERAGE_FIXATION_DURATION', 'FIXATION_COUNT', 'FIXATIONS_PER_WORD',
            'AVERAGE_SACCADE_AMPLITUDE', 'SACCADE_COUNT', 'SACCADES_PER_WORD',
            'AVERAGE_BLINK_DURATION', 'BLINK_COUNT',
            'PUPIL_SIZE_MEAN', 'PUPIL_SIZE_MAX',
            'DURATION_PER_WORD', 'TEXT_COVERAGE_PERCENT',
            'REGRESSIVE_SACCADES', 'REGRESSIVE_SACCADES_PERCENT',
            'FIXATIONS_PER_SECOND', 'SACCADES_PER_SECOND'
        ]
        
        # Анализируем каждый показатель
        results = []
        
        for measure in key_measures:
            stress_values = actual_stress_data[measure].dropna()
            no_stress_values = no_stress_response_data[measure].dropna()
            
            if len(stress_values) > 0 and len(no_stress_values) > 0:
                # Статистический тест
                try:
                    t_stat, p_value = ttest_ind(stress_values, no_stress_values)
                    test_name = "t-тест"
                except:
                    u_stat, p_value = mannwhitneyu(stress_values, no_stress_values, alternative='two-sided')
                    test_name = "Mann-Whitney U"
                
                # Cohen's d
                pooled_std = np.sqrt(((len(stress_values) - 1) * stress_values.var() + 
                                     (len(no_stress_values) - 1) * no_stress_values.var()) / 
                                    (len(stress_values) + len(no_stress_values) - 2))
                cohens_d = (stress_values.mean() - no_stress_values.mean()) / pooled_std if pooled_std > 0 else 0
                
                # Процентное изменение
                if no_stress_values.mean() != 0:
                    percent_change = ((stress_values.mean() - no_stress_values.mean()) / no_stress_values.mean()) * 100
                else:
                    percent_change = 0
                
                result = {
                    'measure': measure,
                    'stress_mean': stress_values.mean(),
                    'stress_std': stress_values.std(),
                    'no_stress_mean': no_stress_values.mean(),
                    'no_stress_std': no_stress_values.std(),
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'percent_change': percent_change,
                    'n_stress': len(stress_values),
                    'n_no_stress': len(no_stress_values),
                    'test_name': test_name
                }
                
                results.append(result)
                
                # Выводим результат
                significance = "✅ ЗНАЧИМО" if p_value < 0.05 else "❌ НЕ ЗНАЧИМО"
                effect_size = "большой" if abs(cohens_d) >= 0.8 else "средний" if abs(cohens_d) >= 0.5 else "малый"
                
                print(f"\n   📊 {measure}:")
                print(f"      Стресс: {stress_values.mean():.2f}±{stress_values.std():.2f}")
                print(f"      Нет стресса: {no_stress_values.mean():.2f}±{no_stress_values.std():.2f}")
                print(f"      Изменение: {percent_change:+.1f}% | d = {cohens_d:.3f} ({effect_size})")
                print(f"      {test_name}: p = {p_value:.4f} | {significance}")
        
        return results
    
    def create_stress_integrated_visualizations(self, data, analysis_results):
        """Создает графики интегрированного анализа"""
        print("\n🎨 СОЗДАНИЕ ГРАФИКОВ ИНТЕГРИРОВАННОГО АНАЛИЗА...")
        
        # График 1: Сравнение групп по ключевым показателям
        self._create_group_comparison_plot(data)
        
        # График 2: Динамика по текстам с учетом реального стресса
        self._create_stress_dynamics_plot(data)
        
        # График 3: Тепловая карта различий между группами
        self._create_stress_heatmap(analysis_results)
        
        # График 4: Индивидуальные траектории респондеров vs нон-респондеров
        self._create_individual_trajectories_plot(data)
        
        # График 5: Анализ специфических паттернов чтения
        self._create_reading_patterns_analysis(data)
    
    def _create_group_comparison_plot(self, data):
        """Создает график сравнения групп"""
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        # Ключевые показатели для сравнения
        measures = [
            'AVERAGE_FIXATION_DURATION', 'FIXATIONS_PER_WORD', 'SACCADES_PER_WORD',
            'REGRESSIVE_SACCADES_PERCENT', 'DURATION_PER_WORD', 'TEXT_COVERAGE_PERCENT',
            'PUPIL_SIZE_MEAN', 'AVERAGE_SACCADE_AMPLITUDE', 'BLINKS_PER_SECOND',
            'FIXATIONS_PER_SECOND', 'SACCADES_PER_SECOND', 'REVISITED_WORDS_PERCENT'
        ]
        
        # Фильтруем данные для стрессовой фазы (тексты 4-6)
        stress_phase_data = data[data['INDEX'] >= 4]
        
        for i, measure in enumerate(measures):
            ax = axes[i]
            
            # Подготавливаем данные для boxplot
            stress_responders_data = stress_phase_data[
                stress_phase_data['stress_group'] == 'stress_responders'
            ][measure].dropna()
            
            non_responders_data = stress_phase_data[
                stress_phase_data['stress_group'] == 'stress_non_responders'
            ][measure].dropna()
            
            if len(stress_responders_data) > 0 and len(non_responders_data) > 0:
                bp = ax.boxplot([non_responders_data, stress_responders_data], 
                              labels=['Нон-респондеры', 'Респондеры'],
                              patch_artist=True)
                
                # Раскрашиваем
                bp['boxes'][0].set_facecolor('#3498DB')  # Синий для нон-респондеров
                bp['boxes'][1].set_facecolor('#E74C3C')  # Красный для респондеров
                
                # Добавляем статистику
                try:
                    t_stat, p_value = ttest_ind(non_responders_data, stress_responders_data)
                    ax.text(0.02, 0.98, f'p = {p_value:.3f}', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                except:
                    pass
            
            ax.set_title(f"{measure}")
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('📊 СРАВНЕНИЕ АЙТРЕКИНГА: РЕСПОНДЕРЫ vs НОН-РЕСПОНДЕРЫ (ТЕКСТЫ 4-6)', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'stress_group_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_stress_dynamics_plot(self, data):
        """Создает график динамики с учетом реального стресса"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        measures = [
            'AVERAGE_FIXATION_DURATION', 'FIXATIONS_PER_WORD', 'REGRESSIVE_SACCADES_PERCENT',
            'DURATION_PER_WORD', 'PUPIL_SIZE_MEAN', 'TEXT_COVERAGE_PERCENT'
        ]
        
        for i, measure in enumerate(measures):
            ax = axes[i]
            
            # Респондеры
            responder_data = data[data['stress_group'] == 'stress_responders']
            responder_means = responder_data.groupby('INDEX')[measure].mean()
            responder_stds = responder_data.groupby('INDEX')[measure].std()
            
            # Нон-респондеры
            non_responder_data = data[data['stress_group'] == 'stress_non_responders']
            non_responder_means = non_responder_data.groupby('INDEX')[measure].mean()
            non_responder_stds = non_responder_data.groupby('INDEX')[measure].std()
            
            # График для респондеров
            texts = sorted(responder_means.index)
            ax.plot(texts, [responder_means[t] for t in texts], 
                   'o-', color='#E74C3C', linewidth=3, markersize=8, 
                   label='Респондеры (реальный стресс)')
            ax.fill_between(texts, 
                           [responder_means[t] - responder_stds[t] for t in texts],
                           [responder_means[t] + responder_stds[t] for t in texts],
                           alpha=0.2, color='#E74C3C')
            
            # График для нон-респондеров
            ax.plot(texts, [non_responder_means[t] for t in texts], 
                   'o-', color='#3498DB', linewidth=3, markersize=8,
                   label='Нон-респондеры (нет реакции)')
            ax.fill_between(texts, 
                           [non_responder_means[t] - non_responder_stds[t] for t in texts],
                           [non_responder_means[t] + non_responder_stds[t] for t in texts],
                           alpha=0.2, color='#3498DB')
            
            # Вертикальная линия разделения
            ax.axvline(x=3.5, color='black', linestyle='--', alpha=0.7)
            ax.text(3.5, ax.get_ylim()[1] * 0.9, 'ИНДУКЦИЯ\nСТРЕССА', 
                   ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            ax.set_title(f"{measure}")
            ax.set_xlabel("Номер текста")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xticks(texts)
        
        plt.suptitle('📈 ДИНАМИКА АЙТРЕКИНГА ПО ТЕКСТАМ: РЕАЛЬНЫЙ СТРЕСС VS НЕТ РЕАКЦИИ', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'stress_dynamics_real.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_stress_heatmap(self, analysis_results):
        """Создает тепловую карту различий"""
        # Создаем матрицу для тепловой карты
        measures = [r['measure'] for r in analysis_results]
        effect_sizes = [r['cohens_d'] for r in analysis_results]
        p_values = [r['p_value'] for r in analysis_results]
        
        # Создаем DataFrame для тепловой карты
        heatmap_data = pd.DataFrame({
            'Показатель': measures,
            'Размер эффекта (Cohen\'s d)': effect_sizes,
            'p-значение': p_values,
            'Значимость': ['✅' if p < 0.05 else '❌' for p in p_values]
        })
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
        
        # Тепловая карта размеров эффектов
        effect_matrix = np.array(effect_sizes).reshape(-1, 1)
        sns.heatmap(effect_matrix, 
                   annot=np.array([[d] for d in effect_sizes]),
                   fmt='.3f',
                   yticklabels=measures,
                   xticklabels=['Cohen\'s d'],
                   cmap='RdBu_r', center=0,
                   ax=ax1, cbar_kws={'label': 'Размер эффекта'})
        ax1.set_title('Размеры эффектов (Cohen\'s d)')
        
        # Тепловая карта p-значений
        p_matrix = np.array([-np.log10(p) for p in p_values]).reshape(-1, 1)
        sns.heatmap(p_matrix,
                   annot=np.array([[p] for p in p_values]),
                   fmt='.3f',
                   yticklabels=measures,
                   xticklabels=['-log10(p)'],
                   cmap='viridis',
                   ax=ax2, cbar_kws={'label': '-log10(p-value)'})
        ax2.set_title('Статистическая значимость')
        ax2.axhline(y=len(measures), color='red', linestyle='--', alpha=0.7)
        
        plt.suptitle('🔥 ТЕПЛОВАЯ КАРТА РАЗЛИЧИЙ В АЙТРЕКИНГЕ: СТРЕСС vs НЕ СТРЕСС', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'stress_effect_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_individual_trajectories_plot(self, data):
        """Создает график индивидуальных траекторий"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        key_measures = [
            'AVERAGE_FIXATION_DURATION', 'REGRESSIVE_SACCADES_PERCENT', 
            'DURATION_PER_WORD', 'PUPIL_SIZE_MEAN'
        ]
        
        for i, measure in enumerate(key_measures):
            ax = axes[i]
            
            # Респондеры
            for participant in self.stress_responders.keys():
                participant_data = data[data['RECORDING_SESSION_LABEL'] == participant]
                if not participant_data.empty:
                    texts = participant_data['INDEX'].values
                    values = participant_data[measure].values
                    
                    responder_type = self.stress_responders[participant]['type']
                    if 'strong' in responder_type:
                        color, alpha, linewidth = '#C0392B', 0.8, 2.5
                    elif 'moderate' in responder_type:
                        color, alpha, linewidth = '#E74C3C', 0.7, 2
                    else:
                        color, alpha, linewidth = '#F1948A', 0.6, 1.5
                    
                    ax.plot(texts, values, 'o-', 
                           color=color, alpha=alpha, linewidth=linewidth,
                           label=f'Респ: {participant}' if i == 0 else "")
            
            # Нон-респондеры
            for participant in self.stress_non_responders.keys():
                participant_data = data[data['RECORDING_SESSION_LABEL'] == participant]
                if not participant_data.empty:
                    texts = participant_data['INDEX'].values
                    values = participant_data[measure].values
                    
                    ax.plot(texts, values, 's-', 
                           color='#3498DB', alpha=0.7, linewidth=1.5,
                           label=f'Нон-респ: {participant}' if i == 0 else "")
            
            # Разделительная линия
            ax.axvline(x=3.5, color='black', linestyle='--', alpha=0.7)
            
            ax.set_title(f"{measure}")
            ax.set_xlabel("Номер текста")
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.suptitle('👥 ИНДИВИДУАЛЬНЫЕ ТРАЕКТОРИИ УЧАСТНИКОВ', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'individual_trajectories_stress.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_reading_patterns_analysis(self, data):
        """Создает анализ специфических паттернов чтения"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Анализ паттернов только для текстов 4-6 (стрессовая фаза)
        stress_data = data[data['INDEX'] >= 4]
        
        # 1. Скорость чтения
        ax = axes[0]
        responders = stress_data[stress_data['stress_group'] == 'stress_responders']['DURATION_PER_WORD']
        non_responders = stress_data[stress_data['stress_group'] == 'stress_non_responders']['DURATION_PER_WORD']
        
        ax.hist([responders.dropna(), non_responders.dropna()], 
               bins=15, alpha=0.7, label=['Респондеры', 'Нон-респондеры'],
               color=['#E74C3C', '#3498DB'])
        ax.set_title('Распределение скорости чтения')
        ax.set_xlabel('Время на слово (мс)')
        ax.legend()
        
        # 2. Эффективность чтения
        ax = axes[1]
        responder_coverage = stress_data[stress_data['stress_group'] == 'stress_responders']['TEXT_COVERAGE_PERCENT']
        non_responder_coverage = stress_data[stress_data['stress_group'] == 'stress_non_responders']['TEXT_COVERAGE_PERCENT']
        
        ax.boxplot([non_responder_coverage.dropna(), responder_coverage.dropna()],
                  labels=['Нон-респондеры', 'Респондеры'])
        ax.set_title('Покрытие текста')
        ax.set_ylabel('Процент прочитанных слов')
        
        # 3. Возвратные саккады
        ax = axes[2]
        resp_regressive = stress_data[stress_data['stress_group'] == 'stress_responders']['REGRESSIVE_SACCADES_PERCENT']
        non_resp_regressive = stress_data[stress_data['stress_group'] == 'stress_non_responders']['REGRESSIVE_SACCADES_PERCENT']
        
        ax.scatter(responders.dropna(), resp_regressive.dropna(), 
                  alpha=0.6, color='#E74C3C', s=60, label='Респондеры')
        ax.scatter(non_responders.dropna(), non_resp_regressive.dropna(), 
                  alpha=0.6, color='#3498DB', s=60, label='Нон-респондеры')
        ax.set_xlabel('Время на слово (мс)')
        ax.set_ylabel('Возвратные саккады (%)')
        ax.set_title('Связь скорости и возвратных саккад')
        ax.legend()
        
        # 4. Размер зрачка vs внимание
        ax = axes[3]
        resp_pupil = stress_data[stress_data['stress_group'] == 'stress_responders']['PUPIL_SIZE_MEAN']
        resp_fixations = stress_data[stress_data['stress_group'] == 'stress_responders']['FIXATIONS_PER_WORD']
        non_resp_pupil = stress_data[stress_data['stress_group'] == 'stress_non_responders']['PUPIL_SIZE_MEAN']
        non_resp_fixations = stress_data[stress_data['stress_group'] == 'stress_non_responders']['FIXATIONS_PER_WORD']
        
        ax.scatter(resp_pupil.dropna(), resp_fixations.dropna(), 
                  alpha=0.6, color='#E74C3C', s=60, label='Респондеры')
        ax.scatter(non_resp_pupil.dropna(), non_resp_fixations.dropna(), 
                  alpha=0.6, color='#3498DB', s=60, label='Нон-респондеры')
        ax.set_xlabel('Размер зрачка (у.е.)')
        ax.set_ylabel('Фиксации на слово')
        ax.set_title('Связь зрачка и внимания')
        ax.legend()
        
        # 5. Временные профили
        ax = axes[4]
        for text_idx in [4, 5, 6]:
            text_data = stress_data[stress_data['INDEX'] == text_idx]
            resp_duration = text_data[text_data['stress_group'] == 'stress_responders']['DURATION_PER_WORD'].mean()
            non_resp_duration = text_data[text_data['stress_group'] == 'stress_non_responders']['DURATION_PER_WORD'].mean()
            
            ax.bar([f'T{text_idx}_Респ', f'T{text_idx}_НонРесп'], 
                  [resp_duration, non_resp_duration],
                  color=['#E74C3C', '#3498DB'], alpha=0.7)
        
        ax.set_title('Временные профили по текстам')
        ax.set_ylabel('Время на слово (мс)')
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        # 6. Сводная метрика стресса
        ax = axes[5]
        # Создаем составной индекс стресса для айтрекинга
        stress_data_clean = stress_data.dropna(subset=['DURATION_PER_WORD', 'REGRESSIVE_SACCADES_PERCENT', 'FIXATIONS_PER_WORD'])
        
        # Нормализуем показатели и создаем индекс
        normalized_duration = (stress_data_clean['DURATION_PER_WORD'] - stress_data_clean['DURATION_PER_WORD'].mean()) / stress_data_clean['DURATION_PER_WORD'].std()
        normalized_regressive = (stress_data_clean['REGRESSIVE_SACCADES_PERCENT'] - stress_data_clean['REGRESSIVE_SACCADES_PERCENT'].mean()) / stress_data_clean['REGRESSIVE_SACCADES_PERCENT'].std()
        normalized_fixations = (stress_data_clean['FIXATIONS_PER_WORD'] - stress_data_clean['FIXATIONS_PER_WORD'].mean()) / stress_data_clean['FIXATIONS_PER_WORD'].std()
        
        eyetracking_stress_index = normalized_duration + normalized_regressive + normalized_fixations
        stress_data_clean['eyetracking_stress_index'] = eyetracking_stress_index
        
        resp_index = stress_data_clean[stress_data_clean['stress_group'] == 'stress_responders']['eyetracking_stress_index']
        non_resp_index = stress_data_clean[stress_data_clean['stress_group'] == 'stress_non_responders']['eyetracking_stress_index']
        
        ax.boxplot([non_resp_index.dropna(), resp_index.dropna()],
                  labels=['Нон-респондеры', 'Респондеры'])
        ax.set_title('Составной индекс стресса (айтрекинг)')
        ax.set_ylabel('Индекс стресса (нормализованный)')
        
        plt.suptitle('📖 АНАЛИЗ СПЕЦИФИЧЕСКИХ ПАТТЕРНОВ ЧТЕНИЯ ПРИ СТРЕССЕ', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'reading_patterns_stress.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_integrated_report(self, analysis_results):
        """Генерирует интегрированный отчет"""
        print("\n📋 ГЕНЕРАЦИЯ ИНТЕГРИРОВАННОГО ОТЧЕТА...")
        print("=" * 60)
        
        # Сортируем результаты по размеру эффекта
        sorted_results = sorted(analysis_results, key=lambda x: abs(x['cohens_d']), reverse=True)
        
        print(f"🏆 ТОП-10 ПОКАЗАТЕЛЕЙ АЙТРЕКИНГА, РАЗЛИЧАЮЩИХ СТРЕСС:")
        for i, result in enumerate(sorted_results[:10], 1):
            significance = "✅" if result['p_value'] < 0.05 else "❌"
            effect_size = ("большой" if abs(result['cohens_d']) >= 0.8 else 
                          "средний" if abs(result['cohens_d']) >= 0.5 else "малый")
            
            print(f"   {i:2d}. {result['measure']}")
            print(f"       Изменение при стрессе: {result['percent_change']:+.1f}%")
            print(f"       Размер эффекта: d = {result['cohens_d']:.3f} ({effect_size})")
            print(f"       Значимость: p = {result['p_value']:.4f} {significance}")
            print()
        
        # Статистика по группам
        significant_results = [r for r in analysis_results if r['p_value'] < 0.05]
        large_effects = [r for r in analysis_results if abs(r['cohens_d']) >= 0.8]
        
        print(f"📊 СТАТИСТИЧЕСКАЯ СВОДКА:")
        print(f"   • Всего проанализировано показателей: {len(analysis_results)}")
        print(f"   • Статистически значимых различий: {len(significant_results)} ({len(significant_results)/len(analysis_results)*100:.1f}%)")
        print(f"   • Показателей с большим размером эффекта: {len(large_effects)}")
        
        # Анализ групп участников
        print(f"\n👥 АНАЛИЗ УЧАСТНИКОВ:")
        print(f"   📈 РЕСПОНДЕРЫ (реагировали на стресс):")
        for participant_id, info in self.stress_responders.items():
            print(f"      • {participant_id}: {info['description']}")
        
        print(f"\n   📉 НОН-РЕСПОНДЕРЫ (не реагировали на стресс):")
        for participant_id, info in self.stress_non_responders.items():
            print(f"      • {participant_id}: {info['description']}")
        
        # Выводы и рекомендации
        print(f"\n💡 ГЛАВНЫЕ ВЫВОДЫ:")
        if len(significant_results) > len(analysis_results) * 0.3:
            print(f"   ✅ Айтрекинг ЭФФЕКТИВЕН для детекции реального стресса")
            print(f"   ✅ Обнаружены значимые различия в {len(significant_results)} показателях")
        else:
            print(f"   ⚠️  Айтрекинг показывает ОГРАНИЧЕННУЮ эффективность")
            print(f"   ⚠️  Необходимы дополнительные исследования")
        
        print(f"\n🎯 РЕКОМЕНДАЦИИ ДЛЯ ДЕТЕКЦИИ СТРЕССА:")
        if large_effects:
            print(f"   📊 Фокусироваться на показателях с большими эффектами:")
            for result in large_effects[:3]:
                print(f"      • {result['measure']}: {result['percent_change']:+.1f}% изменение")
        
        print(f"   🔬 Использовать комбинацию айтрекинга и физиологических данных")
        print(f"   📈 Учитывать индивидуальные различия участников")
        
        print("=" * 60)
    
    def run_integrated_analysis(self):
        """Запускает полный интегрированный анализ"""
        print("🔬 ИНТЕГРИРОВАННЫЙ АНАЛИЗ: СТРЕСС (ПОЛИГРАФ) + АЙТРЕКИНГ")
        print("=" * 70)
        
        # 1. Загружаем данные айтрекинга
        eyetracking_data = self.load_eyetracking_data()
        
        # 2. Анализируем различия между группами
        analysis_results = self.analyze_stress_vs_eyetracking(eyetracking_data)
        
        # 3. Создаем визуализации
        self.create_stress_integrated_visualizations(eyetracking_data, analysis_results)
        
        # 4. Генерируем отчет
        self.generate_integrated_report(analysis_results)
        
        print(f"\n✅ ИНТЕГРИРОВАННЫЙ АНАЛИЗ ЗАВЕРШЕН!")
        print(f"📁 Результаты сохранены в: {self.results_dir}")
        print(f"🎨 Создано графиков: 5")
        print(f"📊 Проанализировано показателей: {len(analysis_results)}")
        
        return {
            'eyetracking_data': eyetracking_data,
            'analysis_results': analysis_results,
            'stress_classification': self.all_participants_stress_data
        }


def main():
    """Главная функция"""
    print("👁️  ИНТЕГРИРОВАННЫЙ АНАЛИЗ СТРЕССА И АЙТРЕКИНГА")
    print("=" * 60)
    
    # Создаем и запускаем анализатор
    analyzer = IntegratedStressEyetrackingAnalyzer()
    results = analyzer.run_integrated_analysis()
    
    return results


if __name__ == "__main__":
    main() 