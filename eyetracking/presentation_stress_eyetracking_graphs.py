#!/usr/bin/env python3
"""
Создание презентационных графиков для значимых показателей айтрекинга при стрессе.

Этот скрипт фокусируется только на статистически значимых различиях между 
участниками, которые реагировали на стресс, и теми, кто не реагировал.
Создает красивые графики для презентаций и отчетов.

Основан на результатах интегрированного анализа:
- 4 статистически значимых показателя (p < 0.05)
- 3 показателя с интересными тенденциями
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import warnings
import scipy.stats

# Настройка matplotlib для русского языка и презентационного качества
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Tahoma', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.style.use('default')

# Подавляем warnings
warnings.filterwarnings('ignore')


class PresentationEyetrackingGraphs:
    """Класс для создания презентационных графиков айтрекинга при стрессе"""
    
    def __init__(self, eyetracking_data_path="eyetracking/by_person/data/trial.xls"):
        self.eyetracking_data_path = pathlib.Path(eyetracking_data_path)
        self.results_dir = pathlib.Path("eyetracking/presentation_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # ТОП-4 ЗНАЧИМЫХ ПОКАЗАТЕЛЯ (p < 0.05)
        self.significant_measures = {
            'AVERAGE_BLINK_DURATION': {
                'title': 'Длительность моргания',
                'ylabel': 'Время (мс)',
                'description': 'При стрессе участники моргают быстрее/короче',
                'change': -38.8,
                'effect_size': -0.891,
                'p_value': 0.0160,
                'rank': 1
            },
            'AVERAGE_FIXATION_DURATION': {
                'title': 'Длительность фиксации',
                'ylabel': 'Время (мс)',
                'description': 'При стрессе участники делают более короткие фиксации',
                'change': -7.0,
                'effect_size': -0.795,
                'p_value': 0.0138,
                'rank': 2
            },
            'FIXATIONS_PER_SECOND': {
                'title': 'Частота фиксаций',
                'ylabel': 'Фиксаций в секунду',
                'description': 'При стрессе взгляд движется быстрее',
                'change': +7.2,
                'effect_size': 0.754,
                'p_value': 0.0191,
                'rank': 3
            },
            'SACCADES_PER_SECOND': {
                'title': 'Частота саккад',
                'ylabel': 'Саккад в секунду',
                'description': 'При стрессе больше движений глаз',
                'change': +7.2,
                'effect_size': 0.752,
                'p_value': 0.0193,
                'rank': 4
            }
        }
        
        # Интересные тенденции (близкие к значимости)
        self.trending_measures = {
            'PUPIL_SIZE_MEAN': {
                'title': 'Размер зрачка',
                'ylabel': 'Относительный размер',
                'description': 'Расширение зрачков при стрессе',
                'change': +14.5,
                'effect_size': 0.473,
                'p_value': 0.1332,
            },
            'TEXT_COVERAGE_PERCENT': {
                'title': 'Покрытие текста',
                'ylabel': 'Процент прочитанного (%)',
                'description': 'Больше покрытие текста при стрессе',
                'change': +4.9,
                'effect_size': 0.355,
                'p_value': 0.15,
            },
            'BLINK_COUNT': {
                'title': 'Количество морганий',
                'ylabel': 'Количество',
                'description': 'Меньше морганий при стрессе',
                'change': -24.7,
                'effect_size': -0.311,
                'p_value': 0.20,
            }
        }
        
        # Классификация участников на основе реального анализа стресса
        self.stress_responders = {
            '1707LTA': 'Самая яркая реакция (+31.8%)',
            '1807KNV': 'Резкий пик в 4-м тексте (+150%)',
            '1607KYA': 'Умеренная реакция',
            '1807OVA': 'Умеренная реакция',
            '1807ZUG': 'Слабая реакция',
            '1807SAV': 'Предполагаемый стресс',
            '1807KAN': 'Предполагаемый стресс',
        }
        
        self.stress_non_responders = {
            '1707KAV': 'Сильное снижение стресса (-60%)',
            '1807HEE': 'Значительное снижение (-66.7%)',
            '1807CAA': 'Неожиданное снижение (-26.1%)',
            '1607LVA': 'Стабильный уровень (0%)',
            '1907ZSI': 'Стабильный уровень (0%)',
            '1707DMA': 'Легкое снижение (-10%)',
            '1707SAA': 'Легкое снижение (-9.1%)',
        }
        
        # Цвета для графиков
        self.colors = {
            'responders': '#E74C3C',        # Красный для респондеров
            'non_responders': '#3498DB',    # Синий для нон-респондеров
            'baseline': '#95A5A6',          # Серый для базовой линии
            'stress_phase': '#F39C12',      # Оранжевый для стрессовой фазы
            'significant': '#27AE60',       # Зеленый для значимых результатов
            'trending': '#9B59B6'           # Фиолетовый для тенденций
        }
        
        print(f"📊 Инициализация презентационных графиков:")
        print(f"   • Значимых показателей: {len(self.significant_measures)}")
        print(f"   • Показателей с тенденциями: {len(self.trending_measures)}")
        print(f"   • Респондеров: {len(self.stress_responders)}")
        print(f"   • Нон-респондеров: {len(self.stress_non_responders)}")
    
    def load_and_prepare_data(self):
        """Загружает и подготавливает данные айтрекинга"""
        print("\n📂 ПОДГОТОВКА ДАННЫХ...")
        
        # Читаем данные аналогично основному скрипту
        data = pd.read_csv(self.eyetracking_data_path, sep="\t", encoding="utf-16", header=0)
        data = data.iloc[1:].reset_index(drop=True)  # Удаляем строку с описаниями
        
        # Преобразуем числовые колонки
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
        
        # Создаем метрики
        data['FIXATIONS_PER_SECOND'] = data['FIXATION_COUNT'] / (data['DURATION'] / 1000)
        data['SACCADES_PER_SECOND'] = data['SACCADE_COUNT'] / (data['DURATION'] / 1000)
        data['TEXT_COVERAGE_PERCENT'] = (data['VISITED_INTEREST_AREA_COUNT'] / data['IA_COUNT']) * 100
        
        # Добавляем классификацию стресса
        def classify_participant(row):
            participant_id = row['RECORDING_SESSION_LABEL']
            text_index = row['INDEX']
            
            if participant_id in self.stress_responders:
                return 'stress_responders' if text_index >= 4 else 'baseline_responders'
            elif participant_id in self.stress_non_responders:
                return 'stress_non_responders' if text_index >= 4 else 'baseline_non_responders'
            else:
                return 'unknown'
        
        data['stress_group'] = data.apply(classify_participant, axis=1)
        
        print(f"   • Загружено записей: {len(data)}")
        print(f"   • Уникальных участников: {data['RECORDING_SESSION_LABEL'].nunique()}")
        print(f"   • Распределение по группам стресса:")
        for group, count in data['stress_group'].value_counts().items():
            print(f"     - {group}: {count}")
        
        return data
    
    def create_top_measures_comparison(self, data):
        """Создает график сравнения ТОП-4 значимых показателей"""
        print("\n🏆 Создание графика ТОП-4 значимых показателей...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Фильтруем данные для стрессовой фазы (тексты 4-6)
        stress_data = data[data['INDEX'] >= 4]
        
        for i, (measure, info) in enumerate(self.significant_measures.items()):
            ax = axes[i]
            
            # Подготавливаем данные
            responders_data = stress_data[
                stress_data['stress_group'] == 'stress_responders'
            ][measure].dropna()
            
            non_responders_data = stress_data[
                stress_data['stress_group'] == 'stress_non_responders'
            ][measure].dropna()
            
            # Создаем красивый boxplot
            bp = ax.boxplot([non_responders_data, responders_data], 
                          labels=['Нон-респондеры', 'Респондеры'],
                          patch_artist=True,
                          boxprops=dict(linewidth=2),
                          whiskerprops=dict(linewidth=2),
                          capprops=dict(linewidth=2),
                          medianprops=dict(linewidth=3, color='white'))
            
            # Раскрашиваем боксы
            bp['boxes'][0].set_facecolor(self.colors['non_responders'])
            bp['boxes'][0].set_alpha(0.8)
            bp['boxes'][1].set_facecolor(self.colors['responders'])
            bp['boxes'][1].set_alpha(0.8)
            
            # Добавляем ранжирование
            rank_text = f"🥇 №{info['rank']}\n{info['change']:+.1f}%"
            
            ax.text(0.02, 0.98, rank_text, 
                   transform=ax.transAxes, 
                   verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', 
                           facecolor=self.colors['significant'], 
                           alpha=0.8),
                   fontsize=12, color='white', fontweight='bold')
            
            # Настройка осей и заголовков
            ax.set_title(f"{info['title']}", fontsize=14, fontweight='bold', pad=20)
            ax.set_ylabel(info['ylabel'], fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#FAFAFA')
            
            # Добавляем описание
            ax.text(0.5, -0.15, info['description'], 
                   transform=ax.transAxes, ha='center',
                   fontsize=10, style='italic',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.suptitle('🏆 ТОП-4 ЗНАЧИМЫХ ПОКАЗАТЕЛЯ АЙТРЕКИНГА ПРИ СТРЕССЕ', 
                     fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, bottom=0.15)
        plt.savefig(self.results_dir / 'top_4_significant_measures.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def create_effect_sizes_chart(self, data):
        """Создает график размеров эффектов"""
        print("\n📊 Создание графика размеров эффектов...")
        
        # Объединяем все показатели
        all_measures = {**self.significant_measures, **self.trending_measures}
        
        # Подготавливаем данные для графика
        measures_names = []
        effect_sizes = []
        p_values = []
        categories = []
        
        for measure, info in all_measures.items():
            measures_names.append(info['title'])
            effect_sizes.append(info['effect_size'])
            p_values.append(info['p_value'])
            if measure in self.significant_measures:
                categories.append('Значимые (p < 0.05)')
            else:
                categories.append('Тенденции (p > 0.05)')
        
        # Создаем график
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Цвета в зависимости от категории
        colors = [self.colors['significant'] if cat == 'Значимые (p < 0.05)' 
                 else self.colors['trending'] for cat in categories]
        
        # Горизонтальный bar chart
        bars = ax.barh(measures_names, effect_sizes, color=colors, alpha=0.8, height=0.6)
        
        # Добавляем значения эффектов на столбцы
        for i, (bar, effect) in enumerate(zip(bars, effect_sizes)):
            width = bar.get_width()
            ax.text(width + (0.05 if width >= 0 else -0.05), bar.get_y() + bar.get_height()/2,
                   f'{effect:.2f}',
                   ha='left' if width >= 0 else 'right', va='center',
                   fontweight='bold', fontsize=11,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        
        # Вертикальные линии для интерпретации размера эффекта
        ax.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
        ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.9)
        ax.axvline(x=-0.2, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=-0.5, color='gray', linestyle='--', alpha=0.7)
        ax.axvline(x=-0.8, color='gray', linestyle='--', alpha=0.9)
        
        # Добавляем текстовые метки для интерпретации
        ax.text(0.2, len(measures_names), 'малый', ha='center', va='bottom', alpha=0.7)
        ax.text(0.5, len(measures_names), 'средний', ha='center', va='bottom', alpha=0.7)
        ax.text(0.8, len(measures_names), 'большой', ha='center', va='bottom', alpha=0.7)
        
        ax.set_xlabel('Размер эффекта (Cohen\'s d)', fontsize=14, fontweight='bold')
        ax.set_title('📏 РАЗМЕРЫ ЭФФЕКТОВ ПО ПОКАЗАТЕЛЯМ АЙТРЕКИНГА', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Легенда
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.colors['significant'], alpha=0.8, label='Значимые различия (p < 0.05)'),
            Patch(facecolor=self.colors['trending'], alpha=0.8, label='Интересные тенденции (p > 0.05)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=12)
        
        ax.grid(axis='x', alpha=0.3)
        ax.set_facecolor('#FAFAFA')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'effect_sizes_chart.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def create_dynamics_timeline(self, data):
        """Создает график динамики по текстам для ТОП-2 показателей"""
        print("\n📈 Создание графика временной динамики...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Берем второй и третий значимые показатели (убираем длительность моргания)
        all_measures = list(self.significant_measures.items())
        top_measures = [all_measures[1], all_measures[2]]  # AVERAGE_FIXATION_DURATION и FIXATIONS_PER_SECOND
        
        for i, (measure, info) in enumerate(top_measures):
            ax = axes[i]
            
            # Респондеры
            responder_data = data[data['RECORDING_SESSION_LABEL'].isin(self.stress_responders.keys())]
            responder_means = responder_data.groupby('INDEX')[measure].mean()
            responder_stds = responder_data.groupby('INDEX')[measure].std()
            responder_counts = responder_data.groupby('INDEX')[measure].count()
            responder_sems = responder_stds / np.sqrt(responder_counts)
            
            # Нон-респондеры
            non_responder_data = data[data['RECORDING_SESSION_LABEL'].isin(self.stress_non_responders.keys())]
            non_responder_means = non_responder_data.groupby('INDEX')[measure].mean()
            non_responder_stds = non_responder_data.groupby('INDEX')[measure].std()
            non_responder_counts = non_responder_data.groupby('INDEX')[measure].count()
            non_responder_sems = non_responder_stds / np.sqrt(non_responder_counts)
            
            # График для респондеров
            texts = sorted(responder_means.index)
            ax.plot(texts, [responder_means[t] for t in texts], 
                   'o-', color=self.colors['responders'], linewidth=4, 
                   markersize=10, markerfacecolor='white', markeredgewidth=2,
                   label='Респондеры (реальный стресс)')
            ax.fill_between(texts, 
                           [responder_means[t] - responder_sems[t] for t in texts],
                           [responder_means[t] + responder_sems[t] for t in texts],
                           alpha=0.2, color=self.colors['responders'])
            
            # График для нон-респондеров
            ax.plot(texts, [non_responder_means[t] for t in texts], 
                   's-', color=self.colors['non_responders'], linewidth=4, 
                   markersize=10, markerfacecolor='white', markeredgewidth=2,
                   label='Нон-респондеры (нет реакции)')
            ax.fill_between(texts, 
                           [non_responder_means[t] - non_responder_sems[t] for t in texts],
                           [non_responder_means[t] + non_responder_sems[t] for t in texts],
                           alpha=0.2, color=self.colors['non_responders'])
            
            # Фоновые зоны
            ax.axvspan(-0.5, 3.5, alpha=0.1, color='green', label='Базовая линия')
            ax.axvspan(3.5, 6.5, alpha=0.1, color='red', label='Стрессовая фаза')
            
            # Вертикальная линия разделения
            ax.axvline(x=3.5, color='black', linestyle='--', linewidth=2, alpha=0.8)
            
            # Аннотация
            y_range = ax.get_ylim()
            ax.annotate('ИНДУКЦИЯ СТРЕССА', xy=(3.5, y_range[1] * 0.9), 
                       xytext=(4.5, y_range[1] * 0.9),
                       arrowprops=dict(arrowstyle='->', color='black', lw=2),
                       fontsize=12, fontweight='bold', ha='center')
            
            # Информация о показателе
            info_text = f"№{info['rank']}\n{info['change']:+.1f}%"
            
            ax.text(0.02, 0.98, info_text, 
                   transform=ax.transAxes, 
                   verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', 
                           facecolor=self.colors['significant'], 
                           alpha=0.9),
                   fontsize=12, color='white', fontweight='bold')
            
            ax.set_title(f"{info['title']}", fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel("Номер текста", fontsize=14)
            ax.set_ylabel(info['ylabel'], fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#FAFAFA')
            ax.set_xticks(texts)
            ax.legend(loc='upper right' if i == 0 else 'lower right', fontsize=10)
        
        plt.suptitle('📈 ВРЕМЕННАЯ ДИНАМИКА КЛЮЧЕВЫХ ПОКАЗАТЕЛЕЙ', 
                     fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'key_dynamics_timeline.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def create_summary_infographic(self):
        """Создает итоговую инфографику с ключевыми результатами"""
        print("\n🎨 Создание итоговой инфографики...")
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Заголовок
        title_ax = fig.add_subplot(gs[0, :])
        title_ax.text(0.5, 0.5, '👁️ КЛЮЧЕВЫЕ РЕЗУЛЬТАТЫ: СТРЕСС И АЙТРЕКИНГ', 
                     ha='center', va='center', fontsize=24, fontweight='bold',
                     transform=title_ax.transAxes)
        title_ax.axis('off')
        
        # Статистика
        stats_ax = fig.add_subplot(gs[1, 0])
        stats_text = ("📊 СТАТИСТИКА\n\n"
                     f"• Всего показателей: 16\n"
                     f"• Значимых различий: 4 (25%)\n"
                     f"• Респондеров: {len(self.stress_responders)}\n"
                     f"• Нон-респондеров: {len(self.stress_non_responders)}")
        
        stats_ax.text(0.1, 0.9, stats_text, ha='left', va='top', 
                     transform=stats_ax.transAxes, fontsize=12,
                     bbox=dict(boxstyle='round,pad=0.5', facecolor=self.colors['significant'], alpha=0.8),
                     color='white', fontweight='bold')
        stats_ax.axis('off')
        
        # ТОП показатель (длительность фиксации вместо моргания)
        top1_ax = fig.add_subplot(gs[1, 1])
        top1_measure = list(self.significant_measures.items())[1]  # Берем второй показатель
        top1_text = (f"🥇 ЛИДЕР\n\n"
                    f"{top1_measure[1]['title']}\n"
                    f"Изменение: {top1_measure[1]['change']:+.1f}%\n"
                    f"Высокая значимость")
        
        top1_ax.text(0.1, 0.9, top1_text, ha='left', va='top',
                    transform=top1_ax.transAxes, fontsize=12,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=self.colors['responders'], alpha=0.8),
                    color='white', fontweight='bold')
        top1_ax.axis('off')
        
        # Интересные тенденции
        trend_ax = fig.add_subplot(gs[1, 2])
        trend_text = ("🤔 ТЕНДЕНЦИИ\n\n"
                     "• Зрачки расширяются (+14.5%)\n"
                     "• Больше покрытие текста (+4.9%)\n"
                     "• Меньше морганий (-24.7%)")
        
        trend_ax.text(0.1, 0.9, trend_text, ha='left', va='top',
                     transform=trend_ax.transAxes, fontsize=12,
                     bbox=dict(boxstyle='round,pad=0.5', facecolor=self.colors['trending'], alpha=0.8),
                     color='white', fontweight='bold')
        trend_ax.axis('off')
        
        # Выводы
        conclusion_ax = fig.add_subplot(gs[1, 3])
        conclusion_text = ("💡 ВЫВОД\n\n"
                          "Айтрекинг показывает\n"
                          "ОГРАНИЧЕННУЮ\n"
                          "эффективность для\n"
                          "детекции стресса")
        
        conclusion_ax.text(0.1, 0.9, conclusion_text, ha='left', va='top',
                          transform=conclusion_ax.transAxes, fontsize=12,
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='orange', alpha=0.8),
                          color='white', fontweight='bold')
        conclusion_ax.axis('off')
        
        # Диаграмма размеров эффектов (упрощенная)
        effects_ax = fig.add_subplot(gs[2:, :2])
        
        measures = [info['title'] for info in self.significant_measures.values()]
        effects = [info['effect_size'] for info in self.significant_measures.values()]
        colors_list = [self.colors['responders'] if e > 0 else self.colors['non_responders'] for e in effects]
        
        bars = effects_ax.barh(measures, effects, color=colors_list, alpha=0.8)
        
        for bar, effect in zip(bars, effects):
            width = bar.get_width()
            effects_ax.text(width + (0.02 if width >= 0 else -0.02), 
                          bar.get_y() + bar.get_height()/2,
                          f'{effect:.2f}', ha='left' if width >= 0 else 'right', 
                          va='center', fontweight='bold', fontsize=10)
        
        effects_ax.set_xlabel('Размер эффекта (Cohen\'s d)', fontsize=12, fontweight='bold')
        effects_ax.set_title('📏 Размеры эффектов значимых показателей', 
                           fontsize=14, fontweight='bold')
        effects_ax.axvline(x=0, color='black', linewidth=1)
        effects_ax.grid(axis='x', alpha=0.3)
        
        # Рекомендации
        recommendations_ax = fig.add_subplot(gs[2:, 2:])
        rec_text = ("🎯 РЕКОМЕНДАЦИИ ДЛЯ ПРАКТИКИ:\n\n"
                   "1. Фокус на временных характеристиках:\n"
                   "   • Длительность морганий ⬇️\n"
                   "   • Длительность фиксаций ⬇️\n"
                   "   • Частота движений глаз ⬆️\n\n"
                   "2. Комбинировать с физиологическими данными\n\n"
                   "3. Учитывать индивидуальные различия\n\n"
                   "4. Создать составной индекс стресса")
        
        recommendations_ax.text(0.05, 0.95, rec_text, ha='left', va='top',
                              transform=recommendations_ax.transAxes, fontsize=11,
                              bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        recommendations_ax.axis('off')
        
        plt.savefig(self.results_dir / 'summary_infographic.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def generate_presentation_summary(self):
        """Генерирует краткую сводку для презентации"""
        print("\n📋 КРАТКАЯ СВОДКА ДЛЯ ПРЕЗЕНТАЦИИ:")
        print("=" * 60)
        
        print("🏆 ТОП-4 ЗНАЧИМЫХ ПОКАЗАТЕЛЯ:")
        for i, (measure, info) in enumerate(self.significant_measures.items(), 1):
            effect_size_desc = ("БОЛЬШОЙ" if abs(info['effect_size']) >= 0.8 else 
                              "средний" if abs(info['effect_size']) >= 0.5 else "малый")
            
            print(f"   {i}. {info['title']} ({measure})")
            print(f"      • Изменение: {info['change']:+.1f}%")
            print(f"      • Размер эффекта: {effect_size_desc}")
            print(f"      • Статистически значимо ✅")
            print(f"      • {info['description']}")
            print()
        
        print("🤔 ИНТЕРЕСНЫЕ ТЕНДЕНЦИИ:")
        for measure, info in self.trending_measures.items():
            print(f"   • {info['title']}: {info['change']:+.1f}%")
            print(f"     {info['description']}")
        
        print("\n💡 ГЛАВНЫЙ ВЫВОД:")
        print("   Айтрекинг показывает ОГРАНИЧЕННУЮ эффективность для детекции стресса.")
        print("   Только 25% показателей (4 из 16) статистически значимо различаются.")
        print("   Наиболее информативны ВРЕМЕННЫЕ характеристики движений глаз.")
        
        print("\n🎯 ДЛЯ ПРЕЗЕНТАЦИИ ИСПОЛЬЗУЙТЕ:")
        print("   1. top_4_significant_measures.png - основной график")
        print("   2. effect_sizes_chart.png - размеры эффектов") 
        print("   3. key_dynamics_timeline.png - временная динамика")
        print("   4. summary_infographic.png - итоговая инфографика")
        
        print("=" * 60)
    
    def run_presentation_analysis(self):
        """Запускает создание презентационных графиков"""
        print("🎨 СОЗДАНИЕ ПРЕЗЕНТАЦИОННЫХ ГРАФИКОВ: АЙТРЕКИНГ И СТРЕСС")
        print("=" * 70)
        
        # 1. Загружаем и подготавливаем данные
        data = self.load_and_prepare_data()
        
        # 2. Создаем презентационные графики
        self.create_top_measures_comparison(data)
        self.create_effect_sizes_chart(data)
        self.create_dynamics_timeline(data)
        self.create_summary_infographic()
        
        # 3. Генерируем сводку
        self.generate_presentation_summary()
        
        print(f"\n✅ ПРЕЗЕНТАЦИОННЫЕ ГРАФИКИ СОЗДАНЫ!")
        print(f"📁 Результаты сохранены в: {self.results_dir}")
        print(f"🎨 Создано графиков: 4")
        print(f"📊 Проанализировано значимых показателей: {len(self.significant_measures)}")
        
        return data


def main():
    """Главная функция"""
    print("🎨 СОЗДАНИЕ ПРЕЗЕНТАЦИОННЫХ ГРАФИКОВ: ЗНАЧИМЫЕ ПОКАЗАТЕЛИ АЙТРЕКИНГА")
    print("=" * 80)
    
    # Создаем и запускаем генератор графиков
    generator = PresentationEyetrackingGraphs()
    data = generator.run_presentation_analysis()
    
    return data


if __name__ == "__main__":
    main() 