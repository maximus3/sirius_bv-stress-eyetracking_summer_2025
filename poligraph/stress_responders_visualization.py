#!/usr/bin/env python3
"""
Скрипт для визуализации физиологических показателей респондеров на стресс.
Показывает разницу между базовой линией (тексты 1-3) и периодом стресса (тексты 4-6).

Запуск: uv run poligraph/stress_responders_visualization.py
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy import stats
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# Настройка matplotlib для русского языка и красивых графиков
plt.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
sns.set_style("whitegrid")
sns.set_palette("husl")

class StressRespondersVisualizer:
    """Класс для визуализации физиологических показателей при стрессе"""
    
    def __init__(self, data_path: str = "poligraph/data"):
        self.data_path = pathlib.Path(data_path)
        
        # Участники для анализа
        self.responders = ['1707LTA', '1807KNV', '1607KYA', '1807OVA', '1807ZUG']
        
        # Настройки графиков
        self.colors = {
            'baseline': '#2E86AB',    # Синий для базовой линии
            'stress': '#F24236',      # Красный для стресса
            'scr': '#A23B72',         # КГР
            'hr': '#F18F01',          # ЧСС 
            'ppg': '#C73E1D'          # ФПГ
        }
        
    def extract_participant_id(self, filename: str) -> str:
        """Извлекает ID участника из имени файла"""
        pattern = r'(\d{4}[A-Z]{3})_exp1'
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
        return None
    
    def load_physiological_data(self) -> pd.DataFrame:
        """Загружает физиологические данные из нормализованного файла"""
        normalized_file = self.data_path / "result" / "Signal_Analysis_Results_Normalized.xlsx"
        
        try:
            df = pd.read_excel(normalized_file)
            df['Participant_ID'] = df['File'].apply(self.extract_participant_id)
            df = df[df['Participant_ID'].notna()]
            
            # Фильтруем только респондеров
            df = df[df['Participant_ID'].isin(self.responders)]
            
            # Конвертируем Label в числовой формат для текстов 1-6
            df['Text_Number'] = pd.to_numeric(df['Label'], errors='coerce')
            df = df[df['Text_Number'].between(1, 6)]
            
            # Добавляем категорию периода
            df['Period'] = df['Text_Number'].apply(
                lambda x: 'Базовая линия (1-3)' if x <= 3 else 'Стресс (4-6)'
            )
            
            print(f"Загружены данные {len(df)} записей для {len(df['Participant_ID'].unique())} респондеров")
            return df
            
        except Exception as e:
            print(f"Ошибка загрузки данных: {e}")
            return pd.DataFrame()
    
    def calculate_stress_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Рассчитывает комплексный индекс стресса"""
        metrics = []
        
        for participant in self.responders:
            if participant not in df['Participant_ID'].values:
                continue
                
            participant_data = df[df['Participant_ID'] == participant]
            
            for text_num in range(1, 7):
                text_data = participant_data[participant_data['Text_Number'] == text_num]
                if text_data.empty:
                    continue
                
                row = text_data.iloc[0]
                
                # Извлекаем нормализованные показатели
                scr_mean = row.get('scr r_Mean', 0)
                scr_line = row.get('scr r_Line_Length', 0)
                hr_mean = row.get('HR (calculated)_Mean', 0)
                hr_line = row.get('HR (calculated)_Line_Length', 0)
                ppg_mean = row.get('ppg r_Mean', 0)
                ppg_line = row.get('ppg r_Line_Length', 0)
                
                # Составной индекс стресса (z-score)
                stress_index = (scr_mean + scr_line + hr_mean + hr_line + ppg_mean + ppg_line) / 6
                
                metrics.append({
                    'Participant_ID': participant,
                    'Text_Number': text_num,
                    'Period': 'Базовая линия (1-3)' if text_num <= 3 else 'Стресс (4-6)',
                    'SCR_Mean': scr_mean,
                    'SCR_LineLength': scr_line,
                    'HR_Mean': hr_mean,
                    'HR_LineLength': hr_line,
                    'PPG_Mean': ppg_mean,
                    'PPG_LineLength': ppg_line,
                    'Stress_Index': stress_index
                })
        
        return pd.DataFrame(metrics)
    
    def create_individual_trajectories_plot(self, metrics_df: pd.DataFrame) -> None:
        """Создает график индивидуальных траекторий стресса"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Индивидуальные траектории физиологических показателей респондеров', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        axes = axes.flatten()
        
        # Показатели для отображения
        metrics_to_plot = [
            ('SCR_Mean', 'Средняя КГР', 'scr'),
            ('SCR_LineLength', 'Активность КГР', 'scr'),
            ('HR_Mean', 'Средняя ЧСС', 'hr'),
            ('HR_LineLength', 'Активность ЧСС', 'hr'),
            ('PPG_Mean', 'Средняя ФПГ', 'ppg'),
            ('Stress_Index', 'Индекс стресса', 'stress')
        ]
        
        for idx, (metric, title, color_key) in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            # Рисуем траектории для каждого респондера
            for participant in self.responders:
                participant_data = metrics_df[metrics_df['Participant_ID'] == participant]
                if not participant_data.empty:
                    ax.plot(participant_data['Text_Number'], participant_data[metric], 
                           marker='o', linewidth=2, alpha=0.7, label=participant)
            
            # Средняя траектория
            mean_data = metrics_df.groupby('Text_Number')[metric].mean()
            ax.plot(mean_data.index, mean_data.values, 
                   color='black', linewidth=3, marker='s', markersize=8, 
                   label='Среднее', alpha=0.9)
            
            # Выделяем периоды цветом фона
            ax.axvspan(1, 3.5, alpha=0.2, color=self.colors['baseline'], label='Базовая линия')
            ax.axvspan(3.5, 6, alpha=0.2, color=self.colors['stress'], label='Стресс')
            
            ax.set_title(title, fontweight='bold', fontsize=12)
            ax.set_xlabel('Номер текста', fontsize=10)
            ax.set_ylabel('Z-score', fontsize=10)
            ax.set_xticks(range(1, 7))
            ax.grid(True, alpha=0.3)
            
            if idx == 0:  # Легенда только на первом графике
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        
        # Сохраняем график
        output_path = self.data_path / "result" / "responders_individual_trajectories.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"График индивидуальных траекторий сохранен: {output_path}")
        plt.show()
    
    def create_period_comparison_plot(self, metrics_df: pd.DataFrame) -> None:
        """Создает график сравнения периодов базовой линии и стресса"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Сравнение физиологических показателей: Базовая линия vs Стресс', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        axes = axes.flatten()
        
        metrics_to_plot = [
            ('SCR_Mean', 'Средняя КГР'),
            ('SCR_LineLength', 'Активность КГР'),
            ('HR_Mean', 'Средняя ЧСС'),
            ('HR_LineLength', 'Активность ЧСС'),
            ('PPG_Mean', 'Средняя ФПГ'),
            ('Stress_Index', 'Индекс стресса')
        ]
        
        for idx, (metric, title) in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            # Box plot для сравнения периодов
            sns.boxplot(data=metrics_df, x='Period', y=metric, ax=ax,
                       palette=[self.colors['baseline'], self.colors['stress']])
            
            # Добавляем точки для каждого участника
            sns.stripplot(data=metrics_df, x='Period', y=metric, ax=ax,
                         color='black', alpha=0.6, size=4)
            
            # Статистический тест
            baseline = metrics_df[metrics_df['Period'] == 'Базовая линия (1-3)'][metric]
            stress = metrics_df[metrics_df['Period'] == 'Стресс (4-6)'][metric]
            
            if len(baseline) > 0 and len(stress) > 0:
                t_stat, p_val = stats.ttest_ind(baseline, stress)
                sig_text = f'p = {p_val:.4f}'
                if p_val < 0.001:
                    sig_text += ' ***'
                elif p_val < 0.01:
                    sig_text += ' **'
                elif p_val < 0.05:
                    sig_text += ' *'
                
                ax.text(0.5, 0.95, sig_text, transform=ax.transAxes, 
                       ha='center', va='top', fontsize=10, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(title, fontweight='bold', fontsize=12)
            ax.set_xlabel('')
            ax.set_ylabel('Z-score', fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Сохраняем график
        output_path = self.data_path / "result" / "responders_period_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"График сравнения периодов сохранен: {output_path}")
        plt.show()
    
    def create_heatmap_plot(self, metrics_df: pd.DataFrame) -> None:
        """Создает тепловую карту изменений показателей"""
        # Подготавливаем данные для тепловой карты
        pivot_data = metrics_df.pivot_table(
            values=['SCR_Mean', 'SCR_LineLength', 'HR_Mean', 'HR_LineLength', 
                   'PPG_Mean', 'Stress_Index'],
            index='Participant_ID',
            columns='Text_Number',
            aggfunc='mean'
        )
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Тепловые карты физиологических показателей по текстам', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        axes = axes.flatten()
        
        metrics_to_plot = [
            ('SCR_Mean', 'Средняя КГР'),
            ('SCR_LineLength', 'Активность КГР'),
            ('HR_Mean', 'Средняя ЧСС'),
            ('HR_LineLength', 'Активность ЧСС'),
            ('PPG_Mean', 'Средняя ФПГ'),
            ('Stress_Index', 'Индекс стресса')
        ]
        
        for idx, (metric, title) in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            # Получаем данные для этого показателя
            heatmap_data = pivot_data[metric]
            
            # Создаем тепловую карту
            sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlBu_r',
                       center=0, ax=ax, cbar_kws={'label': 'Z-score'})
            
            ax.set_title(title, fontweight='bold', fontsize=12)
            ax.set_xlabel('Номер текста', fontsize=10)
            ax.set_ylabel('Участник', fontsize=10)
            
            # Добавляем разделительную линию между периодами
            ax.axvline(x=3, color='white', linewidth=3)
        
        plt.tight_layout()
        
        # Сохраняем график
        output_path = self.data_path / "result" / "responders_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Тепловая карта сохранена: {output_path}")
        plt.show()
    
    def create_summary_statistics_plot(self, metrics_df: pd.DataFrame) -> None:
        """Создает график с суммарной статистикой"""
        # Рассчитываем средние значения по периодам
        summary_stats = metrics_df.groupby('Period').agg({
            'SCR_Mean': ['mean', 'std'],
            'SCR_LineLength': ['mean', 'std'],
            'HR_Mean': ['mean', 'std'], 
            'HR_LineLength': ['mean', 'std'],
            'PPG_Mean': ['mean', 'std'],
            'Stress_Index': ['mean', 'std']
        })
        
        # Упрощаем названия колонок
        summary_stats.columns = [f'{metric}_{stat}' for metric, stat in summary_stats.columns]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Суммарная статистика физиологических показателей', 
                     fontsize=16, fontweight='bold')
        
        # График средних значений
        ax1 = axes[0]
        metrics = ['SCR_Mean', 'SCR_LineLength', 'HR_Mean', 'HR_LineLength', 'PPG_Mean', 'Stress_Index']
        
        baseline_means = [summary_stats.loc['Базовая линия (1-3)', f'{m}_mean'] for m in metrics]
        stress_means = [summary_stats.loc['Стресс (4-6)', f'{m}_mean'] for m in metrics]
        baseline_stds = [summary_stats.loc['Базовая линия (1-3)', f'{m}_std'] for m in metrics]
        stress_stds = [summary_stats.loc['Стресс (4-6)', f'{m}_std'] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax1.bar(x - width/2, baseline_means, width, yerr=baseline_stds, 
               label='Базовая линия (1-3)', color=self.colors['baseline'], alpha=0.7)
        ax1.bar(x + width/2, stress_means, width, yerr=stress_stds,
               label='Стресс (4-6)', color=self.colors['stress'], alpha=0.7)
        
        ax1.set_xlabel('Физиологические показатели')
        ax1.set_ylabel('Среднее значение (Z-score)')
        ax1.set_title('Средние значения по периодам')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['КГР\nсредняя', 'КГР\nактивность', 'ЧСС\nсредняя', 
                            'ЧСС\nактивность', 'ФПГ\nсредняя', 'Индекс\nстресса'], 
                           rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # График изменений (разности)
        ax2 = axes[1]
        changes = [stress_means[i] - baseline_means[i] for i in range(len(metrics))]
        colors_bars = [self.colors['stress'] if change > 0 else self.colors['baseline'] for change in changes]
        
        bars = ax2.bar(x, changes, color=colors_bars, alpha=0.7)
        ax2.set_xlabel('Физиологические показатели')
        ax2.set_ylabel('Изменение (Z-score)')
        ax2.set_title('Изменения при переходе к стрессу')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['КГР\nсредняя', 'КГР\nактивность', 'ЧСС\nсредняя', 
                            'ЧСС\nактивность', 'ФПГ\nсредняя', 'Индекс\nстресса'], 
                           rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.8)
        
        # Добавляем значения на столбцы
        for bar, change in zip(bars, changes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{change:.2f}',
                    ha='center', va='bottom' if height > 0 else 'top')
        
        plt.tight_layout()
        
        # Сохраняем график
        output_path = self.data_path / "result" / "responders_summary_statistics.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"График суммарной статистики сохранен: {output_path}")
        plt.show()
    
    def print_summary_report(self, metrics_df: pd.DataFrame) -> None:
        """Выводит текстовый отчет по результатам"""
        print("\n" + "="*60)
        print("ОТЧЕТ: ФИЗИОЛОГИЧЕСКИЕ ПОКАЗАТЕЛИ РЕСПОНДЕРОВ НА СТРЕСС")
        print("="*60)
        
        print(f"\n🎯 Участники-респондеры ({len(self.responders)}):")
        for i, participant in enumerate(self.responders, 1):
            print(f"  {i}. {participant}")
        
        print("\n📊 Сравнение периодов:")
        
        # Статистика по периодам
        baseline_data = metrics_df[metrics_df['Period'] == 'Базовая линия (1-3)']
        stress_data = metrics_df[metrics_df['Period'] == 'Стресс (4-6)']
        
        metrics_to_analyze = [
            ('SCR_Mean', 'КГР средняя'),
            ('SCR_LineLength', 'КГР активность'),
            ('HR_Mean', 'ЧСС средняя'),
            ('HR_LineLength', 'ЧСС активность'),
            ('PPG_Mean', 'ФПГ средняя'),
            ('Stress_Index', 'Индекс стресса')
        ]
        
        for metric, name in metrics_to_analyze:
            baseline_mean = baseline_data[metric].mean()
            stress_mean = stress_data[metric].mean()
            change = stress_mean - baseline_mean
            change_pct = (change / abs(baseline_mean) * 100) if abs(baseline_mean) > 0.001 else 0
            
            # Статистический тест
            t_stat, p_val = stats.ttest_ind(baseline_data[metric], stress_data[metric])
            
            direction = "↑" if change > 0 else "↓"
            significance = ""
            if p_val < 0.001:
                significance = " ***"
            elif p_val < 0.01:
                significance = " **"
            elif p_val < 0.05:
                significance = " *"
            
            print(f"\n  • {name}:")
            print(f"    Базовая линия: {baseline_mean:.3f}")
            print(f"    Стресс:        {stress_mean:.3f}")
            print(f"    Изменение:     {change:+.3f} ({change_pct:+.1f}%) {direction}")
            print(f"    p-значение:    {p_val:.4f}{significance}")
        
        print(f"\n🔬 Общие выводы:")
        print(f"  • Проанализировано записей: {len(metrics_df)}")
        print(f"  • Период базовой линии: {len(baseline_data)} записей")
        print(f"  • Период стресса: {len(stress_data)} записей")
        
        # Определяем наиболее значимые изменения
        significant_increases = []
        for metric, name in metrics_to_analyze:
            baseline_mean = baseline_data[metric].mean()
            stress_mean = stress_data[metric].mean()
            change = stress_mean - baseline_mean
            t_stat, p_val = stats.ttest_ind(baseline_data[metric], stress_data[metric])
            
            if change > 0 and p_val < 0.05:
                significant_increases.append((name, change, p_val))
        
        if significant_increases:
            print(f"\n  ✅ Значимые увеличения при стрессе:")
            for name, change, p_val in sorted(significant_increases, key=lambda x: x[1], reverse=True):
                print(f"    - {name}: +{change:.3f} (p={p_val:.4f})")
        
        print("\n" + "="*60)
    
    def create_presentation_plots(self, df: pd.DataFrame) -> None:
        """Создает самый информативный презентационный график"""
        
        # Цвета для презентации
        colors = {
            'baseline': '#3498db',  # Синий для базовой линии  
            'stress': '#e74c3c'     # Красный для стресса
        }
        
        print("🎯 Создание самого информативного графика...")
        
        # ГРАФИК СРАВНЕНИЯ: До vs После для всех показателей  
        self._create_comparison_plot(df, colors)
    
    def _create_comparison_plot(self, df: pd.DataFrame, colors: Dict[str, str]) -> None:
        """Создает график сравнения до/после для всех показателей"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Изменения физиологических показателей при стрессе', 
                    fontsize=16, fontweight='bold', y=0.96)
        
        # Показатели для анализа
        metrics = [
            ('scr r_Mean_Real', 'КГР среднее', 'отклонение от личной нормы'),
            ('scr r_Line_Length_Real', 'КГР активность', 'длина траектории'),
            ('HR (calculated)_Mean_Real', 'ЧСС среднее', 'удары/мин'),
            ('ppg r_Mean_Real', 'ФПГ среднее', 'амплитуда сигнала')
        ]
        
        axes = axes.flatten()
        
        for idx, (metric_col, metric_name, units) in enumerate(metrics):
            ax = axes[idx]
            
            # Собираем данные для всех участников
            baseline_values = []
            stress_values = []
            participant_labels = []
            
            for participant in self.responders:
                participant_data = df[df['Participant_ID'] == participant]
                if not participant_data.empty and metric_col in participant_data.columns:
                    baseline = participant_data[participant_data['Period'] == 'Базовая линия (1-3)'][metric_col].mean()
                    stress = participant_data[participant_data['Period'] == 'Стресс (4-6)'][metric_col].mean()
                    
                    baseline_values.append(baseline)
                    stress_values.append(stress)
                    participant_labels.append(participant)
            
            if baseline_values and stress_values:
                # Создаем grouped bar chart
                x = np.arange(len(participant_labels))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, baseline_values, width, 
                              color=colors['baseline'], alpha=0.8, label='Базовая линия')
                bars2 = ax.bar(x + width/2, stress_values, width,
                              color=colors['stress'], alpha=0.8, label='Стресс')
                
                # Добавляем стрелки изменений
                for i, (baseline, stress) in enumerate(zip(baseline_values, stress_values)):
                    change = stress - baseline
                    change_pct = (change / abs(baseline) * 100) if abs(baseline) > 0.001 else 0
                    
                    # Стрелка
                    ax.annotate('', xy=(i + width/2, stress), xytext=(i - width/2, baseline),
                               arrowprops=dict(arrowstyle='->', 
                                             color='red' if change > 0 else 'blue', 
                                             lw=2, alpha=0.7))
                    
                    # Процент изменения
                    max_val = max(baseline, stress)
                    ax.text(i, max_val * 1.1, f'{change_pct:+.0f}%',
                           ha='center', va='bottom', fontsize=10, fontweight='bold',
                           color='red' if change > 0 else 'blue')
                
                ax.set_title(metric_name, fontsize=14, fontweight='bold')
                ax.set_ylabel(f'{units}', fontsize=12)
                ax.set_xticks(x)
                ax.set_xticklabels(participant_labels, rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                
                # Статистический тест
                t_stat, p_val = stats.ttest_rel(baseline_values, stress_values)
                sig_text = ''
                # if p_val < 0.001:
                #     sig_text = '***'
                # elif p_val < 0.01:
                #     sig_text = '**'  
                # elif p_val < 0.05:
                #     sig_text = '*'
                
                if sig_text:  # Показываем только если есть значимость
                    ax.text(0.02, 0.98, sig_text, transform=ax.transAxes, 
                           ha='left', va='top', fontsize=14, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout(pad=3.0)
        
        # Сохраняем
        output_path = self.data_path / "result" / "физиологические_показатели_стресс.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 График сохранен: {output_path}")
        plt.show()

    def run_analysis(self) -> None:
        """Запускает полный анализ и создает все графики"""
        print("🔍 Загрузка данных...")
        
        # Загружаем данные
        physio_data = self.load_physiological_data()
        if physio_data.empty:
            print("❌ Не удалось загрузить данные")
            return
        
        # Рассчитываем метрики стресса
        print("📊 Расчет индексов стресса...")
        metrics_df = self.calculate_stress_metrics(physio_data)
        
        if metrics_df.empty:
            print("❌ Не удалось рассчитать метрики стресса")
            return
        
        # Создаем результирующую папку если её нет
        results_dir = self.data_path / "result"
        results_dir.mkdir(exist_ok=True)
        
        print("🎨 Создание информативного графика...")
        
        # Создаем самый информативный график
        self.create_presentation_plots(physio_data)
        
        print("\n✅ Анализ завершен! График сохранен в папке result/")
        print("🎯 Создан график:")
        print("   • физиологические_показатели_стресс.png - Сравнение показателей до/после стресса")

def main():
    """Главная функция"""
    print("🧠 Анализ физиологических показателей при стрессе")
    print("=" * 50)
    
    visualizer = StressRespondersVisualizer()
    visualizer.run_analysis()

if __name__ == "__main__":
    main() 