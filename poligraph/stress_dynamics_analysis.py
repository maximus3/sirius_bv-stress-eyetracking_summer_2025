import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import re
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# Настройка matplotlib для русского языка
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Tahoma', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class StressDynamicsAnalyzer:
    """Класс для анализа динамики стресса до и после индукции стресса"""
    
    def __init__(self, data_path="poligraph/data"):
        self.data_path = pathlib.Path(data_path)
        
    def extract_participant_id(self, filename):
        """Извлекает ID участника из имени файла"""
        pattern = r'(\d{4}[A-Z]{3})_exp1'
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
        return None
    
    def load_normalized_data(self):
        """Загружает нормализованные данные"""
        normalized_file = self.data_path / "result" / "Signal_Analysis_Results_Normalized.xlsx"
        try:
            norm_df = pd.read_excel(normalized_file)
            norm_df['Participant_ID'] = norm_df['File'].apply(self.extract_participant_id)
            norm_df = norm_df[norm_df['Participant_ID'].notna()]
            
            # Конвертируем Label в числовой формат для текстов 1-6
            norm_df['Text_Number'] = pd.to_numeric(norm_df['Label'], errors='coerce')
            norm_df = norm_df[norm_df['Text_Number'].between(1, 6)]
            
            print(f"Загружены нормализованные данные: {len(norm_df)} записей")
            return norm_df
        except Exception as e:
            print(f"Ошибка загрузки нормализованных данных: {e}")
            return pd.DataFrame()
    
    def load_scr_data(self):
        """Загружает SCR данные"""
        scr_file = self.data_path / "result" / "SCR_analysis_results.csv"
        try:
            scr_df = pd.read_csv(scr_file, sep=';', decimal=',')
            scr_df['Participant_ID'] = scr_df['File'].apply(self.extract_participant_id)
            scr_df = scr_df[scr_df['Participant_ID'].notna()]
            print(f"Загружены SCR данные: {len(scr_df)} записей")
            return scr_df
        except Exception as e:
            print(f"Ошибка загрузки SCR данных: {e}")
            return pd.DataFrame()
    
    def calculate_stress_metrics(self, norm_df):
        """Рассчитывает метрики стресса для каждого участника и текста"""
        stress_data = []
        
        for participant in norm_df['Participant_ID'].unique():
            participant_data = norm_df[norm_df['Participant_ID'] == participant]
            
            for text_num in range(1, 7):
                text_data = participant_data[participant_data['Text_Number'] == text_num]
                
                if not text_data.empty:
                    # Извлекаем ключевые показатели стресса
                    scr_line_length = text_data['scr r_Line_Length'].iloc[0] if 'scr r_Line_Length' in text_data.columns else 0
                    scr_mean = text_data['scr r_Mean'].iloc[0] if 'scr r_Mean' in text_data.columns else 0
                    hr_line_length = text_data['HR (calculated)_Line_Length'].iloc[0] if 'HR (calculated)_Line_Length' in text_data.columns else 0
                    hr_mean = text_data['HR (calculated)_Mean'].iloc[0] if 'HR (calculated)_Mean' in text_data.columns else 0
                    
                    # Рассчитываем составной индекс стресса (более чувствительный)
                    stress_score = self.calculate_text_stress_score(scr_line_length, scr_mean, hr_line_length, hr_mean)
                    
                    stress_data.append({
                        'Participant_ID': participant,
                        'Text_Number': text_num,
                        'SCR_Line_Length': scr_line_length,
                        'SCR_Mean': scr_mean,
                        'HR_Line_Length': hr_line_length,
                        'HR_Mean': hr_mean,
                        'Stress_Score': stress_score,
                        'Phase': 'Базовая линия' if text_num <= 3 else 'После индукции стресса'
                    })
        
        return pd.DataFrame(stress_data)
    
    def calculate_text_stress_score(self, scr_line_length, scr_mean, hr_line_length, hr_mean):
        """Рассчитывает более чувствительный индекс стресса для текста"""
        score = 0
        
        # Используем более низкие пороги для выявления тонких различий
        if scr_line_length > 0.2:  # Снижен порог
            score += 2
        elif scr_line_length > 0.0:
            score += 1
            
        if scr_mean > 0.2:  # Снижен порог
            score += 2
        elif scr_mean > 0.0:
            score += 1
            
        if hr_line_length > 0.2:  # Снижен порог
            score += 1
        elif hr_line_length > 0.0:
            score += 0.5
            
        if hr_mean > 0.2:
            score += 1
        elif hr_mean > 0.0:
            score += 0.5
        
        return score
    
    def analyze_stress_dynamics(self, stress_df):
        """Анализирует динамику стресса до и после индукции"""
        
        # Группируем по фазам эксперимента
        baseline_stress = stress_df[stress_df['Text_Number'].isin([1, 2, 3])].groupby('Participant_ID')['Stress_Score'].mean()
        post_induction_stress = stress_df[stress_df['Text_Number'].isin([4, 5, 6])].groupby('Participant_ID')['Stress_Score'].mean()
        
        # Анализируем изменения
        dynamics_data = []
        for participant in baseline_stress.index:
            if participant in post_induction_stress.index:
                baseline = baseline_stress[participant]
                post_induction = post_induction_stress[participant]
                change = post_induction - baseline
                change_percent = (change / baseline * 100) if baseline > 0 else 0
                
                # Специальный анализ 4-го текста (должен быть пик стресса)
                text4_data = stress_df[(stress_df['Participant_ID'] == participant) & (stress_df['Text_Number'] == 4)]
                text4_stress = text4_data['Stress_Score'].iloc[0] if not text4_data.empty else 0
                
                dynamics_data.append({
                    'Participant_ID': participant,
                    'Baseline_Stress': baseline,
                    'Post_Induction_Stress': post_induction,
                    'Stress_Change': change,
                    'Stress_Change_Percent': change_percent,
                    'Text4_Stress': text4_stress,
                    'Responded_to_Induction': change > 0.5  # Порог реакции на индукцию
                })
        
        return pd.DataFrame(dynamics_data)
    
    def create_dynamics_visualizations(self, stress_df, dynamics_df):
        """Создает графики динамики стресса"""
        
        results_path = pathlib.Path("poligraph/stress_dynamics_results")
        results_path.mkdir(exist_ok=True)
        
        plt.style.use('default')
        
        # График 1: Средняя динамика стресса по текстам
        plt.figure(figsize=(12, 8))
        mean_stress_by_text = stress_df.groupby('Text_Number')['Stress_Score'].mean()
        std_stress_by_text = stress_df.groupby('Text_Number')['Stress_Score'].std()
        
        plt.plot(mean_stress_by_text.index, mean_stress_by_text.values, 'o-', linewidth=3, markersize=8, color='red')
        plt.fill_between(mean_stress_by_text.index, 
                        mean_stress_by_text.values - std_stress_by_text.values,
                        mean_stress_by_text.values + std_stress_by_text.values,
                        alpha=0.3, color='red')
        
        # Отмечаем момент индукции стресса
        plt.axvline(x=3.5, color='black', linestyle='--', linewidth=2, alpha=0.7)
        plt.text(3.6, plt.ylim()[1]*0.9, 'Индукция\nстресса', fontsize=12, fontweight='bold')
        
        plt.title('Динамика стресса по текстам (средние значения)', fontsize=16, fontweight='bold')
        plt.xlabel('Номер текста', fontsize=12)
        plt.ylabel('Индекс стресса', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(range(1, 7))
        
        # Выделяем фазы
        plt.axvspan(0.5, 3.5, alpha=0.2, color='blue', label='Базовая линия')
        plt.axvspan(3.5, 6.5, alpha=0.2, color='red', label='После индукции стресса')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(results_path / 'stress_dynamics_timeline.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # График 2: Индивидуальные траектории участников
        plt.figure(figsize=(14, 10))
        
        participants_with_response = dynamics_df[dynamics_df['Responded_to_Induction']]['Participant_ID'].tolist()
        participants_no_response = dynamics_df[~dynamics_df['Responded_to_Induction']]['Participant_ID'].tolist()
        
        # Рисуем траектории респондеров
        for participant in participants_with_response:
            participant_data = stress_df[stress_df['Participant_ID'] == participant]
            plt.plot(participant_data['Text_Number'], participant_data['Stress_Score'], 
                    'o-', linewidth=2, alpha=0.7, color='red', 
                    label='Реагировали на стресс' if participant == participants_with_response[0] else "")
        
        # Рисуем траектории нон-респондеров
        for participant in participants_no_response:
            participant_data = stress_df[stress_df['Participant_ID'] == participant]
            plt.plot(participant_data['Text_Number'], participant_data['Stress_Score'], 
                    'o-', linewidth=1, alpha=0.5, color='blue',
                    label='Не реагировали на стресс' if participant == participants_no_response[0] else "")
        
        plt.axvline(x=3.5, color='black', linestyle='--', linewidth=2, alpha=0.7)
        plt.text(3.6, plt.ylim()[1]*0.9, 'Индукция\nстресса', fontsize=12, fontweight='bold')
        
        plt.title('Индивидуальные траектории стресса участников', fontsize=16, fontweight='bold')
        plt.xlabel('Номер текста', fontsize=12)
        plt.ylabel('Индекс стресса', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(range(1, 7))
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_path / 'individual_stress_trajectories.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # График 3: Сравнение до и после индукции
        plt.figure(figsize=(12, 8))
        
        x = np.arange(len(dynamics_df))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, dynamics_df['Baseline_Stress'], width, 
                       label='Базовая линия (тексты 1-3)', alpha=0.8, color='blue')
        bars2 = plt.bar(x + width/2, dynamics_df['Post_Induction_Stress'], width,
                       label='После индукции (тексты 4-6)', alpha=0.8, color='red')
        
        # Выделяем участников, которые реагировали на стресс
        for i, responded in enumerate(dynamics_df['Responded_to_Induction']):
            if responded:
                bars1[i].set_edgecolor('green')
                bars1[i].set_linewidth(3)
                bars2[i].set_edgecolor('green')
                bars2[i].set_linewidth(3)
        
        plt.title('Сравнение стресса до и после индукции по участникам', fontsize=16, fontweight='bold')
        plt.xlabel('Участники', fontsize=12)
        plt.ylabel('Средний индекс стресса', fontsize=12)
        plt.xticks(x, dynamics_df['Participant_ID'], rotation=45)
        plt.legend()
        
        # Добавляем линии для участников, которые увеличили стресс
        for i, (baseline, post_induction) in enumerate(zip(dynamics_df['Baseline_Stress'], dynamics_df['Post_Induction_Stress'])):
            if post_induction > baseline:
                plt.plot([i - width/2, i + width/2], [baseline, post_induction], 'g-', linewidth=2, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(results_path / 'before_after_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # График 4: Фокус на 4-й текст (должен быть пик стресса)
        plt.figure(figsize=(10, 6))
        
        text4_data = stress_df[stress_df['Text_Number'] == 4].sort_values('Stress_Score', ascending=False)
        colors = ['red' if score > stress_df['Stress_Score'].mean() else 'blue' for score in text4_data['Stress_Score']]
        
        plt.bar(range(len(text4_data)), text4_data['Stress_Score'], color=colors, alpha=0.7)
        plt.axhline(y=stress_df['Stress_Score'].mean(), color='black', linestyle='--', alpha=0.7, label='Средний уровень стресса')
        
        plt.title('Уровень стресса в 4-м тексте (первый после индукции)', fontsize=16, fontweight='bold')
        plt.xlabel('Участники (отсортированы по уровню стресса)', fontsize=12)
        plt.ylabel('Индекс стресса', fontsize=12)
        plt.xticks(range(len(text4_data)), text4_data['Participant_ID'], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(results_path / 'text4_stress_focus.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # График 5: Тепловая карта по фазам
        plt.figure(figsize=(12, 8))
        
        # Создаем данные для тепловой карты
        heatmap_data = stress_df.pivot(index='Participant_ID', columns='Text_Number', values='Stress_Score')
        
        # Создаем кастомную colormap
        cmap = plt.cm.RdYlBu_r
        
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap=cmap, 
                   cbar_kws={'label': 'Индекс стресса'}, linewidths=0.5)
        
        # Добавляем разделитель между фазами
        plt.axvline(x=3, color='black', linewidth=3)
        plt.text(1.5, len(heatmap_data) + 0.5, 'Базовая линия', ha='center', fontweight='bold')
        plt.text(4.5, len(heatmap_data) + 0.5, 'После индукции стресса', ha='center', fontweight='bold')
        
        plt.title('Тепловая карта стресса по участникам и текстам', fontsize=16, fontweight='bold')
        plt.xlabel('Номер текста', fontsize=12)
        plt.ylabel('ID участника', fontsize=12)
        plt.tight_layout()
        plt.savefig(results_path / 'stress_heatmap_phases.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_statistical_analysis(self, stress_df, dynamics_df):
        """Проводит статистический анализ эффективности индукции стресса"""
        
        results_path = pathlib.Path("poligraph/stress_dynamics_results")
        results_path.mkdir(exist_ok=True)
        
        print("\n=== СТАТИСТИЧЕСКИЙ АНАЛИЗ ===")
        
        # T-test для сравнения базовой линии и пост-индукции
        baseline_scores = dynamics_df['Baseline_Stress']
        post_induction_scores = dynamics_df['Post_Induction_Stress']
        
        t_stat, p_value = stats.ttest_rel(post_induction_scores, baseline_scores)
        
        print(f"\nПарный t-test (до vs после индукции стресса):")
        print(f"t-статистика: {t_stat:.4f}")
        print(f"p-значение: {p_value:.4f}")
        print(f"Значимость: {'ДА' if p_value < 0.05 else 'НЕТ'} (p < 0.05)")
        
        # Эффект индукции по участникам
        responders = dynamics_df[dynamics_df['Responded_to_Induction']]
        non_responders = dynamics_df[~dynamics_df['Responded_to_Induction']]
        
        print(f"\n=== АНАЛИЗ РЕАКЦИИ НА ИНДУКЦИЮ СТРЕССА ===")
        print(f"Участники, реагировавшие на стресс: {len(responders)} ({len(responders)/len(dynamics_df)*100:.1f}%)")
        print(f"Участники, не реагировавшие на стресс: {len(non_responders)} ({len(non_responders)/len(dynamics_df)*100:.1f}%)")
        
        if len(responders) > 0:
            print(f"\nСредний прирост стресса у респондеров: {responders['Stress_Change'].mean():.2f}")
            print(f"Респондеры: {', '.join(responders['Participant_ID'].tolist())}")
        
        if len(non_responders) > 0:
            print(f"Средний прирост стресса у нон-респондеров: {non_responders['Stress_Change'].mean():.2f}")
            print(f"Нон-респондеры: {', '.join(non_responders['Participant_ID'].tolist())}")
        
        # Анализ пика стресса в 4-м тексте
        text4_stress = stress_df[stress_df['Text_Number'] == 4]['Stress_Score']
        other_texts_stress = stress_df[stress_df['Text_Number'] != 4]['Stress_Score']
        
        t_stat_text4, p_value_text4 = stats.ttest_ind(text4_stress, other_texts_stress)
        
        print(f"\n=== АНАЛИЗ 4-ГО ТЕКСТА (ПИК СТРЕССА) ===")
        print(f"Средний стресс в 4-м тексте: {text4_stress.mean():.2f}")
        print(f"Средний стресс в остальных текстах: {other_texts_stress.mean():.2f}")
        print(f"t-статистика: {t_stat_text4:.4f}")
        print(f"p-значение: {p_value_text4:.4f}")
        print(f"4-й текст значимо стрессовее: {'ДА' if p_value_text4 < 0.05 and text4_stress.mean() > other_texts_stress.mean() else 'НЕТ'}")
        
        # Создаем итоговую таблицу
        summary_stats = {
            'Metric': [
                'Средний стресс (базовая линия)',
                'Средний стресс (после индукции)',
                'Изменение стресса',
                'Участников с реакцией на стресс',
                'Средний стресс в 4-м тексте',
                'p-значение (до vs после)',
                'p-значение (4-й текст vs остальные)'
            ],
            'Value': [
                f"{baseline_scores.mean():.2f}",
                f"{post_induction_scores.mean():.2f}",
                f"{(post_induction_scores.mean() - baseline_scores.mean()):.2f}",
                f"{len(responders)}/{len(dynamics_df)} ({len(responders)/len(dynamics_df)*100:.1f}%)",
                f"{text4_stress.mean():.2f}",
                f"{p_value:.4f}",
                f"{p_value_text4:.4f}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_excel(results_path / 'statistical_summary.xlsx', index=False)
        
        print(f"\n=== СТАТИСТИЧЕСКИЙ ОТЧЕТ СОХРАНЕН ===")
        print("Файл: poligraph/stress_dynamics_results/statistical_summary.xlsx")
        
        return summary_df, responders, non_responders
    
    def run_dynamics_analysis(self):
        """Запускает полный анализ динамики стресса"""
        print("=== АНАЛИЗ ДИНАМИКИ СТРЕССА ===\n")
        
        # Загрузка данных
        print("1. Загрузка данных...")
        norm_df = self.load_normalized_data()
        scr_df = self.load_scr_data()
        
        if norm_df.empty:
            print("Ошибка: не удалось загрузить данные")
            return
        
        # Расчет метрик стресса
        print("\n2. Расчет метрик стресса по текстам...")
        stress_df = self.calculate_stress_metrics(norm_df)
        print(f"Проанализировано записей: {len(stress_df)}")
        
        # Анализ динамики
        print("\n3. Анализ динамики стресса...")
        dynamics_df = self.analyze_stress_dynamics(stress_df)
        print(f"Проанализировано участников: {len(dynamics_df)}")
        
        # Создание визуализаций
        print("\n4. Создание графиков динамики...")
        self.create_dynamics_visualizations(stress_df, dynamics_df)
        
        # Статистический анализ
        print("\n5. Статистический анализ...")
        summary_stats, responders, non_responders = self.create_statistical_analysis(stress_df, dynamics_df)
        
        # Сохранение детальных результатов
        results_path = pathlib.Path("poligraph/stress_dynamics_results")
        stress_df.to_excel(results_path / 'detailed_stress_by_text.xlsx', index=False)
        dynamics_df.to_excel(results_path / 'stress_dynamics_summary.xlsx', index=False)
        
        print("\n=== АНАЛИЗ ДИНАМИКИ ЗАВЕРШЕН ===")
        print("Результаты сохранены в папке: poligraph/stress_dynamics_results/")
        
        return {
            'stress_by_text': stress_df,
            'dynamics_summary': dynamics_df,
            'responders': responders,
            'non_responders': non_responders,
            'statistical_summary': summary_stats
        }

# Запуск анализа
if __name__ == "__main__":
    analyzer = StressDynamicsAnalyzer()
    results = analyzer.run_dynamics_analysis() 