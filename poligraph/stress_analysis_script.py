import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import re
import warnings
warnings.filterwarnings('ignore')

# Настройка matplotlib для русского языка
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Tahoma', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class StressAnalyzer:
    """Класс для анализа данных стресса участников"""
    
    def __init__(self, data_path="poligraph/data"):
        self.data_path = pathlib.Path(data_path)
        self.participant_data = {}
        self.stress_results = pd.DataFrame()
        
    def extract_participant_id(self, filename):
        """Извлекает ID участника из имени файла"""
        pattern = r'(\d{4}[A-Z]{3})_exp1'
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
        return None
    
    def load_spilberg_data(self):
        """Загружает данные теста Спилберга"""
        try:
            spilberg_file = self.data_path / "spilberg.xlsx"
            df = pd.read_excel(spilberg_file)
            # Удаляем технические столбцы
            df = df.drop(columns=[col for col in df.columns if 'Unnamed' in str(col)])
            print(f"Загружены данные Спилберга: {len(df)} участников")
            return df
        except Exception as e:
            print(f"Ошибка загрузки данных Спилберга: {e}")
            return pd.DataFrame()
    
    def load_physiological_data(self):
        """Загружает физиологические данные"""
        # Загрузка SCR данных
        scr_file = self.data_path / "result" / "SCR_analysis_results.csv"
        try:
            scr_df = pd.read_csv(scr_file, sep=';', decimal=',')
            scr_df['Participant_ID'] = scr_df['File'].apply(self.extract_participant_id)
            scr_df = scr_df[scr_df['Participant_ID'].notna()]
            print(f"Загружены SCR данные: {len(scr_df)} записей")
        except Exception as e:
            print(f"Ошибка загрузки SCR данных: {e}")
            scr_df = pd.DataFrame()
        
        # Загрузка нормализованных данных
        normalized_file = self.data_path / "result" / "Signal_Analysis_Results_Normalized.xlsx"
        try:
            norm_df = pd.read_excel(normalized_file)
            norm_df['Participant_ID'] = norm_df['File'].apply(self.extract_participant_id)
            norm_df = norm_df[norm_df['Participant_ID'].notna()]
            print(f"Загружены нормализованные данные: {len(norm_df)} записей")
        except Exception as e:
            print(f"Ошибка загрузки нормализованных данных: {e}")
            norm_df = pd.DataFrame()
        
        return scr_df, norm_df
    
    def calculate_stress_indicators(self, scr_df, norm_df):
        """Вычисляет индикаторы стресса для каждого участника"""
        stress_indicators = []
        
        # Группировка по участникам
        participants = scr_df['Participant_ID'].unique()
        
        for participant in participants:
            participant_scr = scr_df[scr_df['Participant_ID'] == participant]
            participant_norm = norm_df[norm_df['Participant_ID'] == participant]
            
            # Анализируем данные SCR (кожно-гальваническая реакция)
            scr_data = participant_scr[participant_scr['Channel'] == 'scr r']
            if not scr_data.empty:
                scr_row = scr_data.iloc[0]
                
                # Извлекаем ключевые показатели стресса
                ns_scr = float(str(scr_row['NS-SCR']).replace(',', '.')) if pd.notna(scr_row['NS-SCR']) else 0
                amp_scr = float(str(scr_row['Amp-SCR']).replace(',', '.')) if pd.notna(scr_row['Amp-SCR']) else 0
                recovery_time = float(str(scr_row['Recovery-Time']).replace(',', '.')) if pd.notna(scr_row['Recovery-Time']) else 0
                line_length = float(str(scr_row['Line-Length']).replace(',', '.')) if pd.notna(scr_row['Line-Length']) else 0
                raw_sd = float(str(scr_row['Raw-SD']).replace(',', '.')) if pd.notna(scr_row['Raw-SD']) else 0
                
                # Анализируем данные HR (частота сердечных сокращений)
                hr_data = participant_scr[participant_scr['Channel'] == 'HR (calculated)']
                hr_line_length = 0
                if not hr_data.empty:
                    hr_line_length = float(str(hr_data.iloc[0]['Line-Length']).replace(',', '.')) if pd.notna(hr_data.iloc[0]['Line-Length']) else 0
                
                # Рассчитываем составной индекс стресса
                stress_score = self.calculate_stress_score(ns_scr, amp_scr, recovery_time, line_length, raw_sd, hr_line_length)
                
                # Анализ по текстам (если есть данные по Label)
                text_analysis = {}
                if not participant_norm.empty:
                    for i in range(1, 7):  # Тексты 1-6
                        text_data = participant_norm[participant_norm['Label'].astype(str).str.contains(str(i), na=False)]
                        if not text_data.empty:
                            # Усредняем показатели по текстам
                            text_stress = text_data[['scr r_Line_Length', 'scr r_Mean', 'HR (calculated)_Line_Length']].mean()
                            text_analysis[f'text_{i}'] = {
                                'scr_line_length': text_stress.get('scr r_Line_Length', 0),
                                'scr_mean': text_stress.get('scr r_Mean', 0),
                                'hr_line_length': text_stress.get('HR (calculated)_Line_Length', 0)
                            }
                
                stress_indicators.append({
                    'Participant_ID': participant,
                    'NS_SCR': ns_scr,
                    'Amp_SCR': amp_scr,
                    'Recovery_Time': recovery_time,
                    'SCR_Line_Length': line_length,
                    'SCR_Raw_SD': raw_sd,
                    'HR_Line_Length': hr_line_length,
                    'Stress_Score': stress_score,
                    'Text_Analysis': text_analysis
                })
        
        return pd.DataFrame(stress_indicators)
    
    def calculate_stress_score(self, ns_scr, amp_scr, recovery_time, line_length, raw_sd, hr_line_length):
        """Рассчитывает составной индекс стресса"""
        # Нормализуем показатели (простая z-score нормализация)
        # Высокие значения NS-SCR, Amp-SCR, Line-Length, Raw-SD указывают на стресс
        # Низкое время восстановления также может указывать на стресс
        
        score = 0
        if ns_scr > 80:  # Много SCR пиков
            score += 2
        elif ns_scr > 60:
            score += 1
            
        if amp_scr > 1.5:  # Высокая амплитуда
            score += 2
        elif amp_scr > 1.0:
            score += 1
            
        if recovery_time < 1.5:  # Быстрое восстановление может указывать на стресс
            score += 1
            
        if line_length > 300:  # Высокая активность сигнала
            score += 2
        elif line_length > 200:
            score += 1
            
        if raw_sd > 10000:  # Высокая вариативность
            score += 2
        elif raw_sd > 5000:
            score += 1
            
        if hr_line_length > 200:  # Высокая вариативность HR
            score += 1
        
        return score
    
    def analyze_text_stress(self, stress_df):
        """Анализирует стресс по текстам"""
        text_stress_results = []
        
        for _, participant in stress_df.iterrows():
            participant_id = participant['Participant_ID']
            text_analysis = participant['Text_Analysis']
            
            for text_num in range(1, 7):
                text_key = f'text_{text_num}'
                if text_key in text_analysis:
                    text_data = text_analysis[text_key]
                    
                    # Определяем стресс по тексту
                    text_stress_score = 0
                    if text_data['scr_line_length'] > 0.5:  # Нормализованные значения
                        text_stress_score += 2
                    if text_data['scr_mean'] > 0.5:
                        text_stress_score += 2
                    if text_data['hr_line_length'] > 0.5:
                        text_stress_score += 1
                    
                    stress_level = "Высокий" if text_stress_score >= 3 else "Средний" if text_stress_score >= 2 else "Низкий"
                    
                    text_stress_results.append({
                        'Participant_ID': participant_id,
                        'Text_Number': text_num,
                        'SCR_Line_Length': text_data['scr_line_length'],
                        'SCR_Mean': text_data['scr_mean'],
                        'HR_Line_Length': text_data['hr_line_length'],
                        'Text_Stress_Score': text_stress_score,
                        'Stress_Level': stress_level
                    })
        
        return pd.DataFrame(text_stress_results)
    
    def create_visualizations(self, stress_df, text_stress_df, spilberg_df):
        """Создает графики"""
        
        # Создаем папку для результатов
        results_path = pathlib.Path("poligraph/analysis_results")
        results_path.mkdir(exist_ok=True)
        
        plt.style.use('default')
        
        # График 1: Общий индекс стресса участников
        plt.figure(figsize=(12, 6))
        plt.bar(stress_df['Participant_ID'], stress_df['Stress_Score'], color='steelblue')
        plt.title('Общий индекс стресса участников', fontsize=14, fontweight='bold')
        plt.xlabel('ID участника')
        plt.ylabel('Индекс стресса')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(results_path / 'overall_stress_index.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # График 2: Корреляция физиологических показателей
        if len(stress_df) > 1:
            plt.figure(figsize=(10, 8))
            corr_data = stress_df[['NS_SCR', 'Amp_SCR', 'SCR_Line_Length', 'HR_Line_Length', 'Stress_Score']].corr()
            sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0)
            plt.title('Корреляция физиологических показателей стресса')
            plt.tight_layout()
            plt.savefig(results_path / 'physiological_correlation.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # График 3: Динамика стресса по текстам
        if not text_stress_df.empty:
            plt.figure(figsize=(14, 8))
            pivot_data = text_stress_df.pivot(index='Participant_ID', columns='Text_Number', values='Text_Stress_Score')
            sns.heatmap(pivot_data, annot=True, cmap='Reds', cbar_kws={'label': 'Индекс стресса'})
            plt.title('Уровень стресса участников по текстам')
            plt.xlabel('Номер текста')
            plt.ylabel('ID участника')
            plt.tight_layout()
            plt.savefig(results_path / 'text_stress_heatmap.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # График 4: Средний стресс по текстам
            plt.figure(figsize=(10, 6))
            text_mean_stress = text_stress_df.groupby('Text_Number')['Text_Stress_Score'].mean()
            plt.plot(text_mean_stress.index, text_mean_stress.values, marker='o', linewidth=2, markersize=8)
            plt.title('Средний уровень стресса по текстам')
            plt.xlabel('Номер текста')
            plt.ylabel('Средний индекс стресса')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(results_path / 'average_text_stress.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # График 5: Сравнение со Спилбергом (если есть данные)
        if not spilberg_df.empty and len(spilberg_df) > 0:
            plt.figure(figsize=(12, 8))
            
            # Пытаемся сопоставить участников
            merged_data = []
            for _, stress_row in stress_df.iterrows():
                participant_id = stress_row['Participant_ID']
                # Ищем в данных Спилберга
                spilberg_match = spilberg_df[spilberg_df.iloc[:, 0].astype(str).str.contains(participant_id[-3:], na=False)]
                if not spilberg_match.empty:
                    spilberg_row = spilberg_match.iloc[0]
                    merged_data.append({
                        'Participant_ID': participant_id,
                        'Physiological_Stress': stress_row['Stress_Score'],
                        'Spilberg_Before': spilberg_row.iloc[1] if len(spilberg_row) > 1 else 0,
                        'Spilberg_After': spilberg_row.iloc[2] if len(spilberg_row) > 2 else 0
                    })
            
            if merged_data:
                merged_df = pd.DataFrame(merged_data)
                
                plt.subplot(2, 1, 1)
                x = np.arange(len(merged_df))
                width = 0.35
                plt.bar(x - width/2, merged_df['Spilberg_Before'], width, label='Спилберг до', alpha=0.8)
                plt.bar(x + width/2, merged_df['Spilberg_After'], width, label='Спилберг после', alpha=0.8)
                plt.xlabel('Участники')
                plt.ylabel('Уровень тревожности')
                plt.title('Тест Спилберга: до и после')
                plt.xticks(x, merged_df['Participant_ID'], rotation=45)
                plt.legend()
                
                plt.subplot(2, 1, 2)
                plt.scatter(merged_df['Physiological_Stress'], merged_df['Spilberg_After'], alpha=0.7)
                plt.xlabel('Физиологический индекс стресса')
                plt.ylabel('Спилберг после')
                plt.title('Корреляция: физиологический стресс vs тест Спилберга')
                
                plt.tight_layout()
                plt.savefig(results_path / 'spilberg_comparison.png', dpi=300, bbox_inches='tight')
                plt.show()
    
    def create_summary_table(self, stress_df, text_stress_df, spilberg_df):
        """Создает итоговую таблицу результатов"""
        results_path = pathlib.Path("poligraph/analysis_results")
        results_path.mkdir(exist_ok=True)
        
        # Общая таблица по участникам
        summary_table = stress_df[['Participant_ID', 'Stress_Score']].copy()
        summary_table['Stress_Level'] = summary_table['Stress_Score'].apply(
            lambda x: 'Высокий' if x >= 6 else 'Средний' if x >= 3 else 'Низкий'
        )
        
        # Добавляем информацию по текстам
        if not text_stress_df.empty:
            for text_num in range(1, 7):
                text_data = text_stress_df[text_stress_df['Text_Number'] == text_num]
                text_dict = dict(zip(text_data['Participant_ID'], text_data['Stress_Level']))
                summary_table[f'Text_{text_num}_Stress'] = summary_table['Participant_ID'].map(text_dict).fillna('Нет данных')
        
        # Сохраняем таблицы
        summary_table.to_excel(results_path / 'stress_analysis_summary.xlsx', index=False, engine='openpyxl')
        
        if not text_stress_df.empty:
            text_stress_df.to_excel(results_path / 'text_level_stress_analysis.xlsx', index=False, engine='openpyxl')
        
        print("Итоговые таблицы:")
        print("\n1. Общий анализ стресса участников:")
        print(summary_table.to_string(index=False))
        
        if not text_stress_df.empty:
            print("\n2. Детальный анализ по текстам:")
            print(text_stress_df.head(10).to_string(index=False))
            print(f"... всего {len(text_stress_df)} записей")
        
        return summary_table, text_stress_df
    
    def run_analysis(self):
        """Запускает полный анализ"""
        print("=== Анализ данных стресса участников ===\n")
        
        # Загрузка данных
        print("1. Загрузка данных...")
        spilberg_df = self.load_spilberg_data()
        scr_df, norm_df = self.load_physiological_data()
        
        if scr_df.empty:
            print("Ошибка: не удалось загрузить физиологические данные")
            return
        
        # Расчет индикаторов стресса
        print("\n2. Расчет индикаторов стресса...")
        stress_df = self.calculate_stress_indicators(scr_df, norm_df)
        print(f"Проанализировано участников: {len(stress_df)}")
        
        # Анализ стресса по текстам
        print("\n3. Анализ стресса по текстам...")
        text_stress_df = self.analyze_text_stress(stress_df)
        print(f"Проанализировано записей по текстам: {len(text_stress_df)}")
        
        # Создание визуализаций
        print("\n4. Создание графиков...")
        self.create_visualizations(stress_df, text_stress_df, spilberg_df)
        
        # Создание итоговых таблиц
        print("\n5. Создание итоговых таблиц...")
        summary_table, detailed_table = self.create_summary_table(stress_df, text_stress_df, spilberg_df)
        
        print("\n=== Анализ завершен ===")
        print("Результаты сохранены в папке: poligraph/analysis_results/")
        
        return {
            'stress_data': stress_df,
            'text_stress_data': text_stress_df,
            'summary_table': summary_table,
            'spilberg_data': spilberg_df
        }

# Запуск анализа
if __name__ == "__main__":
    analyzer = StressAnalyzer()
    results = analyzer.run_analysis() 