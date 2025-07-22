#!/usr/bin/env python3
"""
Демонстрация разницы между реальными значениями КГР и z-score нормализацией.
Показывает зачем нужна нормализация для сравнения между участниками.
ТЕПЕРЬ ИСПОЛЬЗУЕТ РЕАЛЬНЫЕ ДАННЫЕ ИЗ ЭКСПЕРИМЕНТА!

Запуск: uv run poligraph/demo_scr_explanation.py
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings

warnings.filterwarnings('ignore')

# Настройка matplotlib для русского языка
plt.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
sns.set_style("whitegrid")

def extract_participant_id(filename):
    """Извлекает ID участника из имени файла"""
    pattern = r'(\d{4}[A-Z]{3})_exp1'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return None

def load_real_data():
    """Загружает реальные данные из эксперимента"""
    data_path = pathlib.Path("poligraph/data")
    
    # Загружаем нормализованные данные
    normalized_file = data_path / "result" / "Signal_Analysis_Results_Normalized.xlsx"
    try:
        df = pd.read_excel(normalized_file)
        df['Participant_ID'] = df['File'].apply(extract_participant_id)
        df = df[df['Participant_ID'].notna()]
        
        # Конвертируем Label в числовой формат для текстов 1-6
        df['Text_Number'] = pd.to_numeric(df['Label'], errors='coerce')
        df = df[df['Text_Number'].between(1, 6)]
        
        # Добавляем категорию периода
        df['Period'] = df['Text_Number'].apply(
            lambda x: 'Базовая линия (1-3)' if x <= 3 else 'Стресс (4-6)'
        )
        
        print(f"✅ Загружены реальные данные: {len(df)} записей для {len(df['Participant_ID'].unique())} участников")
        return df
    except Exception as e:
        print(f"❌ Ошибка загрузки реальных данных: {e}")
        return pd.DataFrame()

def demo_line_length_concept_real_data(df):
    """Демонстрирует концепцию Line Length на реальных данных"""
    if df.empty:
        print("❌ Нет данных для демонстрации Line Length")
        return 0, 0, 0, 0
    
    # Респондеры и нон-респондеры
    responders = ['1707LTA', '1807KNV', '1607KYA', '1807OVA', '1807ZUG']
    non_responders = ['1707KAV', '1807HEE', '1807CAA', '1607LVA', '1907ZSI']
    
    # Выбираем ярких представителей
    responder_example = '1707LTA'  # Самый яркий респондер
    non_responder_example = '1707KAV'  # Самое сильное снижение
    
    # Данные респондера
    responder_data = df[df['Participant_ID'] == responder_example]
    baseline_resp = responder_data[responder_data['Period'] == 'Базовая линия (1-3)']
    stress_resp = responder_data[responder_data['Period'] == 'Стресс (4-6)']
    
    # Данные нон-респондера
    non_resp_data = df[df['Participant_ID'] == non_responder_example]
    baseline_non = non_resp_data[non_resp_data['Period'] == 'Базовая линия (1-3)']
    stress_non = non_resp_data[non_resp_data['Period'] == 'Стресс (4-6)']
    
    if baseline_resp.empty or stress_resp.empty or baseline_non.empty or stress_non.empty:
        print("❌ Недостаточно данных для выбранных участников")
        return 0, 0, 0, 0
    
    # Извлекаем показатели КГР
    resp_baseline_mean = baseline_resp['scr r_Mean_Real'].mean() if 'scr r_Mean_Real' in baseline_resp.columns else 0
    resp_stress_mean = stress_resp['scr r_Mean_Real'].mean() if 'scr r_Mean_Real' in stress_resp.columns else 0
    resp_baseline_line = baseline_resp['scr r_Line_Length_Real'].mean() if 'scr r_Line_Length_Real' in baseline_resp.columns else 0
    resp_stress_line = stress_resp['scr r_Line_Length_Real'].mean() if 'scr r_Line_Length_Real' in stress_resp.columns else 0
    
    non_baseline_mean = baseline_non['scr r_Mean_Real'].mean() if 'scr r_Mean_Real' in baseline_non.columns else 0
    non_stress_mean = stress_non['scr r_Mean_Real'].mean() if 'scr r_Mean_Real' in stress_non.columns else 0
    non_baseline_line = baseline_non['scr r_Line_Length_Real'].mean() if 'scr r_Line_Length_Real' in baseline_non.columns else 0
    non_stress_line = stress_non['scr r_Line_Length_Real'].mean() if 'scr r_Line_Length_Real' in stress_non.columns else 0
    
    # Создаем график
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Концепция Line Length на РЕАЛЬНЫХ ДАННЫХ эксперимента', 
                fontsize=16, fontweight='bold')
    
    # График данных по текстам для респондера
    responder_texts = responder_data.sort_values('Text_Number')
    if not responder_texts.empty and 'scr r_Mean_Real' in responder_texts.columns:
        axes[0,0].plot(responder_texts['Text_Number'], responder_texts['scr r_Mean_Real'], 
                      'o-', color='red', linewidth=2, markersize=8, label='Среднее КГР')
        axes[0,0].axhline(resp_baseline_mean, color='blue', linestyle='--', alpha=0.7)
        axes[0,0].axvspan(1, 3.5, alpha=0.2, color='blue', label='Базовая линия')
        axes[0,0].axvspan(3.5, 6, alpha=0.2, color='red', label='Стресс')
        axes[0,0].set_title(f'{responder_example} (РЕСПОНДЕР)\nСреднее КГР по текстам', fontweight='bold')
        axes[0,0].set_ylabel('КГР (реальные единицы)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
    
    # График Line Length для респондера
    if not responder_texts.empty and 'scr r_Line_Length_Real' in responder_texts.columns:
        axes[0,1].plot(responder_texts['Text_Number'], responder_texts['scr r_Line_Length_Real'], 
                      'o-', color='red', linewidth=2, markersize=8, label='Активность КГР')
        axes[0,1].axhline(resp_baseline_line, color='blue', linestyle='--', alpha=0.7)
        axes[0,1].axvspan(1, 3.5, alpha=0.2, color='blue', label='Базовая линия')
        axes[0,1].axvspan(3.5, 6, alpha=0.2, color='red', label='Стресс')
        axes[0,1].set_title(f'{responder_example} (РЕСПОНДЕР)\nАктивность КГР (Line Length)', fontweight='bold')
        axes[0,1].set_ylabel('Line Length (реальные единицы)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
    
    # График данных по текстам для нон-респондера
    non_resp_texts = non_resp_data.sort_values('Text_Number')
    if not non_resp_texts.empty and 'scr r_Mean_Real' in non_resp_texts.columns:
        axes[1,0].plot(non_resp_texts['Text_Number'], non_resp_texts['scr r_Mean_Real'], 
                      'o-', color='green', linewidth=2, markersize=8, label='Среднее КГР')
        axes[1,0].axhline(non_baseline_mean, color='blue', linestyle='--', alpha=0.7)
        axes[1,0].axvspan(1, 3.5, alpha=0.2, color='blue', label='Базовая линия')
        axes[1,0].axvspan(3.5, 6, alpha=0.2, color='red', label='Стресс')
        axes[1,0].set_title(f'{non_responder_example} (НОН-РЕСПОНДЕР)\nСреднее КГР по текстам', fontweight='bold')
        axes[1,0].set_xlabel('Номер текста')
        axes[1,0].set_ylabel('КГР (реальные единицы)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    
    # График Line Length для нон-респондера
    if not non_resp_texts.empty and 'scr r_Line_Length_Real' in non_resp_texts.columns:
        axes[1,1].plot(non_resp_texts['Text_Number'], non_resp_texts['scr r_Line_Length_Real'], 
                      'o-', color='green', linewidth=2, markersize=8, label='Активность КГР')
        axes[1,1].axhline(non_baseline_line, color='blue', linestyle='--', alpha=0.7)
        axes[1,1].axvspan(1, 3.5, alpha=0.2, color='blue', label='Базовая линия')
        axes[1,1].axvspan(3.5, 6, alpha=0.2, color='red', label='Стресс')
        axes[1,1].set_title(f'{non_responder_example} (НОН-РЕСПОНДЕР)\nАктивность КГР (Line Length)', fontweight='bold')
        axes[1,1].set_xlabel('Номер текста')
        axes[1,1].set_ylabel('Line Length (реальные единицы)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Сохраняем
    output_path = pathlib.Path("poligraph/data/result/demo_real_line_length_concept.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"График концепции Line Length на реальных данных сохранен: {output_path}")
    plt.show()
    
    return resp_baseline_mean, resp_stress_mean, resp_baseline_line, resp_stress_line

def demo_zscore_normalization_real_data(df):
    """Демонстрирует необходимость z-score нормализации на реальных данных"""
    if df.empty:
        print("❌ Нет данных для демонстрации z-score")
        return
    
    # Выбираем 3 участников с разными базовыми уровнями КГР
    selected_participants = ['1707LTA', '1707KAV', '1607LVA']  # Респондер, нон-респондер со снижением, стабильный
    
    # Подготавливаем данные для анализа
    comparison_data = []
    
    for participant in selected_participants:
        participant_data = df[df['Participant_ID'] == participant]
        
        if not participant_data.empty and 'scr r_Mean_Real' in participant_data.columns:
            # Реальные значения
            baseline_real = participant_data[participant_data['Period'] == 'Базовая линия (1-3)']['scr r_Mean_Real']
            stress_real = participant_data[participant_data['Period'] == 'Стресс (4-6)']['scr r_Mean_Real']
            
            # Z-score значения
            baseline_zscore = participant_data[participant_data['Period'] == 'Базовая линия (1-3)']['scr r_Mean']
            stress_zscore = participant_data[participant_data['Period'] == 'Стресс (4-6)']['scr r_Mean']
            
            # Добавляем данные
            for val in baseline_real:
                comparison_data.append({
                    'Participant': participant, 
                    'Condition': 'Базовая линия',
                    'Real_Value': val,
                    'Z_Score': np.nan  # Заполним позже
                })
            for val in stress_real:
                comparison_data.append({
                    'Participant': participant,
                    'Condition': 'Стресс', 
                    'Real_Value': val,
                    'Z_Score': np.nan
                })
            
            # Добавляем z-score значения
            idx = len(comparison_data) - len(baseline_real) - len(stress_real)
            for i, val in enumerate(baseline_zscore):
                comparison_data[idx + i]['Z_Score'] = val
            for i, val in enumerate(stress_zscore):  
                comparison_data[idx + len(baseline_real) + i]['Z_Score'] = val
    
    if not comparison_data:
        print("❌ Недостаточно данных для выбранных участников")
        return
    
    # Создаем DataFrame
    compare_df = pd.DataFrame(comparison_data)
    
    # Создаем графики сравнения
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Сравнение РЕАЛЬНЫХ ДАННЫХ: Реальные значения vs Z-score', 
                fontsize=16, fontweight='bold')
    
    # График 1: Реальные значения
    sns.boxplot(data=compare_df, x='Participant', y='Real_Value', hue='Condition', ax=axes[0])
    axes[0].set_title('Реальные значения КГР из эксперимента\n(сложно сравнивать между участниками)', 
                     fontsize=14, fontweight='bold')
    axes[0].set_ylabel('КГР (реальные единицы)', fontsize=12)
    axes[0].set_xlabel('ID участника', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    
    # График 2: Z-score значения  
    sns.boxplot(data=compare_df, x='Participant', y='Z_Score', hue='Condition', ax=axes[1])
    axes[1].set_title('Z-score нормализованные значения\n(легко сравнивать реакцию на стресс)', 
                     fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Z-score', fontsize=12)
    axes[1].set_xlabel('ID участника', fontsize=12)
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Сохраняем
    output_path = pathlib.Path("poligraph/data/result/demo_real_zscore_normalization.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"График z-score нормализации на реальных данных сохранен: {output_path}")
    plt.show()
    
    # Рассчитываем статистики по реальным участникам
    print("\n" + "="*60)
    print("СРАВНЕНИЕ РЕАЛЬНЫХ УЧАСТНИКОВ: РЕАЛЬНЫЕ ЗНАЧЕНИЯ vs Z-SCORE")
    print("="*60)
    
    for participant in selected_participants:
        participant_df = compare_df[compare_df['Participant'] == participant]
        
        if not participant_df.empty:
            # Реальные значения
            baseline_real = participant_df[participant_df['Condition'] == 'Базовая линия']['Real_Value'].mean()
            stress_real = participant_df[participant_df['Condition'] == 'Стресс']['Real_Value'].mean()
            change_real = stress_real - baseline_real
            change_pct = (change_real / baseline_real * 100) if baseline_real > 0 else 0
            
            # Z-score значения  
            baseline_zscore = participant_df[participant_df['Condition'] == 'Базовая линия']['Z_Score'].mean()
            stress_zscore = participant_df[participant_df['Condition'] == 'Стресс']['Z_Score'].mean()
            change_zscore = stress_zscore - baseline_zscore
            
            print(f"\n{participant}:")
            print(f"  Реальные значения:")
            print(f"    Базовая линия: {baseline_real:.4f} единиц")
            print(f"    Стресс:        {stress_real:.4f} единиц")  
            print(f"    Изменение:     {change_real:+.4f} единиц ({change_pct:+.1f}%)")
            print(f"  Z-score значения:")
            print(f"    Базовая линия: {baseline_zscore:.2f}")
            print(f"    Стресс:        {stress_zscore:.2f}")
            print(f"    Изменение:     {change_zscore:+.2f} (стандартных отклонений)")

def main():
    """Главная функция демонстрации на реальных данных"""
    print("🧠 ДЕМОНСТРАЦИЯ на РЕАЛЬНЫХ ДАННЫХ: Что такое КГР и зачем нужна z-score нормализация")
    print("=" * 80)
    
    # Загружаем реальные данные
    print("\n🔄 Загрузка реальных данных из эксперимента...")
    real_data = load_real_data()
    
    if real_data.empty:
        print("❌ Не удалось загрузить данные. Проверьте путь к файлам.")
        return
    
    print("\n1️⃣ Демонстрация концепции 'Line Length' на реальных данных...")
    resp_base, resp_stress, resp_base_line, resp_stress_line = demo_line_length_concept_real_data(real_data)
    
    print(f"\n📊 Результаты с реальными участниками:")
    print(f"   1707LTA (респондер):")
    print(f"     Среднее КГР: {resp_base:.4f} → {resp_stress:.4f} (изменение: {resp_stress-resp_base:+.4f})")
    print(f"     Line Length: {resp_base_line:.1f} → {resp_stress_line:.1f} (изменение: {resp_stress_line-resp_base_line:+.1f})")
    
    print("\n2️⃣ Демонстрация необходимости z-score нормализации на реальных данных...")
    demo_zscore_normalization_real_data(real_data)
    
    print("\n" + "="*80)
    print("📝 ВЫВОДЫ НА ОСНОВЕ РЕАЛЬНЫХ ДАННЫХ:")
    print("="*80)
    print("🎯 Средняя КГР - показывает общий уровень стресса участника")
    print("🎯 Активность КГР (Line Length) - показывает изменчивость/активность сигнала")  
    print("🎯 Z-score нужен потому что:")
    print("   • У реальных людей очень разные базовые уровни КГР")
    print("   • Без нормализации нельзя справедливо сравнивать участников") 
    print("   • Z-score показывает: насколько изменился сигнал относительно нормы участника")
    print("   • Формула: Z-score = (значение - среднее_участника) / стд_отклонение_участника")
    print("\n✅ В ваших графиках используются именно z-score значения")
    print("   для корректного выявления респондеров и нон-респондеров на стресс!")
    
    # Показываем участников
    responders = ['1707LTA', '1807KNV', '1607KYA', '1807OVA', '1807ZUG']  
    non_responders = ['1707KAV', '1807HEE', '1807CAA', '1607LVA', '1907ZSI']
    
    print(f"\n🎯 В эксперименте выделено:")
    print(f"   • Респондеры ({len(responders)}): {', '.join(responders)}")
    print(f"   • Нон-респондеры ({len(non_responders)}): {', '.join(non_responders)}")

if __name__ == "__main__":
    # Создаем результирующую папку
    pathlib.Path("poligraph/data/result").mkdir(parents=True, exist_ok=True)
    main() 