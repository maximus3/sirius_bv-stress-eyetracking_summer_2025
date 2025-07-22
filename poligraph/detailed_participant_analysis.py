import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import warnings
warnings.filterwarnings('ignore')

# Настройка matplotlib для русского языка
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Tahoma', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def analyze_stress_responders():
    """Детальный анализ участников, реагировавших на индукцию стресса"""
    
    # Загружаем данные из предыдущего анализа
    results_path = pathlib.Path("poligraph/stress_dynamics_results")
    
    try:
        stress_by_text = pd.read_excel(results_path / 'detailed_stress_by_text.xlsx')
        dynamics_summary = pd.read_excel(results_path / 'stress_dynamics_summary.xlsx')
        
        print("=== ДЕТАЛЬНЫЙ АНАЛИЗ УЧАСТНИКОВ ===\n")
        
        # Разделяем участников на группы
        responders = dynamics_summary[dynamics_summary['Responded_to_Induction'] == True]
        non_responders = dynamics_summary[dynamics_summary['Responded_to_Induction'] == False]
        
        print("📊 РЕСПОНДЕРЫ (реагировали на стресс):")
        for _, participant in responders.iterrows():
            participant_id = participant['Participant_ID']
            baseline = participant['Baseline_Stress']
            post_induction = participant['Post_Induction_Stress']
            text4_stress = participant['Text4_Stress']
            change_percent = participant['Stress_Change_Percent']
            
            print(f"  • {participant_id}: Базовая линия = {baseline:.2f}, После индукции = {post_induction:.2f}")
            print(f"    Стресс в 4-м тексте = {text4_stress:.2f}, Изменение = {change_percent:.1f}%")
        
        print(f"\n📊 НОН-РЕСПОНДЕРЫ (не реагировали на стресс):")
        for _, participant in non_responders.iterrows():
            participant_id = participant['Participant_ID']
            baseline = participant['Baseline_Stress']
            post_induction = participant['Post_Induction_Stress']
            text4_stress = participant['Text4_Stress']
            change_percent = participant['Stress_Change_Percent']
            
            print(f"  • {participant_id}: Базовая линия = {baseline:.2f}, После индукции = {post_induction:.2f}")
            print(f"    Стресс в 4-м тексте = {text4_stress:.2f}, Изменение = {change_percent:.1f}%")
        
        # Анализируем паттерны по текстам
        print(f"\n=== АНАЛИЗ ПАТТЕРНОВ ПО ТЕКСТАМ ===")
        
        # Средние значения по текстам для каждой группы
        responder_ids = responders['Participant_ID'].tolist()
        non_responder_ids = non_responders['Participant_ID'].tolist()
        
        responder_data = stress_by_text[stress_by_text['Participant_ID'].isin(responder_ids)]
        non_responder_data = stress_by_text[stress_by_text['Participant_ID'].isin(non_responder_ids)]
        
        print("\nСредние значения стресса по текстам:")
        print("Текст | Респондеры | Нон-респондеры | Разница")
        print("-" * 50)
        
        for text_num in range(1, 7):
            resp_mean = responder_data[responder_data['Text_Number'] == text_num]['Stress_Score'].mean()
            non_resp_mean = non_responder_data[non_responder_data['Text_Number'] == text_num]['Stress_Score'].mean()
            diff = resp_mean - non_resp_mean
            
            print(f"  {text_num}   |    {resp_mean:.2f}     |      {non_resp_mean:.2f}      |  {diff:.2f}")
        
        # Создаем дополнительные графики
        create_detailed_visualizations(stress_by_text, responders, non_responders, results_path)
        
        # Анализируем самых ярких представителей каждой группы
        analyze_extreme_cases(stress_by_text, responders, non_responders)
        
        return responders, non_responders
        
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None, None

def create_detailed_visualizations(stress_by_text, responders, non_responders, results_path):
    """Создает детальные графики для анализа групп участников"""
    
    # График 1: Сравнение траекторий респондеров и нон-респондеров
    plt.figure(figsize=(14, 10))
    
    # Данные для групп
    responder_ids = responders['Participant_ID'].tolist()
    non_responder_ids = non_responders['Participant_ID'].tolist()
    
    responder_data = stress_by_text[stress_by_text['Participant_ID'].isin(responder_ids)]
    non_responder_data = stress_by_text[stress_by_text['Participant_ID'].isin(non_responder_ids)]
    
    # Средние траектории
    resp_means = responder_data.groupby('Text_Number')['Stress_Score'].agg(['mean', 'std'])
    non_resp_means = non_responder_data.groupby('Text_Number')['Stress_Score'].agg(['mean', 'std'])
    
    # Рисуем средние с доверительными интервалами
    plt.plot(resp_means.index, resp_means['mean'], 'ro-', linewidth=3, markersize=8, 
             label=f'Респондеры (n={len(responders)})', color='red')
    plt.fill_between(resp_means.index, 
                    resp_means['mean'] - resp_means['std'],
                    resp_means['mean'] + resp_means['std'],
                    alpha=0.3, color='red')
    
    plt.plot(non_resp_means.index, non_resp_means['mean'], 'bo-', linewidth=3, markersize=8, 
             label=f'Нон-респондеры (n={len(non_responders)})', color='blue')
    plt.fill_between(non_resp_means.index, 
                    non_resp_means['mean'] - non_resp_means['std'],
                    non_resp_means['mean'] + non_resp_means['std'],
                    alpha=0.3, color='blue')
    
    # Отмечаем момент индукции стресса
    plt.axvline(x=3.5, color='black', linestyle='--', linewidth=2, alpha=0.7)
    plt.text(3.6, plt.ylim()[1]*0.9, 'Индукция\nстресса', fontsize=12, fontweight='bold')
    
    # Выделяем фазы
    plt.axvspan(0.5, 3.5, alpha=0.2, color='gray', label='Базовая линия')
    plt.axvspan(3.5, 6.5, alpha=0.2, color='orange', label='После индукции')
    
    plt.title('Сравнение траекторий стресса: респондеры vs нон-респондеры', fontsize=16, fontweight='bold')
    plt.xlabel('Номер текста', fontsize=12)
    plt.ylabel('Средний индекс стресса', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 7))
    plt.tight_layout()
    plt.savefig(results_path / 'responders_vs_non_responders.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # График 2: Детальный анализ 4-го текста по группам
    plt.figure(figsize=(12, 8))
    
    text4_responders = responder_data[responder_data['Text_Number'] == 4]['Stress_Score']
    text4_non_responders = non_responder_data[non_responder_data['Text_Number'] == 4]['Stress_Score']
    
    # Box plot для сравнения
    data_for_boxplot = [text4_responders, text4_non_responders]
    labels = ['Респондеры', 'Нон-респондеры']
    
    plt.boxplot(data_for_boxplot, labels=labels, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    
    # Добавляем точки для каждого участника
    for i, data in enumerate(data_for_boxplot):
        y = data
        x = np.random.normal(i+1, 0.04, size=len(y))
        plt.scatter(x, y, alpha=0.7, s=50)
    
    plt.title('Сравнение стресса в 4-м тексте между группами', fontsize=16, fontweight='bold')
    plt.ylabel('Индекс стресса в 4-м тексте', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_path / 'text4_comparison_groups.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # График 3: Индивидуальные профили самых ярких представителей
    plt.figure(figsize=(15, 10))
    
    # Выбираем по 3 самых ярких представителя каждой группы
    top_responders = responders.nlargest(3, 'Stress_Change')['Participant_ID'].tolist()
    top_non_responders = non_responders.nsmallest(3, 'Stress_Change')['Participant_ID'].tolist()
    
    plt.subplot(2, 1, 1)
    for participant in top_responders:
        participant_data = stress_by_text[stress_by_text['Participant_ID'] == participant]
        plt.plot(participant_data['Text_Number'], participant_data['Stress_Score'], 
                'o-', linewidth=2, markersize=6, label=participant)
    
    plt.axvline(x=3.5, color='black', linestyle='--', alpha=0.7)
    plt.title('ТОП-3 респондера (наибольший прирост стресса)', fontsize=14, fontweight='bold')
    plt.xlabel('Номер текста')
    plt.ylabel('Индекс стресса')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 7))
    
    plt.subplot(2, 1, 2)
    for participant in top_non_responders:
        participant_data = stress_by_text[stress_by_text['Participant_ID'] == participant]
        plt.plot(participant_data['Text_Number'], participant_data['Stress_Score'], 
                'o-', linewidth=2, markersize=6, label=participant)
    
    plt.axvline(x=3.5, color='black', linestyle='--', alpha=0.7)
    plt.title('ТОП-3 нон-респондера (наибольшее снижение стресса)', fontsize=14, fontweight='bold')
    plt.xlabel('Номер текста')
    plt.ylabel('Индекс стресса')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 7))
    
    plt.tight_layout()
    plt.savefig(results_path / 'individual_extreme_profiles.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_extreme_cases(stress_by_text, responders, non_responders):
    """Анализирует самых ярких представителей каждой группы"""
    
    print("\n=== АНАЛИЗ ЯРКИХ СЛУЧАЕВ ===")
    
    # Самый яркий респондер
    top_responder = responders.loc[responders['Stress_Change'].idxmax()]
    print(f"\n🔥 САМЫЙ ЯРКИЙ РЕСПОНДЕР: {top_responder['Participant_ID']}")
    print(f"   Прирост стресса: {top_responder['Stress_Change']:.2f} ({top_responder['Stress_Change_Percent']:.1f}%)")
    print(f"   Базовая линия: {top_responder['Baseline_Stress']:.2f}")
    print(f"   После индукции: {top_responder['Post_Induction_Stress']:.2f}")
    print(f"   Стресс в 4-м тексте: {top_responder['Text4_Stress']:.2f}")
    
    # Детальный профиль по текстам для самого яркого респондера
    top_resp_profile = stress_by_text[stress_by_text['Participant_ID'] == top_responder['Participant_ID']]
    print("   Профиль по текстам:")
    for _, row in top_resp_profile.iterrows():
        text_num = int(row['Text_Number'])
        stress_score = row['Stress_Score']
        phase = "Базовая" if text_num <= 3 else "Стресс"
        print(f"     Текст {text_num} ({phase}): {stress_score:.2f}")
    
    # Самый яркий нон-респондер
    top_non_responder = non_responders.loc[non_responders['Stress_Change'].idxmin()]
    print(f"\n❄️ САМЫЙ ЯРКИЙ НОН-РЕСПОНДЕР: {top_non_responder['Participant_ID']}")
    print(f"   Изменение стресса: {top_non_responder['Stress_Change']:.2f} ({top_non_responder['Stress_Change_Percent']:.1f}%)")
    print(f"   Базовая линия: {top_non_responder['Baseline_Stress']:.2f}")
    print(f"   После индукции: {top_non_responder['Post_Induction_Stress']:.2f}")
    print(f"   Стресс в 4-м тексте: {top_non_responder['Text4_Stress']:.2f}")
    
    # Детальный профиль по текстам для самого яркого нон-респондера
    top_non_resp_profile = stress_by_text[stress_by_text['Participant_ID'] == top_non_responder['Participant_ID']]
    print("   Профиль по текстам:")
    for _, row in top_non_resp_profile.iterrows():
        text_num = int(row['Text_Number'])
        stress_score = row['Stress_Score']
        phase = "Базовая" if text_num <= 3 else "Стресс"
        print(f"     Текст {text_num} ({phase}): {stress_score:.2f}")
    
    # Анализ средних показателей по группам
    print(f"\n=== ГРУППОВЫЕ ХАРАКТЕРИСТИКИ ===")
    print(f"Респондеры:")
    print(f"  • Средний прирост стресса: {responders['Stress_Change'].mean():.2f}")
    print(f"  • Средняя базовая линия: {responders['Baseline_Stress'].mean():.2f}")
    print(f"  • Средний стресс после индукции: {responders['Post_Induction_Stress'].mean():.2f}")
    print(f"  • Средний стресс в 4-м тексте: {responders['Text4_Stress'].mean():.2f}")
    
    print(f"\nНон-респондеры:")
    print(f"  • Среднее изменение стресса: {non_responders['Stress_Change'].mean():.2f}")
    print(f"  • Средняя базовая линия: {non_responders['Baseline_Stress'].mean():.2f}")
    print(f"  • Средний стресс после индукции: {non_responders['Post_Induction_Stress'].mean():.2f}")
    print(f"  • Средний стресс в 4-м тексте: {non_responders['Text4_Stress'].mean():.2f}")

def create_summary_report(responders, non_responders):
    """Создает итоговый отчет о результатах эксперимента"""
    
    results_path = pathlib.Path("poligraph/stress_dynamics_results")
    
    print(f"\n" + "="*60)
    print("ИТОГОВЫЙ ОТЧЕТ О РЕЗУЛЬТАТАХ ЭКСПЕРИМЕНТА")
    print("="*60)
    
    print(f"\n📊 ОБЩИЕ РЕЗУЛЬТАТЫ:")
    total_participants = len(responders) + len(non_responders)
    responder_percentage = len(responders) / total_participants * 100
    
    print(f"• Всего участников: {total_participants}")
    print(f"• Респондеры (реагировали на стресс): {len(responders)} ({responder_percentage:.1f}%)")
    print(f"• Нон-респондеры (не реагировали): {len(non_responders)} ({100-responder_percentage:.1f}%)")
    
    print(f"\n🎯 ЭФФЕКТИВНОСТЬ ИНДУКЦИИ СТРЕССА:")
    if responder_percentage >= 50:
        effectiveness = "ВЫСОКАЯ"
    elif responder_percentage >= 30:
        effectiveness = "СРЕДНЯЯ"
    else:
        effectiveness = "НИЗКАЯ"
    
    print(f"• Эффективность: {effectiveness}")
    print(f"• {responder_percentage:.1f}% участников показали увеличение стресса после индукции")
    
    print(f"\n📈 КЛЮЧЕВЫЕ НАХОДКИ:")
    print(f"• Средний прирост стресса у респондеров: {responders['Stress_Change'].mean():.2f}")
    print(f"• Максимальный прирост стресса: {responders['Stress_Change'].max():.2f}")
    print(f"• Участник с наибольшим приростом: {responders.loc[responders['Stress_Change'].idxmax()]['Participant_ID']}")
    
    if len(non_responders) > 0:
        print(f"• Среднее изменение у нон-респондеров: {non_responders['Stress_Change'].mean():.2f}")
        print(f"• Участник с наибольшим снижением: {non_responders.loc[non_responders['Stress_Change'].idxmin()]['Participant_ID']}")
    
    print(f"\n💡 РЕКОМЕНДАЦИИ:")
    if responder_percentage < 50:
        print("• Рассмотреть усиление протокола индукции стресса")
        print("• Проанализировать индивидуальные различия участников")
    
    print("• Изучить факторы, влияющие на чувствительность к стрессу")
    print("• Рассмотреть дополнительные физиологические маркеры")
    
    # Сохраняем отчет в файл
    report_data = {
        'Показатель': [
            'Всего участников',
            'Респондеры',
            'Нон-респондеры',
            'Эффективность индукции (%)',
            'Средний прирост у респондеров',
            'Максимальный прирост',
            'Лучший респондер'
        ],
        'Значение': [
            total_participants,
            len(responders),
            len(non_responders),
            f"{responder_percentage:.1f}%",
            f"{responders['Stress_Change'].mean():.2f}",
            f"{responders['Stress_Change'].max():.2f}",
            responders.loc[responders['Stress_Change'].idxmax()]['Participant_ID']
        ]
    }
    
    report_df = pd.DataFrame(report_data)
    report_df.to_excel(results_path / 'final_experiment_report.xlsx', index=False)
    
    print(f"\n📄 Итоговый отчет сохранен: {results_path}/final_experiment_report.xlsx")

if __name__ == "__main__":
    responders, non_responders = analyze_stress_responders()
    if responders is not None and non_responders is not None:
        create_summary_report(responders, non_responders) 