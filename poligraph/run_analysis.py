#!/usr/bin/env python3
"""
Скрипт для запуска анализа данных стресса участников

Использование:
    python run_analysis.py
"""

from stress_analysis_script import StressAnalyzer

def main():
    """Главная функция"""
    try:
        print("Запуск анализа данных стресса...")
        analyzer = StressAnalyzer()
        results = analyzer.run_analysis()
        
        print("\n" + "="*50)
        print("АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
        print("="*50)
        print("\nКлючевые результаты:")
        
        if 'stress_data' in results:
            stress_df = results['stress_data']
            print(f"• Проанализировано участников: {len(stress_df)}")
            
            high_stress = len(stress_df[stress_df['Stress_Score'] >= 6])
            medium_stress = len(stress_df[(stress_df['Stress_Score'] >= 3) & (stress_df['Stress_Score'] < 6)])
            low_stress = len(stress_df[stress_df['Stress_Score'] < 3])
            
            print(f"• Участники с высоким стрессом: {high_stress}")
            print(f"• Участники со средним стрессом: {medium_stress}")
            print(f"• Участники с низким стрессом: {low_stress}")
        
        if 'text_stress_data' in results:
            text_df = results['text_stress_data']
            if not text_df.empty:
                high_text_stress = len(text_df[text_df['Stress_Level'] == 'Высокий'])
                print(f"• Случаев высокого стресса при чтении: {high_text_stress}")
        
        print("\nВсе файлы результатов сохранены в папке: poligraph/analysis_results/")
        
    except Exception as e:
        print(f"ОШИБКА при выполнении анализа: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 