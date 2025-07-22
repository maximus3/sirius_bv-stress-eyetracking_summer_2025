import pathlib
import numpy as np
import pandas as pd
import codecs


def load_data(file_path):
    """Загрузка данных из .npy файла"""
    data = np.load(file_path, allow_pickle=True).item()
    return data["signals"], data["labels"], data["sampling_rate"]


# Он считывает  файл, .npy, который у вас появился в папке Resulte после предобработки
def normalize_features(df, feature_columns):
    """Z-нормализация признаков"""
    for col in feature_columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val > 0:
            df[col] = (df[col] - mean_val) / std_val
        else:
            df[col] = 0  # Обработка случая с нулевым стандартным отклонением
    return df


def calculate_line_length(signal):
    """Вычисление длины линии как суммы абсолютных разностей между соседними точками"""
    return np.sum(np.abs(np.diff(signal)))


def calculate_mean(signal):
    """Вычисление среднего значения сигнала"""
    return np.mean(signal)


def parse_log_file(log_path):
    """Парсинг лог-файла с метками стимулов"""
    intervals = []
    current_start = None
    current_label = None

    try:
        # Чтение файла с обработкой BOM
        with codecs.open(log_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                # Удаление BOM и лишних пробелов
                line = line.strip().replace("\ufeff", "")
                if not line:
                    continue

                # Разделение строки с учетом табуляции и пробелов
                parts = line.split()
                if len(parts) < 3:
                    continue

                try:
                    # Обработка времени и маркера
                    time = float(parts[0])
                    marker = int(parts[2])

                    # Обработка метки (может состоять из нескольких слов)
                    label = " ".join(parts[3:]) if len(parts) > 3 else None

                    if marker == 5:  # Начало стимула
                        if current_start is not None:
                            print(
                                f"Предупреждение: незакрытый интервал для метки '{current_label}'"
                            )
                        current_start = time
                        current_label = label
                    elif marker == 6 and current_start is not None:  # Конец стимула
                        intervals.append(
                            {
                                "label": current_label,
                                "start": current_start,
                                "end": time,
                            }
                        )
                        current_start = None
                        current_label = None
                except (ValueError, IndexError) as e:
                    print(f"Ошибка обработки строки: {line} - {str(e)}")
                    continue

    except Exception as e:
        print(f"Ошибка чтения лог-файла {log_path}: {str(e)}")

    return intervals


def process_file(file_path, log_path):
    """Обработка одного файла с использованием лог-файла"""
    try:
        # Загрузка данных полиграфа
        signals, labels, sr = load_data(file_path)
        signals = np.array(signals)

        # Парсинг лог-файла
        intervals = parse_log_file(log_path)
        if not intervals:
            print(f"Не найдено интервалов в файле: {log_path}")
            return None

        # Обработка каждого интервала
        results = []
        base_name = pathlib.Path(file_path).name.replace("_processed.npy", "")

        for interval in intervals:
            start_idx = int(interval["start"] * sr)
            end_idx = int(interval["end"] * sr)

            # Проверка корректности интервала
            if start_idx < 0 or end_idx > signals.shape[1] or start_idx >= end_idx:
                print(f"Некорректный интервал: {interval}")
                continue

            # Вырезаем интервал
            interval_signals = signals[:, start_idx:end_idx]

            # ВЫЧИСЛЕНИЕ ПАРАМЕТРОВ НА ИСХОДНЫХ СИГНАЛАХ (без нормализации)
            line_lengths_real = [calculate_line_length(chan) for chan in interval_signals]
            mean_values_real = [calculate_mean(chan) for chan in interval_signals]

            # Собираем результаты
            result = {
                "File": base_name,
                "Label": interval["label"],
                "Start_Time": interval["start"],
                "End_Time": interval["end"],
                "Duration": interval["end"] - interval["start"],
            }

            # Добавляем реальные характеристики для каждого канала
            for i, label in enumerate(labels):
                result[f"{label}_Line_Length_Real"] = line_lengths_real[i]
                result[f"{label}_Mean_Real"] = mean_values_real[i]

            # Добавляем нормализованные характеристики (заполним позже)
            # (Пока просто копируем реальные значения, нормализация будет после)
            for i, label in enumerate(labels):
                result[f"{label}_Line_Length"] = line_lengths_real[i]
                result[f"{label}_Mean"] = mean_values_real[i]

            results.append(result)

        return results

    except Exception as e:
        print(f"Ошибка обработки {file_path}: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def main():
    # Настройки путей
    current_dir = pathlib.Path(__file__).parent.resolve()
    data_dir = current_dir / "data/result/"
    log_dir = current_dir / "data/prepared_txt/"

    # Сбор и обработка файлов
    all_results = []

    for file_path in data_dir.glob("*_processed.npy"):
        # Формируем пути к файлам
        base_name = file_path.name.replace("_processed.npy", "")
        log_path = log_dir / f"{base_name}.txt"

        # Проверяем существование лог-файла
        if not log_path.exists():
            print(f"Лог-файл не найден: {log_path}")
            continue

        # Обрабатываем файл
        print(f"Обработка файла: {file_path.name}")
        file_results = process_file(file_path, log_path)
        if file_results:
            all_results.extend(file_results)
            print(f"  Найдено интервалов: {len(file_results)}")

    # Сохранение результатов
    if all_results:
        df = pd.DataFrame(all_results)

        # Упорядочиваем столбцы: мета, реальные, нормированные
        meta_cols = ["File", "Label", "Start_Time", "End_Time", "Duration"]
        real_cols = [col for col in df.columns if col.endswith("_Real")]
        feature_cols = [col for col in df.columns if col not in meta_cols + real_cols]
        df = df[meta_cols + real_cols + feature_cols]

        # НОРМАЛИЗАЦИЯ ПРИЗНАКОВ ПОСЛЕ ВЫЧИСЛЕНИЯ (только для нормированных)
        print("\nПрименение Z-нормализации к признакам...")
        df = normalize_features(df, feature_cols)

        # Сохранение результатов
        output_path = data_dir / "Signal_Analysis_Results_Normalized.xlsx"
        df.to_excel(str(output_path), index=False)
        print(f"\nРезультаты сохранены в: {output_path}")
        print(f"Обработано файлов: {len(set(df['File']))}")
        print(f"Обработано интервалов: {len(df)}")

        # Краткая статистика
        print("\nСтатистика по длине линии (реальные значения):")
        line_cols_real = [col for col in df.columns if "Line_Length_Real" in col]
        print(df[line_cols_real].describe().transpose())

        print("\nСтатистика по средним значениям (реальные значения):")
        mean_cols_real = [col for col in df.columns if "_Mean_Real" in col and "Line_Length" not in col]
        print(df[mean_cols_real].describe().transpose())

        print("\nСтатистика по длине линии (после нормализации):")
        line_cols = [col for col in df.columns if "Line_Length" in col and not col.endswith("_Real")]
        print(df[line_cols].describe().transpose())

        print("\nСтатистика по средним значениям (после нормализации):")
        mean_cols = [col for col in df.columns if "_Mean" in col and not col.endswith("_Real") and "Line_Length" not in col]
        print(df[mean_cols].describe().transpose())
    else:
        print("Нет данных для сохранения")


if __name__ == "__main__":
    main()
