import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import shutil


def batch_smart_resize(input_folder, output_folder, max_size_mb=1, max_dim=1920):
    """
    Функция для пакетной обработки изображений с сохранением структуры папок

    Args:
        input_folder: исходная папка с изображениями
        output_folder: папка для сохранения результатов
        max_size_mb: максимальный размер файла в МБ (используется для контроля качества)
        max_dim: максимальная длинная сторона в пикселях
    """
    # Создаем выходную папку
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Поддерживаемые форматы (добавил больше форматов)
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.ico')

    # Собираем все файлы рекурсивно из всех подпапок
    images = []
    input_path = Path(input_folder)

    # Используем rglob для рекурсивного поиска
    for ext in extensions:
        # Ищем файлы с разным регистром
        images.extend(input_path.rglob(f'*{ext}'))
        images.extend(input_path.rglob(f'*{ext.upper()}'))

    print(f"Найдено {len(images)} изображений")

    # Статистика
    stats = {
        'processed': 0,
        'failed': 0,
        'skipped': 0,
        'total_size_before': 0,
        'total_size_after': 0
    }

    # Обрабатываем каждое изображение
    for img_path in tqdm(images, desc="Обработка"):
        try:
            # Получаем относительный путь от исходной папки
            relative_path = img_path.relative_to(input_path)

            # Формируем путь для сохранения с сохранением структуры папок
            # Меняем расширение на .png
            output_path = Path(output_folder) / relative_path.parent / f"{img_path.stem}.png"

            # Создаем подпапки если их нет
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Получаем размер исходного файла
            original_size = os.path.getsize(img_path)
            stats['total_size_before'] += original_size

            # Открываем и обрабатываем
            with Image.open(img_path) as img:
                # Уменьшаем размер при необходимости
                if max(img.size) > max_dim:
                    ratio = max_dim / max(img.size)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)

                # Сохраняем в PNG (поддерживает прозрачность)
                # PNG сохраняется без потери качества, но мы можем оптимизировать

                # Оптимизация PNG
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    # Изображение с прозрачностью - сохраняем как RGBA
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    img.save(output_path, 'PNG', optimize=True)
                else:
                    # Изображение без прозрачности - можно конвертировать в RGB для экономии места
                    # Но для единообразия все равно сохраняем как PNG
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(output_path, 'PNG', optimize=True)

                # Проверяем размер после обработки
                new_size = os.path.getsize(output_path)
                stats['total_size_after'] += new_size

                # Если размер все еще больше лимита, пробуем дополнительную оптимизацию
                if new_size > max_size_mb * 1024 * 1024:
                    # Пробуем уменьшить качество (для PNG это сложнее, но можно попробовать)
                    # Пересохраняем с меньшей глубиной цвета если возможно
                    try:
                        with Image.open(output_path) as optimized_img:
                            # Пробуем конвертировать в P режим (палитра) если нет прозрачности
                            if optimized_img.mode == 'RGBA':
                                # Сохраняем прозрачность но пробуем оптимизировать палитру
                                optimized_img = optimized_img.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
                                optimized_img = optimized_img.convert('RGBA')
                            elif optimized_img.mode == 'RGB':
                                optimized_img = optimized_img.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
                                optimized_img = optimized_img.convert('RGB')

                            optimized_img.save(output_path, 'PNG', optimize=True)
                    except:
                        pass  # Если не получилось, оставляем как есть

                stats['processed'] += 1

        except Exception as e:
            print(f"\nОшибка при обработке {img_path.name}: {e}")
            stats['failed'] += 1

    # Выводим статистику
    print("\n" + "=" * 50)
    print("СТАТИСТИКА ОБРАБОТКИ:")
    print(f"Успешно обработано: {stats['processed']}")
    print(f"Ошибок: {stats['failed']}")
    print(f"Пропущено: {stats['skipped']}")

    if stats['total_size_before'] > 0:
        saved_mb = (stats['total_size_before'] - stats['total_size_after']) / (1024 * 1024)
        saved_percent = (1 - stats['total_size_after'] / stats['total_size_before']) * 100
        print(f"Размер до: {stats['total_size_before'] / (1024 * 1024):.2f} МБ")
        print(f"Размер после: {stats['total_size_after'] / (1024 * 1024):.2f} МБ")
        print(f"Сэкономлено: {saved_mb:.2f} МБ ({saved_percent:.1f}%)")
    print("=" * 50)


# Улучшенная версия с дополнительными опциями
def batch_smart_resize_advanced(input_folder, output_folder,
                                max_size_mb, max_dim,
                                preserve_structure=True,
                                copy_if_smaller=True):
    """
    Расширенная версия с дополнительными опциями

    Args:
        input_folder: исходная папка
        output_folder: папка для результатов
        max_size_mb: максимальный размер файла в МБ
        max_dim: максимальная длинная сторона
        preserve_structure: сохранять структуру папок
        copy_if_smaller: копировать оригинал если он уже меньше лимита
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff')
    input_path = Path(input_folder)

    # Собираем файлы
    images = []
    for ext in extensions:
        images.extend(input_path.rglob(f'*{ext}'))
        images.extend(input_path.rglob(f'*{ext.upper()}'))

    print(f"Найдено {len(images)} изображений")

    for img_path in tqdm(images, desc="Обработка"):
        try:
            # Определяем путь для сохранения
            if preserve_structure:
                relative_path = img_path.relative_to(input_path)
                output_path = Path(output_folder) / relative_path.parent / f"{img_path.stem}.png"
            else:
                output_path = Path(output_folder) / f"{img_path.stem}.png"

            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Проверяем размер оригинала
            original_size = os.path.getsize(img_path)
            size_limit = max_size_mb * 1024 * 1024

            # Если оригинал уже меньше лимита и это PNG, просто копируем
            if copy_if_smaller and original_size <= size_limit and img_path.suffix.lower() == '.png':
                shutil.copy2(img_path, output_path)
                continue

            # Обрабатываем изображение
            with Image.open(img_path) as img:
                # Уменьшаем размер
                if max(img.size) > max_dim:
                    ratio = max_dim / max(img.size)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)

                # Сохраняем с учетом прозрачности
                has_alpha = img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info)

                if has_alpha:
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                else:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                # Сохраняем с оптимизацией
                img.save(output_path, 'PNG', optimize=True)

        except Exception as e:
            print(f"\nОшибка при обработке {img_path.name}: {e}")


# Использование
if __name__ == "__main__":
    # Простой вариант
    # batch_smart_resize(
    #     input_folder='исходная_папка',  # Замените на ваш путь
    #     output_folder='optimized_png',  # Папка для результатов
    #     max_size_mb=1,  # Максимальный размер 1 МБ
    #     max_dim=1920  # Максимальная сторона 1920px
    # )

    # Или расширенный вариант:
    batch_smart_resize_advanced(
        input_folder='1',
        output_folder='optimized_png',
        max_size_mb=0.5,
        max_dim=600,
        preserve_structure=True,  # Сохраняем структуру папок
        copy_if_smaller=True       # Копируем маленькие PNG без обработки
    )