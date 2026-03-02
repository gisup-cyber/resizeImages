import os
from PIL import Image, ImageOps
from pathlib import Path
from tqdm import tqdm
import shutil
import numpy as np

def crop_to_content(image, padding=5, bg_tolerance=30):
    """
    Обрезает изображение до границ непрозрачного содержимого

    Args:
        image: PIL Image объект
        padding: количество пикселей отступа от края объекта
        bg_tolerance: допуск для определения прозрачности (0-255)

    Returns:
        обрезанное PIL Image изображение
    """
    # Конвертируем в RGBA если нужно
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    # Получаем альфа-канал
    alpha = image.split()[3]

    # Конвертируем в numpy array для анализа
    alpha_array = np.array(alpha)

    # Находим границы непрозрачных пикселей
    # Считаем пиксель непрозрачным, если его альфа > tolerance
    non_transparent = np.where(alpha_array > bg_tolerance)

    if len(non_transparent[0]) == 0:
        # Изображение полностью прозрачное - возвращаем как есть
        return image

    # Находим минимальные и максимальные координаты
    y_min, y_max = np.min(non_transparent[0]), np.max(non_transparent[0])
    x_min, x_max = np.min(non_transparent[1]), np.max(non_transparent[1])

    # Добавляем отступы
    y_min = max(0, y_min - padding)
    y_max = min(image.height, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(image.width, x_max + padding)

    # Обрезаем изображение
    cropped = image.crop((x_min, y_min, x_max, y_max))

    return cropped

def crop_with_content_detection(image, padding=5, bg_color=None, tolerance=30):
    """
    Обрезает изображение до границ содержимого.
    Для изображений с фоном - определяет границы по цвету фона.

    Args:
        image: PIL Image объект
        padding: отступ от края объекта
        bg_color: цвет фона (если известен), иначе определяется автоматически
        tolerance: допуск при определении фона
    """
    # Если есть прозрачность, используем альфа-канал
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        return crop_to_content(image, padding, tolerance)

    # Для изображений без прозрачности пытаемся определить фон по краям
    # Конвертируем в RGB для анализа
    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_array = np.array(image)

    # Определяем цвет фона (берем цвет из углов)
    if bg_color is None:
        # Анализируем 4 угла изображения
        corners = [
            img_array[0, 0],  # верхний левый
            img_array[0, -1],  # верхний правый
            img_array[-1, 0],  # нижний левый
            img_array[-1, -1]  # нижний правый
        ]
        # Усредняем цвета углов
        bg_color = np.mean(corners, axis=0).astype(int)

    # Создаем маску для пикселей, отличающихся от фона
    color_diffs = np.abs(img_array - bg_color)
    is_foreground = np.any(color_diffs > tolerance, axis=2)

    # Находим границы переднего плана
    rows = np.any(is_foreground, axis=1)
    cols = np.any(is_foreground, axis=0)

    if not np.any(rows) or not np.any(cols):
        # Изображение скорее всего однотонное
        return image

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Добавляем отступы
    y_min = max(0, y_min - padding)
    y_max = min(image.height, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(image.width, x_max + padding)

    # Обрезаем
    cropped = image.crop((x_min, y_min, x_max, y_max))

    return cropped

def batch_smart_resize_advanced(input_folder, output_folder,
                                max_size_mb, max_dim,
                                preserve_structure=True,
                                copy_if_smaller=True,
                                crop_content=True,
                                crop_padding=5,
                                crop_tolerance=30):
    """
    Расширенная версия с дополнительными опциями

    Args:
        input_folder: исходная папка
        output_folder: папка для результатов
        max_size_mb: максимальный размер файла в МБ
        max_dim: максимальная длинная сторона
        preserve_structure: сохранять структуру папок
        copy_if_smaller: копировать оригинал если он уже меньше лимита
        crop_content: обрезать до границ содержимого
        crop_padding: отступ от границ объекта в пикселях
        crop_tolerance: допуск для определения фона/прозрачности
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

    # Статистика
    stats = {
        'processed': 0,
        'failed': 0,
        'skipped': 0,
        'cropped': 0,
        'total_size_before': 0,
        'total_size_after': 0
    }

    for img_path in tqdm(images, desc="Обработка"):
        try:
            # Определяем путь для сохранения
            if preserve_structure:
                relative_path = img_path.relative_to(input_path)
                output_path = Path(output_folder) / relative_path.parent / f"{img_path.stem}.png"
            else:
                output_path = Path(output_folder) / f"{img_path.stem}.png"

            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Получаем размер исходного файла
            original_size = os.path.getsize(img_path)
            stats['total_size_before'] += original_size

            size_limit = max_size_mb * 1024 * 1024

            # Если оригинал уже меньше лимита и это PNG, просто копируем
            if copy_if_smaller and original_size <= size_limit and img_path.suffix.lower() == '.png':
                shutil.copy2(img_path, output_path)
                stats['skipped'] += 1
                continue

            # Обрабатываем изображение
            with Image.open(img_path) as img:
                # Обрезаем до содержимого если нужно
                if crop_content:
                    original_size_before_crop = img.size
                    img = crop_with_content_detection(img, padding=crop_padding, tolerance=crop_tolerance)
                    if img.size != original_size_before_crop:
                        stats['cropped'] += 1

                # Уменьшаем размер если превышает max_dim
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

                # Обновляем статистику
                new_size = os.path.getsize(output_path)
                stats['total_size_after'] += new_size
                stats['processed'] += 1

        except Exception as e:
            print(f"\nОшибка при обработке {img_path.name}: {e}")
            stats['failed'] += 1

    # Выводим статистику
    print("\n" + "=" * 50)
    print("СТАТИСТИКА ОБРАБОТКИ:")
    print(f"Успешно обработано: {stats['processed']}")
    print(f"Обрезано до содержимого: {stats['cropped']}")
    print(f"Ошибок: {stats['failed']}")
    print(f"Пропущено (уже оптимизированы): {stats['skipped']}")

    if stats['total_size_before'] > 0:
        saved_mb = (stats['total_size_before'] - stats['total_size_after']) / (1024 * 1024)
        saved_percent = (1 - stats['total_size_after'] / stats['total_size_before']) * 100
        print(f"Размер до: {stats['total_size_before'] / (1024 * 1024):.2f} МБ")
        print(f"Размер после: {stats['total_size_after'] / (1024 * 1024):.2f} МБ")
        print(f"Сэкономлено: {saved_mb:.2f} МБ ({saved_percent:.1f}%)")
    print("=" * 50)


# Функция для предварительного просмотра результатов обрезки
def preview_crop(image_path, padding=5, tolerance=30, save_preview=False):
    """
    Показывает предварительный результат обрезки
    """
    with Image.open(image_path) as img:
        print(f"Оригинал: {img.size}")

        # Обрезаем
        cropped = crop_with_content_detection(img, padding=padding, tolerance=tolerance)
        print(f"После обрезки: {cropped.size}")

        if save_preview:
            preview_path = f"preview_crop_{Path(image_path).stem}.png"
            cropped.save(preview_path)
            print(f"Предварительный просмотр сохранен в {preview_path}")

        return cropped


# Использование
if __name__ == "__main__":
    # Пример с обрезкой по содержимому
    batch_smart_resize_advanced(
        input_folder='1',
        output_folder='optimized_png_cropped',
        max_size_mb=0.7,
        max_dim=1024,
        preserve_structure=True,
        copy_if_smaller=True,
        crop_content=True,        # Включаем обрезку
        crop_padding=10,           # Отступ 10 пикселей от края объекта
        crop_tolerance=25          # Чувствительность определения границ
    )

    # Для тестирования на одном изображении:
    # preview_crop('test_image.png', padding=15, save_preview=True)