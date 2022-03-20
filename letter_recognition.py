import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

from typing import Any
from pathlib import Path
from collections import defaultdict
from skimage.measure import label, regionprops


def bbox_center(bbox) -> tuple:
    """Центр bbox

    Параметры
    ---------
    bbox : list
        Bounding box

    Возвращаемое значение
    ---------------------
    tuple
        Центр x и y
    """
    return (bbox[0] + (bbox[2] - bbox[0]) / 2, bbox[1] + (bbox[3] - bbox[1]) / 2)


class ImageToText:
    def __init__(self, image: np.ndarray = None) -> None:
        if not image is None:
            self.__image: np.ndarray = self.set_image(image)
        else:
            self.__image: np.ndarray = None

        self.__knn: Any = None
        pass

    def __extract_features(self, image: np.ndarray) -> list:
        """Извлечение особенности у каждой буквы

        Параметры
        ---------
        image : np.ndarray
            Бинарная картинка

        Возвращаемое значение
        ---------------------
        list
            Список особенных признаков у буквы
        """
        features: list = []

        # [Next, Previous, First_child, Parent]
        _, hierachy = cv2.findContours(
            image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        ext_cnt = 0
        int_cnt = 0
        for i in range(len(hierachy[0])):
            if hierachy[0][i][-1] == -1:
                ext_cnt += 1
            elif hierachy[0][i][-1] == 0:
                int_cnt += 1

        features.extend([ext_cnt, int_cnt])
        labeled: np.ndarray = label(image)

        region = regionprops(labeled)[0]
        filling_factor = region.area / region.bbox_area

        features.append(filling_factor)

        centroid: np.ndarray = np.array(region.local_centroid) / \
            np.array(region.image.shape)
        features.extend(centroid)
        features.append(region.eccentricity)
        features.append(region.orientation)

        return features

    def __label_english_letters(self) -> np.ndarray:
        """Макрировка английских букв

        Возвращаемое значение
        ---------------------
        np.ndarray
            Маркированное изображение
        """

        image_copy: np.ndarray = self.__image.copy()
        labeled: np.ndarray = label(image_copy)
        regions: list = regionprops(labeled)

        for region1 in regions:

            bbox_region_1: tuple = region1.bbox  # y1, x1, y2, x2
            bbox_center1: tuple = bbox_center(bbox_region_1)
            for region2 in regions:

                bbox_region_2: tuple = region2.bbox
                bbox_center_2: tuple = bbox_center(bbox_region_2)

                if bbox_region_1[0] > bbox_region_2[2] and abs(bbox_center1[1] - bbox_center_2[1]) < 10 and bbox_center1[0] > bbox_center_2[0]:
                    for y in range(labeled.shape[0]):
                        for x in range(labeled.shape[1]):
                            if labeled[y][x] == region2.label:
                                labeled[y][x] = region1.label

                    break

        return labeled

    def __sort_letters(self, image: np.ndarray) -> list:
        """Сортировка букв по порядку, как на картинке

        Параметры
        ---------
        image : np.ndarray
            Бинарная картинка

        Возвращаемое значение
        ---------------------
        list
            Отсортированный bounding box
        """
        bboxs: list = [region.bbox for region in regionprops(image)]

        return sorted(
            bboxs, key=lambda bbox: image.shape[1] - bbox[3], reverse=True)

    def __search_spaces(self, bboxs: list) -> list:
        """Поиск на картинке пробелы

        Параметры
        ---------
        bboxs : list
            Bounding boxs букв

        Возвращаемое значение
        ---------------------
        list
            Двухмерный список слов
        """

        distances: list = [bboxs[i + 1][1] - bboxs[i][3]
                           for i in range(0, len(bboxs) - 1, 1)]
        threshold: float = np.std(distances) * 0.5 + np.mean(distances)

        words: list = []
        current_word: list = [bboxs[0]]

        for i in range(len(distances)):
            if distances[i] > threshold:
                words.append(current_word)
                current_word = []

            current_word.append(bboxs[i + 1])

        words.append(current_word)

        return words

    def __image_processing(self, img_path: str) -> np.ndarray:
        """Обработка картинки к нужному формату

        Параметры
        ---------
        img_path : str
            Путь к картинке

        Возвращаемое значение
        ---------------------
        np.ndarray
            Бирарное изображение
        """
        gray = cv2.imread(str(img_path), 0)
        binary = gray.copy()
        binary[binary > 0] = 1
        return binary

    def setup_knn(self, train_dir_path: str) -> None:
        """Обучение разпознавания текста с помощью метода k-ближайших соседей

        Параметры
        ---------
        train_dir_path : str
            Путь к картинкам для машинного обучения метода k-ближайших соседей

        Возвращаемое значение
        ---------------------
        None
        """
        train_dir: Path = Path(train_dir_path)
        train_data: list = defaultdict(list)
        features_array: list = []
        responses: list = []

        # Сопоставление название папки с картинками
        for path in sorted(train_dir.glob("*")):
            if path.is_dir():
                for img_path in path.glob("*.png"):
                    symbol = path.name[-1]
                    train_data[symbol].append(
                        self.__image_processing(img_path))

        # Извлекаем от каждой буквы особые признаки
        for i, symbol in enumerate(train_data):
            for img in train_data[symbol]:
                features = self.__extract_features(img)
                features_array.append(features)
                responses.append(ord(symbol))

        features_array: np.ndarray = np.array(features_array, dtype="f4")
        responses: np.ndarray = np.array(responses)

        # Метод k-ближайших соседей
        self.__knn = cv2.ml.KNearest_create()
        self.__knn.train(features_array, cv2.ml.ROW_SAMPLE, responses)

    def set_image(self, img_path: str) -> np.ndarray:
        """Добавить/Изменить картинку для разпознавания

        Параметры
        ---------
        image : str
            Путь к картинке

        Возвращаемое значение
        ---------------------
        None
        """
        self.__image = self.__image_processing(img_path)

    def get_image(self) -> np.ndarray:
        """Текущая картинку

        Возвращаемое значение
        ---------------------
        np.ndarray
            Картинка
        """
        return self.__image

    def image_to_text(self):
        """Перевести картинку к текстовый формат

        Возвращаемое значение
        ---------------------
        str
            Разпознанное изображение в текстовом формате
        """

        labeled: np.ndarray = self.__label_english_letters()

        test2: list = self.__sort_letters(labeled)
        words: list = self.__search_spaces(test2)

        plt.imshow(labeled)
        plt.show()

        fig = plt.figure()

        x_plt, y_plt = math.ceil(
            len(test2) / 2), math.floor(len(test2) / 2)

        for i, bbox in enumerate(test2):
            plt.subplot(x_plt, y_plt, i+1)
            top_left_y1, top_left_x1, bottom_right_y1, bottom_right_x1 = bbox
            plt.imshow(labeled[top_left_y1:bottom_right_y1,
                       top_left_x1:bottom_right_x1])

        plt.show()

        return labeled


if __name__ == "__main__":
    image_to_text = ImageToText()
    image_to_text.setup_knn("./data/out/train/")

    # Тестовые данные
    test_dir = Path("./data/out/")
    test_data = defaultdict(list)

    # for img_path in sorted(test_dir.glob("*.png")):
    #     print(img_path)
    # symbol = img_path.name[0]
    # print(symbol)
    # gray = cv2.imread(str(img_path), 0)
    # binary = gray.copy()
    # binary[binary > 0] = 1
    # test_data[symbol].append(binary)

    image_to_text.set_image("data/out/5.png")

    # image = test_data["4"][0]

    image_to_text.image_to_text()
    # plt.imshow(image_to_text.image_to_text())
    # plt.show()

    pass
