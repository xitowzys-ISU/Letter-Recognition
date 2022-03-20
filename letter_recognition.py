import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

from typing import Any
from pathlib import Path
from collections import defaultdict
from skimage.measure import label, regionprops


def bbox_center(bbox):
    """Центр bbox"""
    return (bbox[0] + (bbox[2] - bbox[0]) / 2, bbox[1] + (bbox[3] - bbox[1]) / 2)


class ImageToText:
    def __init__(self, image: np.ndarray = None) -> None:
        if not image is None:
            self.__image: np.ndarray = self.set_image(image)
            self.__labeled_image: np.ndarray = label(self.__image.copy())
            self.__regions_image: list = regionprops(self.__labeled_image)

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
        features = []

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
        labeled = label(image)

        region = regionprops(labeled)[0]
        filling_factor = region.area / region.bbox_area

        features.append(filling_factor)

        centroid = np.array(region.local_centroid) / \
            np.array(region.image.shape)
        features.extend(centroid)
        features.append(region.eccentricity)
        features.append(region.orientation)

        return features

    def __label_english_letters(self) -> list:
        """Макрировка английских букв"""

        image_copy = self.__image.copy()
        labeled = label(image_copy)
        regions = regionprops(labeled)

        for region1 in regions:

            bbox_region_1 = region1.bbox  # y1, x1, y2, x2
            bbox_center1 = bbox_center(bbox_region_1)
            for region2 in regions:

                bbox_region_2 = region2.bbox
                bbox_center_2 = bbox_center(bbox_region_2)

                if bbox_region_1[0] > bbox_region_2[2] and abs(bbox_center1[1] - bbox_center_2[1]) < 10 and bbox_center1[0] > bbox_center_2[0]:
                    for y in range(labeled.shape[0]):
                        for x in range(labeled.shape[1]):
                            if labeled[y][x] == region2.label:
                                labeled[y][x] = region1.label

                    break

        return labeled

    def __sort_letters(self, labeled: np.ndarray) -> list:
        """Сортировка букв по порядку, как на картинке

        Параметры
        ---------
        labeled : np.ndarray
            Отмаркированная картинка

        Возвращаемое значение
        ---------------------
        list
            Отсортированный bounding box
        """
        bboxs = [region.bbox for region in regionprops(labeled)]

        return sorted(
            bboxs, key=lambda bbox: labeled.shape[1] - bbox[3], reverse=True)

    def __search_spaces(self):
        pass

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
        train_dir = Path(train_dir_path)
        train_data = defaultdict(list)
        features_array = []
        responses = []

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

        features_array = np.array(features_array, dtype="f4")
        responses = np.array(responses)

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
        return self.__image

    def image_to_text(self):
        """Перевести картинку к текстовый формат

        Возвращаемое значение
        ---------------------
        str
            Разпознанное изображение в текстовом формате
        """

        labeled = self.__label_english_letters()

        # plt.imshow(labeled)
        # plt.show()

        regions = regionprops(labeled)
        test2 = self.__sort_letters(labeled)
        test = labeled

        plt.imshow(test)
        plt.show()

        fig = plt.figure()

        x_plt, y_plt = math.ceil(
            len(regions) / 2), math.floor(len(regions) / 2)

        for i, bbox in enumerate(test2):
            plt.subplot(x_plt, y_plt, i+1)
            top_left_y1, top_left_x1, bottom_right_y1, bottom_right_x1 = bbox
            plt.imshow(test[top_left_y1:bottom_right_y1,
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