import pandas as pd
import numpy as np
from sklearn import preprocessing

okr = {'ЦАО': 0, 'САО': 1, 'СВАО': 2, 'ВАО': 3, 'ЮВАО': 4, 'ЮАО': 5, 'ЮЗАО': 6, 'ЗАО': 7, 'СЗАО': 8, 'МО': 9}

zod = {'Овен': 1, 'Телец': 2, 'Близнецы': 3, 'Рак': 4, 'Лев': 5, 'Дева': 6, 'Весы': 6, 'Скорпион': 5, 'Стрелец': 4, 'Козерог': 3, 'Водолей': 2, 'Рыбы': 1}
kf = {'К': 1, 'Ч': 0}

def read_and_normalize():
    """
    Функция чтения и нормализации данных
    """
    ds = pd.read_excel('datalist.xlsx')
    pf = ds.replace({'okr': okr, 'zod': zod, 'kf': kf})
    pf = pf.drop('color', axis=1)

    # scaler_df = (pf - pf.min()) / (pf.max() - pf.min())
    # scaler_df = scaler_df.astype(np.float64)
    scaler = preprocessing.MinMaxScaler()
    names = pf.columns
    d = scaler.fit_transform(pf)
    scaled_df = pd.DataFrame(d, columns=names)

    return scaled_df

def k_nearest(k, dataset, x):
    """
    Алгоритм ближайшего k-соседа
    k - количество соседей, dataset - база данных, x - данные исследуемого объекта
    """
    y = dataset.copy()
    y = y.drop('kf', axis=1)
    dataset['r'] = np.linalg.norm(y - x, axis=1)
    knn = dataset.sort_values(by='r')[:k]

    return knn.value_counts('kf').index[0]


def start():
    x = []
    a = read_and_normalize()

    print('Действующие округа: ', okr.keys())
    x.append(input("Выберите ваш округ: "))

    print('Действующие знаки зодиака: ', zod.keys())
    x.append(input('Введите ваш знак зодиака: '))
    x.append(float(input('Работаете ли вы? (1-да, 2-нет): ')))
    x.append(input('Введите ваше время подъема утром: '))
    x.append(input('Введите вашу обычную продолжительность сна: '))
    k = int(input('Количество ближайших соседей: '))
    x = pd.DataFrame(x)
    x = x.replace(okr)
    x = x.replace(zod)
    x = x.astype(np.float64)

    x = (x - x.min()) / (x.max() - x.min())

    result = k_nearest(k, a, x)
    results = {0.0: 'чай', 1.0: 'кофе'}
    print("Судя по нашей скромной базе данных вы предпочитаете ",{results[result]})

start()


