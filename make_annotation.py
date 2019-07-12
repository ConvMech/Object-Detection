import pathlib

import pandas as pd


def clean_space(x: str) -> str:
    return x.replace('\n', '')


def load_text(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        data = {'label': [], 'bbox': []}
        for line in f.readlines():
            lead = line.split(' ')[0]
            try:
                second = line.split(' ')[1]
            except:
                continue

            if second == 'filename':
                data['filename'] = clean_space(line.split('"')[1])
            if second == 'size':
                data['img_size'] = clean_space(line.split(': ')[1]).split(' x ')
            if lead == 'Original':
                data['label'].append(clean_space(line.split('"')[1]))
            if lead == 'Bounding':
                bbox_str = clean_space(line.split(': ')[1]).split(' - ')
                bbox = [eval(i) for i in bbox_str]
                data['bbox'].append(bbox)

    rows = []
    for label, bbox in zip(data['label'], data['bbox']):
        row = {}
        row['filename'] = data['filename']
        row['img_size'] = data['img_size']
        row['label'] = label
        row['bbox'] = bbox
        rows.append(row)
    return rows


if __name__ == '__main__':
    rows = []
    for path in pathlib.Path('./PennFudanPed/Annotation/').iterdir():
        rows += load_text(path)

    df = pd.DataFrame(rows)

    df['x_min'] = df['bbox'].apply(lambda x: x[0][0])
    df['y_min'] = df['bbox'].apply(lambda x: x[0][1])
    df['x_max'] = df['bbox'].apply(lambda x: x[1][0])
    df['y_max'] = df['bbox'].apply(lambda x: x[1][1])
    df.drop(['bbox'], axis=1, inplace=True)
    df.to_csv('annotation.csv', index=False)