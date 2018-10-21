# farmer
farmer is an automated machine learning library.

## Classifier
### Input data : 
- array: 
```
2D image array: (num_samples, height, width, channel)
3D image array: (num_samples, depth, height, width, channel)
```
- classified folder
```
target_dir
  |-----banana
  |       |---- *.jpg
  |       |---- *.png
  |       |
  |
  |-----apple
  |       |---- *.tif
  |       |---- *.dcm
  |       |
  |
```
- annotation data (csv file)
```csv
image_path, class_idx
*.jpg, 0
*.png, 1
```
### How to use it

- python scripts
```python
from farmer import classifier

classifier.fit_from_array(x_train, y_train)
classifier.fit_from_directory(target_dir)
classifier.fit_from_annotation(file_path)
```

- command line
```bash
$ python classifier.py --mode=folder, --target-dir='~/'
$ python classifier.py --mode=annotation, --file-path='~.csv'
```

## Object Detection

## Segmentation

## Regression
