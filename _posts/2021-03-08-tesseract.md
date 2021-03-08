---
layout: post
title:  "Tesseract OCR 安裝 & 訓練"
---

## 1. Install

### Get the Source Code
```bash=
git clone --recurse-submodules https://github.com/tesseract-ocr/tesseract.git
```

### Building the Training Tools
```bash=
./autogen.sh
./configure
make
sudo make install
sudo ldconfig
make training
sudo make training-install
```

## 2. Get tesstrain
任意位置都可以
```bash=
git clone https://github.com/tesseract-ocr/tesstrain.git
```

## 3. Get Training Data
使用
```bash=
wget https://github.com/tesseract-ocr/tessdata_best/raw/master/eng.traineddata
```
把你要的`trained data`載下來，如`eng`，`chi_tra`等等

放至(路徑可能不同)
```bash=
usr/local/share/tessdata/
```

## 4. Create Training Data
將圖片和文字檔放在(`<MODEL_NAME>`: 自己的模型名稱)
```bash=
tesstrain/data/<MODEL_NAME>-ground-truth
```
底下
格式為
```shell=
<FILE_NAME>.png
<FILE_NAME>.gt.txt
```

***至少十樣***

少於十樣會有錯誤，因為訓練和驗證集為9：1

## 5. Start Traning

在`tesstrain`根目錄下

```bash=
make training MODEL_NAME=<MODEL_NAME> \
START_MODEL=<LANG> \
PSM=7 \
TESSDATA=/usr/local/share/tessdata
```

`<MODEL_NAME>`: 自己的模型名稱

`<LANG>`: 語言基底，如：chi_tra

`PSM=7` => 訓練一行文字，可依照自己需求調整參數

訓練完成後，`tesstrain/data/`底下會出現`<MODEL_NAME>.traineddata`。

## References

根據`tesseract --help-extra`可以得知。

```shell=
Page segmentation modes:
  0    Orientation and script detection (OSD) only.
  1    Automatic page segmentation with OSD.
  2    Automatic page segmentation, but no OSD, or OCR. (not implemented)
  3    Fully automatic page segmentation, but no OSD. (Default)
  4    Assume a single column of text of variable sizes.
  5    Assume a single uniform block of vertically aligned text.
  6    Assume a single uniform block of text.
  7    Treat the image as a single text line.
  8    Treat the image as a single word.
  9    Treat the image as a single word in a circle.
 10    Treat the image as a single character.
 11    Sparse text. Find as much text as possible in no particular order.
 12    Sparse text with OSD.
 13    Raw line. Treat the image as a single text line,
       bypassing hacks that are Tesseract-specific.
```
依照自己需求，調整`PSM`參數。
