# HIDRID
# 面向真实图像去雾的雾霾图像数据集合成方法

## 1 环境

matlab2016a、python3.X、windows或ubuntu

## 2 运行代码

### 2.1 dng转png

`python dng_to_png.py`

### 2.2 图像大小标准化


`python crop_dataset.py`

### 2.3 超像素分割[Wu J, Liu C, Li B. Texture-aware and Structure-preserving Superpixel Segmentation]

#### 运行入口(需要matlab2016a)

`TSSPsuperpixel/TSSPdemo.m`

### 2.4 超像素合并

`python merge_superpixel.py`

### 2.5 转为传输图并引导滤波

`python guidedfilter.py`

### 2.6 获得雾霾图像

`python get_hazy.py`

## 3 FiveK-Haze

## 3.1 下载链接

https://pan.baidu.com/s/1997wfUk4DO9qXhEohY-WHg  提取码：nis9 

## 3.2 数据集说明

/FiveK-Haze\n
    ----/gt （目标图像\基准图像\Ground Truth）\n
    ----/t （透射图\传输图\Transmission Map）\n
    ----/test_haze（测试集雾霾图像）\n
    ----/train_haze（训练集雾霾图像）\n
    ----/val_haze（验证集雾霾图像）\n
