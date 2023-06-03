# statistics_final_project

## 描述
這是一個很亂的統計學期末專案。

## 專案架構
```
├── requirements.txt # python library
├── app.py # 網頁程式進入點
├── globals.py # 全域變數，存放模型使用
├── readme.md # 使用手冊
├── .gitignore # git ignore
├── .gitattributes # git large file track
├── templates
│   ├── 404.html # Not found page
│   ├── estimation.html # 問卷 page
│   └── index.html # 統計分析 page
├── module
│   ├── __init__.py # 命名空间管理檔
│   ├── __pycache__ # python cache 
│   ├── AI.py # 一些小小的 function 給網頁程式調用
│   ├── regerssion.ipynb # 線性回歸訓練程式
│   ├── t_test.py # 顧名思義
│   └── relative.ipynb # 相關性評估程式
└── statics
    ├── css # style
    ├── js # script
    ├── png # 簡報圖片原檔
    ├── scss # style
    ├── vendor # 一些小模板
    ├── SEXTOIND.sas # 讀檔(行政院主計處 - 家庭收支調查)
    ├── URBANTOIND.sas # 讀檔(行政院主計處 - 家庭收支調查)
    └── models # 訓練好的模型
```

## 使用方法
## step one
下載專案
```sh
git clone https://github.com/Xunhaoz/statistics_final_project.git
cd statistics_final_project
```
## step two
建立虛擬環境
```sh
python3 -m venv env
.\env\Script\activate # windows
source .\env\bin\activate # linux
```
## step three
下載依賴套件
```sh
pip install -r requirements.txt
```
> 註：請依據自己的cuda硬體下載 pytorch 版本。
## step four
跑起來！！
```sh
python3 .\app.py
```

## 一些 DEMO
### 問卷畫面
![螢幕擷取畫面 2023-05-30 225318](https://github.com/Xunhaoz/statistics_final_project/assets/84084535/da613ce8-ba4c-40c3-90be-3ee0afcbe118)
### 計算結果
![螢幕擷取畫面 2023-05-30 225404](https://github.com/Xunhaoz/statistics_final_project/assets/84084535/d5db7784-b3f5-4545-9ea9-0a1c9130e3dc)
### t-test
```py
NEWCITY.xlsx : 14.482086042259391
NEWCITYTO64.xlsx : 4.359681353242153
NEWCITYTO69.xlsx : 2.286154744455917
NEWCITYTO90.xlsx : 4.439952508658127
SEX.xlsx : 53.29668111429349
SEXTO64.xlsx : 12.751162519383037
SEXTO69.xlsx : 20.078205770748532
SEXTO90.xlsx : 11.030410208437726
STUDY.xlsx : 45.5310830787382
STUDYTO64.xlsx : 3.103131235309686
STUDYTO69.xlsx : 5.631261158287068
STUDYTO90.xlsx : 3.7936549393669323
FINISH
```

## 警告：不德在未經同意下私自下載、使用，僅供 2023 統計學 - 葉錦輝 評分使用。
### 聯絡方式
110502528 張勛皓 資工2A
EMAIL：leo20020529@gmail.com
