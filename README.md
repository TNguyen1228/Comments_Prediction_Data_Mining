# Blog comment prediction

## Introduction 
Perform a regression task to predict the number of comments on a post after a certain period, applying it to [Reddit](https://www.reddit.com/)

Demo: https://comment-reddits.streamlit.app/

## Data 
The BlogFeedback data can be downloaded from [here](https://archive.ics.uci.edu/dataset/304/blogfeedback)

## Setup

### 1. Clone this repository 

```
git clone https://github.com/TranMinhDuc190103/Data_mining_finals.git
```

or download directly instead.

### 2. Install Dependencies

Create a virtual environment and install the required packages:

```
pip install -r requirements.txt
```

## Run Jupyter notebooks

1. **Training the Model**: Use the following notebook to train your model.

In [Models](https://github.com/TranMinhDuc190103/Data_mining_finals/tree/main/Models) folder we provide 3 pre-trained models saved as `.joblib` and 3 Jupyter notebooks used to train model. You can run each notebook to get the pre-trained model or use it instead.

2. **Crawl data from [Reddit](https://www.reddit.com/)**

You can self crawl some data from Reddit by running [reddit-crawler.ipynb](https://github.com/TranMinhDuc190103/Data_mining_finals/blob/main/crawl/reddit-crawler.ipynb) in folder `crawl` to crawl data from Reddit. However you need some key from Reddit app to continue.

The `credentials.py` contain some infomation to interact with Reddit API. Due to security concerns, we are unable to provide complete information. Please contact us for further details.

## Run the app

After installing important libraries and storing your infomation about Reddit app in `credentials.py`, you can run the app with following command 

```
streamlit run app_T.py
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## Contact

If you have any question, please contact us via phone or email below:

Trần Minh Đức, 0344794259, tranminhduc5_t66@hus.edu.vn

Nguyễn Mạnh Tuấn, 0349292753, nguyenmanhtuan_t66@hus.edu.vn

Lê Quốc Lâm, 0337213192, lequoclam_t66@hus.edu.vn

Lê Gia Huy, 0984588603, legiahuy_t66@hus.edu.vn
