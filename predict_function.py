import re
import jieba
import joblib
import numpy as np


# 加载中文停用词表
def load_stopwords():
    stopwords = set()
    with open('chinese_stopwords.txt', 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip())
    return stopwords


# 初始化停用词表和 jieba 分词器
stop_words = load_stopwords()


# 中文邮件预处理函数
def preprocess_email_chinese(email_text):
    # 1. 去除标点符号
    email_text = re.sub(r'[^\w\s]', '', email_text)  # 去除所有标点符号

    # 2. 去除数字
    email_text = re.sub(r'\d+', '', email_text)  # 去除所有数字

    # 3. 去除多余空格
    email_text = re.sub(r'\s+', ' ', email_text).strip()  # 去除多余空格

    # 4. 使用 jieba 进行中文分词
    words = jieba.cut(email_text)  # jieba.cut 返回一个生成器，可以使用 list() 转换为列表

    # 5. 去除停用词
    words = [word for word in words if word not in stop_words and len(word) > 1]  # 忽略长度小于2的词

    # 6. 将词列表拼接为处理后的文本
    processed_text = ' '.join(words)

    return processed_text


# 加载模型
models = [
    joblib.load('svm_model.pkl'),
    joblib.load('naive_bayes_model.pkl'),
    joblib.load('logistic_regression_model.pkl'),
    joblib.load('randomforest_model.pkl'),
    joblib.load('xgboost_model.pkl')
]

vectorizers = [
    joblib.load('countvectorizer_svm.pkl'),
    joblib.load('countvectorizer_bayes.pkl'),
    joblib.load('countvectorizer_logreg.pkl'),
    joblib.load('countvectorizer_randomforest.pkl'),
    joblib.load('countvectorizer_xgboost.pkl')
]


# 预测函数：使用多个模型的投票法进行预测
def predict_with_ensemble(email_text):
    # 预处理邮件文本
    preprocessed_text = preprocess_email_chinese(email_text)

    # 获取每个模型和矢量化器的预测结果
    predictions = []
    for model, vectorizer in zip(models, vectorizers):
        # 转换为特征向量
        features = vectorizer.transform([preprocessed_text])
        # 预测结果
        prediction = model.predict(features)
        predictions.append(prediction[0])  # 预测结果是一个数组，取第一个元素

    # 使用投票法来决定最终分类
    # 0 表示 "ham" (非垃圾邮件), 1 表示 "spam" (垃圾邮件)

    # 计算"垃圾邮件" (1)的票数
    spam_count = predictions.count('spam')
    ham_count = predictions.count('ham')

    # 投票法：垃圾邮件票数大于等于非垃圾邮件票数则判断为垃圾邮件
    if spam_count >= ham_count:
        return '垃圾邮件'
    else:
        return '有意义的邮件'


# 测试新的邮件
new_texts = [
    "恭喜你中奖了！快来领取你的iPhone！点击此处查看详情。",
    "你好，我们明天的会议时间定了吗？请确认一下。"
]

for text in new_texts:
    result = predict_with_ensemble(text)
    print(f"邮件内容：{text}\n分类结果：{result}\n")
