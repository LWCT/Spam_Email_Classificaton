import re
import time

import jieba
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier


class ChineseEmailClassifier:
    """中文邮件分类器，用于识别垃圾邮件和正常邮件"""

    def __init__(self):
        """初始化分类器，设置停用词表和模型相关变量"""
        self.stop_words = self._load_stopwords()
        self.load_models()

    # 加载中文停用词表，返回包含所有停用词的集合
    def _load_stopwords(self):
        """从文件中加载中文停用词表"""
        stopwords = set()
        with open('chinese_stopwords.txt', 'r', encoding='utf-8') as f:
            for line in f:
                stopwords.add(line.strip())
        return stopwords

    # 对中文邮件文本进行预处理
    def _preprocess_email_chinese(self, email_text):
        # 1. 去除标点符号
        email_text = re.sub(r'[^\w\s]', '', email_text)

        # 2. 去除数字
        email_text = re.sub(r'\d+', '', email_text)

        # 3. 去除多余空格
        email_text = re.sub(r'\s+', ' ', email_text).strip()

        # 4. 使用 jieba 进行中文分词
        words = jieba.cut(email_text)

        # 5. 去除停用词
        words = [word for word in words if word not in self.stop_words and len(word) > 1]

        # 6. 将词列表拼接为处理后的文本
        return ' '.join(words)
    # 加载训练数据集
    def load_data(self):
        """加载训练数据"""
        start_time = time.time()

        with open('data/spam_data.txt', 'r', encoding='utf-8') as f:
            spam_texts = f.readlines()
        with open('data/ham_data.txt', 'r', encoding='utf-8') as f:
            ham_texts = f.readlines()

        spam_data = pd.DataFrame({'text': spam_texts, 'label': 'spam'})
        ham_data = pd.DataFrame({'text': ham_texts, 'label': 'ham'})
        data = pd.concat([spam_data, ham_data], ignore_index=True)

        X = data['text']
        y = data['label'].map({'ham': 0, 'spam': 1})

        print(f"数据加载完成，耗时: {time.time() - start_time:.2f} 秒")
        return X, y
    
    def load_models(self):
        # 加载模型和向量器
        self.models = [
            joblib.load('models/svm_model.pkl'),
            joblib.load('models/naive_bayes_model.pkl'),
            joblib.load('models/logistic_regression_model.pkl'),
            joblib.load('models/random_forest_model.pkl'),
            joblib.load('models/xgboost_model.pkl'),
            joblib.load('models/lightgbm_model.pkl')
        ]

        self.vectorizers = [
            joblib.load('models/countvectorizer_svm.pkl'),
            joblib.load('models/countvectorizer_naive_bayes.pkl'),
            joblib.load('models/countvectorizer_logistic_regression.pkl'),
            joblib.load('models/countvectorizer_random_forest.pkl'),
            joblib.load('models/countvectorizer_xgboost.pkl'),
            joblib.load('models/countvectorizer_lightgbm.pkl')
        ]   

    """训练所有模型"""
    def train_models(self):
        """
        训练多个机器学习模型用于邮件分类并保存模型
        包括：XGBoost、LightGBM、Random Forest、Logistic Regression、Naive Bayes、SVM
        返回：
            results: 包含每个模型训练结果的字典，包括准确率和分类报告
        """
        X, y = self.load_data()
        
        # 数据集划分
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 文本向量化
        self.vectorizer = CountVectorizer()
        X_train_counts = self.vectorizer.fit_transform(X_train).astype('float32')
        X_test_counts = self.vectorizer.transform(X_test).astype('float32')

        # 定义模型
        self.models = {
            "XGBoost": XGBClassifier(eval_metric='logloss'),
            "LightGBM": LGBMClassifier(),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Naive Bayes": MultinomialNB(),
            "SVM": svm.SVC(kernel='linear')
        }

        # 训练模型并保存
        results = {}
        for name, model in self.models.items():
            print(f"\n训练模型：{name}")
            start_time = time.time()
            
            model.fit(X_train_counts, y_train)
            y_pred = model.predict(X_test_counts)
            
            # 保存模型和对应的向量器
            model_filename = f'models/{name.lower().replace(" ", "_")}_model.pkl'
            vectorizer_filename = f'models/countvectorizer_{name.lower().replace(" ", "_")}.pkl'
            joblib.dump(model, model_filename)
            joblib.dump(self.vectorizer, vectorizer_filename)
            print(f"模型已保存到: {model_filename}")
            
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            results[name] = {"accuracy": accuracy, "report": report}
            print(f"{name} 训练完成，准确率: {accuracy:.2f}")
            print(f"耗时: {time.time() - start_time:.2f} 秒")

        return results

    def predict(self, email_text):
        """预测邮件类型"""
        # 预处理邮件文本
        preprocessed_text = self._preprocess_email_chinese(email_text)

        # 获取每个模型和矢量化器的预测结果
        predictions = []
        for model, vectorizer in zip(self.models, self.vectorizers):
            features = vectorizer.transform([preprocessed_text]).astype('float32')  # 确保数据类型为float32
            prediction = model.predict(features)
            predictions.append(int(prediction[0]))  # 确保预测结果为整数

        # 计算投票结果
        spam_count = sum(1 for p in predictions if p == 1)
        ham_count = len(predictions) - spam_count

        return '垃圾邮件' if spam_count >= ham_count else '有意义的邮件'

    """重新训练所有模型"""
    def reTrain(self):
        result  = self.train_models()
        self.load_models()


        return result


# 使用示例
if __name__ == '__main__':
    classifier = ChineseEmailClassifier()
    
    test_emails = [
        "恭喜你中奖了！快来领取你的iPhone！点击此处查看详情。",
        "你好，我们明天的会议时间定了吗？请确认一下。"
    ]
    
    for email in test_emails:
        result = classifier.predict(email)
        print(f"邮件内容：{email}\n分类结果：{result}\n")
