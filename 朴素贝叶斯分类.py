import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import handle_data


class SpamEmailClassifierNB:
    def __init__(self, model_file='naive_bayes_model.pkl', vectorizer_file='countvectorizer_bayes.pkl'):
        self.model_file = model_file
        self.vectorizer_file = vectorizer_file
        self.vectorizer = None
        self.model = None

    def train(self, X, y):
        """
        训练朴素贝叶斯分类器
        :param X: 训练数据特征
        :param y: 训练数据标签
        """
        # 文本向量化
        self.vectorizer = CountVectorizer(max_features=10000)
        X_vectorized = self.vectorizer.fit_transform(X)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

        # 训练朴素贝叶斯分类器
        self.model = MultinomialNB()
        self.model.fit(X_train, y_train)

        # 保存模型和矢量器
        joblib.dump(self.model, self.model_file)
        joblib.dump(self.vectorizer, self.vectorizer_file)

        # 在测试集上进行预测
        y_pred = self.model.predict(X_test)

        # 输出结果
        print("Naive Bayes Classifier Results:")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

    def load_model(self):
        """
        加载已训练的模型和向量化器
        """
        self.model = joblib.load(self.model_file)
        self.vectorizer = joblib.load(self.vectorizer_file)

    def predict(self, new_texts):
        """
        对新的文本进行预测
        :param new_texts: 输入的文本列表
        :return: 预测结果
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model and vectorizer must be loaded before prediction.")

        new_texts_counts = self.vectorizer.transform(new_texts)
        predictions = self.model.predict(new_texts_counts)

        return predictions


# 使用类
if __name__ == "__main__":
    # 创建数据处理类和模型类实例
    data_handler = handle_data.SpamEmailDataHandler_local()
    classifier = SpamEmailClassifierNB()

    # 加载数据
    X_all, y_all = data_handler.load_data()

    # 训练朴素贝叶斯模型
    classifier.train(X_all, y_all)

    # 测试新的文本
    new_texts = ["专属优惠，立即赢取奖品！", "你好，我们明天几点见面？"]
    predictions = classifier.predict(new_texts)
    print("Predictions for new texts:", predictions)
