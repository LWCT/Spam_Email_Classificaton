import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
import handle_data



class SpamEmailClassifierSVM:
    def __init__(self, model_file='svm_model.pkl', vectorizer_file='countvectorizer_svm.pkl'):
        self.model_file = model_file
        self.vectorizer_file = vectorizer_file
        self.vectorizer = None
        self.model = None

    def train(self, X, y):
        """
        训练SVM分类器
        :param X: 训练数据特征
        :param y: 训练数据标签
        """
        # 文本向量化
        self.vectorizer = TfidfVectorizer(max_features=10000)
        X_vectorized = self.vectorizer.fit_transform(X)

        # 转换为 float32 类型（SVM 也可以接受这种类型）
        X_vectorized = X_vectorized.astype('float32')

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

        # 训练SVM模型
        self.model = svm.SVC(kernel='linear')
        self.model.fit(X_train, y_train)

        # 保存模型和矢量器
        joblib.dump(self.model, self.model_file)
        joblib.dump(self.vectorizer, self.vectorizer_file)

        # 在测试集上进行预测
        y_pred = self.model.predict(X_test)

        # 输出结果
        print("SVM Classifier Results:")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

    def load_model(self):
        """
        加载已训练的模型和矢量器
        """
        self.model = joblib.load(self.model_file)
        self.vectorizer = joblib.load(self.vectorizer_file)

    def predict(self, new_texts):
        """
        使用训练好的模型对新的文本进行预测
        :param new_texts: 待预测的文本列表
        :return: 预测的结果
        """
        new_texts_counts = self.vectorizer.transform(new_texts)
        predictions = self.model.predict(new_texts_counts)
        return predictions


if __name__ == "__main__":
    # 创建数据处理类和模型类实例
    data_handler = handle_data.SpamEmailDataHandler_local()
    classifier = SpamEmailClassifierSVM()

    # 加载数据
    X_all, y_all = data_handler.load_all_data()

    # 训练SVM模型
    classifier.train(X_all, y_all)

    # 测试新的文本
    new_texts = ["专属优惠，立即赢取奖品！", "你好，我们明天几点见面？"]
    predictions = classifier.predict(new_texts)
    print("Predictions for new texts:", predictions)
