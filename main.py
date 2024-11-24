import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm


# 加载数据并记录时间
def load_data():
    """
    加载垃圾邮件和正常邮件数据，并返回特征（文本）和标签。
    这里将文本数据加载并合并为一个DataFrame，标签值（0表示正常邮件，1表示垃圾邮件）也会映射好。
    同时记录数据加载的时间。
    """
    start_time = time.time()  # 记录开始时间

    # 读取垃圾邮件和正常邮件的文本数据
    with open('data/spam_data.txt', 'r', encoding='utf-8') as f:
        spam_texts = f.readlines()  # 垃圾邮件内容
    with open('data/ham_data.txt', 'r', encoding='utf-8') as f:
        ham_texts = f.readlines()  # 正常邮件内容

    # 将文本数据和标签整合成DataFrame
    spam_data = pd.DataFrame({'text': spam_texts, 'label': 'spam'})
    ham_data = pd.DataFrame({'text': ham_texts, 'label': 'ham'})
    data = pd.concat([spam_data, ham_data], ignore_index=True)  # 合并数据

    # 特征（X）为文本数据，标签（y）为0或1（正常邮件或垃圾邮件）
    X = data['text']
    y = data['label'].map({'ham': 0, 'spam': 1})

    end_time = time.time()  # 记录结束时间
    print(f"数据加载完成，耗时: {end_time - start_time:.2f} 秒")  # 输出耗时

    return X, y  # 返回文本和标签


# 训练并评估模型并记录时间
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    训练并评估指定的机器学习模型，输出准确率和分类报告。
    训练过程中会记录模型训练的时间。
    """
    start_time = time.time()  # 记录训练开始时间

    # 训练模型
    model.fit(X_train, y_train)

    # 预测结果
    y_pred = model.predict(X_test)

    # 计算准确率和生成分类报告
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    end_time = time.time()  # 记录训练结束时间
    print(f"{model.__class__.__name__} 模型训练和评估完成，耗时: {end_time - start_time:.2f} 秒")  # 输出模型训练和评估的时间

    return accuracy, report  # 返回准确率和分类报告


# 比较不同模型的性能
def compare_models():
    """
    加载数据、训练不同的模型并评估它们的表现。
    会输出每个模型的准确率、分类报告，并记录每个核心步骤的耗时。
    """
    X, y = load_data()  # 加载数据

    start_time = time.time()  # 记录文本向量化开始时间

    # 数据集划分：80%训练，20%测试
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 文本数据转换为数字特征：使用CountVectorizer将文本转换为词频矩阵
    vectorizer = CountVectorizer()
    X_train_counts = vectorizer.fit_transform(X_train).astype('float32')  # 转换训练集文本
    X_test_counts = vectorizer.transform(X_test).astype('float32')  # 转换测试集文本

    end_time = time.time()  # 记录文本向量化结束时间
    print(f"文本向量化完成，耗时: {end_time - start_time:.2f} 秒")  # 输出文本向量化的时间

    # 定义要训练的多个模型
    models = {
        "XGBoost": XGBClassifier(eval_metric='logloss'),
        "LightGBM": LGBMClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "SVM": svm.SVC(kernel='linear')
    }

    # 存储模型结果
    results = {}

    # 循环训练并评估每个模型
    for name, model in models.items():
        print(f"\n正在训练和评估模型：{name}")
        accuracy, report = train_and_evaluate_model(model, X_train_counts, X_test_counts, y_train, y_test)
        results[name] = {"Accuracy": accuracy, "Report": report}  # 保存结果

        # 输出每个模型的评估结果
        print(f"准确率: {accuracy:.2f}")
        print("分类报告:\n", report)

    total_end_time = time.time()  # 记录总的训练和评估时间
    print(f"\n所有模型训练和评估完成，全部耗时: {total_end_time - start_time:.2f} 秒")  # 输出总耗时


# 执行模型比较
compare_models()
