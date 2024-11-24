import sys

import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm

from UI.Ui_main_my import Ui_main_my


# 加载数据
def load_data():
    with open('data/spam_data.txt', 'r', encoding='utf-8') as f:
        spam_texts = f.readlines()
    with open('data/ham_data.txt', 'r', encoding='utf-8') as f:
        ham_texts = f.readlines()
    spam_data = pd.DataFrame({'text': spam_texts, 'label': 'spam'})
    ham_data = pd.DataFrame({'text': ham_texts, 'label': 'ham'})
    data = pd.concat([spam_data, ham_data], ignore_index=True)
    X = data['text']
    y = data['label'].map({'ham': 0, 'spam': 1})
    return X, y

# 训练并评估模型
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

# 比较不同模型的性能
def compare_models():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = CountVectorizer()
    X_train_counts = vectorizer.fit_transform(X_train).astype('float32')
    X_test_counts = vectorizer.transform(X_test).astype('float32')

    models = {
        "XGBoost": XGBClassifier(eval_metric='logloss'),
        "LightGBM": LGBMClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "SVM": svm.SVC(kernel='linear')
    }

    results = {}
    for name, model in models.items():
        accuracy, report = train_and_evaluate_model(model, X_train_counts, X_test_counts, y_train, y_test)
        results[name] = {"Accuracy": accuracy, "Report": report}
        print(f"\n模型：{name}")
        print(f"准确率: {accuracy:.2f}")
        print("分类报告:\n", report)

    return models,vectorizer

models, vectorizer = compare_models()

#创建ui窗口
def main():
    # 创建应用程序实例
    app = QApplication(sys.argv)


    # 创建主窗口
    window = QMainWindow()
    main_ui = Ui_main_my()
    main_ui.setupUi(window)
    window.show()

    # 运行事件循环
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
