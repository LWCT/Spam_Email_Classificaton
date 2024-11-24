import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
import joblib
# 1. 读取 spam 和 ham 数据文件
with open('data/spam_data.txt', 'r', encoding= 'utf-8') as f:
    spam_texts = f.readlines()

with open('data/ham_data.txt', 'r', encoding= 'utf-8') as f:
    ham_texts = f.readlines()

# 2. 创建数据框，并添加标签
spam_data = pd.DataFrame({'text': spam_texts, 'label': 'spam'})
ham_data = pd.DataFrame({'text': ham_texts, 'label': 'ham'})

# 3. 合并数据集
data = pd.concat([spam_data, ham_data], ignore_index=True)

# 4. 分离文本和标签
X = data['text']
y = data['label']

# 5. 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 文本特征提取 (词袋模型)
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)#训练集 词汇表和词频矩阵
X_test_counts = vectorizer.transform(X_test)#测试集 词频矩阵

# 7. 训练随机森林分类器
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_counts, y_train)

joblib.dump(clf, 'randomforest_model.pkl')  # 保存训练好的RandomForest模型
joblib.dump(vectorizer, 'countvectorizer_randomforest.pkl')  # 保存词频向量化器
# 8. 在测试集上进行预测
y_pred = clf.predict(X_test_counts)

# 9. 输出结果
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 测试新的文本
new_texts = ["专属优惠，立即赢取奖品！", "你好，我们明天几点见面？"]
new_texts_counts = vectorizer.transform(new_texts)
predictions = clf.predict(new_texts_counts)
print("Predictions for new texts:", predictions)