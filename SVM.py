from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
import joblib
# 加载数据
with open('data/ham_data.txt', 'r', encoding= 'utf-8') as f:
    ham_data = f.readlines()

with open('data/spam_data.txt', 'r', encoding= 'utf-8') as f:
    spam_data = f.readlines()

# 创建标签
X = ham_data + spam_data
y = [0] * len(ham_data) + [1] * len(spam_data)  # 0为非垃圾邮件，1为垃圾邮件

# 文本向量化
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练SVM模型
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)
#保存模型和矢量器
joblib.dump(model, 'svm_model.pkl')
joblib.dump(vectorizer, 'countvectorizer_svm.pkl')

# 预测并评估模型
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
