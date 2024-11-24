from UI.main import Ui_main
from PyQt5.QtWidgets import QMessageBox
import joblib  # 用于加载模型
import re      # 用于文本预处理

class Ui_main_my(Ui_main):
	def __init__(self):
		super().__init__()
		self.models = None
		self.vectorizer = None
		self._load_model()

	def setupUi(self, main):
		super().setupUi(main)
		# 连接按钮点击事件
		self.checkButton.clicked.connect(self.check_email)  # 假设检查邮件按钮的对象名是pushButton

	def _load_model(self):
		try:
			# 加载训练好的模型和向量器
			#self.model = joblib.load('models/spam_model.pkl')
			#self.vectorizer = joblib.load('models/vectorizer.pkl')
			from main import models, vectorizer
			self.models = models  
			self.vectorizer = vectorizer
		except Exception as e:
			print(f"模型加载失败: {str(e)}")



	def check_email(self):
		try:
			# 获取左侧文本框的内容
			email_content = self.emailText.toPlainText()
			
			if not email_content.strip():
				QMessageBox.warning(None, "警告", "请输入邮件内容！")
				return
			
			for name, model in self.models.items():
				# 测试新的文本
				new_texts = [email_content]
				new_texts_counts = self.vectorizer.transform(new_texts).astype('float32')
				predictions = model.predict(new_texts_counts)
				print("Predictions for new texts:", predictions)
			
				# 显示结果
				result = "垃圾邮件" if predictions[0] == 1 else "正常邮件"
				# 在右侧文本框显示结果
				result_text = f"模型{name}\t预测结果：{result}\n"
				self.checkResultText.appendPlainText(result_text)  # 假设右侧文本框的对象名是checkResultText
			

		except Exception as e:
			QMessageBox.critical(None, "错误", f"预测过程出错：{str(e)}")