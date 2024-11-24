from UI.main import Ui_main
from PyQt5.QtWidgets import QMessageBox
from predict_function import ChineseEmailClassifier

class Ui_main_my(Ui_main):
	def __init__(self):
		super().__init__()
		self.classifier = ChineseEmailClassifier()

	def setupUi(self, main):
		super().setupUi(main)
		self.checkButton.clicked.connect(self.check_email)
		self.reTrainButton.clicked.connect(self.reTrain)

	def reTrain(self):
		results = self.classifier.reTrain()
		for name, result in results.items():
			accuracy = result["accuracy"]
			report = result["report"]
			
			# 将结果添加到文本框中
			self.trainResultText.append(f"\n模型：{name}")
			self.trainResultText.append(f"准确率: {accuracy:.2f}")
			self.trainResultText.append(f"分类报告:\n{report}")
			self.trainResultText.append("-" * 50)  # 添加分隔线

	def check_email(self):
		try:
			email_content = self.emailText.toPlainText()

			if not email_content.strip():
				QMessageBox.warning(None, "警告", "请输入邮件内容！")
				return

			results = self.classifier.predict(email_content)
			self.checkResultText.appendPlainText(results)

		except Exception as e:
			QMessageBox.critical(None, "错误", f"预测过程出错：{str(e)}")

