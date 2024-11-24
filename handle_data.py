import re


class SpamEmailDataHandler: # 处理新训练集
    def __init__(self):
        pass

    def get_path_label(self):
        """
        根据index文件获取数据文件的路径和标签
        :return: 数据文件路径、数据文件标签
        """
        label_path_list = open("trec06c/full/index", "r", encoding="gb2312", errors="ignore")
        label_path = [data for data in label_path_list]
        label_path_split = [data.split() for data in label_path if len(data.split()) == 2]
        label_list = [1 if data[0] == "spam" else 0 for data in label_path_split]
        path_list = [data[1].replace("..", "trec06c") for data in label_path_split]
        return path_list, label_list

    def get_data(self, path):
        """
        根据数据文件路径打开数据文件，提取每个邮件的正文
        :param path: 邮件文件路径
        :return: 提取的邮件正文
        """
        with open(path, "r", encoding="gb2312", errors="ignore") as mail:
            mail_text = [line for line in mail]

        # 查找正文部分
        mail_head_index = [i for i, line in enumerate(mail_text) if re.match("[a-zA-Z0-9]", line)]
        if mail_head_index:
            text = ''.join(mail_text[max(mail_head_index) + 1:])
        else:
            text = ''.join(mail_text)

        # 清洗文本
        text = re.sub('\s+', '', re.sub("\u3000", "", re.sub("\n", "", text)))
        return text



    def load_data_trec(self):
        """
        加载TREC数据集的路径和标签
        :return: TREC数据集的邮件文本和标签
        """
        # 获取TREC数据集的路径和标签
        path_lists, label_lists = self.get_path_label()

        # 获取邮件内容
        mail_texts = [self.get_data(path) for path in path_lists]

        # 合并TREC邮件内容和标签
        X = mail_texts
        y = label_lists
        return X, y
class SpamEmailDataHandler_local : # 处理本地数据集
    def __init__(self):
        pass
    def load_data(self):
        """
        加载本地的垃圾邮件和非垃圾邮件数据
        :return: 本地数据的邮件文本和标签
        """
        with open('data/ham_data.txt', 'r', encoding='utf-8') as f:
            ham_data = f.readlines()

        with open('data/spam_data.txt', 'r', encoding='utf-8') as f:
            spam_data = f.readlines()

        X_local = ham_data + spam_data
        y_local = [0] * len(ham_data) + [1] * len(spam_data)

        return X_local, y_local
    def load_all_data(self):
        """
        加载TREC数据集和本地数据并合并
        :return: 合并后的邮件文本和标签
        """
        X_local, y_local = self.load_data()
        X, y = SpamEmailDataHandler().load_data_trec()

        # 合并TREC数据和本地数据
        X_all = X + X_local
        y_all = y + y_local
        return X_all, y_all
