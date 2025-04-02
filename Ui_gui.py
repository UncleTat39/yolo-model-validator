from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1200, 800)  # 加大窗口尺寸
        
        # 主佈局
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        
        # 上半部分：原有的控件和新增控件
        self.topLayout = QtWidgets.QVBoxLayout()
        
        # 路徑選擇區域
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.pathLabel = QtWidgets.QLabel("圖片路徑：")
        self.horizontalLayout.addWidget(self.pathLabel)
        self.pathText = QtWidgets.QLineEdit()
        self.horizontalLayout.addWidget(self.pathText)
        self.selectPath = QtWidgets.QPushButton("選擇文件夾")
        self.horizontalLayout.addWidget(self.selectPath)
        self.topLayout.addLayout(self.horizontalLayout)
        
        # 新增 - 模型選擇區域
        self.modelLayout = QtWidgets.QHBoxLayout()
        self.modelLabel = QtWidgets.QLabel("AI模型路徑：")
        self.modelLayout.addWidget(self.modelLabel)
        self.modelText = QtWidgets.QLineEdit()
        self.modelLayout.addWidget(self.modelText)
        self.selectModel = QtWidgets.QPushButton("選擇模型")
        self.modelLayout.addWidget(self.selectModel)
        self.topLayout.addLayout(self.modelLayout)
        
        # 新增 - 類別名稱輸入區域
        self.classNamesLabel = QtWidgets.QLabel("類別名稱列表（以逗號分隔）：")
        self.topLayout.addWidget(self.classNamesLabel)
        self.classNamesText = QtWidgets.QTextEdit()
        self.classNamesText.setMaximumHeight(80)  # 控制文本框高度
        self.topLayout.addWidget(self.classNamesText)
        
        # 新增 - 閾值設置區域
        self.thresholdLayout = QtWidgets.QHBoxLayout()
        
        # Confidence閾值
        self.confidenceLabel = QtWidgets.QLabel("置信度閾值：")
        self.thresholdLayout.addWidget(self.confidenceLabel)
        self.confidenceThreshold = QtWidgets.QDoubleSpinBox()
        self.confidenceThreshold.setRange(0.01, 0.99)
        self.confidenceThreshold.setSingleStep(0.05)
        self.confidenceThreshold.setValue(0.3)
        self.thresholdLayout.addWidget(self.confidenceThreshold)
        
        # IOU閾值
        self.iouLabel = QtWidgets.QLabel("IOU閾值：")
        self.thresholdLayout.addWidget(self.iouLabel)
        self.iouThreshold = QtWidgets.QDoubleSpinBox()
        self.iouThreshold.setRange(0.01, 0.99)
        self.iouThreshold.setSingleStep(0.05)
        self.iouThreshold.setValue(0.3)
        self.thresholdLayout.addWidget(self.iouThreshold)
        
        self.topLayout.addLayout(self.thresholdLayout)
        
        # 按鈕區域
        self.buttonLayout = QtWidgets.QHBoxLayout()
        
        # 開始按鈕
        self.startButton = QtWidgets.QPushButton("開始檢測")
        self.buttonLayout.addWidget(self.startButton)
        
        # 新增 - 查看圖表按鈕
        self.visualizeButton = QtWidgets.QPushButton("查看結果圖表")
        self.visualizeButton.setEnabled(False)  # 初始禁用，直到有結果
        self.buttonLayout.addWidget(self.visualizeButton)
        
        # 新增 - 查看錯誤識別按鈕
        self.viewErrorsButton = QtWidgets.QPushButton("查看錯誤識別")
        self.viewErrorsButton.setEnabled(False)  # 初始禁用，直到有結果
        self.buttonLayout.addWidget(self.viewErrorsButton)
        
        self.topLayout.addLayout(self.buttonLayout)
        
        # 總體結果顯示區域
        self.groupBox = QtWidgets.QGroupBox("整體檢測結果")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        
        # 測試數量
        self.label_test = QtWidgets.QLabel("測試總數：")
        self.gridLayout.addWidget(self.label_test, 0, 0)
        self.testNumber = QtWidgets.QLabel("0")
        self.gridLayout.addWidget(self.testNumber, 0, 1)
        
        # 準確率
        self.label_accuracy = QtWidgets.QLabel("準確率：")
        self.gridLayout.addWidget(self.label_accuracy, 1, 0)
        self.accuracyNumber = QtWidgets.QLabel("0%")
        self.gridLayout.addWidget(self.accuracyNumber, 1, 1)
        
        # 檢測率
        self.label_positive = QtWidgets.QLabel("檢測率：")
        self.gridLayout.addWidget(self.label_positive, 2, 0)
        self.positiveNumber = QtWidgets.QLabel("0%")
        self.gridLayout.addWidget(self.positiveNumber, 2, 1)
        
        # 新增 - 錯誤識別數量
        self.label_errors = QtWidgets.QLabel("錯誤識別數量：")
        self.gridLayout.addWidget(self.label_errors, 0, 2)
        self.errorsNumber = QtWidgets.QLabel("0")
        self.gridLayout.addWidget(self.errorsNumber, 0, 3)
        
        # 新增 - 專業術語解釋按鈕
        self.helpButton = QtWidgets.QPushButton("專業術語解釋")
        self.gridLayout.addWidget(self.helpButton, 2, 2, 1, 2)
        
        self.topLayout.addWidget(self.groupBox)
        
        # 進度條
        self.progressBar = QtWidgets.QProgressBar()
        self.topLayout.addWidget(self.progressBar)
        
        self.verticalLayout.addLayout(self.topLayout)
        
        # 下半部分：分割器
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        
        # 左側：類別詳細信息表格
        self.classTable = QtWidgets.QTableWidget()
        self.classTable.setColumnCount(7)  # 增加一列用於顯示標籤數量
        self.classTable.setHorizontalHeaderLabels(['類別', '標籤數量', 'TP', 'FP', 'FN', '準確率', '召回率'])
        self.classTable.horizontalHeader().setStretchLastSection(True)
        self.classTable.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.splitter.addWidget(self.classTable)
        
        # 右側：結果文本顯示
        self.resultList = QtWidgets.QTextEdit()
        self.resultList.setReadOnly(True)
        self.splitter.addWidget(self.resultList)
        
        self.verticalLayout.addWidget(self.splitter)
        
        # 設置窗口標題
        Dialog.setWindowTitle("多類別目標檢測評估器")