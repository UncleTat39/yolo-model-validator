import sys
import os
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView, QTextBrowser, QScrollArea
from PyQt5.QtCore import Qt
from Ui_gui import Ui_Dialog
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from terminology_text import TERMINOLOGY_TEXT  # 導入術語解釋文本

class TerminologyDialog(QDialog):
    def __init__(self, parent=None):
        super(TerminologyDialog, self).__init__(parent)
        self.setWindowTitle("專業術語解釋")
        self.resize(600, 600)
        
        layout = QVBoxLayout(self)
        
        textBrowser = QTextBrowser(self)
        textBrowser.setOpenExternalLinks(True)
        layout.addWidget(textBrowser)
        
        # 使用從術語文件導入的文本
        textBrowser.setHtml(TERMINOLOGY_TEXT)
        self.setLayout(layout)


class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(fig)

class ErrorsDialog(QDialog):
    def __init__(self, parent=None, error_records=None):
        super(ErrorsDialog, self).__init__(parent)
        self.setWindowTitle("Error Identification Records")
        self.resize(900, 600)
        
        layout = QVBoxLayout(self)
        
        # Create table
        self.tableWidget = QTableWidget(self)
        self.tableWidget.setColumnCount(5)
        self.tableWidget.setHorizontalHeaderLabels(['Image File', 'Actual Class', 'Predicted Class', 'Confidence', 'Bounding Box'])
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        if error_records:
            self.tableWidget.setRowCount(len(error_records))
            for i, record in enumerate(error_records):
                self.tableWidget.setItem(i, 0, QTableWidgetItem(record['Image_File']))
                self.tableWidget.setItem(i, 1, QTableWidgetItem(record['Actual_Class']))
                self.tableWidget.setItem(i, 2, QTableWidgetItem(record['Predicted_Class']))
                self.tableWidget.setItem(i, 3, QTableWidgetItem(f"{record['Confidence']:.2f}"))
                self.tableWidget.setItem(i, 4, QTableWidgetItem(record['Box']))
                
                # Set tooltip
                for j in range(5):
                    item = self.tableWidget.item(i, j)
                    if item:
                        item.setToolTip(f"Click to open image: {record['Image_File']}")
                        
        self.tableWidget.cellClicked.connect(self.open_image)
        
        layout.addWidget(self.tableWidget)
        self.setLayout(layout)
        self.error_records = error_records
        
    def open_image(self, row, column):
        """Open the annotated image showing error identification"""
        if self.error_records and row < len(self.error_records):
            record = self.error_records[row]
            image_path = os.path.join(
                os.path.dirname(record['Image_Path']), 
                'annotated_results', 
                f"annotated_{record['Image_File']}"
            )
            if os.path.exists(image_path):
                # Open image in system default program
                os.startfile(image_path) if os.name == 'nt' else os.system(f'open "{image_path}"')
            else:
                QMessageBox.warning(self, "Error", f"Cannot find annotated image: {image_path}")

class VisualizationDialog(QDialog):
    def __init__(self, parent=None, class_names=None, confusion_matrix=None, class_metrics=None):
        super(VisualizationDialog, self).__init__(parent)
        self.setWindowTitle("Result Visualization")
        self.resize(1000, 800)
        
        self.class_names = class_names
        self.confusion_matrix = confusion_matrix
        self.class_metrics = class_metrics
        
        layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tabWidget = QtWidgets.QTabWidget(self)
        
        # Add confusion matrix chart
        if confusion_matrix is not None and class_names:
            self.confusionMatrixTab = QtWidgets.QWidget()
            confusionLayout = QVBoxLayout(self.confusionMatrixTab)
            
            self.confusionCanvas = MatplotlibCanvas(self, width=9, height=8)
            self.plot_confusion_matrix()
            confusionLayout.addWidget(self.confusionCanvas)
            
            self.tabWidget.addTab(self.confusionMatrixTab, "Confusion Matrix")
        
        # Add precision/recall chart
        if class_metrics is not None and class_names:
            self.precisionRecallTab = QtWidgets.QWidget()
            prLayout = QVBoxLayout(self.precisionRecallTab)
            
            self.prCanvas = MatplotlibCanvas(self, width=9, height=8)
            self.plot_precision_recall()
            prLayout.addWidget(self.prCanvas)
            
            self.tabWidget.addTab(self.precisionRecallTab, "Precision/Recall")
            
            # Add F1 score chart
            self.f1ScoreTab = QtWidgets.QWidget()
            f1Layout = QVBoxLayout(self.f1ScoreTab)
            
            self.f1Canvas = MatplotlibCanvas(self, width=9, height=8)
            self.plot_f1_scores()
            f1Layout.addWidget(self.f1Canvas)
            
            self.tabWidget.addTab(self.f1ScoreTab, "F1 Score")
        
        layout.addWidget(self.tabWidget)
        self.setLayout(layout)
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix visualization"""
        ax = self.confusionCanvas.axes
        im = ax.imshow(self.confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Set axis labels
        ax.set(xticks=np.arange(len(self.class_names)),
               yticks=np.arange(len(self.class_names)),
               xticklabels=self.class_names, 
               yticklabels=self.class_names,
               ylabel='Actual Class',
               xlabel='Predicted Class',
               title='Confusion Matrix')
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add value annotations
        thresh = self.confusion_matrix.max() / 2.
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                ax.text(j, i, format(self.confusion_matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if self.confusion_matrix[i, j] > thresh else "black")
        
        self.confusionCanvas.draw()
    
    def plot_precision_recall(self):
        """Plot precision and recall bar chart"""
        ax = self.prCanvas.axes
        
        precision_values = []
        recall_values = []
        
        for i in range(len(self.class_names)):
            metrics = self.class_metrics[i]
            precision = metrics['TP'] / (metrics['TP'] + metrics['FP']) if (metrics['TP'] + metrics['FP']) > 0 else 0
            recall = metrics['TP'] / (metrics['TP'] + metrics['FN']) if (metrics['TP'] + metrics['FN']) > 0 else 0
            
            precision_values.append(precision)
            recall_values.append(recall)
        
        x = np.arange(len(self.class_names))
        width = 0.35
        
        ax.bar(x - width/2, precision_values, width, label='Precision')
        ax.bar(x + width/2, recall_values, width, label='Recall')
        
        ax.set_ylabel('Score')
        ax.set_title('Precision and Recall by Class')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Add value annotations
        for i, v in enumerate(precision_values):
            ax.text(i - width/2, v + 0.02, f"{v:.2f}", ha='center')
        
        for i, v in enumerate(recall_values):
            ax.text(i + width/2, v + 0.02, f"{v:.2f}", ha='center')
            
        ax.figure.tight_layout()
        self.prCanvas.draw()
    
    def plot_f1_scores(self):
        """Plot F1 score bar chart"""
        ax = self.f1Canvas.axes
        
        f1_scores = []
        
        for i in range(len(self.class_names)):
            metrics = self.class_metrics[i]
            precision = metrics['TP'] / (metrics['TP'] + metrics['FP']) if (metrics['TP'] + metrics['FP']) > 0 else 0
            recall = metrics['TP'] / (metrics['TP'] + metrics['FN']) if (metrics['TP'] + metrics['FN']) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            f1_scores.append(f1)
        
        ax.bar(self.class_names, f1_scores, color='green')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Scores by Class')
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        
        # Add value annotations
        for i, v in enumerate(f1_scores):
            ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
            
        ax.figure.tight_layout()
        self.f1Canvas.draw()

class MultiClassDetectorApp(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        # 初始化YOLO模型為None，稍後讓用戶選擇
        self.yolo_model = None
        
        # 預設類別名稱列表
        self.default_class_names = [
            'MTR', 'bus', 'bus_stop', 'car',
            'ferry', 'light_rail', 'mini_bus', 'pedestrian',
            'pedestrian_crossing', 'taxi', 'traffic_light', 'tram'
        ]
        
        # 設置默認類別到文本框
        self.classNamesText.setText(', '.join(self.default_class_names))
        
        # 類別名稱將從GUI獲取
        self.class_names = self.default_class_names
        
        # 初始化計數器和評估指標
        self.total_tests = 0
        self.class_metrics = {i: {'TP': 0, 'FP': 0, 'FN': 0} for i in range(len(self.class_names))}
        self.label_statistics = {i: 0 for i in range(len(self.class_names))}
        self.selected_path = ""
        
        # 閾值將從GUI獲取
        self.confidence_threshold = self.confidenceThreshold.value()
        self.iou_threshold = self.iouThreshold.value()
        
        # 初始化混淆矩陣
        self.confusion_matrix = np.zeros((len(self.class_names), len(self.class_names)), dtype=int)
        
        # 新增 - 錯誤識別記錄
        self.error_records = []
        
        # 為每個類別設置顏色 (BGR格式)
        self.update_class_colors()
        
        # 連接按鈕信號
        self.selectPath.clicked.connect(self.select_folder)
        self.selectModel.clicked.connect(self.select_model)
        self.startButton.clicked.connect(self.start_detection)
        self.visualizeButton.clicked.connect(self.show_visualization)
        self.viewErrorsButton.clicked.connect(self.show_errors)
        self.helpButton.clicked.connect(self.show_terminology_help)
        
        # 連接閾值變化的信號
        self.confidenceThreshold.valueChanged.connect(self.update_thresholds)
        self.iouThreshold.valueChanged.connect(self.update_thresholds)
        
        # 類別名稱文本框變化信號
        self.classNamesText.textChanged.connect(self.update_class_names)
        
        # 初始化結果列表
        self.detection_results = []

    def show_terminology_help(self):
        """顯示專業術語解釋對話框"""
        dialog = TerminologyDialog(self)
        dialog.exec_()

    def update_class_colors(self):
        """為每個類別設置隨機顏色"""
        self.class_colors = {
            i: (np.random.randint(0, 255), 
                np.random.randint(0, 255), 
                np.random.randint(0, 255)) 
            for i in range(len(self.class_names))
        }

    def update_class_names(self):
        """從文本框更新類別名稱列表"""
        try:
            # 從文本框獲取類別名稱（以逗號分隔）
            text = self.classNamesText.toPlainText()
            if text.strip():
                new_class_names = [name.strip() for name in text.split(',')]
                # 確保至少有一個類別
                if new_class_names and all(new_class_names):
                    self.class_names = new_class_names
                    # 更新相關數據結構
                    self.class_metrics = {i: {'TP': 0, 'FP': 0, 'FN': 0} for i in range(len(self.class_names))}
                    self.label_statistics = {i: 0 for i in range(len(self.class_names))}
                    self.confusion_matrix = np.zeros((len(self.class_names), len(self.class_names)), dtype=int)
                    self.update_class_colors()
                    print(f"類別名稱已更新: {self.class_names}")
        except Exception as e:
            print(f"更新類別名稱時發生錯誤: {str(e)}")

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "選擇測試圖片文件夾")
        if folder_path:
            self.selected_path = folder_path
            self.pathText.setText(folder_path)

    def select_model(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "選擇YOLO模型", "", "模型文件 (*.pt);;所有文件 (*)")
        if model_path:
            try:
                # 嘗試加載模型
                self.modelText.setText(model_path)
                # 不在此處加載模型，而是在開始檢測時加載，避免選擇不同模型時反覆加載
                QMessageBox.information(self, "成功", "模型路徑已設置，開始檢測時將加載模型")
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"無法設置模型路徑：{str(e)}")

    def update_thresholds(self):
        """更新閾值設置"""
        self.confidence_threshold = self.confidenceThreshold.value()
        self.iou_threshold = self.iouThreshold.value()
        print(f"閾值已更新: 置信度={self.confidence_threshold}, IOU={self.iou_threshold}")

    def count_total_labels(self):
        """統計所有標籤文件中的類別數量"""
        self.label_statistics = {i: 0 for i in range(len(self.class_names))}
        if not self.selected_path:
            return
        
        for file in os.listdir(self.selected_path):
            if file.endswith('.txt') and not file.startswith('metrics_'):
                label_path = os.path.join(self.selected_path, file)
                try:
                    with open(label_path, 'r') as f:
                        for line in f:
                            data = line.strip().split()
                            if len(data) == 5:
                                class_id = int(data[0])
                                if class_id < len(self.class_names):  # 確保類別ID在範圍內
                                    self.label_statistics[class_id] += 1
                except Exception as e:
                    print(f"讀取標籤文件 {file} 時發生錯誤: {str(e)}")

    def calculate_iou(self, box1, box2):
        """計算兩個邊界框的IOU"""
        # 確保輸入格式統一 (轉換為 x1,y1,x2,y2 格式)
        if len(box1) > 4:  # yolo格式 [class, x_center, y_center, width, height]
            x1_1 = float(box1[1]) - float(box1[3])/2
            y1_1 = float(box1[2]) - float(box1[4])/2
            x2_1 = float(box1[1]) + float(box1[3])/2
            y2_1 = float(box1[2]) + float(box1[4])/2
        else:  # [x1,y1,x2,y2] 格式
            x1_1, y1_1, x2_1, y2_1 = box1[:4]

        if len(box2) > 4:
            x1_2 = float(box2[1]) - float(box2[3])/2
            y1_2 = float(box2[2]) - float(box2[4])/2
            x2_2 = float(box2[1]) + float(box2[3])/2
            y2_2 = float(box2[2]) + float(box2[4])/2
        else:
            x1_2, y1_2, x2_2, y2_2 = box2[:4]

        # 計算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0

    def read_yolo_label(self, label_path, img_width, img_height):
        """讀取YOLO格式的標籤文件"""
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    if len(data) == 5:
                        class_id = int(data[0])
                        # 確保類別ID在範圍內
                        if class_id < len(self.class_names):
                            x_center = float(data[1]) * img_width
                            y_center = float(data[2]) * img_height
                            width = float(data[3]) * img_width
                            height = float(data[4]) * img_height
                            boxes.append([class_id, x_center, y_center, width, height])
        return boxes

    def update_results(self):
        """更新評估指標顯示"""
        try:
            # 計算總體指標
            total_tp = sum(metrics['TP'] for metrics in self.class_metrics.values())
            total_fp = sum(metrics['FP'] for metrics in self.class_metrics.values())
            total_fn = sum(metrics['FN'] for metrics in self.class_metrics.values())
            
            if total_tp + total_fp > 0:
                precision = total_tp / (total_tp + total_fp)
            else:
                precision = 0
                
            if total_tp + total_fn > 0:
                recall = total_tp / (total_tp + total_fn)
            else:
                recall = 0
                
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # 更新總體指標顯示
            self.testNumber.setText(str(self.total_tests))
            self.accuracyNumber.setText(f"{precision*100:.2f}%")
            self.positiveNumber.setText(f"{recall*100:.2f}%")
            self.errorsNumber.setText(str(len(self.error_records)))
            
            # 更新類別表格
            self.classTable.setRowCount(len(self.class_names))
            for i, class_name in enumerate(self.class_names):
                metrics = self.class_metrics[i]
                
                # 計算該類別的準確率和召回率
                class_precision = metrics['TP'] / (metrics['TP'] + metrics['FP']) if (metrics['TP'] + metrics['FP']) > 0 else 0
                class_recall = metrics['TP'] / (metrics['TP'] + metrics['FN']) if (metrics['TP'] + metrics['FN']) > 0 else 0
                
                # 設置表格內容
                self.classTable.setItem(i, 0, QtWidgets.QTableWidgetItem(class_name))
                self.classTable.setItem(i, 1, QtWidgets.QTableWidgetItem(str(self.label_statistics[i])))
                self.classTable.setItem(i, 2, QtWidgets.QTableWidgetItem(str(metrics['TP'])))
                self.classTable.setItem(i, 3, QtWidgets.QTableWidgetItem(str(metrics['FP'])))
                self.classTable.setItem(i, 4, QtWidgets.QTableWidgetItem(str(metrics['FN'])))
                self.classTable.setItem(i, 5, QtWidgets.QTableWidgetItem(f"{class_precision*100:.2f}%"))
                self.classTable.setItem(i, 6, QtWidgets.QTableWidgetItem(f"{class_recall*100:.2f}%"))
            
            # 更新進度條
            if hasattr(self, 'image_files') and len(self.image_files) > 0:
                progress = (self.total_tests / len(self.image_files)) * 100
                self.progressBar.setValue(int(progress))
            
            # 更新結果文本
            total_labels = sum(self.label_statistics.values())
            current_text = (f"處理進度: {self.total_tests}/{len(self.image_files) if hasattr(self, 'image_files') else 0}\n"
                          f"總體準確率: {precision*100:.2f}%\n"
                          f"總體召回率: {recall*100:.2f}%\n"
                          f"總體F1分數: {f1_score*100:.2f}%\n"
                          f"錯誤識別數量: {len(self.error_records)}\n"
                          f"置信度閾值: {self.confidence_threshold}\n"
                          f"IOU閾值: {self.iou_threshold}\n"
                          f"\n標籤統計:\n"
                          f"總標籤數量: {total_labels}\n")
            
            # 添加每個類別的標籤數量
            for i, class_name in enumerate(self.class_names):
                current_text += f"{class_name}: {self.label_statistics[i]}\n"
            
            self.resultList.setText(current_text)
            
        except Exception as e:
            print(f"更新結果時發生錯誤: {str(e)}")

    def show_visualization(self):
        """顯示結果可視化對話框"""
        if hasattr(self, 'confusion_matrix') and hasattr(self, 'class_metrics'):
            dialog = VisualizationDialog(
                self, 
                class_names=self.class_names, 
                confusion_matrix=self.confusion_matrix,
                class_metrics=self.class_metrics
            )
            dialog.exec_()
        else:
            QMessageBox.warning(self, "提示", "沒有可用的檢測結果數據進行可視化")

    def show_errors(self):
        """顯示錯誤識別記錄對話框"""
        if hasattr(self, 'error_records') and self.error_records:
            dialog = ErrorsDialog(self, self.error_records)
            dialog.exec_()
        else:
            QMessageBox.information(self, "提示", "沒有發現錯誤識別記錄")

    def start_detection(self):
        if not self.selected_path:
            QMessageBox.warning(self, "警告", "請先選擇測試圖片文件夾")
            return
            
        if not self.modelText.text():
            QMessageBox.warning(self, "警告", "請先選擇AI模型文件")
            return

        try:
            # 再次更新類別名稱，確保使用最新輸入
            self.update_class_names()
            
            # 加載模型
            model_path = self.modelText.text()
            try:
                self.yolo_model = YOLO(model_path)
                print(f"成功加載模型: {model_path}")
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"無法加載模型：{str(e)}")
                return
                
            # 獲取最新的閾值設置
            self.confidence_threshold = self.confidenceThreshold.value()
            self.iou_threshold = self.iouThreshold.value()
            
            # 重置所有計數器
            self.total_tests = 0
            self.class_metrics = {i: {'TP': 0, 'FP': 0, 'FN': 0} for i in range(len(self.class_names))}
            self.confusion_matrix = np.zeros((len(self.class_names), len(self.class_names)), dtype=int)
            self.detection_results = []
            self.error_records = []  # 重置錯誤記錄
            
            # 統計所有標籤
            self.count_total_labels()
            
            # 獲取圖片文件列表
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
            self.image_files = [f for f in os.listdir(self.selected_path) 
                             if f.lower().endswith(image_extensions)]
            
            # 重置進度條
            self.progressBar.setValue(0)
            self.resultList.clear()

            result_dir = os.path.join(self.selected_path, 'annotated_results')
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            for image_file in self.image_files:
                image_path = os.path.join(self.selected_path, image_file)
                label_path = os.path.join(
                    self.selected_path, 
                    os.path.splitext(image_file)[0] + '.txt'
                )

                try:
                    img = cv2.imread(image_path)
                    if img is None:
                        continue

                    self.total_tests += 1
                    height, width = img.shape[:2]

                    # 讀取標籤文件
                    ground_truth = self.read_yolo_label(label_path, width, height)

                    # YOLO檢測
                    start_time = time.time()
                    results = self.yolo_model(img)
                    processing_time = time.time() - start_time

                    result_image = img.copy()
                    predictions = []

                    # 處理檢測結果
                    for detection in results[0].boxes.data:
                        confidence = float(detection[4])
                        if confidence > self.confidence_threshold:
                            x1, y1, x2, y2 = map(int, detection[:4])
                            class_id = int(detection[5])
                            
                            # 確保類別ID在有效範圍內
                            if class_id < len(self.class_names):
                                predictions.append([
                                    class_id,
                                    (x1 + x2)/2,  # x_center
                                    (y1 + y2)/2,  # y_center
                                    x2 - x1,      # width
                                    y2 - y1       # height
                                ])

                                # 繪製預測框和標籤
                                color = self.class_colors[class_id]
                                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                                label = f"{self.class_names[class_id]} {confidence:.2f}"
                                cv2.putText(result_image, label, (x1, y1-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                                # 記錄檢測結果
                                self.detection_results.append({
                                    'Image_File': image_file,
                                    'Class': self.class_names[class_id],
                                    'Confidence': confidence,
                                    'Box': f"{x1},{y1},{x2},{y2}",
                                    'Processing_Time': processing_time
                                })

                    # 評估預測結果
                    matched_gt = [False] * len(ground_truth)
                    matched_pred = [False] * len(predictions)

                    # 匹配預測框和真實框
                    for pred_idx, pred_box in enumerate(predictions):
                        best_iou = 0
                        best_gt_idx = -1

                        for gt_idx, gt_box in enumerate(ground_truth):
                            if not matched_gt[gt_idx]:
                                iou = self.calculate_iou(pred_box, gt_box)
                                if iou > best_iou and iou >= self.iou_threshold:
                                    best_iou = iou
                                    best_gt_idx = gt_idx

                        if best_gt_idx >= 0:
                            matched_gt[best_gt_idx] = True
                            matched_pred[pred_idx] = True

                            pred_class = pred_box[0]
                            gt_class = ground_truth[best_gt_idx][0]

                            # 更新混淆矩陣
                            self.confusion_matrix[gt_class][pred_class] += 1

                            # 更新評估指標
                            if pred_class == gt_class:
                                self.class_metrics[pred_class]['TP'] += 1
                            else:
                                # 錯誤識別情況，記錄下來
                                self.class_metrics[pred_class]['FP'] += 1
                                self.class_metrics[gt_class]['FN'] += 1
                                
                                # 保存錯誤識別記錄
                                x1 = int(pred_box[1] - pred_box[3]/2)
                                y1 = int(pred_box[2] - pred_box[4]/2)
                                x2 = int(pred_box[1] + pred_box[3]/2)
                                y2 = int(pred_box[2] + pred_box[4]/2)
                                
                                self.error_records.append({
                                    'Image_File': image_file,
                                    'Image_Path': image_path,
                                    'Actual_Class': self.class_names[gt_class],
                                    'Predicted_Class': self.class_names[pred_class],
                                    'Confidence': confidence,
                                    'Box': f"{x1},{y1},{x2},{y2}",
                                    'IOU': best_iou
                                })
                                
                                # 在原圖上標註錯誤
                                # 紅色表示錯誤識別
                                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                error_label = f"ERROR! Actual:{self.class_names[gt_class]} Pred:{self.class_names[pred_class]}"
                                cv2.putText(result_image, error_label, (x1, y1-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # 處理未匹配的預測框和真實框
                    for pred_idx, matched in enumerate(matched_pred):
                        if not matched:
                            pred_class = predictions[pred_idx][0]
                            self.class_metrics[pred_class]['FP'] += 1

                    for gt_idx, matched in enumerate(matched_gt):
                        if not matched:
                            gt_class = ground_truth[gt_idx][0]
                            self.class_metrics[gt_class]['FN'] += 1

                    # 保存標註後的圖片
                    result_path = os.path.join(result_dir, f'annotated_{image_file}')
                    cv2.imwrite(result_path, result_image)

                    # 更新界面
                    self.update_results()
                    QtWidgets.QApplication.processEvents()

                except Exception as e:
                    print(f"處理圖片 {image_file} 時發生錯誤: {str(e)}")
                    continue

            # 保存結果
            self.save_results()
            
            # 啟用可視化按鈕
            self.visualizeButton.setEnabled(True)
            self.viewErrorsButton.setEnabled(True)
            
            # 顯示完成消息和詳細統計
            self.show_final_statistics()

        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"處理過程中發生錯誤：{str(e)}")

    def show_final_statistics(self):
        """顯示詳細的評估統計結果"""
        summary = "檢測完成！\n\n"
        summary += f"使用置信度閾值: {self.confidence_threshold}\n"
        summary += f"使用IOU閾值: {self.iou_threshold}\n"
        summary += f"錯誤識別數量: {len(self.error_records)}\n\n"
        
        summary += "標籤統計：\n"
        total_labels = sum(self.label_statistics.values())
        summary += f"總標籤數量: {total_labels}\n"
        
        for i, class_name in enumerate(self.class_names):
            summary += f"{class_name}: {self.label_statistics[i]}\n"
        
        summary += "\n類別統計：\n"
        
        for i in range(len(self.class_names)):
            metrics = self.class_metrics[i]
            total = metrics['TP'] + metrics['FP'] + metrics['FN']
            if total > 0:
                precision = metrics['TP'] / (metrics['TP'] + metrics['FP']) if (metrics['TP'] + metrics['FP']) > 0 else 0
                recall = metrics['TP'] / (metrics['TP'] + metrics['FN']) if (metrics['TP'] + metrics['FN']) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                summary += f"\n{self.class_names[i]}:\n"
                summary += f"TP: {metrics['TP']}, FP: {metrics['FP']}, FN: {metrics['FN']}\n"
                summary += f"Precision: {precision:.2%}\n"
                summary += f"Recall: {recall:.2%}\n"
                summary += f"F1-Score: {f1:.2%}\n"
        
        QMessageBox.information(self, "評估結果", summary)

    def save_results(self):
        results_dir = os.path.join(self.selected_path, 'detection_results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存評估指標
        metrics_path = os.path.join(results_dir, f'metrics_{timestamp}_conf{self.confidence_threshold:.2f}_iou{self.iou_threshold:.2f}.txt')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write(f"Evaluation Time: {timestamp}\n")
            f.write(f"Total Tests: {self.total_tests}\n")
            f.write(f"Confidence Threshold: {self.confidence_threshold}\n")
            f.write(f"IOU Threshold: {self.iou_threshold}\n")
            f.write(f"Model Path: {self.modelText.text()}\n")
            f.write(f"Class Names: {', '.join(self.class_names)}\n")
            f.write(f"Error Count: {len(self.error_records)}\n\n")
            
            # Label statistics
            f.write("Label Statistics:\n")
            total_labels = sum(self.label_statistics.values())
            f.write(f"Total Labels: {total_labels}\n")
            for i, class_name in enumerate(self.class_names):
                f.write(f"{class_name}: {self.label_statistics[i]}\n")
            f.write("\n")
            
            f.write("Class Metrics:\n")
            for i in range(len(self.class_names)):
                metrics = self.class_metrics[i]
                f.write(f"\n{self.class_names[i]}:\n")
                f.write(f"True Positives: {metrics['TP']}\n")
                f.write(f"False Positives: {metrics['FP']}\n")
                f.write(f"False Negatives: {metrics['FN']}\n")
                
                if metrics['TP'] + metrics['FP'] > 0:
                    precision = metrics['TP'] / (metrics['TP'] + metrics['FP'])
                    f.write(f"Precision: {precision:.2%}\n")
                
                if metrics['TP'] + metrics['FN'] > 0:
                    recall = metrics['TP'] / (metrics['TP'] + metrics['FN'])
                    f.write(f"Recall: {recall:.2%}\n")
                
                if metrics['TP'] + metrics['FP'] > 0 and metrics['TP'] + metrics['FN'] > 0:
                    precision = metrics['TP'] / (metrics['TP'] + metrics['FP'])
                    recall = metrics['TP'] / (metrics['TP'] + metrics['FN'])
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    f.write(f"F1-Score: {f1:.2%}\n")
            
            f.write("\nConfusion Matrix:\n")
            f.write("Predicted →\nActual ↓\n")
            f.write("\t" + "\t".join(self.class_names) + "\n")
            
            for i in range(len(self.class_names)):
                f.write(f"{self.class_names[i]}\t")
                f.write("\t".join(str(x) for x in self.confusion_matrix[i]) + "\n")

        # 保存CSV結果
        csv_path = os.path.join(results_dir, f'detection_results_{timestamp}_conf{self.confidence_threshold:.2f}_iou{self.iou_threshold:.2f}.csv')
        df = pd.DataFrame(self.detection_results)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 新增 - 保存錯誤識別記錄
        if self.error_records:
            error_csv_path = os.path.join(results_dir, f'error_records_{timestamp}_conf{self.confidence_threshold:.2f}_iou{self.iou_threshold:.2f}.csv')
            error_df = pd.DataFrame(self.error_records)
            error_df.to_csv(error_csv_path, index=False, encoding='utf-8-sig')
            print(f"已保存錯誤識別記錄到: {error_csv_path}")

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MultiClassDetectorApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()