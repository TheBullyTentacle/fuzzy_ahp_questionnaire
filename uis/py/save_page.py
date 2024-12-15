# Form implementation generated from reading ui file 'uis\ui\save_page.ui'
#
# Created by: PyQt6 UI code generator 6.8.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QFileDialog


class Ui_savepage(object):
    def setupUi(self, savepage):
        savepage.setObjectName("savepage")
        savepage.resize(600, 300)
        self.centralwidget = QtWidgets.QWidget(parent=savepage)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(parent=self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 591, 261))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.verticalLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.greetings = QtWidgets.QLabel(parent=self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.greetings.setFont(font)
        self.greetings.setWordWrap(True)
        self.greetings.setObjectName("greetings")
        self.gridLayout.addWidget(self.greetings, 0, 1, 1, 2)
        self.toolButton = QtWidgets.QToolButton(parent=self.verticalLayoutWidget)
        self.toolButton.setObjectName("toolButton")
        self.gridLayout.addWidget(self.toolButton, 1, 2, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(130, 20, QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout.addItem(spacerItem, 0, 0, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(parent=self.verticalLayoutWidget)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 1, 0, 1, 2)
        self.start = QtWidgets.QPushButton(parent=self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.start.sizePolicy().hasHeightForWidth())
        self.start.setSizePolicy(sizePolicy)
        self.start.setObjectName("start")
        self.gridLayout.addWidget(self.start, 2, 0, 1, 3)
        savepage.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=savepage)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 600, 22))
        self.menubar.setObjectName("menubar")
        savepage.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=savepage)
        self.statusbar.setObjectName("statusbar")
        savepage.setStatusBar(self.statusbar)
        self.toolButton.clicked.connect(self.select_folder)

        self.retranslateUi(savepage)
        QtCore.QMetaObject.connectSlotsByName(savepage)

    def retranslateUi(self, savepage):
        _translate = QtCore.QCoreApplication.translate
        savepage.setWindowTitle(_translate("savepage", "Save"))
        self.greetings.setText(_translate("savepage", "Выберете, куда сохранить ответы"))
        self.toolButton.setText(_translate("savepage", "..."))
        self.start.setText(_translate("savepage", "Выбрать"))
    def select_folder(self):
        # Open the folder selection dialog and store the selected folder path
        folder_path = QFileDialog.getExistingDirectory()

        if folder_path:
            # Set the folder path to the QLineEdit
            self.lineEdit.setText(folder_path)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    savepage = QtWidgets.QMainWindow()
    ui = Ui_savepage()
    ui.setupUi(savepage)
    savepage.show()
    sys.exit(app.exec())