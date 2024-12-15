# This is a sample Python script.
import pandas
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import profile
import numpy as np
import uis.py.welcome_page as wp

def find_weights(importances):
    #dim = len(importances.keys())
    matrix = importances.copy()
    dim = matrix.shape[0]
    #заполняем
    for i in range(dim):
        for j in range(dim):
            if i < j:
                matrix[i][j] = 1/matrix[j][i]
            elif i == j:
                matrix[i][j] = 1
    #нормализуем
    for i in range(dim):
        matrix[:, i] /= np.sum(matrix[:, i])
    #составляем вектор весов
    weights = [np.mean(row) for row in matrix]
    return weights


def find_fuzzy(answers):
    pass

from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget

import sys

from random import randint


class AnotherWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self):
        super().__init__()
        ui = wp.Ui_Welcome()
        ui.setupUi(self)




class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.w = None  # No external window yet.
        # self.button = QPushButton("Push for Window")
        # self.button.clicked.connect(self.show_new_window)
        # self.setCentralWidget(self.button)
        self.ui = wp.Ui_Welcome()
        self.ui.setupUi(QMainWindow(self))

    def show_new_window(self, checked):
        if self.w is None:
            #self.close()
            self.w = AnotherWindow()
            self.w.show()

        else:
            self.w.close()  # Close window.
            self.w = None  # Discard reference.


app = QApplication(sys.argv)
w = MainWindow()
w.show()
app.exec()