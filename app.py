import sys
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel,
    QFileDialog, QComboBox, QTextEdit
)
from PyQt5.QtCore import Qt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

class LoanRepaymentApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Loan Repayment Prediction")
        self.setGeometry(200, 200, 600, 400)

        self.dataset = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.uploadButton = QPushButton("Upload Dataset")
        self.uploadButton.clicked.connect(self.load_dataset)
        layout.addWidget(self.uploadButton)

        self.modelSelector = QComboBox()
        self.modelSelector.addItems(["Decision Tree", "Random Forest", "Gradient Boosting"])
        layout.addWidget(self.modelSelector)

        self.runButton = QPushButton("Run Model")
        self.runButton.clicked.connect(self.run_model)
        layout.addWidget(self.runButton)

        self.resultArea = QTextEdit()
        self.resultArea.setReadOnly(True)
        layout.addWidget(self.resultArea)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_dataset(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv)")
        if file_path:
            self.dataset = pd.read_csv(file_path)
            self.resultArea.append(f"Dataset loaded: {file_path}")
            self.resultArea.append(f"Dataset Head:\n{self.dataset.head()}")

    def run_model(self):
        if self.dataset is None:
            self.resultArea.append("Please upload a dataset first!")
            return

        # Preprocess dataset
        df = self.dataset.copy()
        df['purpose'] = LabelEncoder().fit_transform(df['purpose'])
        X = df.drop('not.fully.paid', axis=1)
        y = df['not.fully.paid']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

        # Select model
        model_name = self.modelSelector.currentText()
        if model_name == "Decision Tree":
            model = DecisionTreeClassifier(max_depth=2)
        elif model_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=100)
        elif model_name == "Gradient Boosting":
            model = GradientBoostingClassifier(learning_rate=0.05)
        else:
            self.resultArea.append("Invalid model selection!")
            return

        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        self.resultArea.append(f"\n<----- {model_name} Results ----->\n")
        self.resultArea.append(f"Accuracy: {accuracy:.2f}")
        self.resultArea.append(f"Classification Report:\n{report}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LoanRepaymentApp()
    window.show()
    sys.exit(app.exec_())
