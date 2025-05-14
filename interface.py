import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QRadioButton, QGroupBox, QMessageBox
)
from tracker import ObjectDetectionStream  # Импорт класса трекинга

class CameraInterface(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Camera Tracker Mini App")
        self.setFixedSize(400, 250)

        # Основной вертикальный layout
        main_layout = QVBoxLayout()

        # Группа радиокнопок для выбора режима подключения
        connection_group = QGroupBox("Connection Mode")
        connection_layout = QHBoxLayout()

        self.radio_direct = QRadioButton("Direct URL")
        self.radio_direct.setChecked(True)
        self.radio_direct.toggled.connect(self.toggle_fields)
        self.radio_ip = QRadioButton("IP Connection")
        self.radio_ip.toggled.connect(self.toggle_fields)

        connection_layout.addWidget(self.radio_direct)
        connection_layout.addWidget(self.radio_ip)
        connection_group.setLayout(connection_layout)
        main_layout.addWidget(connection_group)

        # Группа для прямого ввода URL
        self.direct_group = QGroupBox("Direct Stream URL")
        direct_layout = QHBoxLayout()
        self.direct_lineedit = QLineEdit()
        self.direct_lineedit.setPlaceholderText("Enter stream URL")
        self.direct_paste_btn = QPushButton("Paste")
        self.direct_paste_btn.clicked.connect(lambda: self.paste_text(self.direct_lineedit))
        direct_layout.addWidget(self.direct_lineedit)
        direct_layout.addWidget(self.direct_paste_btn)
        self.direct_group.setLayout(direct_layout)
        main_layout.addWidget(self.direct_group)

        # Группа для ввода IP адреса и порта
        self.ip_group = QGroupBox("IP Connection")
        ip_layout = QHBoxLayout()
        ip_label = QLabel("IP:")
        self.ip_lineedit = QLineEdit()
        self.ip_lineedit.setPlaceholderText("Enter IP Address")
        self.ip_paste_btn = QPushButton("Paste")
        self.ip_paste_btn.clicked.connect(lambda: self.paste_text(self.ip_lineedit))
        port_label = QLabel("Port:")
        self.port_lineedit = QLineEdit()
        self.port_lineedit.setPlaceholderText("80")
        self.port_paste_btn = QPushButton("Paste")
        self.port_paste_btn.clicked.connect(lambda: self.paste_text(self.port_lineedit))

        ip_layout.addWidget(ip_label)
        ip_layout.addWidget(self.ip_lineedit)
        ip_layout.addWidget(self.ip_paste_btn)
        ip_layout.addWidget(port_label)
        ip_layout.addWidget(self.port_lineedit)
        ip_layout.addWidget(self.port_paste_btn)
        self.ip_group.setLayout(ip_layout)
        main_layout.addWidget(self.ip_group)

        # Изначально скрываем группу для IP, если выбран режим Direct URL
        self.ip_group.hide()

        # Кнопка старта трекинга
        self.start_btn = QPushButton("Start Tracking")
        self.start_btn.clicked.connect(self.start_tracking)
        main_layout.addWidget(self.start_btn)

        self.setLayout(main_layout)

    def toggle_fields(self):
        """Переключает видимость полей ввода в зависимости от выбранного режима."""
        if self.radio_direct.isChecked():
            self.direct_group.show()
            self.ip_group.hide()
        else:
            self.direct_group.hide()
            self.ip_group.show()

    def paste_text(self, line_edit: QLineEdit):
        """Вставляет текст из буфера обмена в заданное поле ввода."""
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        if text:
            line_edit.setText(text)
        else:
            QMessageBox.warning(self, "Warning", "Clipboard is empty!")

    def start_tracking(self):
        """Собирает данные из полей ввода и запускает процесс трекинга."""
        if self.radio_direct.isChecked():
            url = self.direct_lineedit.text().strip()
            if not url:
                QMessageBox.warning(self, "Error", "Please enter a Stream URL.")
                return
            stream_url = url
        else:
            ip_addr = self.ip_lineedit.text().strip()
            port = self.port_lineedit.text().strip() or "80"
            if not ip_addr:
                QMessageBox.warning(self, "Error", "Please enter an IP address.")
                return
            stream_url = f"http://{ip_addr}:{port}/videostream"

        self.close()  # Закрываем окно приложения

        # Запуск трекинга
        detector = ObjectDetectionStream(stream_url)
        detector()

def main():
    app = QApplication(sys.argv)
    window = CameraInterface()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
