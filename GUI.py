import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QMessageBox, QFileDialog, QWidget, QVBoxLayout, QLabel,
QButtonGroup, QPushButton, QLineEdit, QMenu, QStackedWidget, QHBoxLayout, QSpacerItem, QSizePolicy, QScrollArea, QCheckBox, QComboBox)
from PyQt6.QtGui import QAction, QPalette, QColor
from PyQt6.QtCore import Qt
from CameraConnection import CameraURLMeneger
import os
import json
from ultralytics import YOLO
try:
    from trainmodel import ModelTrainer, train_from_settings
    TRAIN_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Модуль обучения не доступен: {e}")
    TRAIN_MODULE_AVAILABLE = False
#import tracker


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.click_info = 1
        self.click_info1 = 1
        self.setWindowTitle("Track Camer")
        self.setGeometry(200, 200, 1200, 800)
        self.currentWidget = 0
        self.dark_mode = False
        self.file_paths = {}
        self.load_initial_paths()
        #self.thremes_logic()
        menu_bar = self.menuBar()

        connect_menu = menu_bar.addMenu("Подключение")

        open_action = QAction("Открыть", self)
        open_action.triggered.connect(self.open_camer_conector)
        connect_menu.addAction(open_action)
        self.connect_camer_window = None
        

        self.settings_menu = menu_bar.addMenu("Настройки")
        self.settingAction = QAction("Открыть", self)
        self.settingAction.triggered.connect(self.openSettingWindow)
        self.settings_menu.addAction(self.settingAction)
        self.openSettingsWindow = None

        control_camer_menu = menu_bar.addMenu('Управление Камерами')

        self.limit_fps_action = QAction(self.limit_fps_text(), self)
        self.limit_fps_action.triggered.connect(self.limit_fps_toggle)
        control_camer_menu.addAction(self.limit_fps_action)

        self.fps_max_action = QAction(self.fps_max_text(), self)
        self.fps_max_action.triggered.connect(self.fps_max_logic)
        control_camer_menu.addAction(self.fps_max_action)

        self.model_track_action = QAction(self.model_track_text(), self)
        self.model_track_action.triggered.connect(self.model_track_logic)
        control_camer_menu.addAction(self.model_track_action)

        self.work_track_action = QAction(self.work_track_text(), self)
        self.work_track_action.triggered.connect(self.work_track_logic)
        control_camer_menu.addAction(self.work_track_action)

        information_camer_action = QAction("Информация о камере", self)
        information_camer_action.triggered.connect(self.information_camer_logic)
        control_camer_menu.addAction(information_camer_action)


        train_modele_menu = menu_bar.addMenu("Обучение Модели")
        path_for_train_model_action = QAction("Путь данных для обучения модели", self)
        path_for_train_model_action.triggered.connect(self.path_for_train_model_logic)
        train_modele_menu.addAction(path_for_train_model_action)

        model_train_path = QAction("Модель которую будете обучать", self)
        model_train_path.triggered.connect(self.model_train_path_logic)
        train_modele_menu.addAction(model_train_path)

        train_model = QAction("Обучение", self)
        train_model.triggered.connect(self.start_model_training)
        train_modele_menu.addAction(train_model)

        view_multipe_camers_menu = menu_bar.addMenu("Просмотр нескольких камер одновременно")


#######################################################################################


        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.page1 = self.createPage("1 Страница")
        self.page2 = self.createPage("2 Страница")
        self.page3 = self.createPage("3 Страница")

        self.stack.addWidget(self.page1)
        self.stack.addWidget(self.page2)
        self.stack.addWidget(self.page3)

    def get_file_path(self, path_key):
        return self.file_paths.get(path_key, "")

    def createPage(self, labelText):
        page = QWidget()
        
        vbox = QVBoxLayout(page)# vertical


        label = QLabel(labelText, page)
        label.setStyleSheet('font-size: 20px')
        label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        vbox.addWidget(label)
        vbox.addStretch(1)


        hbox = QHBoxLayout()# horizontal

        btnPrev = QPushButton("<-", page)
        btnPrev.setFixedSize(40,20)
        btnPrev.clicked.connect(self.prevPage)
        hbox.addWidget(btnPrev, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)


        btnNext = QPushButton("->", page)
        btnNext.setFixedSize(40, 20)
        btnNext.clicked.connect(self.nextPage)
        hbox.addWidget(btnNext, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        vbox.addLayout(hbox)

        page.btnPrev = btnPrev
        page.btnNext = btnNext
        if self.currentWidget == 0:
            page.btnPrev.setVisible(False) 
        else:
            page.btnPrev.setVisible(True)
        return page
    
    def buttonsUpdate(self):
        page = self.stack.widget(self.currentWidget)

        page.btnPrev.setVisible(self.currentWidget > 0)

        page.btnNext.setVisible(self.currentWidget < self.stack.count() - 1)

        self.stack.setCurrentIndex(self.currentWidget)
        
    def nextPage(self):
        if self.currentWidget < self.stack.count() - 1:
            self.currentWidget += 1
            self.buttonsUpdate()
    
    def prevPage(self):
        if self.currentWidget > 0:
            self.currentWidget -=1
            self.buttonsUpdate()


    def open_camer_conector(self):
        self.connect_camer_window = ConnectWindowCamer(
            palette=self.palette(),
            dark_mode=getattr(self, "dark_mode", False)
        )
        self.connect_camer_window.show()

    def openSettingWindow(self):
        self.openSettingsWindow = settingsWindow(
            parent=self, 
            palette=self.palette(),
            dark_mode=getattr(self, "dark_mode", False)
        )
        self.openSettingsWindow.show()

    def save_file(self):
        QMessageBox.information(self, "Сохранить", "Здесь будет логика сохранения файла")

    def upgrade_logic(self):
        QMessageBox.information(self, "Обновления", "обновления")

    def thremes_logic(self, theme_name=None):
        if not hasattr(self, "dark_mode"):
            self.dark_mode = False
        if not hasattr(self, "current_theme"):
            self.current_theme = "Светлая"

        if theme_name is None:
            themes = list(self.get_theme_palettes().keys())
            current_index = themes.index(self.current_theme) if self.current_theme in themes else 0
            next_index = (current_index + 1) % len(themes)
            theme_name = themes[next_index]

        palette = QPalette()
        theme_palettes = self.get_theme_palettes()
        
        if theme_name in theme_palettes:
            theme_data = theme_palettes[theme_name]
            for color_role, color_value in theme_data.items():
                palette.setColor(color_role, QColor(*color_value))
            
            self.current_theme = theme_name
            self.dark_mode = (theme_name == "Темная")

        self.setPalette(palette)
        QApplication.instance().setPalette(palette)

    def get_theme_palettes(self):
        return {
            "Светлая": {
                QPalette.ColorRole.Window: (240, 240, 240),
                QPalette.ColorRole.WindowText: (0, 0, 0),
                QPalette.ColorRole.Base: (255, 255, 255),
                QPalette.ColorRole.Button: (220, 220, 220),
                QPalette.ColorRole.ButtonText: (0, 0, 0),
                QPalette.ColorRole.Text: (0, 0, 0),
                QPalette.ColorRole.Highlight: (100, 149, 237), 
                QPalette.ColorRole.HighlightedText: (255, 255, 255)
            },
            "Темная": {
                QPalette.ColorRole.Window: (45, 45, 45),
                QPalette.ColorRole.WindowText: (255, 255, 255),
                QPalette.ColorRole.Base: (45, 45, 45),
                QPalette.ColorRole.Button: (60, 60, 60),
                QPalette.ColorRole.ButtonText: (255, 255, 255),
                QPalette.ColorRole.Text: (255, 255, 255),
                QPalette.ColorRole.Highlight: (100, 149, 237),
                QPalette.ColorRole.HighlightedText: (255, 255, 255)
            },
            "Синяя": {
                QPalette.ColorRole.Window: (240, 245, 255),
                QPalette.ColorRole.WindowText: (0, 0, 80),
                QPalette.ColorRole.Base: (255, 255, 255),
                QPalette.ColorRole.Button: (200, 220, 255),
                QPalette.ColorRole.ButtonText: (0, 0, 80),
                QPalette.ColorRole.Text: (0, 0, 80),
                QPalette.ColorRole.Highlight: (70, 130, 180),  # SteelBlue
                QPalette.ColorRole.HighlightedText: (255, 255, 255)
            }
        }
    
    def activate_logic(self):
        QMessageBox.information(self, "Активация", "тут будет ключ активации")

    def visual_display_camer_logic(self):
        QMessageBox.information(self, "Визуальное отображение камер", "Визуальное отображение камер")

    def user_logic(self):
        QMessageBox.information(self, "Пользователь", "Пользователь")


    def limit_fps_text(self):
        return f"Ограниечение по кадрам:{'Вкл' if getattr(self, 'limit_fps_toggle', False) else 'Выкл'}"

    def limit_fps_toggle(self):
        self.limit_fps_toggle = not getattr(self, 'limit_fps_toggle', False)
        self.limit_fps_action.setText(self.limit_fps_text())

    def fps_max_text(self):
        fps_options = ["15", "30", "60"]
        if self.click_info == 0:
            fps = ""
        else:
            fps = fps_options[self.click_info - 1]
        return f"Частота обновления кадров: {fps}"
    
    def fps_max_logic(self):
        self.click_info = (self.click_info + 1) % 4
        self.fps_max_action.setText(self.fps_max_text())


    def model_track_text(self):
        model_track = ["small", "medium", "hight"]
        if self.click_info1 == 0:
            model = ""
        else:
            model = model_track[self.click_info1 - 1]
        return f'Модель трекинга {model}'
    def model_track_logic(self):
        self.click_info1 = (self.click_info1 + 1) % 4
        self.model_track_action.setText(self.model_track_text())


    def work_track_text(self):
        return f"Ограниечение по кадрам:{'Вкл' if getattr(self, 'work_track_logic', False) else 'Выкл'}"

    def work_track_logic(self):
        self.work_track_logic = not getattr(self, 'work_track_logic', False)
        self.work_track_action.setText(self.work_track_text())

    def path_for_train_model_logic(self):
        current_path = self.file_paths.get("train_data", "")
        
        path = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку с данными для обучения модели",
            current_path
        )
        
        if path:
            self.file_paths["train_data"] = path
            
            self.save_all_paths()
            
            QMessageBox.information(self, "Путь сохранён", f"Путь к данным:\n{path}")
        else:
            QMessageBox.warning(self, "Отмена", "Вы не выбрали папку.")
    
    def model_train_path_logic(self):
        current_path = self.file_paths.get("model", "")
        
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Выберите модель которую будете обучать",
            os.path.dirname(current_path) if current_path else "",
            "PyTorch Model Files (*.pt);;All Files (*)"
        )
        
        if path:
            self.file_paths["model"] = path
            self.save_all_paths()
            QMessageBox.information(self, "Модель сохранена", f"Путь к модели:\n{path}")
        else:
            QMessageBox.warning(self, "Отмена", "Вы не выбрали модель.")


    def save_all_paths(self):
        try:
            with open("file_paths.json", "w", encoding="utf-8") as f:
                json.dump(self.file_paths, f, ensure_ascii=False, indent=2)
            
            if hasattr(self, 'openSettingsWindow') and self.openSettingsWindow:
                self.openSettingsWindow.file_paths = self.file_paths.copy()
                
        except Exception as e:
            print(f"❌ Ошибка сохранения путей: {e}")



    def information_camer_logic(self):
        QMessageBox.information(self, "Информация о камере", "123")


    def file_location_logic(self):
        QMessageBox.information(self, "Расположение файлов", "Тут открывается расположение файлов")


    def show_about(self):
        QMessageBox.information(self, "О программе", "П")


    def load_initial_paths(self):
        try:
            if os.path.exists("file_paths.json"):
                with open("file_paths.json", "r", encoding="utf-8") as f:
                    self.file_paths = json.load(f)
            else:
                if getattr(sys, "frozen", False):
                    base_dir = os.path.dirname(sys.executable)
                else:
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                
                self.file_paths = {
                    "config": os.path.join(base_dir, "data", "config"),
                    "logs": os.path.join(base_dir, "data", "logs"),
                    "temp": os.path.join(base_dir, "data", "temp"),
                    "model": os.path.join(base_dir, "data", "model", "base.pt"),
                    "train_data": os.path.join(base_dir, "data", "train_data"),
                }
                
            if os.path.exists("train_data_path.txt"):
                with open("train_data_path.txt", "r", encoding="utf-8") as f:
                    self.file_paths["train_data"] = f.read().strip()
                    
            if os.path.exists("model_path.txt"):
                with open("model_path.txt", "r", encoding="utf-8") as f:
                    self.file_paths["train_model"] = f.read().strip()
                    
        except Exception as e:
            print(f"❌ Ошибка загрузки путей при запуске: {e}")



    def start_model_training(self):
        try:
            if not TRAIN_MODULE_AVAILABLE:
                QMessageBox.warning(self, "Ошибка", 
                                "Модуль обучения не доступен!\n"
                                "Убедитесь что файл trainmodel.py находится в той же папке.")
                return
            
            train_data_path = self.file_paths.get("train_data")
            model_path = self.file_paths.get("train_model")
            
            if not train_data_path or not model_path:
                QMessageBox.warning(self, "Ошибка", "Не указаны пути для обучения!\nПроверьте настройки путей.")
                return
            
            if not os.path.exists(train_data_path):
                QMessageBox.warning(self, "Ошибка", f"Папка с данными не найдена:\n{train_data_path}")
                return
                
            if not os.path.exists(model_path):
                QMessageBox.warning(self, "Ошибка", f"Файл модели не найден:\n{model_path}")
                return
            
            training_params = {
                'epochs': 20,
                'imgsz': 640,
                'batch': 4,
                'device': 'cuda',
                'name': 'my_training',
                'fliplr': 0.55,
                'hsv_h': 0.020,
                'hsv_s': 0.75,
                'hsv_v': 0.45,
                'scale': 0.55,
                'translate': 0.15,
                'auto_augment': 'randaugment',
                'erasing': 0.35
            }
            
            QMessageBox.information(self, "Обучение", "Обучение модели запущено...")
            
            results = train_from_settings(
                train_data_path=train_data_path,
                model_path=model_path,
                **training_params 
            )
            
            QMessageBox.information(self, "Успех", "Обучение завершено!")
            
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Ошибка при обучении:\n{str(e)}")



class ConnectWindowCamer(QWidget):
    def __init__(self, palette=None, dark_mode=False):
        super().__init__()
        self.setWindowTitle("Окно подключения")
        self.setGeometry(600, 600, 300, 150)
        self.setMinimumSize(300,150)
        self.setMaximumSize(300,150)
        self.dark_mode = dark_mode

        self.urlManager = CameraURLMeneger()
        if palette:
            self.setPalette(palette)

        self.select_connect_camer = QPushButton(" Способ подключения камеры ", self)
        menu = QMenu(self)
        menu.addAction("URL", self.URL)
        menu.addAction("IP", self.IP)
        menu.addAction("COM", self.COM)


        self.select_connect_camer.setMenu(menu)
        self.select_connect_camer.setFixedHeight(30)
        self.select_connect_camer.setFixedWidth(250)
        self.select_connect_camer.move(25, 30)

        self.path_input = QLineEdit(self)
        self.path_input.setFixedWidth(250)
        self.path_input.move(25, 70)
        

        self.connect_buttom = QPushButton("Подключить",self)
        self.connect_buttom.setFixedHeight(30)
        self.connect_buttom.setFixedWidth(100)
        self.connect_buttom.move(100, 100)
        self.connect_buttom.clicked.connect(self.saveConnectionDate)


        self.connectionType = None

    def setConnectType(self, conn_type: str):
        self.connectType = conn_type
        self.select_connect_camer.setText(conn_type)

    def URL(self):
        self.setConnectType("URL")
        saved_url = self.urlManager.get_url("default")
        if saved_url:
            self.path_input.setText(saved_url)

    def IP(self):
        self.setConnectType("IP")
        self.path_input.clear()

    def COM(self):
        self.setConnectType("COM")
        self.path_input.clear()

    def saveConnectionDate(self):
        if not self.connectionType:
            QMessageBox.warning(self, "Выберите тип подключения!")
            return

        value = self.path_input.text().strip()
        if not value:
            QMessageBox.warning(self, "Ведите значения для подючения")
            return

        if self.connection_type == "URL":
            self.urlManager.save_url("default", value)
            QMessageBox.information(self, "Сохранено", f"URL '{value}' сохранён!")



class settingsWindow(QWidget):
    def __init__(self, parent=None, palette=None, dark_mode=False):
        super().__init__()
        self.parent_window = parent
        self.setWindowTitle("Настройки")
        self.setGeometry(300, 300, 1200, 600)
        self.setMinimumSize(1200, 600)
        self.dark_mode = dark_mode
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        self.create_left_panel(main_layout)
        
        self.create_right_panel(main_layout)
        
        self.show_section("activation")
    
    def create_left_panel(self, main_layout):
        left_container = QWidget()
        left_container.setFixedWidth(230)
        left_container.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border-right: 1px solid #dee2e6;
            }
        """)
        
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 20, 0, 20)
        left_layout.setSpacing(0)
        nav_buttons = [
            ("Активация", "activation"),
            ("Расположение файлов", "filepath"),
            ("Тема", "theme"),
            ("Обновления", "updates"),
            ("О приложении", "about"),
            ("Визуальное отображение камер", "camera_display"),
            ("Пользователь", "user")
        ]
        
        self.nav_buttons = {}
        
        for text, section_id in nav_buttons:
            btn = QPushButton(text)
            btn.setFixedHeight(30)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    border: none;
                    padding: 8px 16px;
                    text-align: left;
                    font-size: 12px;
                    color: #495057;
                }
                QPushButton:hover {
                    background-color: #e9ecef;
                }
                QPushButton:checked {
                    background-color: #007bff;
                    color: white;
                    font-weight: bold;
                }
            """)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, s=section_id: self.show_section(s))
            
            self.nav_buttons[section_id] = btn
            left_layout.addWidget(btn)
        
        left_layout.addStretch()
        main_layout.addWidget(left_container)
    
    def create_right_panel(self, main_layout):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    
        self.right_container = QWidget()
        self.right_layout = QVBoxLayout(self.right_container)
        self.right_layout.setContentsMargins(30, 30, 30, 30)
        self.right_layout.setSpacing(20)
        
        scroll_area.setWidget(self.right_container)
        main_layout.addWidget(scroll_area)
        
        self.create_activation_section()
        self.create_filepath_section()
        self.create_theme_section()
        self.create_updates_section()
        self.create_about_section()
        self.create_camera_display_section()
        self.create_user_section()
    
    def show_section(self, section_id):
        for btn in self.nav_buttons.values():
            btn.setChecked(False)

        self.nav_buttons[section_id].setChecked(True)
        
        for i in reversed(range(self.right_layout.count())):
            widget = self.right_layout.itemAt(i).widget()
            if widget:
                widget.setVisible(False)
        
        section_widget = getattr(self, f"{section_id}_section", None)
        if section_widget:
            section_widget.setVisible(True)
    
    def create_section_header(self, title):
        header = QLabel(title)
        header.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #212529;
                margin-bottom: 10px;
            }
        """)
        return header
    
    def create_setting_row(self, label, control):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        
        label_widget = QLabel(label)
        label_widget.setStyleSheet("font-size: 12px; color: #495057;")
        label_widget.setFixedWidth(300)
        
        row.addWidget(label_widget)
        row.addWidget(control)
        row.addStretch()
        
        return row
    
    def create_activation_section(self):
        self.activation_section = QWidget()
        layout = QVBoxLayout(self.activation_section)
        
        layout.addWidget(self.create_section_header("Активация"))
        
        activation_status = QLabel("Статус: Не активировано")
        activation_status.setStyleSheet("color: #dc3545; font-size: 12px;")
        layout.addWidget(activation_status)
        
        activate_btn = QPushButton("Активировать")
        activate_btn.setFixedSize(120, 30)
        layout.addWidget(activate_btn)
        
        layout.addStretch()
        self.right_layout.addWidget(self.activation_section)
        self.activation_section.setVisible(False)
    
    def create_filepath_section(self):
        self.filepath_section = QWidget()
        layout = QVBoxLayout(self.filepath_section)
        
        layout.addWidget(self.create_section_header("Расположение файлов"))
        

        self.file_paths = {
            "config": "",
            "logs": "",
            "temp": ""
        }
        self.load_file_paths()

        paths = [
            ("Путь к файлам конфигурации", "config", "Выберите папку для конфигураций"),
            ("Путь к логам", "logs", "Выберите папку для логов"),
            ("Путь к временным файлам", "temp", "Выберите папку для временных файлов")
        ]

        self.path_labels = {}

        for label_text, path_key, dialog_title in paths:
            row_layout = QHBoxLayout()
            row_layout.setContentsMargins(0, 0, 0, 0)
            label = QLabel(label_text)
            label.setFixedWidth(250)
            label.setStyleSheet('font-size: 12px;')
            path_label = QLabel(self.file_paths.get(path_key, "Не выбрано"))
            path_label.setStyleSheet("""
                Qlabel{
                    color: #6c757d;
                    font-size: 12px;
                    background-color: #f8f9fa;
                    border: 1px solid #dee2e6;
                    padding: 4px 8px;
                    border-radius: 3px;
                    min-width: 300px;             
                }
            """)
            path_label.setWordWrap(True)
            change_button = QPushButton("Изменить")
            change_button.setFixedSize(80, 25)
            change_button.clicked.connect(lambda checked, key=path_key, title=dialog_title: self.change_file_path(key, title))


            reset_button = QPushButton("Сбросить")
            reset_button.setFixedSize(60, 25)
            reset_button.clicked.connect(lambda checded, key=path_key: self.reset_file_path(key))

            row_layout.addWidget(label)
            row_layout.addWidget(path_label, 1)
            row_layout.addWidget(change_button)
            row_layout.addWidget(reset_button)

            layout.addLayout(row_layout)

            self.path_labels[path_key] = path_label

        apply_button = QPushButton("Применить")
        apply_button.setFixedSize(150, 30)
        apply_button.clicked.connect(self.apply_all_paths)
        layout.addWidget(apply_button)

        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.right_layout.addWidget(self.filepath_section)
        self.filepath_section.setVisible(False)

    def load_file_paths(self):
        try:
            try:
                with open("file_paths.json", "r", encoding="utf-8") as f:
                    saved_path = json.load(f)
                    self.file_paths.update(saved_path)
            except FileNotFoundError:
                if getattr(sys, "frozen", False):
                    base_dir = os.path.dirname(sys.executable)
                else:
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                default_path = {
                    "config": os.path.join(base_dir, "data", "config"),
                    "logs": os.path.join(base_dir, "data", "logs"),
                    "temp": os.path.join(base_dir, "data", "temp"),
                    "model": os.path.join(base_dir, "data", "model", "base.pt"),
                    "train_data": os.path.join(base_dir, "data", "train_data")
                }
                self.file_paths.update(default_path)
        except Exception as e:
            print(f"Ошибка загрузки путей{e}")

    def save_file_paths(self):
        try:
            with open("file_paths.json", "w", encoding="utf-8") as f:
                json.dump(self.file_paths, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения путей{e}")

    def change_file_path(self, path_key, dialog_title):
        current_path = self.file_paths.get(path_key, "")
        new_path = QFileDialog.getExistingDirectory(
            self,
            dialog_title,
            current_path if current_path and current_path != "не выбрано" else ""
        )
        if new_path:
            self.file_paths[path_key] = new_path
            self.path_labels[path_key].setText(new_path)
            self.save_file_paths()
            QMessageBox.information(self, "Путь изменен", f"Успешно изменен на {new_path}")

    def reset_file_path(self, path_key):
        if getattr(sys, "frozen", False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        default_paths = {
            "config": os.path.join(base_dir, "data", "config"),
            "logs": os.path.join(base_dir, "data", "logs"),
            "temp": os.path.join(base_dir, "data", "temp"),
            "model": os.path.join(base_dir, "data", "model", "base.pt"),
            "train_data": os.path.join(base_dir, "data", "train_data")
        }
        default_paths = default_paths.get(path_key, "")
        self.file_paths[path_key] = default_paths
        self.path_labels[path_key].setText(default_paths)
        self.save_file_paths()
        QMessageBox.information(self, "Путь сброшен", f"Успешно изменен на{default_paths}")

    def apply_all_paths(self):
        try:
            for path_key, path_value in self.file_paths.items():
                if path_value and path_value != "не выбрано":
                    os.makedirs(path_value, exist_ok=True)
            self.save_file_paths()

            QMessageBox.information(
                self,
                "Пути Применены",
                "Все пути успешно применены и папки созданы"
            )
            if self.parent_window:
                self.parent_window.file_path = self.file_paths.copy()
        except Exception as e:
            QMessageBox.warning(
                self,
                "Ошибка",
                f"Не удалось применить пути {str(e)}"
            )
        
    def get_file_path(self, path_key):
        return self.file_paths.get(path_key, "")

    
    def create_theme_section(self):
        self.theme_section = QWidget()
        layout = QVBoxLayout(self.theme_section)
        
        layout.addWidget(self.create_section_header("Тема"))
        
        self.theme_combo = QComboBox()
        
        if self.parent_window and hasattr(self.parent_window, 'get_theme_palettes'):
            themes = list(self.parent_window.get_theme_palettes().keys())
        else:
            themes = ["Светлая", "Темная", "Синяя", "Зеленая", "Темно-синяя", "Контрастная", "Серая"]
        
        self.theme_combo.addItems(themes)
        
        if self.parent_window and hasattr(self.parent_window, "current_theme"):
            current_theme = getattr(self.parent_window, "current_theme", "Светлая")
            if current_theme in themes:
                self.theme_combo.setCurrentText(current_theme)
        else:
            if self.parent_window and getattr(self.parent_window, "dark_mode", False):
                self.theme_combo.setCurrentText("Темная")
            else:
                self.theme_combo.setCurrentText("Светлая")
        
        self.theme_combo.currentTextChanged.connect(self.apply_theme)
        layout.addLayout(self.create_setting_row("Цветовая схема:", self.theme_combo))
        
        preview_label = QLabel("Предпросмотр:")
        preview_label.setStyleSheet("font-size: 12px; color: #495057;")
        layout.addWidget(preview_label)
        
        preview_widget = QWidget()
        preview_widget.setFixedHeight(60)
        preview_widget.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
            }
        """)
        layout.addWidget(preview_widget)
        
        apply_btn = QPushButton("Применить тему")
        apply_btn.setFixedSize(120, 30)
        apply_btn.clicked.connect(lambda: self.apply_theme(self.theme_combo.currentText()))
        layout.addWidget(apply_btn)
        
        layout.addStretch()
        self.right_layout.addWidget(self.theme_section)
        self.theme_section.setVisible(False)

    def apply_theme(self, theme_name):
        if not self.parent_window or not hasattr(self.parent_window, 'thremes_logic'):
            return
        
        if hasattr(self.parent_window, 'thremes_logic'):

            import inspect
            sig = inspect.signature(self.parent_window.thremes_logic)
            if 'theme_name' in sig.parameters:
                self.parent_window.thremes_logic(theme_name=theme_name)
            else:

                themes = ["Светлая", "Темная"]
                if theme_name in themes:
                    current_dark = getattr(self.parent_window, "dark_mode", False)
                    if theme_name == "Темная" and not current_dark:
                        self.parent_window.thremes_logic()
                    elif theme_name == "Светлая" and current_dark:
                        self.parent_window.thremes_logic()
    
    def get_current_theme(self):
        return getattr(self, "current_theme", "Светлая")


    def create_updates_section(self):
        self.updates_section = QWidget()
        layout = QVBoxLayout(self.updates_section)
        
        layout.addWidget(self.create_section_header("Обновления"))
        
        auto_update = QCheckBox("Автоматически проверять обновления")
        auto_update.setChecked(True)
        layout.addWidget(auto_update)
        
        beta_updates = QCheckBox("Разрешить бета-версии")
        layout.addWidget(beta_updates)
        
        check_now_btn = QPushButton("Проверить обновления")
        check_now_btn.setFixedSize(150, 30)
        layout.addWidget(check_now_btn)
        
        layout.addStretch()
        self.right_layout.addWidget(self.updates_section)
        self.updates_section.setVisible(False)
    
    def create_about_section(self):
        self.about_section = QWidget()
        layout = QVBoxLayout(self.about_section)
        
        layout.addWidget(self.create_section_header("О приложении"))
        
        info_text = """
        <p><b>Версия:</b> 1.0.0</p>
        <p><b>Сборка:</b> 2024.01.001</p>
        <p><b>Разработчик:</b> Ваша компания</p>
        <p><b>Лицензия:</b> Проприетарная</p>
        <p><b>Веб-сайт:</b> <a href="https://example.com">https://example.com</a></p>
        """
        
        info_label = QLabel(info_text)
        info_label.setStyleSheet("font-size: 12px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        layout.addStretch()
        self.right_layout.addWidget(self.about_section)
        self.about_section.setVisible(False)
    
    def create_camera_display_section(self):
        self.camera_display_section = QWidget()
        layout = QVBoxLayout(self.camera_display_section)
        
        layout.addWidget(self.create_section_header("Визуальное отображение камер"))
        
        show_grid = QCheckBox("Показывать сетку")
        show_grid.setChecked(True)
        layout.addWidget(show_grid)
        
        show_names = QCheckBox("Показывать имена камер")
        layout.addWidget(show_names)
        
        layout.addStretch()
        self.right_layout.addWidget(self.camera_display_section)
        self.camera_display_section.setVisible(False)
    
    def create_user_section(self):
        self.user_section = QWidget()
        layout = QVBoxLayout(self.user_section)
        
        layout.addWidget(self.create_section_header("Пользователь"))
        
        user_info = """
        <p><b>Имя пользователя:</b> user123</p>
        <p><b>Роль:</b> Администратор</p>
        <p><b>Последний вход:</b> 2024-01-15 14:30</p>
        """
        
        user_label = QLabel(user_info)
        user_label.setStyleSheet("font-size: 12px;")
        layout.addWidget(user_label)
        
        logout_btn = QPushButton("Выйти")
        logout_btn.setFixedSize(100, 30)
        layout.addWidget(logout_btn)
        
        layout.addStretch()
        self.right_layout.addWidget(self.user_section)
        self.user_section.setVisible(False)
        






app = QApplication(sys.argv)
window = MyWindow()
window.show()
sys.exit(app.exec())