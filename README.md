# Система трекинга людей и машин
# Описание проекта
- Проект направлен на создание интеллектуальной системы с использованием искусственного интеллекта для распознавания и трекинга объектов в видеопотоке. 
# Как работает проект
- подлючается к камере или видео и с помощью ИИ сначало происходит детекция обьектов: Людей и машин, после уже происходит сам трекинг людей
# как установить проект
```
git clone https://github.com/Yasomy/Machin_eye.git
```
```
python -m venv myenv
```
- после пробуем
```bash
myenv\Scripts\activate
```
- если сработало - качаем библеотеки в проект, нет то пробуем в powershell
```powershell
Set-ExecutionPolicy Unrestricted -Scope Process
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
myenv\Scripts\activate
```
- после этого должно заработать
# установки библеотек
- важно чтобы библеотеки устанавливались в виртуальное окружение
```
pip install -r requiremets.txt
```
# Как сделать так чтобы гпу заработало???
- сначала устанавливаем драйвера CUDA
```python
python -c "import torch; print(torch.cuda.is_available())"
```
- смотрим работает ли у нас GPU false значит нет
тогда сначала проверяем есть ли вообще драва у нас на CUDA
```python
nvidia-smi
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
3000/4000 серия GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"
```


