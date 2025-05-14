# Machin_eye

# Бибилиотеки
```python
pip install opencv-python
pip install ultralytics
pip install imageio[ffmpeg]
```

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
# Как сделать так чтобы гпу заработало???
- сначала устанавливаем драйвера Куда
```python
python -c "import torch; print(torch.cuda.is_available())"
```
- смотрим работает ли у нас гпу false значит нет
тогда сначала проверяем есть ли вообще драва у нас на куда
```python
nvidia-smi
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
3000/4000 серия GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"
```


