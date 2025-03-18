# Machin_eye

# Бибилиотеки
```python
- pip install opencv-python
- pip install ultralytics
- pip install imageio[ffmpeg]
```

# как установить проект
```
git clone https://github.com/Yasomy/Machin_eye.git
```
# После того как скачали zip надо установить папку myenv 
```bash
cd C:\Users\student\Mathineeye
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
- после нужно с библеотеками повозиться
```python
python -c "import torch; print(torch.cuda.is_available())"
```
- смотрит работает ли у нас гпу false значит нет
тогда сначала проверяем есть ли вообще драва у нас на куда
```python
nvidia-smi
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
3000/4000 серия GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"
```
- после если есть то удаляем старые библеотеки торча
- после устанавливаем новые библеотеки под куду
- после проверяем опять
- если да, то все работает! победка!
