# Machin_eye

// Бибилиотеки //
pip install opencv-python
pip install ultralytics
pip install imageio[ffmpeg]
pip install opencv-python



После того как скачали zip надо установить папку myenv - cd C:\Users\student\Mathineeye ( аля сам путь к директории указать) и после python -m venv myenv( сама установка этой папки)
после пробуем myenv\Scripts\activate если сработало - качаем библеотеки в проект, нет то - 
пробуем в powershell Set-ExecutionPolicy Unrestricted -Scope Process тут Y
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser тут Y
myenv\Scripts\activate
после этого должно заработать



Как сделать так чтобы гпу заработало???
сначала устанавливаем драйвера Куда
после нужно с библеотеками повозиться
python -c "import torch; print(torch.cuda.is_available())" - смотрит работает ли у нас гпу -false значит нет
тогда сначала проверяем есть ли вообще драва у нас на куда nvidia-smi
после если есть то удаляем старые библеотеки торча
pip uninstall torch torchvision torchaudio
после устанавливаем новые библеотеки под куду
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
а если 3000 или 4000 серии
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
после проверяем опять
python -c "import torch; print(torch.cuda.is_available())"
если да, то все работает! победка!
