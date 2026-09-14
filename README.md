# Tesla Detector de Placas

Projeto acadêmico de visão computacional para detecção e reconhecimento de placas veiculares.

O sistema utiliza YOLO, EasyOCR, OpenCV, NumPy e Flask para detectar a placa, reconhecer os caracteres, identificar o padrão da placa, estimar a cor do veículo e encaminhar casos de baixa confiança para revisão humana.

## Como instalar

Crie o ambiente virtual:

```powershell
python -m venv .venv
```

Ative a venv:

```powershell
.venv\Scripts\activate
```

Instale as dependências:

```powershell
pip install -r requirements.txt
```

## Como gerar o requirements.txt

Com a venv ativada:

```powershell
pip freeze > requirements.txt
```

As bibliotecas listadas no `requirements.txt` podem apresentar diferenças de comportamento entre versões. 

## Como rodar o projeto

Execute:

```powershell
python app.py
```

Depois acesse no navegador:

```text
http://127.0.0.1:5000
```

## Modelo treinado

O modelo utilizado para detecção de placas deve estar em:

```text
models/best.pt
```

Na primeira execução, o sistema também pode baixar automaticamente o modelo:

```text
yolo11n-seg.pt
```
