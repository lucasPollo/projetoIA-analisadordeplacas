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

## API REST

Além da interface Web, o projeto também possui uma API REST para permitir que outros sistemas enviem uma imagem e recebam o resultado da análise em formato JSON.

### Endpoint

```text
POST /api/analyze
```

Com o Flask rodando, o endereço completo é:

```text
http://127.0.0.1:5000/api/analyze
```

### Como executar a API

Primeiro, inicie normalmente o projeto:

```powershell
python app.py
```

Depois, em outro terminal, envie uma imagem para a API.

Exemplo com `curl`:

```powershell
curl.exe -X POST -F "image=@test_images\mercosul.jpg" http://127.0.0.1:5000/api/analyze
```

Troque:

```text
test_images\mercosul.jpg
```

pelo caminho da imagem que deseja analisar.

### Exemplo de resposta

```json
{
  "latencia_ms": 63997.34,
  "placas": [
    {
      "placa": "ITA1354",
      "tipo": "Padrão brasileiro antigo",
      "cor_veiculo": "Laranja",
      "confianca_yolo": 0.9074,
      "confianca_ocr": 0.7658,
      "confianca_veiculo": 0.3842,
      "confianca_cor": 0.837,
      "tipo_veiculo": "Carro",
      "decisao": "APROVADO AUTOMATICAMENTE"
    }
  ]
}
```

A API utiliza o mesmo pipeline da interface Web:

```text
Imagem
  ↓
YOLO placas
  ↓
OCR
  ↓
YOLO veículo
  ↓
Estimativa de cor
  ↓
Validação
  ↓
Decisão
  ↓
Resposta JSON
```