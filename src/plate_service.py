from pathlib import Path
import re
import time
import uuid

import cv2
import easyocr
import numpy as np
from ultralytics import YOLO


class PlateRecognitionService:

    VEHICLE_CLASSES = {
        2: "Carro",
        3: "Motocicleta",
        5: "Ônibus",
        7: "Caminhão",
    }

    def __init__(
        self,
        model_path: str,
        results_dir: str,
        candidate_threshold: float = 0.45,
        detection_threshold: float = 0.60,
        ocr_threshold: float = 0.70,
        classification_threshold: float = 0.50,
        ocr_candidate_threshold: float = 0.35,
    ):
        print("Carregando modelo de placas...")

        self.model = YOLO(model_path)

        print("Carregando modelo de veículos...")

        # Modelo pré-treinado para segmentar veículos.
        # Não exige novo treinamento.
        self.vehicle_model = YOLO(
            "yolo11n-seg.pt"
        )

        print("Carregando EasyOCR...")

        self.reader = easyocr.Reader(
            ["en"],
            gpu=False
        )

        self.results_dir = Path(
            results_dir
        )

        self.results_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.candidate_threshold = (
            candidate_threshold
        )

        self.detection_threshold = (
            detection_threshold
        )

        self.ocr_threshold = (
            ocr_threshold
        )

        self.classification_threshold = (
            classification_threshold
        )

        self.ocr_candidate_threshold = (
            ocr_candidate_threshold
        )

        print("Sistema de IA carregado!")

    # ========================================================
    # LIMPEZA DE TEXTO
    # ========================================================

    @staticmethod
    def limpar_texto(
        texto: str
    ) -> str:

        if not texto:
            return ""

        return re.sub(
            r"[^A-Z0-9]",
            "",
            texto.upper()
        )

    # ========================================================
    # REMOVER BR / BRASIL DAS PLACAS MERCOSUL
    # ========================================================

    @staticmethod
    def remover_textos_auxiliares(
        texto: str
    ) -> str:

        texto = texto.upper()

        while len(texto) > 7:

            if texto.startswith(
                "BRASIL"
            ):
                texto = texto[6:]
                continue

            if texto.startswith(
                "BR"
            ):
                texto = texto[2:]
                continue

            break

        while len(texto) > 7:

            if texto.endswith(
                "BRASIL"
            ):
                texto = texto[:-6]
                continue

            if texto.endswith(
                "BR"
            ):
                texto = texto[:-2]
                continue

            break

        return texto

    # ========================================================
    # IDENTIFICAÇÃO DO TIPO DE PLACA
    # ========================================================

    @staticmethod
    def identificar_tipo_placa(
        placa: str
    ) -> str:

        mercosul = (
            r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$"
        )

        antiga = (
            r"^[A-Z]{3}[0-9]{4}$"
        )

        if re.fullmatch(
            mercosul,
            placa
        ):
            return "Mercosul"

        if re.fullmatch(
            antiga,
            placa
        ):
            return (
                "Padrão brasileiro antigo"
            )

        return "Formato não reconhecido"

    # ========================================================
    # CORREÇÕES PARA MERCOSUL
    # ========================================================

    @staticmethod
    def corrigir_mercosul(
        texto: str
    ) -> str:

        if len(texto) != 7:
            return texto

        caracteres = list(
            texto
        )

        posicoes_letras = [
            0,
            1,
            2,
            4
        ]

        posicoes_numeros = [
            3,
            5,
            6
        ]

        numero_para_letra = {
            "0": "O",
            "1": "I",
            "2": "Z",
            "5": "S",
            "6": "G",
            "8": "B",
        }

        letra_para_numero = {
            "O": "0",
            "Q": "0",
            "D": "0",
            "I": "1",
            "L": "1",
            "Z": "2",
            "S": "5",
            "G": "6",
            "B": "8",
        }

        for posicao in posicoes_letras:

            caractere = (
                caracteres[
                    posicao
                ]
            )

            if (
                caractere
                in numero_para_letra
            ):
                caracteres[
                    posicao
                ] = numero_para_letra[
                    caractere
                ]

        for posicao in posicoes_numeros:

            caractere = (
                caracteres[
                    posicao
                ]
            )

            if (
                caractere
                in letra_para_numero
            ):
                caracteres[
                    posicao
                ] = letra_para_numero[
                    caractere
                ]

        return "".join(
            caracteres
        )

    # ========================================================
    # CORREÇÕES PARA PLACA ANTIGA
    # ========================================================

    @staticmethod
    def corrigir_antiga(
        texto: str
    ) -> str:

        if len(texto) != 7:
            return texto

        caracteres = list(
            texto
        )

        posicoes_letras = [
            0,
            1,
            2
        ]

        posicoes_numeros = [
            3,
            4,
            5,
            6
        ]

        numero_para_letra = {
            "0": "O",
            "1": "I",
            "2": "Z",
            "5": "S",
            "6": "G",
            "8": "B",
        }

        letra_para_numero = {
            "O": "0",
            "Q": "0",
            "D": "0",
            "I": "1",
            "L": "1",
            "Z": "2",
            "S": "5",
            "G": "6",
            "B": "8",
        }

        for posicao in posicoes_letras:

            caractere = (
                caracteres[
                    posicao
                ]
            )

            if (
                caractere
                in numero_para_letra
            ):
                caracteres[
                    posicao
                ] = numero_para_letra[
                    caractere
                ]

        for posicao in posicoes_numeros:

            caractere = (
                caracteres[
                    posicao
                ]
            )

            if (
                caractere
                in letra_para_numero
            ):
                caracteres[
                    posicao
                ] = letra_para_numero[
                    caractere
                ]

        return "".join(
            caracteres
        )

    # ========================================================
    # QUANTIDADE DE CORREÇÕES
    # ========================================================

    @staticmethod
    def contar_alteracoes(
        original: str,
        corrigido: str
    ) -> int:

        if (
            len(original)
            != len(corrigido)
        ):
            return 99

        return sum(
            1
            for a, b in zip(
                original,
                corrigido
            )
            if a != b
        )

    # ========================================================
    # AVALIAR CANDIDATO
    # ========================================================

    def avaliar_candidato(
        self,
        texto: str,
        confianca: float
    ):

        texto = self.limpar_texto(
            texto
        )

        texto = (
            self.remover_textos_auxiliares(
                texto
            )
        )

        if len(texto) != 7:
            return None

        if (
            confianca
            < self.ocr_candidate_threshold
        ):
            return None

        tipo_original = (
            self.identificar_tipo_placa(
                texto
            )
        )

        if (
            tipo_original
            != "Formato não reconhecido"
        ):

            return {
                "placa": texto,
                "tipo": tipo_original,
                "confianca":
                    confianca,
                "alteracoes": 0,
                "score":
                    confianca + 0.20,
            }

        mercosul = (
            self.corrigir_mercosul(
                texto
            )
        )

        tipo_mercosul = (
            self.identificar_tipo_placa(
                mercosul
            )
        )

        antiga = (
            self.corrigir_antiga(
                texto
            )
        )

        tipo_antiga = (
            self.identificar_tipo_placa(
                antiga
            )
        )

        possibilidades = []

        if (
            tipo_mercosul
            == "Mercosul"
        ):

            alteracoes = (
                self.contar_alteracoes(
                    texto,
                    mercosul
                )
            )

            possibilidades.append({
                "placa":
                    mercosul,
                "tipo":
                    "Mercosul",
                "confianca":
                    confianca,
                "alteracoes":
                    alteracoes,
                "score":
                    confianca
                    - alteracoes * 0.05,
            })

        if (
            tipo_antiga
            == "Padrão brasileiro antigo"
        ):

            alteracoes = (
                self.contar_alteracoes(
                    texto,
                    antiga
                )
            )

            possibilidades.append({
                "placa":
                    antiga,
                "tipo":
                    "Padrão brasileiro antigo",
                "confianca":
                    confianca,
                "alteracoes":
                    alteracoes,
                "score":
                    confianca
                    - alteracoes * 0.05,
            })

        if not possibilidades:
            return None

        return max(
            possibilidades,
            key=lambda item:
                item["score"]
        )

    # ========================================================
    # EXTRAIR PLACA DO OCR
    # ========================================================

    def extrair_placa(
        self,
        resultados_ocr
    ):

        candidatos = []

        for (
            _,
            texto,
            confianca
        ) in resultados_ocr:

            confianca = float(
                confianca
            )

            texto = (
                self.limpar_texto(
                    texto
                )
            )

            if not texto:
                continue

            texto = (
                self.remover_textos_auxiliares(
                    texto
                )
            )

            if texto in {
                "BR",
                "BRA",
                "BRASIL",
            }:
                continue

            if len(texto) == 7:

                resultado = (
                    self.avaliar_candidato(
                        texto,
                        confianca
                    )
                )

                if resultado:
                    candidatos.append(
                        resultado
                    )

            elif len(texto) > 7:

                for inicio in range(
                    len(texto) - 6
                ):

                    trecho = texto[
                        inicio:
                        inicio + 7
                    ]

                    resultado = (
                        self.avaliar_candidato(
                            trecho,
                            confianca
                        )
                    )

                    if resultado:
                        candidatos.append(
                            resultado
                        )

        textos_validos = []
        confiancas = []

        for (
            _,
            texto,
            confianca
        ) in resultados_ocr:

            texto = (
                self.limpar_texto(
                    texto
                )
            )

            texto = (
                self.remover_textos_auxiliares(
                    texto
                )
            )

            if not texto:
                continue

            if texto in {
                "BR",
                "BRA",
                "BRASIL",
            }:
                continue

            textos_validos.append(
                texto
            )

            confiancas.append(
                float(
                    confianca
                )
            )

        if textos_validos:

            texto_completo = "".join(
                textos_validos
            )

            confianca_media = (
                sum(confiancas)
                / len(confiancas)
            )

            if (
                len(texto_completo)
                >= 7
            ):

                for inicio in range(
                    len(
                        texto_completo
                    ) - 6
                ):

                    trecho = (
                        texto_completo[
                            inicio:
                            inicio + 7
                        ]
                    )

                    resultado = (
                        self.avaliar_candidato(
                            trecho,
                            confianca_media
                        )
                    )

                    if resultado:
                        candidatos.append(
                            resultado
                        )

        if not candidatos:
            return "", 0.0

        melhor = max(
            candidatos,
            key=lambda item:
                item["score"]
        )

        return (
            melhor["placa"],
            melhor["confianca"]
        )

    # ========================================================
    # VARIAÇÕES PARA OCR
    # ========================================================

    @staticmethod
    def gerar_variacoes_ocr(
        imagem
    ):

        variacoes = []

        if (
            imagem is None
            or imagem.size == 0
        ):
            return variacoes

        ampliada = cv2.resize(
            imagem,
            None,
            fx=4,
            fy=4,
            interpolation=cv2.INTER_CUBIC
        )

        variacoes.append(
            ampliada
        )

        cinza = cv2.cvtColor(
            ampliada,
            cv2.COLOR_BGR2GRAY
        )

        variacoes.append(
            cinza
        )

        filtrada = cv2.bilateralFilter(
            cinza,
            9,
            75,
            75
        )

        variacoes.append(
            filtrada
        )

        _, binaria = cv2.threshold(
            filtrada,
            0,
            255,
            cv2.THRESH_BINARY
            + cv2.THRESH_OTSU
        )

        variacoes.append(
            binaria
        )

        invertida = cv2.bitwise_not(
            binaria
        )

        variacoes.append(
            invertida
        )

        altura, largura = (
            imagem.shape[:2]
        )

        inicio_y = int(
            altura * 0.16
        )

        recorte = imagem[
            inicio_y:altura,
            0:largura
        ]

        if recorte.size > 0:

            recorte = cv2.resize(
                recorte,
                None,
                fx=4,
                fy=4,
                interpolation=cv2.INTER_CUBIC
            )

            recorte_cinza = (
                cv2.cvtColor(
                    recorte,
                    cv2.COLOR_BGR2GRAY
                )
            )

            variacoes.append(
                recorte_cinza
            )

        return variacoes

    # ========================================================
    # EXECUTAR OCR
    # ========================================================

    def executar_ocr(
        self,
        placa_crop
    ):

        variacoes = (
            self.gerar_variacoes_ocr(
                placa_crop
            )
        )

        melhor_placa = ""
        melhor_confianca = 0.0
        melhores_resultados = []

        for variacao in variacoes:

            resultados = (
                self.reader.readtext(
                    variacao,
                    allowlist=(
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                        "0123456789"
                    ),
                    paragraph=False
                )
            )

            if not resultados:
                continue

            (
                placa,
                confianca
            ) = self.extrair_placa(
                resultados
            )

            if (
                placa
                and confianca
                > melhor_confianca
            ):

                melhor_placa = (
                    placa
                )

                melhor_confianca = (
                    confianca
                )

                melhores_resultados = (
                    resultados
                )

        return (
            melhor_placa,
            melhor_confianca,
            melhores_resultados
        )

    # ========================================================
    # DETECTAR VEÍCULOS
    # ========================================================

    def detectar_veiculos(
        self,
        imagem
    ):
        """
        Detecta e segmenta os veículos presentes
        na imagem usando YOLO11n-seg.

        Não utiliza o modelo treinado de placas.
        """

        altura_imagem, largura_imagem = (
            imagem.shape[:2]
        )

        resultados = (
            self.vehicle_model.predict(
                imagem,
                conf=0.25,
                classes=list(
                    self.VEHICLE_CLASSES.keys()
                ),
                verbose=False
            )
        )

        veiculos = []

        for resultado in resultados:

            for indice, box in enumerate(
                resultado.boxes
            ):

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                x1 = max(
                    0,
                    min(
                        x1,
                        largura_imagem - 1
                    )
                )

                y1 = max(
                    0,
                    min(
                        y1,
                        altura_imagem - 1
                    )
                )

                x2 = max(
                    x1 + 1,
                    min(
                        x2,
                        largura_imagem
                    )
                )

                y2 = max(
                    y1 + 1,
                    min(
                        y2,
                        altura_imagem
                    )
                )

                classe = int(
                    box.cls[0]
                )

                confianca = float(
                    box.conf[0]
                )

                mascara = None

                # Se o YOLO-seg forneceu máscara,
                # convertemos o polígono para máscara
                # no tamanho original da imagem.
                if (
                    resultado.masks
                    is not None
                    and indice
                    < len(
                        resultado.masks.xy
                    )
                ):

                    poligono = (
                        resultado.masks.xy[
                            indice
                        ]
                    )

                    if (
                        poligono is not None
                        and len(poligono) >= 3
                    ):

                        mascara = np.zeros(
                            (
                                altura_imagem,
                                largura_imagem
                            ),
                            dtype=np.uint8
                        )

                        poligono = np.array(
                            poligono,
                            dtype=np.int32
                        )

                        cv2.fillPoly(
                            mascara,
                            [poligono],
                            255
                        )

                veiculos.append({
                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2
                    ],
                    "classe":
                        classe,
                    "tipo":
                        self.VEHICLE_CLASSES.get(
                            classe,
                            "Veículo"
                        ),
                    "confianca":
                        confianca,
                    "mascara":
                        mascara,
                })

        return veiculos

    # ========================================================
    # ASSOCIAR PLACA AO VEÍCULO
    # ========================================================

    @staticmethod
    def encontrar_veiculo_da_placa(
        bbox_placa,
        veiculos
    ):
        """
        Procura o veículo que contém o centro
        da placa.

        Isso é útil quando há mais de um veículo
        na mesma fotografia.
        """

        if not veiculos:
            return None

        (
            px1,
            py1,
            px2,
            py2
        ) = bbox_placa

        centro_placa_x = (
            px1 + px2
        ) / 2

        centro_placa_y = (
            py1 + py2
        ) / 2

        candidatos = []

        for veiculo in veiculos:

            (
                vx1,
                vy1,
                vx2,
                vy2
            ) = veiculo["bbox"]

            if (
                vx1
                <= centro_placa_x
                <= vx2
                and
                vy1
                <= centro_placa_y
                <= vy2
            ):

                area = (
                    (vx2 - vx1)
                    * (vy2 - vy1)
                )

                candidatos.append(
                    (
                        area,
                        veiculo
                    )
                )

        if candidatos:

            candidatos.sort(
                key=lambda item:
                    item[0]
            )

            return candidatos[0][1]

        # Se houver apenas um veículo na foto,
        # ele é o melhor candidato mesmo que a
        # bounding box tenha ficado um pouco curta.
        if len(veiculos) == 1:
            return veiculos[0]

        # Com vários veículos, encontra o mais
        # próximo do centro da placa.
        melhor_veiculo = None
        menor_distancia = None

        for veiculo in veiculos:

            (
                vx1,
                vy1,
                vx2,
                vy2
            ) = veiculo["bbox"]

            centro_veiculo_x = (
                vx1 + vx2
            ) / 2

            centro_veiculo_y = (
                vy1 + vy2
            ) / 2

            distancia = (
                (
                    centro_placa_x
                    - centro_veiculo_x
                ) ** 2
                +
                (
                    centro_placa_y
                    - centro_veiculo_y
                ) ** 2
            )

            if (
                menor_distancia is None
                or distancia
                < menor_distancia
            ):

                menor_distancia = (
                    distancia
                )

                melhor_veiculo = (
                    veiculo
                )

        return melhor_veiculo

    # ========================================================
    # ESTIMAR COR DO VEÍCULO
    # ========================================================

    @staticmethod
    def estimar_cor_veiculo(
        imagem,
        veiculo
    ):
        """
        Estima a cor predominante do veículo.

        Utiliza:
        - bounding box do veículo;
        - máscara de segmentação, quando disponível;
        - região central da carroceria;
        - espaço de cores HSV.

        O resultado é uma estimativa visual e
        não um dado cadastral oficial.
        """

        if veiculo is None:
            return (
                "Não identificada",
                0.0
            )

        (
            x1,
            y1,
            x2,
            y2
        ) = veiculo["bbox"]

        altura_imagem, largura_imagem = (
            imagem.shape[:2]
        )

        x1 = max(
            0,
            min(
                x1,
                largura_imagem - 1
            )
        )

        y1 = max(
            0,
            min(
                y1,
                altura_imagem - 1
            )
        )

        x2 = max(
            x1 + 1,
            min(
                x2,
                largura_imagem
            )
        )

        y2 = max(
            y1 + 1,
            min(
                y2,
                altura_imagem
            )
        )

        largura = (
            x2 - x1
        )

        altura = (
            y2 - y1
        )

        # Região mais provável de conter
        # pintura da carroceria.
        roi_x1 = (
            x1
            + int(
                largura * 0.08
            )
        )

        roi_x2 = (
            x1
            + int(
                largura * 0.92
            )
        )

        roi_y1 = (
            y1
            + int(
                altura * 0.28
            )
        )

        roi_y2 = (
            y1
            + int(
                altura * 0.80
            )
        )

        roi = imagem[
            roi_y1:roi_y2,
            roi_x1:roi_x2
        ]

        if roi.size == 0:
            return (
                "Não identificada",
                0.0
            )

        # ----------------------------------------------------
        # MÁSCARA DE SEGMENTAÇÃO
        # ----------------------------------------------------

        mascara = (
            veiculo.get(
                "mascara"
            )
        )

        if mascara is not None:

            mascara_roi = mascara[
                roi_y1:roi_y2,
                roi_x1:roi_x2
            ]

            mascara_roi = (
                mascara_roi > 0
            )

        else:

            mascara_roi = np.ones(
                roi.shape[:2],
                dtype=bool
            )

        if (
            np.count_nonzero(
                mascara_roi
            )
            < 50
        ):

            mascara_roi = np.ones(
                roi.shape[:2],
                dtype=bool
            )

        # ----------------------------------------------------
        # HSV
        # ----------------------------------------------------

        hsv = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2HSV
        )

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        h = h[
            mascara_roi
        ]

        s = s[
            mascara_roi
        ]

        v = v[
            mascara_roi
        ]

        if len(h) == 0:
            return (
                "Não identificada",
                0.0
            )

        # Remove pixels praticamente sem
        # informação visual.
        pixels_validos = (
            v > 20
        )

        h = h[
            pixels_validos
        ]

        s = s[
            pixels_validos
        ]

        v = v[
            pixels_validos
        ]

        if len(h) == 0:
            return (
                "Preto",
                0.5
            )

        # ----------------------------------------------------
        # VERIFICA SE O VEÍCULO É COLORIDO
        # ----------------------------------------------------

        mascara_colorida = (
            (s >= 55)
            &
            (v >= 45)
        )

        proporcao_colorida = float(
            np.mean(
                mascara_colorida
            )
        )

        # ----------------------------------------------------
        # CORES COLORIDAS
        # ----------------------------------------------------

        if (
            proporcao_colorida
            >= 0.18
        ):

            tons = h[
                mascara_colorida
            ].astype(
                np.int32
            )

            saturacoes = s[
                mascara_colorida
            ].astype(
                np.float32
            )

            valores = v[
                mascara_colorida
            ].astype(
                np.float32
            )

            # Pixels mais saturados têm mais
            # peso na decisão.
            pesos = (
                saturacoes
                / 255.0
            ) * (
                0.5
                + valores
                / 510.0
            )

            histograma = np.bincount(
                tons,
                weights=pesos,
                minlength=180
            ).astype(
                np.float32
            )

            # Suavização circular do Hue.
            expandido = np.concatenate([
                histograma[-4:],
                histograma,
                histograma[:4]
            ])

            kernel = np.ones(
                9,
                dtype=np.float32
            )

            suavizado = np.convolve(
                expandido,
                kernel,
                mode="same"
            )

            suavizado = (
                suavizado[
                    4:-4
                ]
            )

            hue_dominante = int(
                np.argmax(
                    suavizado
                )
            )

            total_histograma = float(
                np.sum(
                    suavizado
                )
            )

            if total_histograma > 0:

                confianca_cor = float(
                    suavizado[
                        hue_dominante
                    ]
                    / total_histograma
                )

                # Escala para valor mais intuitivo.
                confianca_cor = min(
                    1.0,
                    confianca_cor * 8
                )

            else:

                confianca_cor = (
                    proporcao_colorida
                )

            valor_mediano = float(
                np.median(
                    valores
                )
            )

            # Vermelho
            if (
                hue_dominante < 10
                or hue_dominante >= 170
            ):
                return (
                    "Vermelho",
                    confianca_cor
                )

            # Laranja / Marrom
            if (
                10
                <= hue_dominante
                < 22
            ):

                if valor_mediano < 130:

                    return (
                        "Marrom",
                        confianca_cor
                    )

                return (
                    "Laranja",
                    confianca_cor
                )

            # Amarelo
            if (
                22
                <= hue_dominante
                < 38
            ):
                return (
                    "Amarelo",
                    confianca_cor
                )

            # Verde
            if (
                38
                <= hue_dominante
                < 85
            ):
                return (
                    "Verde",
                    confianca_cor
                )

            # Azul
            if (
                85
                <= hue_dominante
                < 135
            ):
                return (
                    "Azul",
                    confianca_cor
                )

            # Roxo
            if (
                135
                <= hue_dominante
                < 170
            ):
                return (
                    "Roxo",
                    confianca_cor
                )

        # ----------------------------------------------------
        # PRETO / BRANCO / CINZA / PRATA
        # ----------------------------------------------------

        mascara_neutra = (
            s < 65
        )

        if (
            np.count_nonzero(
                mascara_neutra
            )
            > 0
        ):

            valores_neutros = v[
                mascara_neutra
            ]

            # Percentil superior reduz a influência
            # de pneus e vidros pretos.
            luminosidade = float(
                np.percentile(
                    valores_neutros,
                    65
                )
            )

            proporcao_neutra = float(
                np.mean(
                    mascara_neutra
                )
            )

        else:

            luminosidade = float(
                np.percentile(
                    v,
                    65
                )
            )

            proporcao_neutra = 0.5

        if luminosidade < 80:

            return (
                "Preto",
                proporcao_neutra
            )

        if luminosidade > 190:

            return (
                "Branco",
                proporcao_neutra
            )

        return (
            "Cinza / Prata",
            proporcao_neutra
        )

    # ========================================================
    # DECISÃO AUTOMÁTICA
    # ========================================================

    def tomar_decisao(
        self,
        conf_yolo: float,
        conf_ocr: float,
        tipo: str
    ) -> str:

        if (
            conf_yolo
            >= self.detection_threshold
            and conf_ocr
            >= self.ocr_threshold
            and tipo not in {
                "Formato não reconhecido",
                "Formato não confirmado",
            }
        ):
            return (
                "APROVADO AUTOMATICAMENTE"
            )

        return "REVISÃO HUMANA"

    # ========================================================
    # PROCESSAMENTO PRINCIPAL
    # ========================================================

    def processar(
        self,
        image_path: str
    ):

        inicio = (
            time.perf_counter()
        )

        image = cv2.imread(
            image_path
        )

        if image is None:

            raise ValueError(
                "Não foi possível abrir "
                "a imagem."
            )

        original = (
            image.copy()
        )

        # ====================================================
        # DETECTAR VEÍCULOS
        # ====================================================

        veiculos = (
            self.detectar_veiculos(
                original
            )
        )

        # ====================================================
        # DETECTAR PLACAS
        # ====================================================

        resultados = (
            self.model.predict(
                image,
                conf=self.candidate_threshold,
                iou=0.50,
                verbose=False
            )
        )

        placas = []

        # ====================================================
        # PROCESSAR PLACAS
        # ====================================================

        for resultado in resultados:

            for box in resultado.boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[
                        0
                    ].tolist()
                )

                confianca_yolo = float(
                    box.conf[0]
                )

                bbox_placa = [
                    x1,
                    y1,
                    x2,
                    y2
                ]

                # ============================================
                # ENCONTRAR VEÍCULO DA PLACA
                # ============================================

                veiculo_associado = (
                    self.encontrar_veiculo_da_placa(
                        bbox_placa,
                        veiculos
                    )
                )

                # ============================================
                # ESTIMAR COR
                # ============================================

                if veiculo_associado:

                    (
                        cor_veiculo,
                        confianca_cor
                    ) = (
                        self.estimar_cor_veiculo(
                            original,
                            veiculo_associado
                        )
                    )

                    tipo_veiculo = (
                        veiculo_associado[
                            "tipo"
                        ]
                    )

                    confianca_veiculo = (
                        veiculo_associado[
                            "confianca"
                        ]
                    )

                else:

                    cor_veiculo = (
                        "Não identificada"
                    )

                    confianca_cor = 0.0

                    tipo_veiculo = (
                        "Não identificado"
                    )

                    confianca_veiculo = 0.0

                # ============================================
                # RECORTE DA PLACA
                # ============================================

                placa_crop = original[
                    y1:y2,
                    x1:x2
                ]

                if (
                    placa_crop.size
                    == 0
                ):
                    continue

                # ============================================
                # OCR
                # ============================================

                (
                    placa_texto,
                    confianca_ocr,
                    _
                ) = self.executar_ocr(
                    placa_crop
                )

                # ============================================
                # TIPO DE PLACA
                # ============================================

                if (
                    placa_texto
                    and confianca_ocr
                    >= self.classification_threshold
                ):

                    tipo = (
                        self.identificar_tipo_placa(
                            placa_texto
                        )
                    )

                else:

                    tipo = (
                        "Formato não confirmado"
                    )

                # ============================================
                # DECISÃO
                # ============================================

                decisao = (
                    self.tomar_decisao(
                        confianca_yolo,
                        confianca_ocr,
                        tipo
                    )
                )

                # ============================================
                # SALVAR RECORTE
                # ============================================

                crop_filename = (
                    f"placa_"
                    f"{uuid.uuid4().hex}.jpg"
                )

                crop_path = (
                    self.results_dir
                    / crop_filename
                )

                cv2.imwrite(
                    str(
                        crop_path
                    ),
                    placa_crop
                )

                # ============================================
                # COR DA BOUNDING BOX
                # ============================================

                if (
                    decisao
                    == "APROVADO AUTOMATICAMENTE"
                ):

                    cor_box = (
                        0,
                        200,
                        0
                    )

                else:

                    cor_box = (
                        0,
                        165,
                        255
                    )

                # ============================================
                # BOUNDING BOX DA PLACA
                # ============================================

                cv2.rectangle(
                    image,
                    (
                        x1,
                        y1
                    ),
                    (
                        x2,
                        y2
                    ),
                    cor_box,
                    3
                )

                # ============================================
                # TEXTO DA DETECÇÃO
                # ============================================

                if placa_texto:

                    label = (
                        f"{placa_texto} "
                        f"{confianca_yolo * 100:.1f}%"
                    )

                else:

                    label = (
                        f"PLACA "
                        f"{confianca_yolo * 100:.1f}%"
                    )

                cv2.putText(
                    image,
                    label,
                    (
                        x1,
                        max(
                            30,
                            y1 - 10
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cor_box,
                    2
                )

                # ============================================
                # RESULTADO
                # ============================================

                placas.append({
                    "placa":
                        placa_texto,

                    "tipo":
                        tipo,

                    "cor_veiculo":
                        cor_veiculo,

                    "confianca_cor":
                        round(
                            confianca_cor,
                            4
                        ),

                    "tipo_veiculo":
                        tipo_veiculo,

                    "confianca_veiculo":
                        round(
                            confianca_veiculo,
                            4
                        ),

                    "confianca_yolo":
                        round(
                            confianca_yolo,
                            4
                        ),

                    "confianca_ocr":
                        round(
                            confianca_ocr,
                            4
                        ),

                    "decisao":
                        decisao,

                    "crop_filename":
                        crop_filename,

                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2
                    ]
                })

        # ====================================================
        # SALVAR IMAGEM COMPLETA
        # ====================================================

        result_filename = (
            f"resultado_"
            f"{uuid.uuid4().hex}.jpg"
        )

        result_path = (
            self.results_dir
            / result_filename
        )

        cv2.imwrite(
            str(
                result_path
            ),
            image
        )

        # ====================================================
        # LATÊNCIA
        # ====================================================

        latencia = (
            (
                time.perf_counter()
                - inicio
            )
            * 1000
        )

        # ====================================================
        # RETORNO PARA FLASK
        # ====================================================

        return {
            "placas":
                placas,

            "resultado_imagem":
                result_filename,

            "latencia_ms":
                round(
                    latencia,
                    2
                )
        }