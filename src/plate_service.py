from pathlib import Path
import re
import time
import uuid

import cv2
import easyocr
from ultralytics import YOLO


class PlateRecognitionService:

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
        print("Carregando YOLO...")

        self.model = YOLO(model_path)

        print("Carregando EasyOCR...")

        self.reader = easyocr.Reader(
            ["en"],
            gpu=False
        )

        self.results_dir = Path(results_dir)

        self.results_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Confiança mínima para o YOLO
        # considerar algo como possível placa.
        self.candidate_threshold = candidate_threshold

        # Confiança mínima do YOLO para
        # aprovação automática.
        self.detection_threshold = detection_threshold

        # Confiança mínima do OCR para
        # aprovação automática.
        self.ocr_threshold = ocr_threshold

        # Confiança mínima para confirmar
        # o tipo da placa.
        self.classification_threshold = (
            classification_threshold
        )

        # Leituras abaixo disso nem são
        # utilizadas como candidatas pelo OCR.
        self.ocr_candidate_threshold = (
            ocr_candidate_threshold
        )

        print("Sistema de IA carregado!")

    # ========================================================
    # LIMPEZA DE TEXTO
    # ========================================================

    @staticmethod
    def limpar_texto(texto: str) -> str:
        """
        Mantém somente letras e números.
        """

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
        """
        Remove BR e BRASIL quando o OCR junta
        esses elementos com a placa principal.

        Exemplo:
        BRAHD9HO7BRASIL
        ->
        AHD9HO7
        """

        texto = texto.upper()

        # Só remove quando há mais de 7 caracteres,
        # evitando modificar uma placa válida.
        while len(texto) > 7:

            if texto.startswith("BRASIL"):
                texto = texto[6:]
                continue

            if texto.startswith("BR"):
                texto = texto[2:]
                continue

            break

        while len(texto) > 7:

            if texto.endswith("BRASIL"):
                texto = texto[:-6]
                continue

            if texto.endswith("BR"):
                texto = texto[:-2]
                continue

            break

        return texto

    # ========================================================
    # IDENTIFICAÇÃO DO TIPO
    # ========================================================

    @staticmethod
    def identificar_tipo_placa(
        placa: str
    ) -> str:
        """
        Identifica os dois formatos brasileiros.

        Mercosul:
        ABC1D23

        Antiga:
        ABC1234
        """

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
            return "Padrão brasileiro antigo"

        return "Formato não reconhecido"

    # ========================================================
    # CORREÇÕES PARA MERCOSUL
    # ========================================================

    @staticmethod
    def corrigir_mercosul(
        texto: str
    ) -> str:
        """
        Corrige erros comuns do OCR usando
        a estrutura:

        L L L N L N N
        """

        if len(texto) != 7:
            return texto

        caracteres = list(texto)

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

            caractere = caracteres[posicao]

            if caractere in numero_para_letra:

                caracteres[posicao] = (
                    numero_para_letra[
                        caractere
                    ]
                )

        for posicao in posicoes_numeros:

            caractere = caracteres[posicao]

            if caractere in letra_para_numero:

                caracteres[posicao] = (
                    letra_para_numero[
                        caractere
                    ]
                )

        return "".join(caracteres)

    # ========================================================
    # CORREÇÕES PARA PLACA ANTIGA
    # ========================================================

    @staticmethod
    def corrigir_antiga(
        texto: str
    ) -> str:
        """
        Corrige erros comuns do OCR usando
        a estrutura:

        L L L N N N N
        """

        if len(texto) != 7:
            return texto

        caracteres = list(texto)

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

            caractere = caracteres[posicao]

            if caractere in numero_para_letra:

                caracteres[posicao] = (
                    numero_para_letra[
                        caractere
                    ]
                )

        for posicao in posicoes_numeros:

            caractere = caracteres[posicao]

            if caractere in letra_para_numero:

                caracteres[posicao] = (
                    letra_para_numero[
                        caractere
                    ]
                )

        return "".join(caracteres)

    # ========================================================
    # QUANTIDADE DE CORREÇÕES
    # ========================================================

    @staticmethod
    def contar_alteracoes(
        original: str,
        corrigido: str
    ) -> int:

        if len(original) != len(corrigido):
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
    # AVALIA CANDIDATO DE 7 CARACTERES
    # ========================================================

    def avaliar_candidato(
        self,
        texto: str,
        confianca: float
    ):
        """
        Primeiro testa o texto exatamente como
        o OCR retornou.

        Somente se não for válido tenta corrigir.

        Isso evita um problema como:

        IEB7001

        ser artificialmente convertido para
        uma placa Mercosul.
        """

        texto = self.limpar_texto(
            texto
        )

        texto = self.remover_textos_auxiliares(
            texto
        )

        if len(texto) != 7:
            return None

        if (
            confianca
            < self.ocr_candidate_threshold
        ):
            return None

        # ----------------------------------------------------
        # 1. VERIFICA TEXTO ORIGINAL PRIMEIRO
        # ----------------------------------------------------

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
                "confianca": confianca,
                "alteracoes": 0,
                "score": confianca + 0.20,
            }

        # ----------------------------------------------------
        # 2. TENTA CORREÇÃO MERCOSUL
        # ----------------------------------------------------

        mercosul = self.corrigir_mercosul(
            texto
        )

        tipo_mercosul = (
            self.identificar_tipo_placa(
                mercosul
            )
        )

        # ----------------------------------------------------
        # 3. TENTA CORREÇÃO ANTIGA
        # ----------------------------------------------------

        antiga = self.corrigir_antiga(
            texto
        )

        tipo_antiga = (
            self.identificar_tipo_placa(
                antiga
            )
        )

        possibilidades = []

        if tipo_mercosul == "Mercosul":

            alteracoes = (
                self.contar_alteracoes(
                    texto,
                    mercosul
                )
            )

            possibilidades.append({
                "placa": mercosul,
                "tipo": "Mercosul",
                "confianca": confianca,
                "alteracoes": alteracoes,
                "score": (
                    confianca
                    - alteracoes * 0.05
                ),
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
                "placa": antiga,
                "tipo":
                    "Padrão brasileiro antigo",
                "confianca": confianca,
                "alteracoes": alteracoes,
                "score": (
                    confianca
                    - alteracoes * 0.05
                ),
            })

        if not possibilidades:
            return None

        return max(
            possibilidades,
            key=lambda item: item["score"]
        )

    # ========================================================
    # EXTRAIR PLACA DOS RESULTADOS DO OCR
    # ========================================================

    def extrair_placa(
        self,
        resultados_ocr
    ):
        """
        Procura o melhor candidato entre
        os textos encontrados pelo OCR.
        """

        candidatos = []

        # ----------------------------------------------------
        # TEXTOS INDIVIDUAIS
        # ----------------------------------------------------

        for (
            _,
            texto,
            confianca
        ) in resultados_ocr:

            confianca = float(
                confianca
            )

            texto = self.limpar_texto(
                texto
            )

            if not texto:
                continue

            texto = (
                self.remover_textos_auxiliares(
                    texto
                )
            )

            # Ignora textos auxiliares
            if texto in {
                "BR",
                "BRA",
                "BRASIL",
            }:
                continue

            # Se tiver exatamente 7 caracteres
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

            # Se tiver mais de 7 caracteres,
            # procura uma janela válida.
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

        # ----------------------------------------------------
        # COMBINAÇÃO DOS TEXTOS
        # ----------------------------------------------------

        textos_validos = []
        confiancas = []

        for (
            _,
            texto,
            confianca
        ) in resultados_ocr:

            texto = self.limpar_texto(
                texto
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
                float(confianca)
            )

        if textos_validos:

            texto_completo = "".join(
                textos_validos
            )

            confianca_media = (
                sum(confiancas)
                / len(confiancas)
            )

            if len(texto_completo) >= 7:

                for inicio in range(
                    len(texto_completo) - 6
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

        # Escolhe o melhor candidato.
        melhor = max(
            candidatos,
            key=lambda item: item["score"]
        )

        return (
            melhor["placa"],
            melhor["confianca"]
        )

    # ========================================================
    # GERA VERSÕES PARA OCR
    # ========================================================

    @staticmethod
    def gerar_variacoes_ocr(
        imagem
    ):
        """
        Gera várias versões da placa.

        Isso melhora principalmente placas antigas,
        sujas, inclinadas ou com pouco contraste.
        """

        variacoes = []

        if (
            imagem is None
            or imagem.size == 0
        ):
            return variacoes

        # ----------------------------------------------------
        # 1. IMAGEM COMPLETA AMPLIADA
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # 2. ESCALA DE CINZA
        # ----------------------------------------------------

        cinza = cv2.cvtColor(
            ampliada,
            cv2.COLOR_BGR2GRAY
        )

        variacoes.append(
            cinza
        )

        # ----------------------------------------------------
        # 3. REDUÇÃO DE RUÍDO
        # ----------------------------------------------------

        filtrada = cv2.bilateralFilter(
            cinza,
            9,
            75,
            75
        )

        variacoes.append(
            filtrada
        )

        # ----------------------------------------------------
        # 4. BINARIZAÇÃO OTSU
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # 5. VERSÃO INVERTIDA
        # ----------------------------------------------------

        invertida = cv2.bitwise_not(
            binaria
        )

        variacoes.append(
            invertida
        )

        # ----------------------------------------------------
        # 6. VERSÃO CORTANDO TOPO
        # Útil para Mercosul BR / BRASIL
        # ----------------------------------------------------

        altura, largura = imagem.shape[:2]

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

            recorte_cinza = cv2.cvtColor(
                recorte,
                cv2.COLOR_BGR2GRAY
            )

            variacoes.append(
                recorte_cinza
            )

        return variacoes

    # ========================================================
    # EXECUTA OCR EM TODAS AS VARIAÇÕES
    # ========================================================

    def executar_ocr(
        self,
        placa_crop
    ):
        """
        Executa o EasyOCR em várias versões
        e escolhe a leitura válida com maior
        confiança.
        """

        variacoes = (
            self.gerar_variacoes_ocr(
                placa_crop
            )
        )

        melhor_placa = ""
        melhor_confianca = 0.0
        melhores_resultados = []

        for variacao in variacoes:

            resultados = self.reader.readtext(
                variacao,
                allowlist=(
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "0123456789"
                ),
                paragraph=False
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

                melhor_placa = placa
                melhor_confianca = confianca
                melhores_resultados = resultados

        return (
            melhor_placa,
            melhor_confianca,
            melhores_resultados
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

        inicio = time.perf_counter()

        # ----------------------------------------------------
        # CARREGA A IMAGEM
        # ----------------------------------------------------

        image = cv2.imread(
            image_path
        )

        if image is None:

            raise ValueError(
                "Não foi possível abrir "
                "a imagem."
            )

        original = image.copy()

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        resultados = self.model.predict(
            image,
            conf=self.candidate_threshold,
            iou=0.50,
            verbose=False
        )

        placas = []

        # ----------------------------------------------------
        # PROCESSA AS DETECÇÕES
        # ----------------------------------------------------

        for resultado in resultados:

            for box in resultado.boxes:

                # Coordenadas
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                confianca_yolo = float(
                    box.conf[0]
                )

                # --------------------------------------------
                # RECORTE
                # --------------------------------------------

                placa_crop = original[
                    y1:y2,
                    x1:x2
                ]

                if placa_crop.size == 0:
                    continue

                # --------------------------------------------
                # OCR
                # --------------------------------------------

                (
                    placa_texto,
                    confianca_ocr,
                    _
                ) = self.executar_ocr(
                    placa_crop
                )

                # --------------------------------------------
                # IDENTIFICAÇÃO DO TIPO
                # --------------------------------------------

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

                # --------------------------------------------
                # DECISÃO
                # --------------------------------------------

                decisao = (
                    self.tomar_decisao(
                        confianca_yolo,
                        confianca_ocr,
                        tipo
                    )
                )

                # --------------------------------------------
                # SALVA RECORTE
                # --------------------------------------------

                crop_filename = (
                    f"placa_"
                    f"{uuid.uuid4().hex}.jpg"
                )

                crop_path = (
                    self.results_dir
                    / crop_filename
                )

                cv2.imwrite(
                    str(crop_path),
                    placa_crop
                )

                # --------------------------------------------
                # COR DA DETECÇÃO
                # --------------------------------------------

                if (
                    decisao
                    == "APROVADO AUTOMATICAMENTE"
                ):

                    cor = (
                        0,
                        200,
                        0
                    )

                else:

                    cor = (
                        0,
                        165,
                        255
                    )

                # --------------------------------------------
                # BOUNDING BOX
                # --------------------------------------------

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
                    cor,
                    3
                )

                # --------------------------------------------
                # TEXTO NA IMAGEM
                # --------------------------------------------

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
                    cor,
                    2
                )

                # --------------------------------------------
                # RESULTADO DA PLACA
                # --------------------------------------------

                placas.append({
                    "placa":
                        placa_texto,

                    "tipo":
                        tipo,

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
        # SALVA IMAGEM COMPLETA
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
            str(result_path),
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
        # RETORNO PARA O FLASK
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