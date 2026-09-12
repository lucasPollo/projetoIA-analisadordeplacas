from pathlib import Path
import re
import time

import cv2
import easyocr
from ultralytics import YOLO


# ============================================================
# CAMINHOS DO PROJETO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best.pt"

IMAGE_PATH = (
    BASE_DIR
    / "test_images"
    / "carro6.jpg"
)

OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FUNÇÕES
# ============================================================

def limpar_placa(texto: str) -> str:
    """
    Remove espaços, hífens e qualquer caractere
    que não seja letra ou número.
    """
    texto = texto.upper()

    return re.sub(
        r"[^A-Z0-9]",
        "",
        texto
    )


def identificar_tipo_placa(placa: str) -> str:
    """
    Identifica se a placa segue o padrão
    brasileiro antigo ou o padrão Mercosul.
    """

    padrao_mercosul = (
        r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$"
    )

    padrao_antigo = (
        r"^[A-Z]{3}[0-9]{4}$"
    )

    if re.fullmatch(
        padrao_mercosul,
        placa
    ):
        return "Mercosul"

    if re.fullmatch(
        padrao_antigo,
        placa
    ):
        return "Padrão brasileiro antigo"

    return "Formato não reconhecido"


def corrigir_caracteres_mercosul(
    texto: str
) -> str:
    """
    Corrige confusões comuns do OCR com base
    na posição esperada da placa Mercosul.

    Formato Mercosul brasileiro:
    L L L N L N N
    """

    texto = limpar_placa(
        texto
    )

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
        caractere = caracteres[
            posicao
        ]

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
        caractere = caracteres[
            posicao
        ]

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


def corrigir_caracteres_placa_antiga(
    texto: str
) -> str:
    """
    Corrige confusões comuns do OCR para
    placas brasileiras antigas.

    Formato:
    L L L N N N N
    """

    texto = limpar_placa(
        texto
    )

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
        caractere = caracteres[
            posicao
        ]

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
        caractere = caracteres[
            posicao
        ]

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


def preparar_imagem_ocr(
    imagem
):
    """
    Prepara a placa para o OCR.

    Remove parte superior da placa Mercosul,
    aumenta a imagem, converte para tons de
    cinza e reduz ruído.
    """

    altura, largura = (
        imagem.shape[:2]
    )

    # Remove aproximadamente 18% da parte superior.
    # Isso ajuda a eliminar BR / BRASIL.
    inicio_y = int(
        altura * 0.18
    )

    # Pequena margem nas laterais.
    inicio_x = int(
        largura * 0.02
    )

    fim_x = int(
        largura * 0.98
    )

    imagem = imagem[
        inicio_y:altura,
        inicio_x:fim_x
    ]

    if imagem.size == 0:
        return None

    imagem = cv2.resize(
        imagem,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    imagem = cv2.cvtColor(
        imagem,
        cv2.COLOR_BGR2GRAY
    )

    imagem = cv2.bilateralFilter(
        imagem,
        9,
        75,
        75
    )

    return imagem


def extrair_placa_do_ocr(
    resultados_ocr
):
    """
    Analisa os textos encontrados pelo OCR
    e tenta localizar um candidato válido
    de placa brasileira.
    """

    candidatos = []

    for (
        _,
        texto,
        confianca
    ) in resultados_ocr:

        texto_limpo = limpar_placa(
            texto
        )

        if not texto_limpo:
            continue

        # Textos auxiliares comuns em placas Mercosul.
        if texto_limpo in {
            "BR",
            "BRASIL",
            "BRA",
            "BRASILBR",
        }:
            continue

        candidatos.append(
            (
                texto_limpo,
                float(
                    confianca
                )
            )
        )

    # --------------------------------------------------------
    # TENTA CADA BLOCO INDIVIDUAL
    # --------------------------------------------------------

    for (
        texto,
        confianca
    ) in candidatos:

        if len(texto) != 7:
            continue

        mercosul = (
            corrigir_caracteres_mercosul(
                texto
            )
        )

        if (
            identificar_tipo_placa(
                mercosul
            )
            == "Mercosul"
        ):
            return (
                mercosul,
                confianca
            )

        antiga = (
            corrigir_caracteres_placa_antiga(
                texto
            )
        )

        if (
            identificar_tipo_placa(
                antiga
            )
            == "Padrão brasileiro antigo"
        ):
            return (
                antiga,
                confianca
            )

    # --------------------------------------------------------
    # TENTA COMBINAR TEXTOS DO OCR
    # --------------------------------------------------------

    texto_completo = "".join(
        texto
        for (
            texto,
            _
        ) in candidatos
    )

    if not texto_completo:
        return "", 0.0

    # Procura sequências de 7 caracteres.
    for inicio in range(
        0,
        max(
            1,
            len(texto_completo) - 6
        )
    ):

        candidato = (
            texto_completo[
                inicio:
                inicio + 7
            ]
        )

        if (
            len(candidato)
            != 7
        ):
            continue

        mercosul = (
            corrigir_caracteres_mercosul(
                candidato
            )
        )

        if (
            identificar_tipo_placa(
                mercosul
            )
            == "Mercosul"
        ):
            confianca_media = (
                sum(
                    confianca
                    for (
                        _,
                        confianca
                    ) in candidatos
                )
                / len(candidatos)
            )

            return (
                mercosul,
                confianca_media
            )

        antiga = (
            corrigir_caracteres_placa_antiga(
                candidato
            )
        )

        if (
            identificar_tipo_placa(
                antiga
            )
            == "Padrão brasileiro antigo"
        ):
            confianca_media = (
                sum(
                    confianca
                    for (
                        _,
                        confianca
                    ) in candidatos
                )
                / len(candidatos)
            )

            return (
                antiga,
                confianca_media
            )

    return "", 0.0


def tomar_decisao(
    confianca_yolo: float,
    confianca_ocr: float,
    tipo_placa: str
) -> str:
    """
    Implementa o fallback do sistema.
    """

    if (
        confianca_yolo >= 0.60
        and confianca_ocr >= 0.70
        and tipo_placa
        != "Formato não reconhecido"
    ):
        return (
            "APROVADO AUTOMATICAMENTE"
        )

    return "REVISÃO HUMANA"


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    print(
        "=" * 60
    )

    print(
        "TESLA PLATE AI"
    )

    print(
        "DETECÇÃO + OCR + VALIDAÇÃO"
    )

    print(
        "=" * 60
    )

    inicio_total = (
        time.perf_counter()
    )

    # --------------------------------------------------------
    # 1. CARREGAR YOLO
    # --------------------------------------------------------

    print(
        "\nCarregando modelo YOLO..."
    )

    model = YOLO(
        str(
            MODEL_PATH
        )
    )

    # --------------------------------------------------------
    # 2. CARREGAR EASYOCR
    # --------------------------------------------------------

    print(
        "Carregando EasyOCR..."
    )

    reader = (
        easyocr.Reader(
            ["en"],
            gpu=False
        )
    )

    # --------------------------------------------------------
    # 3. CARREGAR IMAGEM
    # --------------------------------------------------------

    print(
        f"Imagem: "
        f"{IMAGE_PATH}"
    )

    image = cv2.imread(
        str(
            IMAGE_PATH
        )
    )

    if image is None:
        raise FileNotFoundError(
            "Não foi possível "
            "carregar a imagem: "
            f"{IMAGE_PATH}"
        )

    original = (
        image.copy()
    )

    # --------------------------------------------------------
    # 4. DETECÇÃO YOLO
    # --------------------------------------------------------

    print(
        "\nDetectando placas..."
    )

    inicio_yolo = (
        time.perf_counter()
    )

    results = (
        model.predict(
            source=image,
            conf=0.25,
            verbose=False
        )
    )

    fim_yolo = (
        time.perf_counter()
    )

    tempo_yolo = (
        (
            fim_yolo
            - inicio_yolo
        )
        * 1000
    )

    placas_processadas = 0

    # --------------------------------------------------------
    # 5. PERCORRER DETECÇÕES
    # --------------------------------------------------------

    for result in results:

        for (
            index,
            box
        ) in enumerate(
            result.boxes,
            start=1
        ):

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[
                    0
                ].tolist()
            )

            confianca_yolo = float(
                box.conf[
                    0
                ]
            )

            # ------------------------------------------------
            # 6. RECORTE
            # ------------------------------------------------

            placa_recortada = (
                original[
                    y1:y2,
                    x1:x2
                ]
            )

            if (
                placa_recortada.size
                == 0
            ):
                continue

            placas_processadas += 1

            arquivo_recorte = (
                OUTPUT_DIR
                / (
                    f"placa_"
                    f"{index}.jpg"
                )
            )

            cv2.imwrite(
                str(
                    arquivo_recorte
                ),
                placa_recortada
            )

            # ------------------------------------------------
            # 7. PRÉ-PROCESSAMENTO PARA OCR
            # ------------------------------------------------

            placa_ocr = (
                preparar_imagem_ocr(
                    placa_recortada
                )
            )

            if placa_ocr is None:
                continue

            arquivo_processado = (
                OUTPUT_DIR
                / (
                    f"placa_"
                    f"{index}"
                    f"_ocr.jpg"
                )
            )

            cv2.imwrite(
                str(
                    arquivo_processado
                ),
                placa_ocr
            )

            # ------------------------------------------------
            # 8. OCR
            # ------------------------------------------------

            inicio_ocr = (
                time.perf_counter()
            )

            resultados_ocr = (
                reader.readtext(
                    placa_ocr,
                    allowlist=(
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                        "0123456789"
                    )
                )
            )

            fim_ocr = (
                time.perf_counter()
            )

            tempo_ocr = (
                (
                    fim_ocr
                    - inicio_ocr
                )
                * 1000
            )

            print()
            print(
                "-" * 60
            )

            print(
                f"PLACA DETECTADA "
                f"#{index}"
            )

            print(
                "-" * 60
            )

            print(
                "\nTextos encontrados "
                "pelo OCR:"
            )

            if not resultados_ocr:

                print(
                    "  Nenhum texto "
                    "reconhecido."
                )

            else:

                for (
                    _,
                    texto,
                    confianca
                ) in resultados_ocr:

                    print(
                        f"  {texto} "
                        f"("
                        f"{confianca * 100:.2f}%"
                        f")"
                    )

            # ------------------------------------------------
            # 9. EXTRAIR PLACA REAL
            # ------------------------------------------------

            (
                texto_final,
                confianca_ocr
            ) = extrair_placa_do_ocr(
                resultados_ocr
            )

            tipo_placa = (
                identificar_tipo_placa(
                    texto_final
                )
            )

            # ------------------------------------------------
            # 10. FALLBACK
            # ------------------------------------------------

            decisao = (
                tomar_decisao(
                    confianca_yolo,
                    confianca_ocr,
                    tipo_placa
                )
            )

            # ------------------------------------------------
            # 11. RESULTADOS
            # ------------------------------------------------

            print()

            print(
                f"Texto final: "
                f"{texto_final or 'Não reconhecido'}"
            )

            print(
                f"Tipo: "
                f"{tipo_placa}"
            )

            print(
                f"Confiança YOLO: "
                f"{confianca_yolo * 100:.2f}%"
            )

            print(
                f"Confiança OCR: "
                f"{confianca_ocr * 100:.2f}%"
            )

            print(
                f"Tempo OCR: "
                f"{tempo_ocr:.2f} ms"
            )

            print(
                f"Decisão: "
                f"{decisao}"
            )

            # ------------------------------------------------
            # 12. DEFINIR COR
            # ------------------------------------------------

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

            # ------------------------------------------------
            # 13. DESENHAR BOUNDING BOX
            # ------------------------------------------------

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

            label = (
                texto_final
                if texto_final
                else "PLACA"
            )

            cv2.putText(
                image,
                label,
                (
                    x1,
                    max(
                        y1 - 10,
                        30
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                cor,
                2
            )

    # --------------------------------------------------------
    # 14. SALVAR IMAGEM FINAL
    # --------------------------------------------------------

    resultado_path = (
        OUTPUT_DIR
        / "resultado_final.jpg"
    )

    cv2.imwrite(
        str(
            resultado_path
        ),
        image
    )

    # --------------------------------------------------------
    # 15. TEMPO TOTAL
    # --------------------------------------------------------

    fim_total = (
        time.perf_counter()
    )

    tempo_total = (
        (
            fim_total
            - inicio_total
        )
        * 1000
    )

    print()
    print(
        "=" * 60
    )

    print(
        "RESULTADO FINAL"
    )

    print(
        "=" * 60
    )

    print(
        f"Placas processadas: "
        f"{placas_processadas}"
    )

    print(
        f"Tempo YOLO: "
        f"{tempo_yolo:.2f} ms"
    )

    print(
        f"Tempo total: "
        f"{tempo_total:.2f} ms"
    )

    print(
        "Imagem final salva em:"
    )

    print(
        resultado_path
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()