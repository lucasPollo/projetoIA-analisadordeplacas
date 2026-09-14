from pathlib import Path
import uuid

from flask import (
    Flask,
    render_template,
    request,
    url_for,
    jsonify
)

from src.plate_service import (
    PlateRecognitionService
)


BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = (
    BASE_DIR
    / "static"
    / "uploads"
)

RESULTS_DIR = (
    BASE_DIR
    / "static"
    / "results"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "best.pt"
)


UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


app = Flask(__name__)

app.config[
    "MAX_CONTENT_LENGTH"
] = 10 * 1024 * 1024


service = PlateRecognitionService(
    model_path=str(MODEL_PATH),
    results_dir=str(RESULTS_DIR)
)


EXTENSOES_PERMITIDAS = {
    "jpg",
    "jpeg",
    "png",
    "webp"
}


def arquivo_permitido(
    filename
):

    return (
        "."
        in filename
        and filename
        .rsplit(
            ".",
            1
        )[1]
        .lower()
        in EXTENSOES_PERMITIDAS
    )


@app.route(
    "/",
    methods=[
        "GET",
        "POST"
    ]
)
def index():

    resultado = None
    imagem_original = None
    erro = None

    if request.method == "POST":

        arquivo = request.files.get(
            "image"
        )

        if not arquivo:

            erro = (
                "Nenhuma imagem enviada."
            )

        elif not arquivo.filename:

            erro = (
                "Selecione uma imagem."
            )

        elif not arquivo_permitido(
            arquivo.filename
        ):

            erro = (
                "Formato de arquivo "
                "não permitido."
            )

        else:

            extensao = (
                arquivo.filename
                .rsplit(
                    ".",
                    1
                )[1]
                .lower()
            )

            filename = (
                f"{uuid.uuid4().hex}"
                f".{extensao}"
            )

            caminho = (
                UPLOAD_DIR
                / filename
            )

            arquivo.save(
                str(caminho)
            )

            resultado = (
                service.processar(
                    str(caminho)
                )
            )

            imagem_original = (
                url_for(
                    "static",
                    filename=(
                        f"uploads/"
                        f"{filename}"
                    )
                )
            )

    return render_template(
        "index.html",
        resultado=resultado,
        imagem_original=imagem_original,
        erro=erro
    )

@app.route(
    "/api/analyze",
    methods=["POST"]
)
def api_analyze():

    arquivo = request.files.get(
        "image"
    )

    # ==========================================
    # VALIDAÇÕES
    # ==========================================

    if not arquivo:

        return jsonify({
            "erro":
                "Nenhuma imagem enviada."
        }), 400

    if not arquivo.filename:

        return jsonify({
            "erro":
                "Arquivo sem nome."
        }), 400

    if not arquivo_permitido(
        arquivo.filename
    ):

        return jsonify({
            "erro":
                "Formato de arquivo não permitido."
        }), 400

    # ==========================================
    # SALVAR IMAGEM
    # ==========================================

    extensao = (
        arquivo.filename
        .rsplit(".", 1)[1]
        .lower()
    )

    filename = (
        f"{uuid.uuid4().hex}."
        f"{extensao}"
    )

    caminho = (
        UPLOAD_DIR
        / filename
    )

    arquivo.save(
        str(caminho)
    )

    try:

        # ======================================
        # EXECUTA IA
        # ======================================

        resultado = (
            service.processar(
                str(caminho)
            )
        )

        # ======================================
        # URLs DOS RESULTADOS
        # ======================================

        resultado["resultado_url"] = (
            url_for(
                "static",
                filename=(
                    "results/"
                    + resultado[
                        "resultado_imagem"
                    ]
                ),
                _external=True
            )
        )

        for placa in resultado[
            "placas"
        ]:

            placa["crop_url"] = (
                url_for(
                    "static",
                    filename=(
                        "results/"
                        + placa[
                            "crop_filename"
                        ]
                    ),
                    _external=True
                )
            )

        return jsonify(
            resultado
        ), 200

    except Exception as erro:

        return jsonify({
            "erro":
                "Erro ao processar imagem.",

            "detalhes":
                str(erro)
        }), 500

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )