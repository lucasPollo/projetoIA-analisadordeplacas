from pathlib import Path
import uuid

from flask import (
    Flask,
    render_template,
    request,
    url_for
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


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )