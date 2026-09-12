from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset" / "data.yaml"
RESULTS_PATH = BASE_DIR / "results"


def main():
    print("=" * 60)
    print("TESLA PLATE AI - TESTE DE TREINAMENTO")
    print("=" * 60)

    print(f"Dataset: {DATASET_PATH}")
    print("Carregando modelo YOLO11n pré-treinado...")

    model = YOLO("yolo11n.pt")

    model.train(
        data=str(DATASET_PATH),
        epochs=3,
        imgsz=640,
        batch=4,
        device="cpu",
        project=str(RESULTS_PATH),
        name="teste_treinamento",
        workers=2,
        plots=True,
        save=True,
        verbose=True,
    )

    print()
    print("=" * 60)
    print("TREINAMENTO DE TESTE CONCLUÍDO")
    print("=" * 60)


if __name__ == "__main__":
    main()