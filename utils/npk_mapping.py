"""Soil type to approximate NPK (kg/ha style units as in crop dataset)."""

SOIL_NPK: dict[str, tuple[int, int, int]] = {
    "Black": (60, 40, 40),
    "Sandy": (30, 20, 20),
    "Clay": (50, 50, 50),
    "Loamy": (55, 45, 45),
}


def soil_to_npk(soil_type: str) -> tuple[int, int, int]:
    if soil_type not in SOIL_NPK:
        raise ValueError(f"Unknown soil type: {soil_type}")
    return SOIL_NPK[soil_type]
