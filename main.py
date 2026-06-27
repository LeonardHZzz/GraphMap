import sys
import logging
import argparse
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("UrbanGraph")

from logica import (
    GraphLoader, TrafficRouter, RouteVisualizer,
    FACTORES_HORARIO, RADIO_M, CENTRO
)


def _pedir_indice(prompt: str, maximo: int) -> int:
    while True:
        try:
            val = int(input(prompt)) - 1
            if 0 <= val < maximo:
                return val
            print(f"  Ingresa un número entre 1 y {maximo}.")
        except (ValueError, KeyboardInterrupt):
            print("  Entrada inválida.")


def modo_cli(usar_cache: bool = True) -> None:
    loader = GraphLoader(centro=CENTRO, radio=RADIO_M)
    loader.cargar(usar_cache=usar_cache)
    log.info(loader.info())

    router     = TrafficRouter(loader)
    visualizer = RouteVisualizer(loader)

    lugares  = sorted(loader.lugares_clave.keys())
    horarios = list(FACTORES_HORARIO.keys())

    print("\n" + "=" * 55)
    print("   UrbanGraph Traffic — Miraflores, Lima")
    print(f"   Radio: {RADIO_M}m | Algoritmo: Bidirectional A*")
    print("=" * 55)

    print("\nLugares disponibles:")
    for i, nombre in enumerate(lugares, 1):
        print(f"  [{i:2d}] {nombre}")

    idx_origen  = _pedir_indice("Selecciona ORIGEN  (numero): ", len(lugares))
    idx_destino = _pedir_indice("Selecciona DESTINO (numero): ", len(lugares))

    print("\nHorarios:")
    for i, h in enumerate(horarios, 1):
        print(f"  [{i}] {h}  (x{FACTORES_HORARIO[h]})")
    idx_horario = _pedir_indice("Selecciona HORARIO (numero): ", len(horarios))

    origen_nombre  = lugares[idx_origen]
    destino_nombre = lugares[idx_destino]
    horario        = horarios[idx_horario]

    if loader.lugares_clave[origen_nombre] == loader.lugares_clave[destino_nombre]:
        print("[!] Origen y destino son el mismo.")
        return

    origen_id  = loader.lugares_clave[origen_nombre]
    destino_id = loader.lugares_clave[destino_nombre]

    print(f"\n{'─'*55}")
    print(f"Ruta: {origen_nombre} → {destino_nombre}")
    print(f"Horario: {horario}  |  factor: x{FACTORES_HORARIO[horario]}")
    print("─" * 55)

    resultado = router.bidirectional_a_star(origen_id, destino_id, horario)

    if resultado.camino:
        print(resultado.resumen())
        visualizer.mapa_folium(resultado, "ruta_optima.html")
        visualizer.diagrama_graphviz(resultado, "ruta_graphviz")
        print("\n[OK] Archivos: ruta_optima.html | ruta_graphviz.png")
    else:
        print("[!] No se encontro una ruta viable.")


def modo_streamlit() -> None:
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "interfaz.py"],
        check=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UrbanGraph Traffic — uso local unicamente"
    )
    parser.add_argument(
        "--modo",
        choices=["web", "cli"],
        default="web",
        help="web: abre Streamlit | cli: terminal interactiva",
    )
    parser.add_argument(
        "--sin-cache",
        action="store_true",
        help="Descarga el grafo de OSM aunque exista cache local",
    )
    args = parser.parse_args()

    if args.modo == "web":
        modo_streamlit()
    else:
        modo_cli(usar_cache=not args.sin_cache)


if __name__ == "__main__":
    main()
