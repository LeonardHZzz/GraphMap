import logging
import sys
from pathlib import Path

import streamlit as st
from logica import GraphLoader, TrafficRouter, RouteVisualizer, FACTORES_HORARIO, RADIO_M, CENTRO

log = logging.getLogger("UrbanGraph")


def _init_loader() -> GraphLoader:
    loader = GraphLoader(centro=CENTRO, radio=RADIO_M)
    loader.cargar(usar_cache=True)
    return loader


def ejecutar_app() -> None:
    st.set_page_config(
        page_title="UrbanGraph Traffic",
        page_icon="🗺",
        layout="wide",
    )

    st.title("UrbanGraph Traffic")
    st.caption(f"Red vial de Miraflores — radio {RADIO_M // 1000} km desde Parque Kennedy")

    with st.spinner("Cargando red vial de Miraflores..."):
        if "loader" not in st.session_state:
            st.session_state.loader     = _init_loader()
            st.session_state.router     = TrafficRouter(st.session_state.loader)
            st.session_state.visualizer = RouteVisualizer(st.session_state.loader)

    loader     = st.session_state.loader
    router     = st.session_state.router
    visualizer = st.session_state.visualizer

    col_info1, col_info2 = st.columns(2)
    col_info1.metric("Intersecciones (nodos)", f"{len(loader.G_osm.nodes):,}")
    col_info2.metric("Cuadras (aristas)",      f"{len(loader.G_osm.edges):,}")

    st.divider()

    lugares  = sorted(loader.lugares_clave.keys())
    horarios = list(FACTORES_HORARIO.keys())

    col1, col2, col3 = st.columns(3)
    with col1:
        origen_nombre  = st.selectbox("Origen",  lugares, index=0)
    with col2:
        destino_nombre = st.selectbox("Destino", lugares, index=1)
    with col3:
        horario_sel = st.selectbox(
            "Horario",
            horarios,
            format_func=lambda h: f"{h}  (x{FACTORES_HORARIO[h]})"
        )

    calcular = st.button("Calcular ruta", type="primary", use_container_width=True)

    if calcular:
        if origen_nombre == destino_nombre:
            st.error("El origen y el destino no pueden ser el mismo lugar.")
            return

        origen_id  = loader.lugares_clave[origen_nombre]
        destino_id = loader.lugares_clave[destino_nombre]

        with st.spinner("Ejecutando Bidirectional A*..."):
            resultado = router.bidirectional_a_star(origen_id, destino_id, horario_sel)

        if not resultado.camino:
            st.warning("No se encontró una ruta viable entre esos puntos.")
            return

        st.success(f"Ruta encontrada: {origen_nombre} → {destino_nombre}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Algoritmo",         resultado.algoritmo)
        m2.metric("Nodos en ruta",     len(resultado.camino))
        m3.metric("Nodos explorados",  f"{resultado.nodos_vistos:,}")
        m4.metric("Tiempo de cálculo", f"{resultado.tiempo_ms:.1f} ms")

        tab_mapa, tab_grafo = st.tabs(["Mapa interactivo", "Diagrama de ruta"])

        with tab_mapa:
            st.caption("Azul: ruta óptima | Naranja: puntos clave | Calor: congestión por arista")
            mapa_path = "ruta_optima.html"
            visualizer.mapa_folium(resultado, mapa_path)
            with open(mapa_path, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=520, scrolling=False)
            with open(mapa_path, "rb") as f:
                st.download_button(
                    label="Descargar mapa HTML",
                    data=f,
                    file_name="ruta_optima.html",
                    mime="text/html",
                )

        with tab_grafo:
            grafo_path = "ruta_graphviz"
            dot = visualizer.diagrama_graphviz(resultado, grafo_path)
            if dot is not None:
                png_path = Path(grafo_path + ".png")
                if png_path.exists():
                    st.image(str(png_path), use_container_width=True)
                    with open(png_path, "rb") as f:
                        st.download_button(
                            label="Descargar diagrama PNG",
                            data=f,
                            file_name="ruta_graphviz.png",
                            mime="image/png",
                        )

    st.divider()
    with st.expander("Resumen del grafo cargado"):
        st.text(loader.info())
        st.text(f"Radio de cobertura: {RADIO_M} m ({RADIO_M / 1000:.1f} km)")
        st.text(f"Centro: {CENTRO}")
        st.text(f"Factores de horario: {FACTORES_HORARIO}")

if __name__ == "__main__":
    ejecutar_app()