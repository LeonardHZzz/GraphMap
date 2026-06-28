import logging
from pathlib import Path

import folium
import streamlit as st
from streamlit_folium import st_folium

from logica import GraphLoader, TrafficRouter, RouteVisualizer, FACTORES_HORARIO, RADIO_M, CENTRO

log = logging.getLogger("UrbanGraph")

# ── Constantes de sesión ──────────────────────────────────────────────────────
_FASE_ORIGEN  = "esperando_origen"
_FASE_DESTINO = "esperando_destino"
_FASE_LISTO   = "listo"


def _init_session() -> None:
    """Inicializa todas las claves de session_state la primera vez."""
    defaults = {
        "loader":          None,
        "router":          None,
        "visualizer":      None,
        "fase":            _FASE_ORIGEN,   # estado del flujo de selección
        "origen_nodo":     None,
        "destino_nodo":    None,
        "origen_latlon":   None,
        "destino_latlon":  None,
        "resultado":       None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _cargar_grafo() -> None:
    if st.session_state.loader is None:
        loader = GraphLoader(centro=CENTRO, radio=RADIO_M)
        loader.cargar(usar_cache=True)
        st.session_state.loader     = loader
        st.session_state.router     = TrafficRouter(loader)
        st.session_state.visualizer = RouteVisualizer(loader)


def _resetear_seleccion() -> None:
    st.session_state.fase           = _FASE_ORIGEN
    st.session_state.origen_nodo    = None
    st.session_state.destino_nodo   = None
    st.session_state.origen_latlon  = None
    st.session_state.destino_latlon = None
    st.session_state.resultado      = None


def _instruccion_actual() -> str:
    fase = st.session_state.fase
    if fase == _FASE_ORIGEN:
        return "🟢 Haz clic en el mapa para fijar el **origen**"
    if fase == _FASE_DESTINO:
        return "🔴 Haz clic en el mapa para fijar el **destino**"
    return "✅ Origen y destino seleccionados — elige horario y calcula"


def _procesar_clic(lat: float, lon: float) -> None:
    loader = st.session_state.loader
    nodo   = loader.nodo_mas_cercano(lat, lon)

    if st.session_state.fase == _FASE_ORIGEN:
        st.session_state.origen_nodo   = nodo
        st.session_state.origen_latlon = (lat, lon)
        st.session_state.fase          = _FASE_DESTINO
        st.session_state.resultado     = None

    elif st.session_state.fase == _FASE_DESTINO:
        if nodo == st.session_state.origen_nodo:
            st.warning("El punto seleccionado coincide con el origen. Elige otro.")
            return
        st.session_state.destino_nodo   = nodo
        st.session_state.destino_latlon = (lat, lon)
        st.session_state.fase           = _FASE_LISTO
        st.session_state.resultado      = None


def _nombre_nodo(nodo: int) -> str:
    """Devuelve el nombre del lugar clave si coincide, si no las coordenadas."""
    loader = st.session_state.loader
    for nombre, nid in loader.lugares_clave.items():
        if nid == nodo:
            return nombre
    d = loader.G_osm.nodes[nodo]
    return f"({d['y']:.5f}, {d['x']:.5f})"


def ejecutar_app() -> None:
    st.set_page_config(
        page_title="UrbanGraph Traffic",
        page_icon="🗺",
        layout="wide",
    )

    _init_session()

    with st.spinner("Cargando red vial de Miraflores..."):
        _cargar_grafo()

    loader     = st.session_state.loader
    router     = st.session_state.router
    visualizer = st.session_state.visualizer

    # ── Cabecera ──────────────────────────────────────────────────────────────
    st.title("UrbanGraph Traffic")
    st.caption(f"Red vial de Miraflores — radio {RADIO_M // 1000} km · Bidirectional A*")

    c1, c2 = st.columns(2)
    c1.metric("Intersecciones (nodos)", f"{len(loader.G_osm.nodes):,}")
    c2.metric("Cuadras (aristas)",      f"{len(loader.G_osm.edges):,}")

    st.divider()

    # ── Panel de estado + controles ───────────────────────────────────────────
    col_estado, col_ctrl = st.columns([3, 1])

    with col_estado:
        st.info(_instruccion_actual())

        if st.session_state.origen_nodo is not None:
            st.markdown(
                f"🟢 **Origen:** {_nombre_nodo(st.session_state.origen_nodo)}"
            )
        if st.session_state.destino_nodo is not None:
            st.markdown(
                f"🔴 **Destino:** {_nombre_nodo(st.session_state.destino_nodo)}"
            )

    with col_ctrl:
        horarios = list(FACTORES_HORARIO.keys())
        horario_sel = st.selectbox(
            "Horario",
            horarios,
            format_func=lambda h: f"{h}  (×{FACTORES_HORARIO[h]})",
            key="horario_sel",
        )
        btn_calcular = st.button(
            "Calcular ruta",
            type="primary",
            disabled=(st.session_state.fase != _FASE_LISTO),
            use_container_width=True,
        )
        st.button(
            "🔄 Reiniciar",
            on_click=_resetear_seleccion,
            use_container_width=True,
        )

    # ── Calcular ruta ─────────────────────────────────────────────────────────
    if btn_calcular and st.session_state.fase == _FASE_LISTO:
        with st.spinner("Ejecutando Bidirectional A*..."):
            resultado = router.bidirectional_a_star(
                st.session_state.origen_nodo,
                st.session_state.destino_nodo,
                horario_sel,
            )
        if not resultado.camino:
            st.error("No se encontró una ruta viable entre esos puntos.")
        else:
            st.session_state.resultado = resultado

    # ── Métricas del resultado ────────────────────────────────────────────────
    if st.session_state.resultado is not None:
        res = st.session_state.resultado
        st.success(
            f"Ruta encontrada — "
            f"{_nombre_nodo(st.session_state.origen_nodo)} → "
            f"{_nombre_nodo(st.session_state.destino_nodo)}"
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Algoritmo",        res.algoritmo)
        m2.metric("Nodos en ruta",    len(res.camino))
        m3.metric("Nodos explorados", f"{res.nodos_vistos:,}")
        m4.metric("Tiempo cálculo",   f"{res.tiempo_ms:.1f} ms")

    st.divider()

    # ── Mapa interactivo ──────────────────────────────────────────────────────
    resultado = st.session_state.resultado

    if resultado is not None:
        # Mapa con ruta ya calculada
        mapa = visualizer.mapa_folium(
            resultado,
            origen_nodo  = st.session_state.origen_nodo,
            destino_nodo = st.session_state.destino_nodo,
            horario      = horario_sel,
        )
    else:
        # Mapa base — solo marcadores de selección parcial
        mapa = visualizer.mapa_base(
            origen_nodo  = st.session_state.origen_nodo,
            destino_nodo = st.session_state.destino_nodo,
        )

    # Renderizar con st_folium — captura clics
    mapa_data = st_folium(
        mapa,
        use_container_width=True,
        height=540,
        returned_objects=["last_clicked"],
        key="mapa_principal",
    )

    # Procesar clic del usuario
    clic = mapa_data.get("last_clicked") if mapa_data else None
    if clic and st.session_state.fase in (_FASE_ORIGEN, _FASE_DESTINO):
        lat = clic.get("lat")
        lon = clic.get("lng")
        if lat is not None and lon is not None:
            _procesar_clic(lat, lon)
            st.rerun()

    # ── Tab de diagrama Graphviz (solo con resultado) ─────────────────────────
    if resultado is not None:
        st.divider()
        with st.expander("📊 Ver diagrama estructural de la ruta"):
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

        with st.expander("🗺 Descargar mapa como HTML"):
            mapa_path = "ruta_optima.html"
            if Path(mapa_path).exists():
                with open(mapa_path, "rb") as f:
                    st.download_button(
                        label="Descargar mapa HTML",
                        data=f,
                        file_name="ruta_optima.html",
                        mime="text/html",
                    )

    # ── Resumen técnico ───────────────────────────────────────────────────────
    with st.expander("ℹ️ Resumen del grafo cargado"):
        st.text(loader.info())
        st.text(f"Radio de cobertura: {RADIO_M} m ({RADIO_M / 1000:.1f} km)")
        st.text(f"Centro: {CENTRO}")
        st.text(f"Factores de horario: {FACTORES_HORARIO}")


if __name__ == "__main__":
    ejecutar_app()
