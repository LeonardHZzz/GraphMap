import logging
from pathlib import Path

import folium
import streamlit as st
from streamlit_folium import st_folium

from logica import GraphLoader, TrafficRouter, RouteVisualizer, FACTORES_HORARIO, RADIO_M, CENTRO

log = logging.getLogger("UrbanGraph")

_FASE_ORIGEN  = "esperando_origen"
_FASE_DESTINO = "esperando_destino"
_FASE_LISTO   = "listo"


def _init_session() -> None:
    defaults = {
        "loader":          None,
        "router":          None,
        "visualizer":      None,
        "fase":            _FASE_ORIGEN,
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
        return "🟢 **Paso 1:** Haz clic en el mapa para establecer el punto de **Origen**."
    if fase == _FASE_DESTINO:
        return "🔴 **Paso 2:** Haz clic para establecer el punto de **Destino**."
    return "🚀 **Listo:** Puntos registrados. Selecciona el horario y ejecuta el cálculo estructural."


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
            st.error("El nodo de destino no puede ser idéntico al de origen.")
            return
        st.session_state.destino_nodo   = nodo
        st.session_state.destino_latlon = (lat, lon)
        st.session_state.fase           = _FASE_LISTO
        st.session_state.resultado      = None


def _nombre_nodo(nodo: int) -> str:
    loader = st.session_state.loader
    for nombre, nid in loader.lugares_clave.items():
        if nid == nodo:
            return nombre
    d = loader.G_osm.nodes[nodo]
    return f"Intersección ID: {nodo} ({d['y']:.4f}, {d['x']:.4f})"


def ejecutar_app() -> None:
    st.set_page_config(
        page_title="UrbanGraph Traffic Analyzer",
        page_icon="🗺",
        layout="wide",
    )

    _init_session()

    with st.spinner("Inicializando topología vial del distrito..."):
        _cargar_grafo()

    loader     = st.session_state.loader
    router     = st.session_state.router
    visualizer = st.session_state.visualizer

    st.title("🗺 UrbanGraph Traffic Analyzer")
    st.caption("Modelado de Sistemas Complejos y Enrutamiento Dinámico Heurístico · Miraflores, Lima")

    # Métrica de infraestructura vial
    m1, m2, m3 = st.columns(3)
    m1.metric("Intersecciones (Nodos V)", f"{len(loader.G_osm.nodes):,}")
    m2.metric("Tramos Viales (Aristas E)", f"{len(loader.G_osm.edges):,}")
    m3.metric("Radio Operacional Mínimo", f"{RADIO_M} metros")

    st.divider()

    col_mapa, col_controles = st.columns([2, 1])

    with col_controles:
        st.subheader("⚙️ Parámetros de Control")
        st.info(_instruccion_actual())

        if st.session_state.origen_nodo is not None:
            st.success(f"**Origen:** {_nombre_nodo(st.session_state.origen_nodo)}")
        if st.session_state.destino_nodo is not None:
            st.error(f"**Destino:** {_nombre_nodo(st.session_state.destino_nodo)}")

        horarios = list(FACTORES_HORARIO.keys())
        horario_sel = st.selectbox(
            "Carga de Tráfico (Escenario Horario)",
            horarios,
            format_func=lambda h: f"{h.replace('_', ' ').title()} (Factor de Peso: ×{FACTORES_HORARIO[h]})",
        )

        btn_calcular = st.button(
            "Calcular Ruta Óptima",
            type="primary",
            disabled=(st.session_state.fase != _FASE_LISTO),
            use_container_width=True,
        )
        
        st.button("🔄 Reiniciar Coordenadas", on_click=_resetear_seleccion, use_container_width=True)

        if btn_calcular and st.session_state.fase == _FASE_LISTO:
            with st.spinner("Computando Bidirectional A* con funciones de coste horarias..."):
                resultado = router.bidirectional_a_star(
                    st.session_state.origen_nodo,
                    st.session_state.destino_nodo,
                    horario_sel,
                )
            if not resultado.camino:
                st.error("Error: Flujo de red interrumpido o aislado.")
            else:
                st.session_state.resultado = resultado

        # Panel de Métricas de Rendimiento Algorítmico
        if st.session_state.resultado is not None:
            res = st.session_state.resultado
            st.markdown("### 📊 Eficiencia de Búsqueda")
            st.json({
                "Algoritmo": res.algoritmo,
                "Tiempo Procesamiento": f"{res.tiempo_ms:.3f} ms",
                "Nodos Solución": len(res.camino),
                "Nodos Totales Evaluados": res.nodos_vistos,
                "Costo de la Función Objetivo": round(res.costo_total, 2)
            })

    with col_mapa:
        resultado = st.session_state.resultado
        mapa = visualizer.mapa_folium(resultado, st.session_state.origen_nodo, st.session_state.destino_nodo, horario_sel) \
               if resultado is not None \
               else visualizer.mapa_base(st.session_state.origen_nodo, st.session_state.destino_nodo)

        mapa_data = st_folium(
            mapa,
            use_container_width=True,
            height=580,
            returned_objects=["last_clicked"],
            key="mapa_vial",
        )

        clic = mapa_data.get("last_clicked") if mapa_data else None
        if clic and st.session_state.fase in (_FASE_ORIGEN, _FASE_DESTINO):
            _procesar_clic(clic.get("lat"), clic.get("lng"))
            st.rerun()

    # Secciones Técnicas de Grafos Estructurales (Graphviz)
    st.divider()
    tab1, tab2 = st.tabs(["📊 Grafo de Caminos Alternativos Descartados", "🏙 Topología de Red General de la Ciudad"])
    
    with tab1:
        if resultado is not None:
            st.caption("Muestra la secuencia óptima elegida (azul/rojo) frente a las opciones de calles adyacentes evaluadas y descartadas (gris punteado) por congestión.")
            grafo_path = "ruta_graphviz"
            dot = visualizer.diagrama_graphviz(resultado, grafo_path)
            if dot is not None and Path(grafo_path + ".png").exists():
                st.image(grafo_path + ".png", use_container_width=True)
        else:
            st.info("Calcula una ruta para ver el análisis de descarte heurístico.")

    with tab2:
        st.caption("Representación abstracta y simplificada de las interconexiones globales del dataset vial mapeado en el backend.")
        with st.spinner("Compilando mapa de adyacencias estructural..."):
            dot_global = visualizer.generar_grafo_global_graphviz(max_nodos=50)
            global_path = "grafo_global_ciudad"
            dot_global.render(global_path, cleanup=True)
            if Path(global_path + ".png").exists():
                st.image(global_path + ".png", use_container_width=True)


if __name__ == "__main__":
    ejecutar_app()