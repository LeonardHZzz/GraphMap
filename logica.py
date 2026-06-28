import heapq
import math
import time
import random
import logging
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Set, Dict

import osmnx as ox
import folium
from folium.plugins import HeatMap
import graphviz as gv

log = logging.getLogger("UrbanGraph")

FACTORES_HORARIO = {
    "normal":       1.0,
    "punta_manana": 2.5,
    "punta_tarde":  3.0,
}

LUGARES_CLAVE_IDS = {
    "Huaca Pucllana (Norte)":  262572626,
    "Larcomar (Sur)":          262726656,
    "Faro la Marina (Oeste)":  4347713104,
    "Parque Kennedy (Centro)": 262577858,
    "Zona Este (Benavides)":   262574499,
    "Parque del Amor":         262574028,
    "Bajada Balta":            263612203,
    "Av. Arequipa / Angamos":  386842559,
    "Playa Redondo":           314493121,
    "Limite San Isidro":       262572619,
}

CENTRO     = "Parque Kennedy, Miraflores, Lima, Peru"
RADIO_M    = 1500  # Radio optimizado para estabilidad en Streamlit Cloud
CACHE_PATH = Path("grafo_miraflores.pkl")


@dataclass
class ResultadoRuta:
    camino:           list
    costo_total:      float
    tiempo_ms:        float
    nodos_vistos:     int
    algoritmo:        str
    horario:          str
    nodos_explorados: Set[int]  # Todos los nodos evaluados en las colas

    def resumen(self) -> str:
        return (
            f"[{self.algoritmo}] horario={self.horario} | "
            f"nodos_ruta={len(self.camino)} | costo={self.costo_total:.1f} | "
            f"explorados={self.nodos_vistos} | tiempo={self.tiempo_ms:.2f}ms"
        )


class GraphLoader:

    def __init__(self, centro: str = CENTRO, radio: int = RADIO_M, semilla: int = 42):
        self.centro = centro
        self.radio  = radio
        random.seed(semilla)
        self.G_osm         = None
        self.grafo_ciudad  = {}
        self.coordenadas   = {}
        self.lugares_clave = {}
        self.latlon        = {}

    def cargar(self, usar_cache: bool = True) -> None:
        if usar_cache and CACHE_PATH.exists():
            self._cargar_cache()
        else:
            self._descargar_osm()
            if usar_cache:
                self._guardar_cache()
        self._resolver_lugares()
        log.info("Grafo listo — nodos: %d | aristas: %d",
                 len(self.grafo_ciudad),
                 sum(len(v) for v in self.grafo_ciudad.values()))

    def info(self) -> str:
        return (f"Intersecciones: {len(self.G_osm.nodes)} | "
                f"Cuadras: {len(self.G_osm.edges)}")

    def nodo_mas_cercano(self, lat: float, lon: float) -> int:
        mejor_nodo = None
        mejor_dist = float("inf")
        for nodo, (nlon, nlat) in self.latlon.items():
            d = math.sqrt((lat - nlat) ** 2 + (lon - nlon) ** 2)
            if d < mejor_dist:
                mejor_dist = d
                mejor_nodo = nodo
        return mejor_nodo

    def bbox(self) -> tuple:
        lats = [v[1] for v in self.latlon.values()]
        lons = [v[0] for v in self.latlon.values()]
        return min(lats), max(lats), min(lons), max(lons)

    def _descargar_osm(self) -> None:
        log.info("Descargando OSM radio=%dm...", self.radio)
        self.G_osm = ox.graph_from_address(self.centro, dist=self.radio, network_type="drive")
        G_proy     = ox.project_graph(self.G_osm)
        for node, data in G_proy.nodes(data=True):
            self.coordenadas[node]  = (data["x"], data["y"])
            self.grafo_ciudad[node] = {}
        for node, data in self.G_osm.nodes(data=True):
            self.latlon[node] = (data["x"], data["y"])
        for u, v, data in G_proy.edges(data=True):
            ruido = max(0.5, random.gauss(1.0, 0.2))
            self.grafo_ciudad[u][v] = {"dist": data["length"], "trafico": ruido}

    def _guardar_cache(self) -> None:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump((self.G_osm, self.grafo_ciudad, self.coordenadas, self.latlon), f)
        log.info("Cache guardado → %s", CACHE_PATH)

    def _cargar_cache(self) -> None:
        log.info("Cargando desde cache...")
        with open(CACHE_PATH, "rb") as f:
            data = pickle.load(f)
        if len(data) == 4:
            self.G_osm, self.grafo_ciudad, self.coordenadas, self.latlon = data
        else:
            self.G_osm, self.grafo_ciudad, self.coordenadas = data
            for node, d in self.G_osm.nodes(data=True):
                self.latlon[node] = (d["x"], d["y"])

    def _resolver_lugares(self) -> None:
        nodos = list(self.G_osm.nodes)
        for i, (nombre, nid) in enumerate(LUGARES_CLAVE_IDS.items()):
            if nid in self.G_osm.nodes:
                self.lugares_clave[nombre] = nid
            else:
                idx = (i * (len(nodos) // 10)) % len(nodos)
                self.lugares_clave[nombre] = nodos[idx]


class TrafficRouter:

    def __init__(self, loader: GraphLoader):
        self._grafo  = loader.grafo_ciudad
        self._coords = loader.coordenadas

    def _peso(self, dist: float, trafico: float, horario: str) -> float:
        return dist * trafico * FACTORES_HORARIO.get(horario, 1.0)

    def _heuristica(self, u: int, v: int) -> float:
        x1, y1 = self._coords[u]
        x2, y2 = self._coords[v]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def bidirectional_a_star(self, inicio: int, destino: int, horario: str = "normal") -> ResultadoRuta:
        t0 = time.perf_counter()

        g_fwd   = {inicio:  0.0}
        g_bwd   = {destino: 0.0}
        pad_fwd = {inicio:  None}
        pad_bwd = {destino: None}
        open_fwd = [(self._heuristica(inicio, destino), inicio)]
        open_bwd = [(self._heuristica(destino, inicio), destino)]
        cerrado_fwd: set = set()
        cerrado_bwd: set = set()
        explorados: set = set()

        mejor_costo    = float("inf")
        nodo_encuentro = None
        vistos         = 0

        grafo_inv = {}
        for u, vecinos in self._grafo.items():
            for v, info in vecinos.items():
                grafo_inv.setdefault(v, {})[u] = info

        def expandir_fwd():
            nonlocal mejor_costo, nodo_encuentro
            _, u = heapq.heappop(open_fwd)
            if u in cerrado_fwd:
                return
            cerrado_fwd.add(u)
            explorados.add(u)
            for v, info in self._grafo.get(u, {}).items():
                explorados.add(v)
                ng = g_fwd[u] + self._peso(info["dist"], info["trafico"], horario)
                if ng < g_fwd.get(v, float("inf")):
                    g_fwd[v]   = ng
                    pad_fwd[v] = u
                    heapq.heappush(open_fwd, (ng + self._heuristica(v, destino), v))
                if v in cerrado_bwd:
                    total = ng + g_bwd[v]
                    if total < mejor_costo:
                        mejor_costo    = total
                        nodo_encuentro = v

        def expandir_bwd():
            nonlocal mejor_costo, nodo_encuentro
            _, u = heapq.heappop(open_bwd)
            if u in cerrado_bwd:
                return
            cerrado_bwd.add(u)
            explorados.add(u)
            for v, info in grafo_inv.get(u, {}).items():
                explorados.add(v)
                ng = g_bwd[u] + self._peso(info["dist"], info["trafico"], horario)
                if ng < g_bwd.get(v, float("inf")):
                    g_bwd[v]   = ng
                    pad_bwd[v] = u
                    heapq.heappush(open_bwd, (ng + self._heuristica(v, inicio), v))
                if v in cerrado_fwd:
                    total = g_fwd[v] + ng
                    if total < mejor_costo:
                        mejor_costo    = total
                        nodo_encuentro = v

        while open_fwd and open_bwd:
            vistos += 1
            if open_fwd[0][0] <= open_bwd[0][0]:
                expandir_fwd()
            else:
                expandir_bwd()
            if open_fwd and open_bwd:
                mu = open_fwd[0][0] + open_bwd[0][0]
                if mu >= mejor_costo:
                    break

        camino = self._reconstruir_bi(pad_fwd, pad_bwd, inicio, destino, nodo_encuentro)
        return ResultadoRuta(
            camino           = camino,
            costo_total      = mejor_costo if mejor_costo < float("inf") else 0.0,
            tiempo_ms        = (time.perf_counter() - t0) * 1000,
            nodos_vistos     = vistos,
            algoritmo        = "Bidirectional A*",
            horario          = horario,
            nodos_explorados = explorados
        )

    def _reconstruir_bi(self, pad_fwd, pad_bwd, inicio, destino, encuentro) -> list:
        if encuentro is None:
            return []
        segmento_fwd = []
        nodo = encuentro
        while nodo is not None:
            segmento_fwd.append(nodo)
            nodo = pad_fwd.get(nodo)
        segmento_fwd.reverse()
        segmento_bwd = []
        nodo = pad_bwd.get(encuentro)
        while nodo is not None:
            segmento_bwd.append(nodo)
            nodo = pad_bwd.get(nodo)
        return segmento_fwd + segmento_bwd


class RouteVisualizer:

    def __init__(self, loader: GraphLoader):
        self._loader  = loader
        self._G_osm   = loader.G_osm
        self._grafo   = loader.grafo_ciudad
        self._lugares = loader.lugares_clave

    def _obtener_centro_real(self):
        return (self._G_osm.graph["center_lat"], self._G_osm.graph["center_lon"]) \
               if "center_lat" in self._G_osm.graph \
               else (-12.1219, -77.0299)

    def _agregar_limite_cobertura(self, mapa: folium.Map) -> None:
        """Dibuja un círculo delimitador sutil del área operativa máxima del Grafo."""
        centro = self._obtener_centro_real()
        folium.Circle(
            location=centro,
            radius=self._loader.radio,
            color="#2c3e50",
            weight=2,
            fill=True,
            fill_color="#bdc3c7",
            fill_opacity=0.08,
            dash_array="5, 10",
            tooltip="Frontera de Búsqueda Vial (Área Máxima)"
        ).add_to(mapa)

    def mapa_base(self, origen_nodo=None, destino_nodo=None) -> folium.Map:
        centro = self._obtener_centro_real()
        m = folium.Map(location=centro, zoom_start=15, tiles="CartoDB positron")
        
        self._agregar_limite_cobertura(m)

        instrucciones = """
        <div style="position:fixed; top:20px; left:50%; transform:translateX(-50%);
                    background:rgba(255, 255, 255, 0.95); padding:10px 22px; border-radius:30px;
                    box-shadow:0 4px 15px rgba(0,0,0,0.15); font-family:'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    font-size:13px; font-weight:600; color:#2c3e50; z-index:9999; pointer-events:none; letter-spacing:0.5px;">
          <span style="color:#27ae60;">🟢 1° Clic:</span> Origen &nbsp;|&nbsp; <span style="color:#c0392b;">🔴 2° Clic:</span> Destino
        </div>"""
        m.get_root().html.add_child(folium.Element(instrucciones))

        if origen_nodo is not None:
            d = self._G_osm.nodes[origen_nodo]
            folium.Marker(
                [d["y"], d["x"]], popup="Punto de Origen",
                icon=folium.Icon(color="green", icon="play", prefix="fa")
            ).add_to(m)

        if destino_nodo is not None:
            d = self._G_osm.nodes[destino_nodo]
            folium.Marker(
                [d["y"], d["x"]], popup="Punto de Destino",
                icon=folium.Icon(color="red", icon="flag", prefix="fa")
            ).add_to(m)

        return m

    def mapa_folium(self, resultado: ResultadoRuta,
                    origen_nodo=None, destino_nodo=None,
                    horario: str = "normal",
                    ruta_html: str = "ruta_optima.html") -> Optional[folium.Map]:
        camino = resultado.camino
        if not camino:
            return None

        puntos = [(self._G_osm.nodes[n]["y"], self._G_osm.nodes[n]["x"]) for n in camino]
        m = folium.Map(location=puntos[0], zoom_start=15, tiles="CartoDB positron")

        self._agregar_limite_cobertura(m)
        self._agregar_heatmap(m, horario)

        folium.PolyLine(
            puntos, color="#2980b9", weight=6, opacity=0.85,
            tooltip=f"Ruta Calculada vía {resultado.algoritmo}"
        ).add_to(m)

        folium.Marker(puntos[0], popup="Origen", icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(m)
        folium.Marker(puntos[-1], popup="Destino", icon=folium.Icon(color="red", icon="flag", prefix="fa")).add_to(m)

        info = f"""
        <div style="position:fixed; bottom:25px; left:25px; background:white;
                    padding:14px 18px; border-radius:12px; border-left: 5px solid #2980b9;
                    box-shadow:0 4px 20px rgba(0,0,0,0.12);
                    font-family:'Courier New', monospace; font-size:12px; color:#34495e; z-index:9999; line-height:1.5;">
          <b style="font-size:14px; color:#2c3e50;">URBAN-GRAPH TRAFFIC</b><br>
          ───────────────────────<br>
          <b>Método:</b> {resultado.algoritmo}<br>
          <b>Métrica Horaria:</b> {resultado.horario}<br>
          <b>Nodos en Trazo:</b> {len(camino)}<br>
          <b>Costo Dinámico:</b> {resultado.costo_total:.1f}<br>
          <b>Latencia Cálculo:</b> {resultado.tiempo_ms:.2f} ms
        </div>"""
        m.get_root().html.add_child(folium.Element(info))
        m.save(ruta_html)
        return m

    def _agregar_heatmap(self, mapa: folium.Map, horario: str) -> None:
        mod = FACTORES_HORARIO.get(horario, 1.0)
        puntos_calor = []
        for u, vecinos in self._grafo.items():
            for v, info in vecinos.items():
                peso_total = info["dist"] * info["trafico"] * mod
                if u in self._G_osm.nodes and v in self._G_osm.nodes:
                    lu = self._G_osm.nodes[u]
                    lv = self._G_osm.nodes[v]
                    puntos_calor.append([(lu["y"] + lv["y"]) / 2, (lu["x"] + lv["x"]) / 2, peso_total])
        if puntos_calor:
            max_p = max(p[2] for p in puntos_calor)
            HeatMap(
                [[p[0], p[1], p[2] / max_p] for p in puntos_calor],
                min_opacity=0.15, radius=11, blur=14,
                gradient={0.2: "#3498db", 0.5: "#2ecc71", 0.8: "#e67e22", 1.0: "#e74c3c"},
                name="Flujo de Congestión",
            ).add_to(mapa)
            folium.LayerControl().add_to(mapa)

    def generar_grafo_global_graphviz(self, max_nodos: int = 45) -> gv.Digraph:
        """Genera la vista topológica general abstracta del dataset completo de la ciudad."""
        dot = gv.Digraph(name="Grafo_Global", format="png")
        dot.attr(rankdir="TN", bgcolor="#f8f9fa", fontname="Arial")
        dot.attr('node', shape="circle", style="filled", fillcolor="#e2e8f0", color="#cbd5e1", fontname="Arial", fontsize="10")
        dot.attr('edge', color="#94a3b8", opacity="0.4", penwidth="1.0")

        nodos_muestra = list(self._grafo.keys())[:max_nodos]
        nombre_por_id = {v: k for k, v in self._lugares.items()}

        for nodo in nodos_muestra:
            if nodo in nombre_por_id:
                dot.node(str(nodo), label=nombre_por_id[nodo], fillcolor="#64748b", fontcolor="white", shape="box", style="filled,rounded")
            else:
                dot.node(str(nodo), label=f"#{str(nodo)[-4:]}")

            for vecino in list(self._grafo.get(nodo, {}).keys())[:2]:
                if vecino in nodos_muestra:
                    dot.edge(str(nodo), str(vecino))
        return dot

    def diagrama_graphviz(self, resultado: ResultadoRuta, ruta_png: str = "ruta_graphviz") -> Optional[object]:
        camino = resultado.camino
        if not camino:
            return None

        camino_set = set(camino)
        nombre_por_id = {v: k for k, v in self._lugares.items()}
        
        dot = gv.Digraph(format="png")
        dot.attr(rankdir="LR", bgcolor="#ffffff", fontname="Arial")
        
        # 1. Pintar nodos de la ruta óptima elegida
        for nodo in camino_set:
            label = nombre_por_id.get(nodo, f"#{str(nodo)[-4:]}")
            es_extremo = nodo in (camino[0], camino[-1])
            dot.node(
                str(nodo), label=label, style="filled,rounded",
                fillcolor="#2ecc71" if es_extremo else "#3498db",
                fontcolor="white", fontname="Arial", shape="box", penwidth="2"
            )

        # 2. Pintar caminos alternativos descartados (Nodos explorados pero no tomados)
        for u in camino_set:
            for v in self._grafo.get(u, {}).keys():
                if v in resultado.nodos_explorados and v not in camino_set:
                    lbl = nombre_por_id.get(v, f"#{str(v)[-4:]}")
                    dot.node(str(v), label=lbl, style="filled,dashed", fillcolor="#f1f2f6", fontcolor="#7f8c8d", shape="box")
                    
                    info = self._grafo[u][v]
                    dist = int(info.get("dist", 0))
                    dot.edge(str(u), str(v), label=f"{dist}m", color="#bdc3c7", style="dotted", penwidth="1.0")

        # 3. Dibujar aristas principales de la ruta óptima
        for i in range(len(camino) - 1):
            u, v = camino[i], camino[i + 1]
            info = self._grafo.get(u, {}).get(v, {})
            dist = int(info.get("dist", 0))
            dot.edge(str(u), str(v), label=f"{dist}m", color="#e74c3c", penwidth="3.0")

        dot.render(ruta_png, cleanup=True)
        return dot