import heapq
import math
import time
import random
import logging
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

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

CENTRO    = "Parque Kennedy, Miraflores, Lima, Peru"
RADIO_M   = 2500
CACHE_PATH = Path("grafo_miraflores.pkl")


@dataclass
class ResultadoRuta:
    camino:       list
    costo_total:  float
    tiempo_ms:    float
    nodos_vistos: int
    algoritmo:    str
    horario:      str

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
        self.G_osm        = None
        self.grafo_ciudad = {}
        self.coordenadas  = {}
        self.lugares_clave = {}

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

    def _descargar_osm(self) -> None:
        log.info("Descargando OSM radio=%dm...", self.radio)
        self.G_osm = ox.graph_from_address(self.centro, dist=self.radio, network_type="drive")
        G_proy     = ox.project_graph(self.G_osm)
        for node, data in G_proy.nodes(data=True):
            self.coordenadas[node]  = (data["x"], data["y"])
            self.grafo_ciudad[node] = {}
        for u, v, data in G_proy.edges(data=True):
            ruido = max(0.5, random.gauss(1.0, 0.2))
            self.grafo_ciudad[u][v] = {"dist": data["length"], "trafico": ruido}

    def _guardar_cache(self) -> None:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump((self.G_osm, self.grafo_ciudad, self.coordenadas), f)
        log.info("Cache guardado → %s", CACHE_PATH)

    def _cargar_cache(self) -> None:
        log.info("Cargando desde cache...")
        with open(CACHE_PATH, "rb") as f:
            self.G_osm, self.grafo_ciudad, self.coordenadas = pickle.load(f)

    def _resolver_lugares(self) -> None:
        nodos = list(self.G_osm.nodes)
        for i, (nombre, nid) in enumerate(LUGARES_CLAVE_IDS.items()):
            if nid in self.G_osm.nodes:
                self.lugares_clave[nombre] = nid
            else:
                idx = (i * (len(nodos) // 10)) % len(nodos)
                self.lugares_clave[nombre] = nodos[idx]
                log.warning("Nodo '%s' no encontrado, usando fallback.", nombre)


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

        g_fwd  = {inicio:  0.0}
        g_bwd  = {destino: 0.0}
        pad_fwd = {inicio:  None}
        pad_bwd = {destino: None}
        open_fwd = [(self._heuristica(inicio, destino), inicio)]
        open_bwd = [(self._heuristica(destino, inicio), destino)]
        cerrado_fwd: set = set()
        cerrado_bwd: set = set()

        mejor_costo = float("inf")
        nodo_encuentro = None
        vistos = 0

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
            for v, info in self._grafo.get(u, {}).items():
                ng = g_fwd[u] + self._peso(info["dist"], info["trafico"], horario)
                if ng < g_fwd.get(v, float("inf")):
                    g_fwd[v]  = ng
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
            for v, info in grafo_inv.get(u, {}).items():
                ng = g_bwd[u] + self._peso(info["dist"], info["trafico"], horario)
                if ng < g_bwd.get(v, float("inf")):
                    g_bwd[v]  = ng
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
            camino       = camino,
            costo_total  = mejor_costo if mejor_costo < float("inf") else 0.0,
            tiempo_ms    = (time.perf_counter() - t0) * 1000,
            nodos_vistos = vistos,
            algoritmo    = "Bidirectional A*",
            horario      = horario,
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
        self._G_osm   = loader.G_osm
        self._grafo   = loader.grafo_ciudad
        self._lugares = loader.lugares_clave

    def mapa_folium(self, resultado: ResultadoRuta, ruta_html: str = "ruta_optima.html") -> Optional[object]:
        camino = resultado.camino
        if not camino:
            log.warning("Camino vacio.")
            return None

        puntos = [(self._G_osm.nodes[n]["y"], self._G_osm.nodes[n]["x"]) for n in camino]
        m = folium.Map(location=puntos[0], zoom_start=15, tiles="CartoDB positron")

        self._agregar_heatmap(m, resultado.horario)

        folium.PolyLine(puntos, color="royalblue", weight=6, opacity=0.9,
                        tooltip=f"{resultado.algoritmo} | {resultado.horario}").add_to(m)

        folium.Marker(puntos[0],  popup="Origen",
                      icon=folium.Icon(color="green", icon="play")).add_to(m)
        folium.Marker(puntos[-1], popup="Destino",
                      icon=folium.Icon(color="red",   icon="stop")).add_to(m)

        nombre_por_id = {v: k for k, v in self._lugares.items()}
        for nid, nombre in nombre_por_id.items():
            if nid in self._G_osm.nodes:
                d = self._G_osm.nodes[nid]
                folium.CircleMarker(
                    location=[d["y"], d["x"]], radius=6,
                    color="orange", fill=True, fill_opacity=0.9,
                    popup=nombre
                ).add_to(m)

        info = f"""
        <div style="position:fixed;bottom:30px;left:30px;background:white;
                    padding:10px 14px;border-radius:8px;
                    box-shadow:2px 2px 6px rgba(0,0,0,.3);
                    font-family:monospace;font-size:13px;z-index:9999;">
          <b>UrbanGraph Traffic</b><br>
          Algoritmo: {resultado.algoritmo}<br>
          Horario: {resultado.horario}<br>
          Nodos en ruta: {len(camino)}<br>
          Costo ponderado: {resultado.costo_total:.0f}<br>
          Tiempo calculo: {resultado.tiempo_ms:.1f} ms
        </div>"""
        m.get_root().html.add_child(folium.Element(info))
        m.save(ruta_html)
        log.info("Mapa guardado → %s", ruta_html)
        return m

    def _agregar_heatmap(self, mapa: object, horario: str) -> None:
        mod = FACTORES_HORARIO.get(horario, 1.0)
        puntos_calor = []
        for u, vecinos in self._grafo.items():
            for v, info in vecinos.items():
                peso_total = info["dist"] * info["trafico"] * mod
                if u in self._G_osm.nodes and v in self._G_osm.nodes:
                    lu = self._G_osm.nodes[u]
                    lv = self._G_osm.nodes[v]
                    lat_mid = (lu["y"] + lv["y"]) / 2
                    lon_mid = (lu["x"] + lv["x"]) / 2
                    puntos_calor.append([lat_mid, lon_mid, peso_total])

        if puntos_calor:
            max_peso = max(p[2] for p in puntos_calor)
            puntos_norm = [[p[0], p[1], p[2] / max_peso] for p in puntos_calor]
            HeatMap(
                puntos_norm,
                min_opacity=0.2,
                radius=12,
                blur=15,
                gradient={0.2: "blue", 0.5: "lime", 0.8: "orange", 1.0: "red"},
                name="Congestion"
            ).add_to(mapa)
            folium.LayerControl().add_to(mapa)

    def diagrama_graphviz(self, resultado: ResultadoRuta, ruta_png: str = "ruta_graphviz") -> Optional[object]:
        camino = resultado.camino
        if not camino:
            log.warning("Camino vacio.")
            return None

        nombre_por_id = {v: k for k, v in self._lugares.items()}
        dot = gv.Digraph(format="png")
        dot.attr(rankdir="LR", bgcolor="white", fontname="Helvetica")

        for nodo in set(camino):
            label     = nombre_por_id.get(nodo, f"#{str(nodo)[-4:]}")
            es_extremo = nodo in (camino[0], camino[-1])
            dot.node(str(nodo), label=label, style="filled",
                     fillcolor="gold" if es_extremo else "lightblue",
                     fontname="Helvetica", shape="box")

        for i in range(len(camino) - 1):
            u, v  = camino[i], camino[i + 1]
            info  = self._grafo.get(u, {}).get(v, {})
            dist  = int(info.get("dist", 0))
            dot.edge(str(u), str(v), label=f"{dist}m", color="crimson", penwidth="2.5")

        dot.render(ruta_png, cleanup=True)
        log.info("Diagrama exportado → %s.png", ruta_png)
        return dot
