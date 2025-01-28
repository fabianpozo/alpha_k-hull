import numpy as np
from scipy.spatial import distance
from math import radians, sin, cos, pi
from itertools import groupby as iterGroupBy
import logging
import pprint
from shapely import Point, LineString, Polygon
from collections import defaultdict

# Configuración del logging
logging.basicConfig(filename='depuracion_cuñas.log', filemode="w", level=logging.DEBUG, format='%(message)s')
#-----------------------------------------------------------convex-hull
def graham_scan(points):
    """Calcular el Convex Hull usando el algoritmo Graham Scan, devolviendo los índices de los puntos en el Convex Hull."""
    
    def polar_angle(p0, p1):
        """Calcular el ángulo polar entre dos puntos."""
        return np.arctan2(p1[1] - p0[1], p1[0] - p0[0])

    def distance(p0, p1):
        """Calcular la distancia cuadrada entre dos puntos."""
        return (p1[0] - p0[0])**2 + (p1[1] - p0[1])**2

    # Ordenar los puntos para iniciar desde el punto más bajo (o más a la izquierda en caso de empate)
    points_with_indices = list(enumerate(points))  # Guardar los índices originales
    points_with_indices.sort(key=lambda p: (p[1][1], p[1][0]))  # Ordenar por y, luego por x
    start = points_with_indices[0]  # El primer punto (el más bajo o más a la izquierda)

    # Ordenar los puntos restantes por ángulo polar respecto al punto inicial
    sorted_points = sorted(points_with_indices[1:], key=lambda p: (polar_angle(start[1], p[1]), distance(start[1], p[1])))

    # Inicializar el stack con los primeros dos puntos (el punto inicial y el primer punto ordenado)
    hull = [start[0], sorted_points[0][0]]  # Guardar solo los índices

    for p in sorted_points[1:]:
        while len(hull) > 1 and np.cross(np.subtract(points[hull[-1]], points[hull[-2]]), np.subtract(p[1], points[hull[-1]])) <= 0:
            hull.pop()  # Eliminar puntos que no están en el Convex Hull
        hull.append(p[0])  # Agregar el índice del punto al Convex Hull

    return hull  # Devolver los índices de los puntos en el Convex Hull

#------------------------------------funcion auxiliar 
def ensure_2d_array(point):
    """Convierte el punto a un arreglo NumPy de dos elementos si no lo es."""
    if not isinstance(point, np.ndarray):
        point = np.array(point)
    if point.shape != (2,):
        raise ValueError(f"El punto {point} no es un arreglo de dos elementos.")
    return point

# Función auxiliar para preprocesar points_in_arc
def preprocess_points_in_arc(points_in_arc):
    """
    Normaliza los datos en `points_in_arc` para asegurar que están en un formato consistente.
    Filtra valores inválidos y ajusta estructuras anidadas.
    """
    processed_points = []
    for point_data in points_in_arc:
        try:
            # Asegura que cada entrada tenga la forma [np.array, marcador]
            if isinstance(point_data, list) and len(point_data) == 2:
                p, marker = point_data
                if isinstance(p, (list, np.ndarray)) and len(p) == 2:
                    p = np.array(p)  # Convertir a arreglo NumPy si no lo es
                if marker is None:
                    marker = np.array([np.nan, np.nan])  # Reemplazar None con np.nan
                processed_points.append([p, marker])
        except Exception as e:
            logging.error(f"Error procesando punto en arco: {point_data}. Error: {e}")
    return processed_points


#------------------------------------------------------------------------------------------wedge

class WedgeCalculator:
    def __init__(self, k, alpha, points, point_scale=1, error=1e-4):
        self.k = k
        self.alpha = radians(alpha)
        self.points = [[p[0] * point_scale, p[1] * point_scale] for p in points]
        self.sin_alpha = sin(self.alpha)
        self.cos_alpha = cos(self.alpha)
        self.cds_alpha = self.cos_alpha / self.sin_alpha
        self.error = error

        # Debug: Initial Configuration
        init_message = (f"Initialized WedgeCalculator with k={k}, alpha={alpha}°, "
                        f"sin_alpha={self.sin_alpha}, cos_alpha={self.cos_alpha}, cds_alpha={self.cds_alpha}")
        print(init_message)
        logging.debug(init_message)

    def orientation_line(self, pa, pb, p, pb_is_direction=False, get_distance=False):
        """
        Determina la orientación de un punto respecto a una línea definida por dos puntos.
        """
        px, py = p[0] - pa[0], p[1] - pa[1]
        dx, dy = (pb[0], pb[1]) if pb_is_direction else (pb[0] - pa[0], pb[1] - pa[1])

        # Calcula orientación o distancia
        if get_distance:
            result = (px * dy - py * dx) / ((dx**2 + dy**2) ** 0.5)
        else:
            result = px * dy - py * dx

        # Logging de orientación
        posicion = "derecha" if result > 0 else "izquierda" if result < 0 else "en la línea"
        logging.debug(f"Orientación del punto {p} respecto a la línea PA: {pa}-> PB: {pb}: \n"
                      f"Resultado={result:.4f}, Posición={posicion}\n")
        return result


    def get_score_rotate(self, pc, pa, pb):
        """
        Calcula un puntaje basado en el ángulo de rotación entre puntos `pc`, `pa`, y `pb`,
        ajustando el `score` para que esté en el rango [0, 4].
        """

        # Calcula los vectores desde `pc` hacia `pa` y `pb`
        ax, ay = pa[0] - pc[0], pa[1] - pc[1]
        bx, by = pb[0] - pc[0], pb[1] - pc[1]

        # Producto escalar y cálculo del coseno del ángulo
        dot_product = ax * bx + ay * by
        magnitude_a = (ax**2 + ay**2) ** 0.5
        magnitude_b = (bx**2 + by**2) ** 0.5
        score = dot_product / (magnitude_a * magnitude_b)

        # Ajusta la puntuación según la orientación
        if self.orientation_line(pc, pa, pb) < 0:
            score = 2 + 1 + score  # Ajuste para orientación a la izquierda
        else:
            score = 1 - score  # Ajuste para orientación a la derecha

        # Mensaje de depuración
        log_message = (f"Rotación calculada para los puntos PC: {pc}, PA: {pa}, PB: {pb} -> Puntaje: {score}\n")
        print(log_message)
        logging.debug(log_message)
        return score
    
    def get_sort_point_rotate(self, pc, pa, points, points_key=lambda p: p, reverse=False):
        """
        Ordena puntos en sentido horario respecto a `pc` y `pa`.

        Parámetros:
        - pc: Punto central desde el que se calcula la rotación.
        - pa: Punto de referencia para determinar la orientación.
        - points: Lista de puntos a ordenar.
        - points_key: Función opcional para extraer datos de los puntos.
        - reverse: Indica si el orden debe ser inverso.

        Retorna:
        - Lista de puntos ordenados en sentido horario.
        """
        logging.debug(f"Ordenando puntos alrededor de pc={pc} con respecto a pa={pa}. Puntos iniciales: {points}\n")

        # Define la función de ordenamiento basada en la puntuación de rotación
        rotate_sort = lambda p: self.get_score_rotate(pc, pa, points_key(p))
        
        # Ordena los puntos usando la puntuación calculada
        sorted_points = sorted(points, key=rotate_sort, reverse=reverse)
        
        logging.debug(f"Puntos ordenados alrededor de pc={pc} con respecto a pa={pa}: {sorted_points}\n")
        return sorted_points
    
    def center(self, pa, pb, get_vector=True):
        """
        Retorna el centro del segmento definido por `pa` -> `pb` usando el ángulo `alpha`
        y verifica el vector perpendicular.

        Parámetros:
        - pa: Punto inicial del segmento.
        - pb: Punto final del segmento.
        - get_vector: Si es True, retorna el vector centrado; si es False, retorna el centro desplazado desde `pa`.

        Devuelve:
        - Centro calculado como vector o punto, según `get_vector`.
        """
        # Calculamos el vector original de pa a pb
        J = [pb[0] - pa[0], pb[1] - pa[1]]
        CS = self.cds_alpha  # Cos/Sin(alpha)
        
        # Calculamos el centro y el vector perpendicular
        C = [J[0] - J[1] * CS, J[1] + J[0] * CS]
        C = [C[0] / 2.0, C[1] / 2.0]
        
        # Determinamos el resultado según si se solicita en forma de vector o desplazado al punto pa
        center_result = C if get_vector else [C[0] + pa[0], C[1] + pa[1]]

        # Calcular el vector perpendicular
        perpendicular_vector = [-J[1], J[0]]  # Perpendicular a (Jx, Jy) es (-Jy, Jx)
        
        # Verificar la perpendicularidad mediante el producto escalar
        dot_product = J[0] * perpendicular_vector[0] + J[1] * perpendicular_vector[1]
        is_perpendicular = abs(dot_product) < 1e-6  # Tolerancia pequeña para verificar ortogonalidad
        
        # Depuración: imprimir los resultados
        logging.debug(f"Centro calculado (con escala) entre PA: {pa} y PB: {pb}: {center_result}, vector perpendicular: {perpendicular_vector}\n")
        logging.debug(f"Producto escalar para verificar perpendicularidad: {dot_product}\n")
        logging.debug(f"¿Es perpendicular? {'Sí' if is_perpendicular else 'No'}\n")

        return center_result

    def get_score_point(self, pc, pa, pb, points):
        """
        Calcula cuántos puntos están dentro del área de cuña formada por los vectores Pc->Pa y Pc->Pb.

        Parámetros:
        - pc: Punto central [x, y].
        - pa: Punto que define el primer vector de la cuña [x, y].
        - pb: Punto que define el segundo vector de la cuña [x, y].
        - points: Lista de puntos a evaluar.

        Retorna:
        - Número de puntos dentro del área de la cuña.
        """
        # Calculamos los vectores izquierdo y derecho de la cuña
        if pc == pa:
            v_der = [pb[0] - pa[0], pb[1] - pa[1]]  # Pa->Pb
            v_izq = [
                v_der[0] * self.cos_alpha + v_der[1] * self.sin_alpha,
                v_der[1] * self.cos_alpha - v_der[0] * self.sin_alpha,
            ]  # Rotación en -alpha
        elif pc == pb:
            v_izq = [pa[0] - pb[0], pa[1] - pb[1]]  # Pb->Pa
            v_der = [
                v_izq[0] * self.cos_alpha - v_izq[1] * self.sin_alpha,
                v_izq[1] * self.cos_alpha + v_izq[0] * self.sin_alpha,
            ]  # Rotación en alpha
        else:
            v_izq = [pa[0] - pc[0], pa[1] - pc[1]]  # Pc->Pa
            v_der = [pb[0] - pc[0], pb[1] - pc[1]]  # Pc->Pb

        # Contamos los puntos dentro de la cuña
        score = 0
        for p in points:
            iq = self.orientation_line(pc, v_izq, p, pb_is_direction=True)
            de = self.orientation_line(pc, v_der, p, pb_is_direction=True)
            if iq <= 0 and de >= 0:
                score += 1

        # Depuración: Mostrar resultados
        logging.debug(f"Cuña generada desde PC={pc}, PA={pa}, PB={pb}\n")
        logging.debug(f"Puntos dentro del área delimitada por los vectores vIzq={v_izq} y vDer={v_der}: {score}\n")

        return score

    def distance(self, pa, pb):
        """
        Calcula la distancia euclidiana entre dos puntos.

        Parámetros:
        - pa: Primer punto como lista [x, y].
        - pb: Segundo punto como lista [x, y].

        Retorna:
        - Distancia entre los puntos pa y pb.
        """
        # Validar que los puntos no sean None y sean listas o tuplas
        if pa is None or pb is None:
            logging.error(f"Puntos inválidos para calcular la distancia: PA={pa}, PB={pb}\n")
            raise ValueError("Los puntos pa y pb no pueden ser None.")
        
        dx = pa[0] - pb[0]
        dy = pa[1] - pb[1]
        result = (dx**2 + dy**2)**0.5

        # Depuración: Mostrar el resultado
        logging.debug(f"Distancia calculada entre PA: {pa} y PB: {pb}: {result}\n")
        return result

    def get_points_in_line(self, pa, pc, points):
        """
        Devuelve una lista de puntos en la línea definida por pa -> pc,
        ordenados por distancia a pa y que son colineales con pc.

        Parámetros:
        - pa: Punto inicial como lista [x, y].
        - pc: Punto final como lista [x, y].
        - points: Lista de puntos para evaluar.

        Retorna:
        - Lista de puntos en la línea en orden de distancia desde pa.
        """
        # Filtrar puntos cercanos a la línea definida por pa -> pc
        collinear_points = [
            p for p in points
            if p != pa and abs(self.orientation_line(pa, pc, p, get_distance=True)) <= self.error
        ]

        # Determinar la dirección de la línea
        sig_dx = (pc[0] - pa[0]) > 0
        sig_dy = (pc[1] - pa[1]) > 0

        # Filtrar puntos que están en la dirección de la línea
        directional_points = [
            p for p in collinear_points
            if sig_dx == (p[0] - pa[0] > 0) and sig_dy == (p[1] - pa[1] > 0)
        ]

        # Ordenar los puntos por distancia a pa
        sorted_points = sorted(directional_points, key=lambda x: self.distance(pa, x))

        # Depuración: Mostrar los puntos resultantes
        logging.debug(f"Puntos en línea desde PA: {pa} -> PC: {pc}: {sorted_points}\n")
        return sorted_points

    def get_point_on_rotate(self, pa, pb, points):
        """
        Encuentra el próximo punto en la rotación de una cuña.

        Parámetros:
        - pa: Punto inicial como lista [x, y].
        - pb: Punto final como lista [x, y].
        - points: Lista de puntos a procesar.

        Retorna:
        - El punto encontrado o None si no se encuentra.
        """

        logging.debug(f"Iniciando get_point_on_rotate con PA={pa}, PB={pb}, puntos={points}\n")

        # Calcular las direcciones izquierda y derecha
        izq_dir = [pb[0] - pa[0], pb[1] - pa[1]]
        der_dir = [
            izq_dir[0] * self.cos_alpha + izq_dir[1] * self.sin_alpha,
            izq_dir[1] * self.cos_alpha - izq_dir[0] * self.sin_alpha,
        ]
        logging.debug(f"Direcciones calculadas: izq_dir={izq_dir}, der_dir={der_dir}\n")

        # Obtener puntos en línea y definir puntos a ignorar
        in_line = list(self.get_points_in_line(pa, [pa[0] + der_dir[0], pa[1] + der_dir[1]], points))
        ignore_points = [pa, pb] + in_line
        logging.debug(f"Puntos en línea ignorados: {in_line}, puntos a ignorar: {ignore_points}\n")

        valid_points = [p for p in points if p not in ignore_points]
        logging.debug(f"Puntos válidos para procesar: {valid_points}")

        # Procesar los puntos válidos en orden
        for candidate_point in self.get_sort_point_rotate(pa, [pa[0] + der_dir[0], pa[1] + der_dir[1]], valid_points):
            logging.debug(f"Procesando punto: {candidate_point}")
            in_line = list(self.get_points_in_line(pa, candidate_point, points))

            # Calcular nuevas direcciones
            der_dir_new = [candidate_point[0] - pa[0], candidate_point[1] - pa[1]]
            izq_dir_new = [
                der_dir_new[0] * self.cos_alpha - der_dir_new[1] * self.sin_alpha,
                der_dir_new[1] * self.cos_alpha + der_dir_new[0] * self.sin_alpha,
            ]

            # Calcular el grado de puntos dentro de la cuña
            grade = self.get_score_point(pa, candidate_point, [pa[0] + izq_dir_new[0], pa[1] + izq_dir_new[1]], points)
            logging.debug(f"Grado calculado para {candidate_point}: {grade}, dirección izquierda: {izq_dir_new}\n")

            # Validar el grado contra k
            if grade == self.k:
                logging.debug(f"Punto encontrado que satisface k={self.k}: {in_line[-1]}\n")
                return in_line[-1]
            elif self.k < grade and len(in_line) > 1:
                adjusted_grade = grade - len(in_line)
                for point in in_line:
                    adjusted_grade += 1
                    if adjusted_grade == self.k:
                        logging.debug(f"Punto {point} satisface k={self.k} en ajuste de grado.\n")
                        return point

        logging.warning("No se encontró ningún punto válido para la rotación.")
        return None

    def get_wedge_coord_x(self, pa, pb, inver_x=False, inver_y=False):
        """
        Calcula la coordenada X de una cuña ajustada con `inver_x` e `inver_y`.

        Argumentos:
        pa (list): Primer punto [x, y].
        pb (list): Segundo punto [x, y].
        invert_x (bool): Indica si la coordenada X debe invertirse.
        invert_y (bool): Indica si la coordenada Y debe invertirse.

        Retorna:
        float: Coordenada X de la cuña.
        """
        # Calcula la diferencia en Y, ajustando según si se invierte la coordenada Y.
        y = (pb[1] - pa[1]) if not inver_y else (pa[1] - pb[1])
        dx = y * self.cds_alpha  # Usamos cds_alpha de la instancia de clase.

        # Calcula la coordenada X de la cuña según si se invierte X.
        wedge_x = pb[0] - dx if not inver_x else pb[0] + dx

        # Log de depuración
        log_message = (f"Calculando coordenada X de cuña entre PA: {pa}, PB: {pb} -> "
                    f"Inversión en X: {inver_x}, Inversión en Y: {inver_y} -> Coordenada X: {wedge_x}\n")
        logging.debug(log_message)

        return wedge_x
    
    def split_list_value(self, values, key=None):
        """
        Filtra los puntos en la lista que cumplen con la condición dada.
        """
        # Verifica si los valores proporcionados son una lista
        if not isinstance(values, list):
            return values

        # Si no se proporciona una clave, retorna la lista sin modificaciones
        if key is None:
            return values

        # Filtra los valores según la clave proporcionada
        filtered_values = [x for x in values if key(x)]

        # Log de depuración
        log_message = f"Filtrando valores: {values} -> Resultados filtrados: {filtered_values}\n"
        logging.debug(log_message)

        return filtered_values

    
    def get_wedges(self, inver_x=False, inver_y=False, test=False):
        """
        Genera las cuñas a partir de los puntos proporcionados, considerando inversiones en X e Y.

        Argumentos:
            inver_x (bool): Indica si se debe invertir en X.
            inver_y (bool): Indica si se debe invertir en Y.
            test (bool): Modo de prueba, muestra más detalles en la salida.

        Retorna:
            list: Lista de cuñas generadas con la estructura definida.
        """
        wedges = []

        # Agrupación de puntos por niveles en Y
        levels = {
            y: list(sorted(group, key=lambda p: -p[0]))
            for y, group in iterGroupBy(sorted(self.points, key=lambda p: -p[1]), key=lambda p: p[1])
        }
        logging.debug(f"Niveles generados: {levels}\n")
        if not test:
            logging.debug(f"Niveles para prueba:\n{pprint.pformat(levels)}\n")

        # Direcciones base de las cuñas
        dir_line1 = [1, 0]  # Derecha
        dir_line2 = [self.cos_alpha, self.sin_alpha]  # Derecha + Arriba acorde al ángulo

        if inver_x:
            dir_line1[0] *= -1
            dir_line2[0] *= -1

        if inver_y:
            dir_line1[1] *= -1
            dir_line2[1] *= -1

        logging.debug(f"Direcciones iniciales: dir_line1={dir_line1}, dir_line2={dir_line2}")

        # Recorrido por niveles para generar cuñas
        sorted_levels = list(sorted(levels.keys(), reverse=not inver_y))
        for level_pos, level_y in enumerate(sorted_levels):
            for point in levels[level_y]:
                if not test:
                    logging.debug(f"Punto en nivel principal: {point}")

                # Direcciones ajustadas por el punto actual
                dir_p_line1 = [point[0] + dir_line1[0], point[1] + dir_line1[1]]
                dir_p_line2 = [point[0] + dir_line2[0], point[1] + dir_line2[1]]

                # Inversiones de direcciones
                if inver_x or inver_y:
                    dir_p_line1, dir_p_line2 = dir_p_line2, dir_p_line1

                # Grado inicial para el punto actual
                grade = self.get_score_point(point, dir_p_line1, dir_p_line2, self.points)
                logging.debug(f"Grado inicial para el punto {point}: {grade}\n")

                # Recorre niveles inferiores
                lower_level_pos = level_pos + 1
                while lower_level_pos < len(sorted_levels):
                    lower_level_y = sorted_levels[lower_level_pos]
                    wedge_coord_x = self.get_wedge_coord_x([0, lower_level_y], point, inver_x, inver_y)

                    lower_points = self.split_list_value(
                        levels[lower_level_y],
                        key=(lambda x: wedge_coord_x <= x[0]) if not inver_x else (lambda x: x[0] <= wedge_coord_x)
                    )

                    grade += len(lower_points)

                    # Verificación y adición de la cuña
                    if len(lower_points) > 0 and grade == self.k:
                        # Calcular centro y puntos inicial/final del arco
                        center = self.center(point, lower_points[0], get_vector=False)
                        start = lower_points[0]
                        end = point
                        
                        wedge = {
                            'wedge': [wedge_coord_x, lower_level_y],
                            'pa_s': lower_points if not (inver_x ^ inver_y) else [point],
                            'pb': point if not (inver_x ^ inver_y) else lower_points[0],
                            'inver_x': inver_x,
                            'inver_y': inver_y,
                            'center': center,  # Centro calculado de la cuña
                            'start': start,    # Punto inicial del arco
                            'end': end         # Punto final del arco
                        }
                        wedges.append(wedge)
                        logging.debug(f"Cuña válida encontrada: {wedge}\n")

                    lower_level_pos += 1

        return wedges
    
    def get_normal(self, point):
        """
        Calcula la norma de un punto en el plano cartesiano.
        """
        return point[0] ** 2 + point[1] ** 2

    def get_normal_diff(self, point_a, point_b):
        """
        Calcula la norma de la diferencia entre dos puntos.
        """
        diff = [point_b[0] - point_a[0], point_b[1] - point_a[1]]
        return self.get_normal(diff)

    def get_points_on_circle(self, pa, pb, points):
        """
        Calcula los puntos en el arco determinado por los puntos pa -> pb.

        Retorna:
            [<Puntos en el arco hacia Pa>, <Puntos en el arco hacia Pb>]

        Nota: Los puntos no están ordenados según orientación.
        """
        logging.debug(f"Entrando a get_points_on_circle con PA={pa}, PB={pb}, puntos={self.points}\n")

        pa_point_on_arc = []
        pb_point_on_arc = []

        # Vector J (A-B)
        J = [pa[0] - pb[0], pa[1] - pb[1]]
        logging.debug(f"Vector J: {J}")

        # Función lambda para calcular Lambda
        get_lambda = lambda V: (V[0] * J[1] - V[1] * J[0]) / (self.sin_alpha * self.get_normal(V))

        # Dirección de L1
        l1_direction = [-J[0], -J[1]]

        # Direcciones de L2A y L2B
        l2a_direction = [
            l1_direction[0] * self.cos_alpha + l1_direction[1] * self.sin_alpha,
            l1_direction[1] * self.cos_alpha - l1_direction[0] * self.sin_alpha,
        ]

        l2b_direction = [
            l1_direction[0] * self.cos_alpha - l1_direction[1] * self.sin_alpha,
            l1_direction[1] * self.cos_alpha + l1_direction[0] * self.sin_alpha,
        ]

        for p in self.points:
            logging.debug(f"Procesando punto {p} para validación de arco.")

            # Validación de regiones
            l1_pos = self.orientation_line(pa, pb, p)
            l1_izq = l1_pos < 0
            l1_der = l1_pos > 0

            l2a_pos = self.orientation_line(pa, l2a_direction, p, pb_is_direction=True)
            l2a_izq = l2a_pos < 0
            l2a_der = l2a_pos > 0

            l2b_pos = self.orientation_line(pb, l2b_direction, p, pb_is_direction=True)
            l2b_izq = l2b_pos < 0
            l2b_der = l2b_pos > 0

            # Caso A
            xa = None
            if (l1_izq and l2a_izq) or (l1_der and l2a_der):
                va = [pa[0] - p[0], pa[1] - p[1]]
                wa = [
                    va[0] * self.cos_alpha - va[1] * self.sin_alpha,
                    va[1] * self.cos_alpha + va[0] * self.sin_alpha,
                ]
                lambda_a = get_lambda(va)
                xa = [pb[0] + wa[0] * lambda_a, pb[1] + wa[1] * lambda_a]

                if l1_izq and l2a_izq:
                    if self.get_normal_diff(pa, xa) <= self.get_normal_diff(pa, p):
                        xa = None

            # Caso B
            xb = None
            if (l1_izq and l2b_izq) or (l1_der and l2b_der):
                vb = [pb[0] - p[0], pb[1] - p[1]]
                wb = [
                    vb[0] * self.cos_alpha + vb[1] * self.sin_alpha,
                    vb[1] * self.cos_alpha - vb[0] * self.sin_alpha,
                ]
                lambda_b = get_lambda(vb)
                xb = [pa[0] + wb[0] * lambda_b, pa[1] + wb[1] * lambda_b]

                if l1_izq and l2b_izq:
                    if self.get_normal_diff(pb, xb) <= self.get_normal_diff(pb, p):
                        xb = None

            # Guardado de puntos
            if xa is not None:
                pa_point_on_arc.append([xa, p])
            if xb is not None:
                pb_point_on_arc.append([xb, p])

        return [pa_point_on_arc, pb_point_on_arc]

    def get_next_point(self, points_in_arc, current_point):
        """
        Encuentra el siguiente punto en el arco después del punto actual.

        Args:
            points_in_arc (list): Lista de puntos en el arco.
            current_point (list): Punto actual.

        Returns:
            list or None: El siguiente punto en el arco, o None si no se encuentra.
        """
        logging.debug(f"Entrando a get_next_point con points_in_arc={points_in_arc} y current_point={current_point}")
        current_pass = False
        for point_data in points_in_arc:
            if current_pass:
                next_point = point_data[0][0]
                if next_point == current_point:
                    logging.debug(f"Punto actual {next_point} ignorado.")
                    continue
                logging.debug(f"Punto siguiente encontrado: {point_data}")
                return point_data
            elif point_data[1] is None:
                logging.debug(f"Marcador encontrado en {point_data}, current_point={current_point}")
                current_pass = True
        logging.warning(f"No se encontró un siguiente punto después de {current_point}")
        # Si no se encuentra ningún punto
        return None
    

    def compute_one_wedge(self, wedge_data, test=False):
        """
        Calcula los arcos para una cuña específica.

        Args:
            wedge_data (dict): Datos de la cuña que incluyen el punto de inicio, los pivotes y el centro inicial.
            test (bool): Si es True, solo realiza pruebas y no genera resultados finales.

        Returns:
            list or dict: Lista de arcos calculados o estadísticas si está en modo test.
        """
        logging.debug(f"Iniciando compute_one_wedge con wedge_data={wedge_data}")

        # Resultado de los arcos
        out_arcs = []
        out_test = {
            'arcs': out_arcs,
            'iters': 0,
            'rotates': 0,
            'changes': 0,
        }

        # Límite de iteraciones
        limit_iter = len(self.points) ** 3
        current_iter = 0

        if not test:
            logging.info('---------------------WEDGE--------------')
            logging.info(f'Processing wedge data: {wedge_data}\n')

        # Estado inicial
        current = {
            'point': wedge_data['wedge'],  # Punto en el arco
            'pa': wedge_data['pa_s'][0],  # Primer pivote
            'pb': wedge_data['pb'],       # Segundo pivote
            'center': self.center(wedge_data['pa_s'][0], wedge_data['pb'], get_vector=False),
        }
        logging.debug(f"Estado inicial: {current}\n")

        previous = {
            'point': None,
            'pa': None,
            'pb': None,
            'center': None,
        }

        # Loop principal
        run = True
        while run:
            current_iter += 1
            logging.debug(f"Iteración {current_iter}, estado actual: {current}")

            # Validar límite de iteraciones
            if current_iter > limit_iter:
                logging.info("STOP - Límite de iteraciones alcanzado.")
                break

            # Buscar puntos que interceptan en el arco
            intercepts = self.get_points_on_circle(current['pa'], current['pb'], self.points)
            logging.debug(f"Intersecciones iniciales: {intercepts}")

            # Filtrar soluciones en los bordes
            intercepts[0] = [x for x in intercepts[0] if x[0] != current['pa']]
            intercepts[1] = [x for x in intercepts[1] if x[0] != current['pb']]
            logging.debug(f"Intersecciones filtradas: {intercepts}")

            # Ordenar los puntos interceptados
            points_in_arc = [[x, 'pa'] for x in intercepts[0]] + [[x, 'pb'] for x in intercepts[1]]
            points_in_arc += [[[current['point'], None], None]]
            points_in_arc = self.get_sort_point_rotate(current['center'], current['pa'], points_in_arc, points_key=lambda p: p[0][0], reverse=True)
            logging.debug(f"Puntos ordenados en arco: {points_in_arc}")

            # Obtener el siguiente punto
            next_point = self.get_next_point(points_in_arc, current['point'])
            logging.debug(f"Siguiente punto: {next_point}")

            # Verificar condiciones de término
            if previous['point'] is not None and wedge_data['pa_s'][0] == current['pa'] and wedge_data['pb'] == current['pb']:
                wedge_score_rotate = self.get_score_rotate(current['center'], current['pa'], wedge_data['wedge'])
                if wedge_score_rotate <= self.get_score_rotate(current['center'], current['pa'], current['point']):
                    if next_point is None or self.get_score_rotate(current['center'], current['pa'], next_point[0][0]) <= wedge_score_rotate:
                        logging.info("STOP - Condición de término satisfecha.")
                        run = False

            if not run:
                break

            # Actualizar estado anterior
            previous.update(current)

            # Decidir el siguiente paso
            if next_point is None:
                # Rotar
                if test:
                    out_test['rotates'] += 1
                current['point'] = current['pa']
                current['pb'] = current['pa']
                current['pa'] = self.get_point_on_rotate(previous['pa'], previous['pb'], self.points)
                current['center'] = self.center(current['pa'], current['pb'], get_vector=False)
            else:
                # Cambio de soportes
                if test:
                    out_test['changes'] += 1
                current['point'] = next_point[0][0]
                if next_point[1] == 'pa':
                    current['pa'] = next_point[0][1]
                else:
                    current['pb'] = next_point[0][1]
                current['center'] = self.center(current['pa'], current['pb'], get_vector=False)

            # Guardar arco calculado
            out_arcs.append([previous['center'], previous['point'], current['point']])

        if test:
            out_test['iters'] = current_iter
            return out_test

        logging.info(f"\nArcos calculados: {out_arcs}")
        return out_arcs

    def get_compute_wedge(self):
        """
        Realiza el cálculo de las cuñas utilizando los valores de la clase.

        Este método calcula todas las cuñas y procesa cada una para obtener los arcos resultantes.

        Retorna:
            list: Lista de todas las cuñas procesadas con sus respectivos arcos.
        """
        # Obtener cuñas iniciales
        print("---- CUÑAS ----")
        wedges = self.get_wedges(inver_x=False)

        # Indicar grupo (pueden existir varias cuñas en un mismo grupo)
        group = 0
        print(f"Se encontraron {len(wedges)} cuñas.")

        # Procesar cada cuña
        all_arcs = []
        print("---- COMPUTO ----")
        for wedge_data in wedges:
            arcs = self.compute_one_wedge(wedge_data)
            all_arcs.append(arcs)
            print(f"Cuña procesada: {wedge_data}")
        
        return all_arcs

    def out_get_wedges_data(self, test=False):
        """
        Genera los datos de las cuñas en todas las combinaciones de inversión de X e Y.

        Argumentos:
            test (bool): Si está en True, retorna datos adicionales para pruebas.

        Retorna:
            list or dict: Lista con datos extendidos si `test=True`, de lo contrario, solo las cuñas en la configuración por defecto.
        """
        logging.debug("---- CUÑAS ----")

        # Obtener cuñas en diferentes configuraciones de inversión
        wedges_ff = self.get_wedges(inver_x=False, inver_y=False, test=test)
        wedges_tf = self.get_wedges(inver_x=True, inver_y=False, test=test)
        wedges_ft = self.get_wedges(inver_x=False, inver_y=True, test=test)
        wedges_tt = self.get_wedges(inver_x=True, inver_y=True, test=test)

        if test:
            return [
                {
                    'inver_x': False,
                    'inver_y': False,
                    'wedges': wedges_ff,
                },
                {
                    'inver_x': True,
                    'inver_y': False,
                    'wedges': wedges_tf,
                },
                {
                    'inver_x': False,
                    'inver_y': True,
                    'wedges': wedges_ft,
                },
                {
                    'inver_x': True,
                    'inver_y': True,
                    'wedges': wedges_tt,
                },
            ]
        
        # Retornar cuñas en configuración por defecto
        return wedges_ff 

    def out_compute_one_wedge(self, wedge_data):
        """
        Computa una única cuña con los parámetros actuales de la instancia.

        Argumentos:
            wedge_data (dict): Datos de la cuña a computar, incluyendo los puntos y configuraciones.

        Retorna:
            list: Lista de arcos calculados para la cuña.
        """
        logging.debug(f"Comenzando el cálculo de una cuña con datos: {wedge_data}")
        return self.compute_one_wedge(wedge_data)
    
    def is_point_in_arc(self, point, arc):
        """
        Verifica si un punto está dentro del rango de un arco.
        """
        center, start, end = map(np.array, arc)

        # Verifica la distancia del punto al centro del arco
        radius = np.linalg.norm(start - center)
        distance_to_center = np.linalg.norm(point - center)
        if abs(distance_to_center - radius) > 1e-6:  # Permite un pequeño margen de error
            return False

        # Calcula el ángulo del punto respecto al centro del arco
        angle_point = np.arctan2(point[1] - center[1], point[0] - center[0])
        angle_start = np.arctan2(start[1] - center[1], start[0] - center[0])
        angle_end = np.arctan2(end[1] - center[1], end[0] - center[0])

        # Normaliza los ángulos al rango [0, 2π]
        angle_point = angle_point % (2 * np.pi)
        angle_start = angle_start % (2 * np.pi)
        angle_end = angle_end % (2 * np.pi)

        # Verifica si el ángulo del punto está dentro del rango del arco
        if angle_start < angle_end:
            return angle_start <= angle_point <= angle_end
        else:  # El rango cruza 0 grados
            return angle_point >= angle_start or angle_point <= angle_end

    def calculate_intersection(self, arc1, arc2):
        """
        Calcula la intersección entre dos arcos usando sus ecuaciones.
        """
        # Representar los arcos como ecuaciones paramétricas o implícitas
        center1, start1, _ = map(np.array, arc1)  # Convertir a np.array
        center2, start2, end2 = map(np.array, arc2)  # Convertir a np.array

        # Calcular distancia entre los centros
        dx, dy = center2[0] - center1[0], center2[1] - center1[1]
        d = np.sqrt(dx**2 + dy**2)

        # Verificar si los arcos se intersectan
        r1 = np.linalg.norm(start1 - center1)  # Radio del primer arco
        r2 = np.linalg.norm(start2 - center2)  # Radio del segundo arco
        if d > r1 + r2 or d < abs(r1 - r2):
            return None  # No hay intersección

        # Cálculo simplificado de coordenadas de intersección
        a = (r1**2 - r2**2 + d**2) / (2 * d)
        h = np.sqrt(r1**2 - a**2)
        midpoint = center1 + a * (center2 - center1) / d
        intersection_x = midpoint[0] + h * (center2[1] - center1[1]) / d
        intersection_y = midpoint[1] - h * (center2[0] - center1[0]) / d
        print(f'puntos de interseccion: X: {intersection_x}, Y: {intersection_y}')
        return (intersection_x, intersection_y)
    
    def line_sweep_for_intersections(self, arcs_data):
        """
        Realiza un barrido de línea para encontrar las intersecciones relevantes de los arcos.
        """
        # Crear eventos (inicio y fin) para cada arco
        intersections = []  # Lista para almacenar las intersecciones válidas

        # Revisa todas las combinaciones de arcos
        for i, arc1 in enumerate(arcs_data):
            for j, arc2 in enumerate(arcs_data):
                if i == j:  
                    continue

                # Calcula la intersección entre los dos arcos
                intersection = self.calculate_intersection(arc1, arc2)
                if intersection is not None:
                    # Filtra intersecciones fuera del rango
                    if self.is_point_in_arc(intersection, arc1) and self.is_point_in_arc(intersection, arc2):
                        intersections.append(intersection)

        return intersections

    def filter_intersection_points(self, intersections, arcs_data):
        """
        Filtra los puntos de intersección que se encuentran en el lado cóncavo de un arco.

        Args:
            intersections (list of tuples): Lista de puntos de intersección (x, y).
            arcs_data (list of Arc): Datos de los arcos, cada uno con su centro, puntos inicial y final.

        Returns:
            list of tuples: Lista de puntos de intersección que están en el lado cóncavo.
        """
        def process_arcs(intersections, arcs_data):
            valid_points = []  # Lista para almacenar puntos válidos junto con sus arcos

            # Iterar sobre los arcos
            for i, arc1 in enumerate(arcs_data):
                center1, start1, _ = map(np.array, arc1)
                radius1 = np.linalg.norm(start1 - center1)

                for j, arc2 in enumerate(arcs_data):
                    if i >= j:  # Evitar comparar el mismo arco o duplicar comparaciones
                        continue
                    center2, start2, _ = map(np.array, arc2)
                    radius2 = np.linalg.norm(start2 - center2)

                    # Verificar cada punto de intersección
                    for intersection in intersections:
                        intersection_point = np.array(intersection)
                        distance1 = np.linalg.norm(intersection_point - center1)
                        distance2 = np.linalg.norm(intersection_point - center2)

                        # Si el punto está en ambos radios, es una intersección válida
                        if np.isclose(distance1, radius1, atol=1e-6) and np.isclose(distance2, radius2, atol=1e-6):
                            valid_points.append((tuple(intersection_point), arc1, arc2))

            return valid_points


        valid_intersections = process_arcs(intersections, arcs_data)

        final_valid_points = []  # Nueva lista para almacenar puntos finales con arcos

        for point, arc1, arc2 in valid_intersections:
            point_array = np.array(point)
            valid = True  # Bandera que asegura si el punto pasa la validación

            for arc in arcs_data:
                center, start, _ = map(np.array, arc)  # Obtener centro y radio
                radius = np.linalg.norm(start - center)
                distance_to_center = np.linalg.norm(point_array - center)

                if np.isclose(distance_to_center, radius, atol=1e-6):
                    valid = True
                    continue
                # Si el punto tiene radio menor al del arco, lo descartamos
                if distance_to_center < radius:
                    valid = False
                    break

            if valid:  # Si el punto es válido, lo guardamos junto con los arcos
                final_valid_points.append((point, arc1, arc2))

        return final_valid_points

    def sort_points_counterclockwise(self, points):
        """
        Ordena los puntos en sentido antihorario alrededor del centroide.
        Args:
            points (list): Lista de puntos ((x, y), arc1, arc2).
        Returns:
            list: Puntos ordenados en sentido antihorario.
        """
        # Extraer solo las coordenadas para el cálculo del centroide
        coordinates = [p[0] for p in points]  # p[0] es la tupla (x, y)
        center = np.mean(coordinates, axis=0)  # Centroide de las coordenadas

        # Ordenar los puntos utilizando las coordenadas (sin perder la estructura original)
        points = sorted(points, key=lambda p: np.arctan2(p[0][1] - center[1], p[0][0] - center[0]))
        return points
    
    def group_intersection_points(self, points):
        """
        Agrupa puntos de intersección en componentes conectadas (áreas) usando un grafo
        y asigna un identificador de área (area_id) a cada punto.

        Args:
            points (list): Lista de puntos con datos [(x, y), arco1, arco2].

        Returns:
            list: Lista de puntos con datos [(x, y), arco1, arco2, area_id].
        """

        def calculate_polygon_area(component_points):
            """Calcula el área de un polígono usando la fórmula del área de Shoelace."""
            n = len(component_points)
            area = 0
            for i in range(n):
                x1, y1 = component_points[i]
                x2, y2 = component_points[(i + 1) % n]  # Circularidad
                area += (x1 * y2) - (x2 * y1)
            return abs(area) / 2.0

        # Construir el grafo
        graph = defaultdict(list)
        for i in range(len(points)):
            _, arc1_i, arc2_i = points[i]
            for j in range(i + 1, len(points)):
                _, arc1_j, arc2_j = points[j]
                if arc1_i == arc1_j or arc1_i == arc2_j or arc2_i == arc1_j or arc2_i == arc2_j:
                    graph[i].append(j)
                    graph[j].append(i)

        # Algoritmo DFS para identificar componentes conectadas
        def dfs(node, visited, component):
            visited[node] = True
            component.append(node)
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    dfs(neighbor, visited, component)

        visited = [False] * len(points)
        components = []

        for i in range(len(points)):
            if not visited[i]:
                component = []
                dfs(i, visited, component)
                components.append(component)

        # Validar y asignar area_id a componentes
        area_id = 0
        valid_points = []

        for component in components:
            component_points = [points[idx][0] for idx in component]
            area = calculate_polygon_area(component_points)
            if area > 1e-3:  # Filtrar áreas insignificantes
                for idx in component:
                    valid_points.append((*points[idx], area_id))
                area_id += 1

        return valid_points