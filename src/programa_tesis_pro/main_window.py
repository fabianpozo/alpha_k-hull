import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QVBoxLayout, QLabel, QGraphicsTextItem, QGraphicsLineItem, QFileDialog, QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsPolygonItem
from PyQt5.QtGui import QPen, QColor, QBrush, QPainter, QTransform, QFont, QPainterPath, QPolygonF
from PyQt5.QtCore import Qt, QLineF, QRectF, QPointF
from design_GUI import Ui_MainWindow  # Importar el diseño generado por PyQt
from algoritmos import *  
import sys



class GraphicsView(QGraphicsView):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window  # Referencia al MainApp para actualizar la status bar y la consola
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)  # Suavizar las líneas
        self.setMouseTracking(True)  # Habilitar el seguimiento del mouse sin hacer clic
        self.point_items = []  # Lista para almacenar los puntos dibujados (punto y texto asociados)
        self.points_array = np.empty((0, 2))  # Matriz para almacenar coordenadas de puntos (NumPy)
        self.is_pencil_mode = None  # Estado para saber si estamos en modo lápiz, goma o ninguno

        self.grid_size = 20  # Tamaño de la cuadrícula
        self.grid_pen = QPen(QColor(200, 200, 200), 1)  # Color gris claro para la cuadrícula
        self.axis_pen = QPen(QColor(0, 0, 255), 2)  # Color azul para los ejes X e Y

        # Configuración inicial del zoom
        self.zoom_factor = 1.15  # Factor de escala para zoom in y zoom out

        # Definir los ejes para que las coordenadas funcionen como en un plano cartesiano
        transform = QTransform()
        transform.scale(1, 1)  # Asignación de signos a los ejes
        self.setTransform(transform)

        # Definir límites extremos de la pizarra
        self.MIN_VAL = -21474
        self.MAX_VAL = 21474

        self.undo_stack = []  # Historial de estados para deshacer
        self.redo_stack = []  # Historial de estados para rehacer

        # Establecer los límites de la escena
        self.setSceneRect(self.MIN_VAL, self.MIN_VAL, self.MAX_VAL - self.MIN_VAL, self.MAX_VAL - self.MIN_VAL)
        self.centerOn(0, 0)  # Asegurar que la vista se centre en (0, 0)

        self.hull_polygon_item = None  # Almacenará el polígono del Convex Hull
        self.point_counter = 1 #enumerar puntos intersecciones de los arcos computados

    def drawBackground(self, painter, rect):
        """Dibujar la cuadrícula y los ejes dentro de los límites definidos."""
        super().drawBackground(painter, rect)

        grid_size = self.grid_size

        # Limitar rectángulo al rango de la escena
        left = max(self.MIN_VAL, int(rect.left()) - (int(rect.left()) % grid_size))
        top = max(self.MIN_VAL, int(rect.top()) - (int(rect.top()) % grid_size))
        right = min(self.MAX_VAL, int(rect.right()))
        bottom = min(self.MAX_VAL, int(rect.bottom()))

        # Dibujar líneas de cuadrícula dentro de los límites
        painter.setPen(self.grid_pen)
        for x in range(left, right, grid_size):
            painter.drawLine(x, top, x, bottom)
        for y in range(top, bottom, grid_size):
            painter.drawLine(left, y, right, y)

        # Dibujar los ejes X e Y si están dentro de los límites
        painter.setPen(self.axis_pen)
        if left <= 0 <= right:
            painter.drawLine(0, top, 0, bottom)  # Eje Y
        if top <= 0 <= bottom:
            painter.drawLine(left, 0, right, 0)  # Eje X

    def resizeEvent(self, event):
        """Mantener la cuadrícula centrada cuando se redimensiona la ventana"""
        super().resizeEvent(event)
        self.centerOn(0, 0)  # Asegurar que la vista se mantenga centrada en (0, 0)

    def wheelEvent(self, event):
        """Controlar el zoom usando la rueda del mouse y la tecla Ctrl."""
        if event.modifiers() == Qt.ControlModifier:  # Verificar si Ctrl está presionado
            zoom_in_factor = 1.15
            zoom_out_factor = 1 / zoom_in_factor

            # Obtener la posición del cursor en la escena antes del zoom
            mouse_pos = self.mapToScene(event.pos())

            # Aplicar el factor de zoom
            if event.angleDelta().y() > 0:  # Zoom in
                self.scale(zoom_in_factor, zoom_in_factor)
            else:  # Zoom out
                self.scale(zoom_out_factor, zoom_out_factor)

            # Centrar la vista en la posición del mouse
            new_mouse_pos = self.mapToScene(event.pos())
            delta = new_mouse_pos - mouse_pos
            self.translate(delta.x(), delta.y())

        else:
            super().wheelEvent(event)  # Comportamiento estándar

    def mousePressEvent(self, event):
        """Lógica cuando se hace clic en la pizarra"""
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().setCursor(Qt.ClosedHandCursor)
            self.last_mouse_pos = event.pos()  # Inicializar la posición del mouse
        elif event.button() == Qt.LeftButton and self.is_pencil_mode is not None:
            pos = self.mapToScene(event.pos())
            pos.setY(-pos.y())
            if self.is_pencil_mode:
                self.add_point(pos)
            else:
                self.remove_point(pos)
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """Restaurar el estado normal después del desplazamiento"""
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().unsetCursor()
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        """Actualizar las coordenadas del mouse y desplazar la vista si está en modo arrastre."""
        if self.dragMode() == QGraphicsView.ScrollHandDrag and event.buttons() == Qt.MiddleButton:
            delta = event.pos() - self.last_mouse_pos
            self.last_mouse_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self.viewport().update()  # Forzar redibujado inmediato
        else:
            super().mouseMoveEvent(event)
            pos = self.mapToScene(event.pos())  # Actualizar coordenadas
            self.main_window.update_status_bar(pos.x(), -pos.y())

    def add_point(self, position):
        """Agregar un punto en la posición dada y actualizar la consola"""
        self.save_state()  # Guardar el estado actual antes de añadir un punto
        point_radius = 6
        pen = QPen(QColor(Qt.black))
        brush = QBrush(QColor(Qt.black))

        # Dibuja el punto con coordenada Y invertida para la visualización
        visual_y = -position.y()
        point = self.scene.addEllipse(
            position.x() - point_radius / 2,
            visual_y - point_radius / 2,
            point_radius,
            point_radius,
            pen,
            brush
        )

        # Dibujar el número del punto debajo de él
        point_number = QGraphicsTextItem(str(len(self.point_items) + 1))
        point_number.setFont(QFont("Arial", 10))
        point_number.setPos(position.x() - point_radius / 2, visual_y + point_radius / 2)
        point_number.setDefaultTextColor(Qt.black)

        # Añadir el punto y el número a la escena y almacenarlos en la lista
        self.point_items.append((point, point_number))
        self.scene.addItem(point_number)

        # Agregar el punto a la matriz NumPy sin invertir el eje Y
        new_point = np.array([[position.x(), position.y()]])  # Almacena la coordenada Y tal cual
        self.points_array = np.vstack([self.points_array, new_point])

        # Mostrar en la consola
        self.main_window.update_console(f"Punto {len(self.point_items)} añadido en: ({int(position.x())}, {int(position.y())})")

        # Recalcular el Convex Hull si está activada la opción de "Ejecutar Convex Hull"
        if self.main_window.ui.actionEjecutar_Convex_Hull.isChecked():
            self.main_window.run_convex_hull()

    def remove_point(self, position):
        """Eliminar un punto si está cerca de la posición clicada y actualizar la consola"""
        # Invertir la coordenada Y para que coincida con la visualización en la pizarra
        self.save_state()  # Guardar el estado actual antes de añadir un punto
        adjusted_position = QPointF(position.x(), -position.y())

        for i, (point_item, point_number_item) in enumerate(self.point_items):
            point_center = point_item.sceneBoundingRect().center()
            distance = (point_center - adjusted_position).manhattanLength()
            if distance < 10:  # Umbral para determinar si un punto está cerca
                # Eliminar el punto y el número de la escena
                self.scene.removeItem(point_item)
                self.scene.removeItem(point_number_item)

                # Mostrar en la consola el punto eliminado con su número
                point_number = point_number_item.toPlainText()  # Obtener el número del punto
                self.main_window.update_console(f"Punto {point_number} eliminado en: ({int(point_center.x())}, {int(-point_center.y())})")

                # Eliminar el punto correspondiente de la lista de puntos
                self.points_array = np.delete(self.points_array, i, axis=0)
                del self.point_items[i]
                break

        # Reenumerar los puntos restantes
        self.renumber_points()

        # Recalcular el Convex Hull si está activada la opción de "Ejecutar Convex Hull"
        if self.main_window.ui.actionEjecutar_Convex_Hull.isChecked():
            self.main_window.run_convex_hull()

    def renumber_points(self):
        """Actualizar los números de los puntos después de eliminar uno"""
        for i, (point_item, point_number_item) in enumerate(self.point_items):
            point_number_item.setPlainText(str(i + 1))

    def save_state(self):
        """Guardar el estado actual de la escena en el historial de deshacer."""
        state = []
        for item in self.scene.items():
            if isinstance(item, QGraphicsEllipseItem):  # Puntos
                state.append({
                    'type': 'ellipse',
                    'rect': item.rect(),
                    'pen': item.pen().color().name(),
                    'brush': item.brush().color().name(),
                })
            elif isinstance(item, QGraphicsTextItem):  # Texto asociado a puntos
                state.append({
                    'type': 'text',
                    'text': item.toPlainText(),
                    'position': item.pos(),
                    'color': item.defaultTextColor().name(),
                    'font': item.font().toString(),
                })
            elif isinstance(item, QGraphicsLineItem):  # Líneas
                line = item.line()
                state.append({
                    'type': 'line',
                    'p1': (line.x1(), line.y1()),
                    'p2': (line.x2(), line.y2()),
                    'pen': item.pen().color().name(),
                    'width': item.pen().width(),
                })
            elif isinstance(item, QGraphicsPolygonItem):  # Polígonos (áreas sombreadas)
                polygon = item.polygon()
                points = [(point.x(), point.y()) for point in polygon]
                state.append({
                    'type': 'polygon',
                    'points': points,
                    'brush': item.brush().color().name(),
                })
        self.undo_stack.append(state)
        self.redo_stack.clear()  # Limpiar el historial de rehacer


    def undo(self):
        """Deshacer la última acción."""
        if not self.undo_stack:
            return
        self.redo_stack.append(self._serialize_scene())  # Guardar el estado actual en rehacer
        state = self.undo_stack.pop()
        self._restore_state(state)

    def redo(self):
        """Rehacer la última acción deshecha."""

        if not self.redo_stack:
            return
        self.undo_stack.append(self._serialize_scene())  # Guardar el estado actual en deshacer
        state = self.redo_stack.pop()
        self._restore_state(state)

    def _serialize_scene(self):
        """Guardar el estado actual de la escena."""
        state = []
        for item in self.scene.items():
            if isinstance(item, QGraphicsEllipseItem):  # Puntos
                state.append({
                    'type': 'ellipse',
                    'rect': item.rect(),
                    'pen': item.pen().color().name(),
                    'brush': item.brush().color().name(),
                })
            elif isinstance(item, QGraphicsTextItem):  # Texto asociado a puntos
                state.append({
                    'type': 'text',
                    'text': item.toPlainText(),
                    'position': item.pos(),
                    'color': item.defaultTextColor().name(),
                    'font': item.font().toString(),
                })
            elif isinstance(item, QGraphicsLineItem):  # Líneas
                line = item.line()
                state.append({
                    'type': 'line',
                    'p1': (line.x1(), line.y1()),
                    'p2': (line.x2(), line.y2()),
                    'pen': item.pen().color().name(),
                    'width': item.pen().width(),
                })
            elif isinstance(item, QGraphicsPolygonItem):  # Polígonos (áreas sombreadas)
                polygon = item.polygon()
                points = [(point.x(), point.y()) for point in polygon]
                state.append({
                    'type': 'polygon',
                    'points': points,
                    'brush': item.brush().color().name(),
                })
        return state

    def _restore_state(self, state):
        """Reconstruir la escena a partir del estado guardado."""
        self.scene.clear()  # Limpiar la escena
        for item_data in state:
            if item_data['type'] == 'ellipse':  # Puntos
                ellipse = QGraphicsEllipseItem(item_data['rect'])
                ellipse.setPen(QPen(QColor(item_data['pen'])))
                ellipse.setBrush(QBrush(QColor(item_data['brush'])))
                self.scene.addItem(ellipse)
            elif item_data['type'] == 'text':  # Texto asociado
                text = QGraphicsTextItem(item_data['text'])
                text.setPos(item_data['position'])
                text.setDefaultTextColor(QColor(item_data['color']))
                font = QFont()
                font.fromString(item_data['font'])
                text.setFont(font)
                self.scene.addItem(text)
            elif item_data['type'] == 'line':  # Líneas
                line = QLineF(*item_data['p1'], *item_data['p2'])
                pen = QPen(QColor(item_data['pen']), item_data['width'])
                self.scene.addLine(line, pen)
            elif item_data['type'] == 'polygon':  # Polígonos
                polygon = QPolygonF([QPointF(x, y) for x, y in item_data['points']])
                item = QGraphicsPolygonItem(polygon)
                item.setBrush(QBrush(QColor(item_data['brush'])))
                self.scene.addItem(item)
                
    def draw_convex_hull(self, hull_indices):
        """Dibujar las líneas que conectan los puntos del Convex Hull en la pizarra."""
        points = self.points_array  # Acceder a la lista de puntos almacenados como NumPy array

        for i in range(len(hull_indices)):
            # Obtener los puntos usando los índices
            p1 = points[hull_indices[i - 1]]
            p2 = points[hull_indices[i]]

            # Crear una línea entre p1 y p2
            line = QLineF(p1[0], -p1[1], p2[0], -p2[1])
            pen = QPen(Qt.red, 2)
            self.scene.addLine(line, pen)

    def clear_convex_hull_lines(self):
        """Borra las líneas del Convex Hull que están actualmente dibujadas en la pizarra."""
        items = self.scene.items()
        for item in items:
            if isinstance(item, QGraphicsLineItem):
                self.scene.removeItem(item)

    def draw_convex_hull_polygon(self, hull_indices):
        """Dibujar el área achurada del Convex Hull en la pizarra."""
        points = self.points_array
        if self.hull_polygon_item:
            # Eliminar el polígono anterior antes de dibujar el nuevo
            self.scene.removeItem(self.hull_polygon_item)
        
        # Crear un polígono con los puntos del Convex Hull
        polygon = QtGui.QPolygonF()
        for index in hull_indices:
            polygon.append(QtCore.QPointF(points[index][0], -points[index][1]))
        
        # Crear el elemento QGraphicsPolygonItem y establecer el color de relleno
        self.hull_polygon_item = QtWidgets.QGraphicsPolygonItem(polygon)
        
        # Establecer un color semiopaco para el relleno
        brush = QBrush(QColor(255, 0, 0, 50))  # Rojo claro con 50 de opacidad (sutil)
        self.hull_polygon_item.setBrush(brush)
        
        # Añadir el polígono a la escena
        self.scene.addItem(self.hull_polygon_item)

    def draw_akhull(self, lines_and_arcs):
        """
        Dibuja las líneas y arcos calculados para el (alpha, k)-hull.
        lines_and_arcs: iterable con líneas o arcos.
        """
        for element in lines_and_arcs:
            if isinstance(element, np.ndarray):
                # Aquí dibujamos líneas, ya que las líneas se representan como dos puntos.
                p1, p2 = element
                line = QLineF(QPointF(p1[0], p1[1]), QPointF(p2[0], p2[1]))
                self.main_window.update_console(f"Dibujando línea entre puntos: {p1} y {p2}")  # Depuración
                self.scene.addLine(line, QPen(QColor(Qt.red), 2))    #self.graphics_view.scene.addLine(line, QPen(QColor(Qt.red), 2))

            elif isinstance(element, Arc):  # Adaptado para los arcos
                # Calcular el centro y el radio del arco
                center = element.circle_center()
                radius = np.linalg.norm(center - element.p)

                 # Depurar el centro y el radio
                self.main_window.update_console(f"Dibujando arco con centro en: {center}, radio: {radius}")

                
                # Convertir a coordenadas de pantalla
                q_center = QPointF(center[0], center[1])
                p1 = QPointF(element.p[0], element.p[1])
                p2 = QPointF(element.q[0], element.q[1])

                # Calcular el ángulo inicial y la amplitud del arco
                start_angle = np.degrees(angle_360(center, element.p))
                arc_length = np.degrees(angle_to(center, element.p, element.q))

                 # Depurar ángulo inicial y longitud del arco
                self.main_window.update_console(f"Ángulo inicial: {start_angle}, Longitud del arco: {arc_length}")

                # Definir el rectángulo donde estará el arco
                rect = QRectF(q_center.x() - radius, q_center.y() - radius, 2 * radius, 2 * radius)

                # Dibujar el arco usando `drawArc`
                arc_item = self.scene.addArc(rect, start_angle * 16, arc_length * 16, QPen(QColor(Qt.blue), 2))
    #arc_item = self.graphics_view.scene.addArc(rect, start_angle * 16, arc_length * 16, QPen(QColor(Qt.blue), 2))
                self.graphics_view.scene.addItem(arc_item)

#------------------------------------------------------------------------------------------------ draw wedge
    def draw_wedges(self, wedges_data, alpha):
        """
        Dibuja las cuñas en la escena, ajustando la dirección de acuerdo con el ángulo alpha y enumerándolas.
        """
        # Inicializar la lista de elementos de las cuñas si no existe
        if not hasattr(self, "wedge_items"):
            self.wedge_items = []

        color_index = 0
        wedge_colors = [QColor("#FF0000"), QColor("#00FF00"), QColor("#0000FF"), QColor("#FFA500")]

        for index, wedge in enumerate(wedges_data, start=1):
            center_x, center_y = wedge['wedge']
            inver_x = wedge.get('inverX', False)
            inver_y = wedge.get('inverY', True)
            center = QPointF(center_x, -center_y)

            # Seleccionar la longitud de la línea en función del estado del botón "extender cuñas"
            if self.main_window.extend_wedges_enabled:
                # Usar los puntos `pa_s` y `pb` para calcular la longitud
                pa_s = wedge.get('pa_s', [[center_x, center_y]])[0]
                pb = wedge.get('pb', [center_x, center_y])
                length_dir1 = np.linalg.norm(np.array([pa_s[0], -pa_s[1]]) - np.array([center_x, -center_y]))
                length_dir2 = np.linalg.norm(np.array([pb[0], -pb[1]]) - np.array([center_x, -center_y]))
            else:
                # Usar longitud predeterminada si no está activado el botón
                length_dir1 = self.grid_size
                length_dir2 = self.grid_size

            # Definir las direcciones de las líneas
            dir_line1 = np.array([length_dir1, 0])
            angle_rad = np.radians(alpha)
            dir_line2 = np.array([length_dir2 * np.cos(angle_rad), length_dir2 * np.sin(angle_rad)])

            # Invertir coordenadas si es necesario
            if inver_x:
                dir_line1[0] = -dir_line1[0]
                dir_line2[0] = -dir_line2[0]
            if inver_y:
                dir_line1[1] = -dir_line1[1]
                dir_line2[1] = -dir_line2[1]

            # Dibujar las líneas de la cuña
            pen = QPen(wedge_colors[color_index % len(wedge_colors)], 2)
            color_index += 1
            line1 = self.scene.addLine(center.x(), center.y(), center.x() + dir_line1[0], center.y() + dir_line1[1], pen)
            line2 = self.scene.addLine(center.x(), center.y(), center.x() + dir_line2[0], center.y() + dir_line2[1], pen)

            # Agregar las líneas a la lista de elementos de las cuñas
            self.wedge_items.append(line1)
            self.wedge_items.append(line2)

            # Agregar el número de la cuña debajo
            wedge_number = QGraphicsTextItem(str(index))
            wedge_number.setFont(QFont("Arial", 10))
            wedge_number.setDefaultTextColor(pen.color())
            wedge_number.setPos(center.x() - 5, center.y() + 10)
            self.scene.addItem(wedge_number)
            # Agregar el número a la lista de elementos de las cuñas
            self.wedge_items.append(wedge_number)

    def correct_orientation(center, p1, p2):
        # Calcula el producto cruzado para determinar la orientación
        v1 = np.array([p1[0] - center[0], p1[1] - center[1]])
        v2 = np.array([p2[0] - center[0], p2[1] - center[1]])
        cross_product = np.cross(v1, v2)
        return cross_product < 0  # True si es horario, False si es antihorario

    def draw_wedge_arcs(self, arcs_data, alpha):
        """
        Dibuja los arcos de cada cuña en la escena.
        """
        self.datos_de_los_arcos = []
        points = self.points_array  # Puntos actuales en la distribución

        for arc in arcs_data:
            print("\nDatos del arco a dibujar:", arc)
            center = [arc[0][0], arc[0][1]]  # Centro de la cuña
            p1 = [arc[1][0], arc[1][1]]  # Punto inicial del arco
            p2 = [arc[2][0], arc[2][1]]  # Punto final del arco

            # Buscar el punto correspondiente al centro de la cuña
            matching_point = None
            for point in points:
                if np.isclose(point[0], p1[0], atol=1e-6):
                    matching_point = point
                    break

            if matching_point is not None:
                # Si encontramos un punto que corresponde al centro, lo usamos como inicio
                p1 = [matching_point[0], matching_point[1]]
                print(f"El arco empezará desde el punto correspondiente al centro: {p1}")

            # Calcular el radio
            norm = lambda x, y: np.sqrt(x**2 + y**2)
            radius = norm(center[0] - p1[0], center[1] - p1[1])

            # Calcular ángulo inicial
            dx_start = p1[0] - center[0]
            dy_start = p1[1] - center[1]
            start_angle_rad = np.arctan2(dy_start, dx_start)
            start_angle = np.degrees(start_angle_rad) % 360
            print(f"draw arc start: punto {p1}, angulo inicial: {start_angle}")

            # Calcular ángulo final
            dx_end = p2[0] - center[0]
            dy_end = p2[1] - center[1]
            end_angle_rad = np.arctan2(dy_end, dx_end)
            end_angle = np.degrees(end_angle_rad) % 360
            print(f"draw arc end: punto {p2}, angulo inicial: {end_angle}")

            # Ajustar la extensión del arco
            if start_angle <= end_angle:
                span_angle = end_angle - start_angle
            else:
                span_angle = 360 - (start_angle - end_angle)

            print(f"span angle: {span_angle}")

            # Invertir Y solo al definir el rectángulo para PyQt
            center_inv = [center[0], -center[1]]

            # Crear un arco en PyQt5
            rect = QRectF(center_inv[0] - radius, center_inv[1] - radius, 2 * radius, 2 * radius)
            path = QPainterPath()
            path.arcMoveTo(rect, start_angle)
            path.arcTo(rect, start_angle, span_angle)

            # Dibujar el arco
            graphics_arc = QGraphicsPathItem(path)
            graphics_arc.setPen(QPen(Qt.red, 2))
            self.scene.addItem(graphics_arc)

            self.datos_de_los_arcos.append((center, p1, p2))  # Guardamos centro, p1 y p2

    def add_intersection_point(self, point):
        """
        Agrega un punto de intersección a la escena con un color distintivo.
        """
        x, y = [point[0], -point[1]]
        radius = 6
        pen = QPen(QColor("#8a2be2"))  # Color magenta para intersecciones
        brush = QBrush(QColor("#FF00FF"))
        self.scene.addEllipse(x - radius / 2, y - radius / 2, radius, radius, pen, brush)


        # Crear un texto con el número del punto
        text_item = QGraphicsTextItem(str(self.point_counter))
        text_item.setFont(QFont("Arial", 10))  # Fuente para el número
        text_item.setDefaultTextColor(QColor("#FF00FF"))  # Color del texto (igual que el punto)
        text_item.setPos(x + radius / 2, y + radius / 2)  # Posición del texto cerca del punto
        self.scene.addItem(text_item)

        # Incrementar el contador para el siguiente punto
        self.point_counter += 1


    def draw_alpha_k_area(self, points):
        """
        Dibuja el área sombreada del polígono solucionado a partir de puntos y arcos.
        Args:
            points (list): Lista de puntos [(x, y)] filtrados.
        """
        pen = QPen(Qt.green, 2)
        brush = QBrush(QColor(0, 255, 0, 50))  # Color verde semitransparente para sombrear


        # Agrupar los puntos por area_id
        grouped_by_area = defaultdict(list)
        for point in points:
            coords, arc1, arc2, area_id = point
            grouped_by_area[area_id].append((coords, arc1, arc2))

        total_area = 0.0  # Para almacenar el área total
        discretized_points = []  # Lista global de puntos discretizados para todas las áreas

        # Función para calcular la diferencia angular mínima
        def angle_difference(a1, a2):
            diff = (a2 - a1) % 360
            return diff if diff <= 180 else diff - 360  # Asegura la menor distancia angular

        # Dibujar cada área separadamente
        for area_id, group in grouped_by_area.items():
            area_discretized_points = []  # Puntos discretizados para esta área específica

            for i in range(len(group)):
                current_coords, current_arc1, current_arc2 = group[i]

                # Conectar el último punto al primero
                if i == len(group) - 1:
                    next_coords, next_arc1, next_arc2 = group[0]  # Conectar al primer punto
                else:
                    next_coords, next_arc1, next_arc2 = group[i + 1]

                # Encontrar el arco compartido
                matching_arc = None
                for arc in [current_arc1, current_arc2]:
                    if arc == next_arc1 or arc == next_arc2:
                        matching_arc = arc
                        break

                if not matching_arc:
                    continue

                # Extraer centro y radio
                center, arc_start, _ = matching_arc
                radius = np.linalg.norm(np.array(center) - np.array(arc_start))

                # Calcular ángulos
                angle_start = np.degrees(np.arctan2(current_coords[1] - center[1], current_coords[0] - center[0])) % 360
                angle_end = np.degrees(np.arctan2(next_coords[1] - center[1], next_coords[0] - center[0])) % 360

                # Calcular la diferencia angular mínima
                span_angle = angle_difference(angle_start, angle_end)

                # Crear y dibujar el arco
                rect = QRectF(center[0] - radius, -center[1] - radius, 2 * radius, 2 * radius)
                path = QPainterPath()
                path.arcMoveTo(rect, angle_start)
                path.arcTo(rect, angle_start, span_angle)
                self.scene.addPath(path, pen)

                # Discretizar el arco para esta área
                resolution = 1000
                angles = np.linspace(angle_start, angle_start + span_angle, resolution)
                for angle in angles:
                    angle_rad = np.radians(angle)
                    x = center[0] + radius * np.cos(angle_rad)
                    y = center[1] + radius * np.sin(angle_rad)
                    area_discretized_points.append((x, y))

            # Agregar los puntos discretizados de esta área a la lista general
            discretized_points.extend(area_discretized_points)

            # Calcular el área del polígono curvilíneo para esta área
            polygon = Polygon(area_discretized_points)
            total_area += polygon.area

            # Dibujar el área sombreada para esta área
            polygon_qt = QtGui.QPolygonF([QtCore.QPointF(x, -y) for x, y in area_discretized_points])
            polygon_item = QGraphicsPolygonItem(polygon_qt)
            polygon_item.setBrush(brush)
            polygon_item.setPen(QPen(Qt.NoPen))
            self.scene.addItem(polygon_item)

        # Mostrar el área total en la consola
        self.main_window.update_console(f"Área total del polígono solución: {total_area:.2f} unidades cuadradas")

    def draw_solution_area(self, points):
        """
        Dibuja el área sombreada del polígono solución a partir de los puntos discretizados.
        Args:
            points (list): Lista de puntos [(x, y)] discretizados.
        """
        # Crear un polígono a partir de los puntos discretizados
        polygon = QtGui.QPolygonF([QtCore.QPointF(x, -y) for x, y in points])

        # Crear el elemento QGraphicsPolygonItem
        polygon_item = QGraphicsPolygonItem(polygon)

        # Establecer un color semiopaco para el área sombreada
        brush = QBrush(QColor(0, 255, 0, 50))  # Verde con opacidad
        polygon_item.setBrush(brush)

        # Añadir el polígono a la escena
        self.scene.addItem(polygon_item)
""""
                                            Con este fragmento de codigo podemos hacer que se achure el area en vez que se cree un sombreado (no se actualiza bien)
def draw_convex_hull(self, hull_indices):
    #Dibujar las líneas que conectan los puntos del Convex Hull en la pizarra y aplicar achurado en el área.
    points = self.points_array  # Acceder a la lista de puntos almacenados como NumPy array

    # Crear un polígono a partir de los puntos del Convex Hull
    polygon = QtGui.QPolygonF()
    for i in hull_indices:
        polygon.append(QtCore.QPointF(points[i][0], points[i][1]))

    # Definir un pincel con un patrón de achurado
    brush = QBrush(Qt.Dense4Pattern)  # Cambiar Dense4Pattern a otro patrón si lo deseas

    # Añadir el polígono achurado a la escena
    self.scene.addPolygon(polygon, QPen(Qt.red, 2), brush)

    # También dibujar las líneas del Convex Hull
    for i in range(len(hull_indices)):
        p1 = points[hull_indices[i - 1]]
        p2 = points[hull_indices[i]]

        # Crear una línea entre p1 y p2
        line = QLineF(p1[0], p1[1], p2[0], p2[1])
        pen = QPen(Qt.red, 2)
        self.scene.addLine(line, pen)

"""


class MainApp(QMainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        self.ui = Ui_MainWindow()  # Instancia del diseño
        self.ui.setupUi(self)  # Configurar la interfaz

        # Inicializamos el GraphicsView personalizado
        self.graphics_view = GraphicsView(self)  # Pasamos una referencia de MainApp a GraphicsView
        # Añadir el GraphicsView al layout "Pizarra" del archivo de diseño
        self.ui.Pizarra.addWidget(self.graphics_view)

        # Definir valores predeterminados
        self.alpha = 90.0  # Alpha inicial
        self.k = 3         # K inicial
        self.wedges_data = None  # Inicializar un contenedor para guardar las cuñas calculadas
        self.currentWedge = None  # Inicializar la cuña activa



        self.update_console(f"Inicialización: k = {self.k}, alpha = {self.alpha}") # Mostrar por consola las variables de k y alpha por default

        # Conectar las acciones de la barra de herramientas
        self.ui.actionPuntos.triggered.connect(self.toggle_pencil_mode)      # Lapiz
        self.ui.actionBorrardor.triggered.connect(self.toggle_eraser_mode)   # Goma
        self.ui.actionBorrar_todos_los_puntos.triggered.connect(self.clear_all_points)  # Basurero
        #self.ui.actionEjecutar_Convex_Hull.triggered.connect(self.run_convex_hull) 
        self.ui.actionEjecutar_Convex_Hull.triggered.connect(self.toggle_convex_hull)  # Conectar el botón navbar convex hull
        self.ui.actionLimpiar_algoritmo_ejecutado.triggered.connect(self.clear_algorithm) #Conectar boton de limpiar algoritmo ejecutado
        self.ui.actionModificar_Alpha.triggered.connect(self.set_alpha) #conectar boton de cambiar valor de alpha
        self.ui.actionModificar_K.triggered.connect(self.set_k) #conectar boton de cambiar valor de k
        self.ui.actionEjecutar_alpha_k_hull.triggered.connect(self.run_solution_area) #Computar (alpha,k)-hull
        self.ui.actionMostrar_Cu_as.triggered.connect(self.show_wedges) #mostrar cuñas
        self.ui.actionComputar_Cu_as.triggered.connect(self.compute_wedges) #,mostrar arcos de las cuñas
        self.ui.actionGuardar_puntos.triggered.connect(self.save_points) #Conectar acciones de cargar puntos
        self.ui.actionCargar_puntos.triggered.connect(self.load_points) #Conectar acciones de guardar puntos
        self.ui.actionExtender_Cu_as.triggered.connect(self.toggle_extend_wedges) # Conectar el botón de extender cuñas
        self.ui.actionBorrar_Cu_as.triggered.connect(self.toggle_hide_wedges) #Conectar el botón de ocultar cuñas 
        self.ui.actionEjecutar_Line_Sweep_Barrido_de_plano.triggered.connect(self.run_line_sweep) #Conectar boton de line sweep
        self.ui.actionDeshacer.triggered.connect(self.undo) #conectar boton de deshacer
        self.ui.actionRehacer.triggered.connect(self.redo) #conectar boton de rehacer
        self.ui.actionSeleccionar_cu_a_a_computar.triggered.connect(self.select_wedge_to_compute) #conectar boton de seleccionar cuña a computar

    
        # Nueva variable para saber si el Convex Hull está activado
        self.convex_hull_active = False  # Por defecto no activado

        self.extend_wedges_enabled = False  # Estado inicial de extender cuñas
        # Desactivar ambas herramientas por defecto
        self.ui.actionPuntos.setChecked(False)
        self.ui.actionBorrardor.setChecked(False)
        self.graphics_view.is_pencil_mode = None  # Ningún modo está activo inicialmente

        # Añadir un QLabel a la barra de estado para mostrar las coordenadas
        self.coordinates_label = QLabel("Coordenadas: (0, 0)")
        self.ui.statusBar.addPermanentWidget(self.coordinates_label)


    def load_points(self):
        """Cargar puntos desde un archivo .txt y agregarlos a la escena."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Cargar Puntos", "", "Text Files (*.txt)")
        if file_path:
            try:
                # Limpiar todos los puntos actuales antes de cargar los nuevos
                self.clear_all_points()

                with open(file_path, 'r') as file:
                    for line in file:
                        # Separar cada línea en dos valores de coordenadas
                        x_str, y_str = line.strip().split()
                        x, y = float(x_str), float(y_str)  # Convertir a float

                        # Crear el punto en la escena
                        pos = QPointF(x, y)
                        self.graphics_view.add_point(pos)

                self.update_console("Puntos cargados correctamente desde el archivo.")
            except Exception as e:
                self.update_console(f"Error al cargar puntos: {e}")

    def save_points(self):
        """Guardar puntos actuales en un archivo .txt"""
        file_path, _ = QFileDialog.getSaveFileName(self, "Guardar Puntos", "", "Text Files (*.txt)")
        if file_path:
            try:
                with open(file_path, "w") as file:
                    for point in self.graphics_view.points_array:
                        file.write(f"{point[0]} {point[1]}\n") #se invierte el eje y para que se guarden bien los puntos 
                self.update_console(f"Puntos guardados en {file_path}.")
            except Exception as e:
                self.update_console(f"Error al guardar puntos: {e}")


    def set_alpha(self):
        # Solicitar nuevo valor de alpha
        value, ok = QtWidgets.QInputDialog.getDouble(self, "Cambiar Alpha", "Nuevo valor de Alpha:", self.alpha, 0, 360, 2)
        if ok:
            self.alpha = value
            self.update_console(f"Alpha actualizado a {self.alpha} grados")

    def set_k(self):
        # Solicitar nuevo valor de k
        value, ok = QtWidgets.QInputDialog.getInt(self, "Cambiar K", "Nuevo valor de K:", self.k, 1, 100)
        if ok:
            self.k = value
            self.update_console(f"K actualizado a {self.k}")


    def undo(self):
        # Implementar el algoritmo de deshacer aquí
        print("Deshacer acción")
        # Aquí puedes llamar a métodos de graphics_view si es necesario
        self.graphics_view.undo()

    def redo(self):
        # Implementar el algoritmo de rehacer aquí
        print("Rehacer acción")
        # Aquí puedes llamar a métodos de graphics_view si es necesario
        self.graphics_view.redo()
        
    def toggle_pencil_mode(self):
        """Activar o desactivar el modo lápiz para dibujar puntos"""
        if self.ui.actionPuntos.isChecked():
            self.ui.actionBorrardor.setChecked(False)
            self.graphics_view.is_pencil_mode = True
        else:
            self.graphics_view.is_pencil_mode = None

    def toggle_eraser_mode(self):
        """Activar o desactivar el modo goma para borrar puntos"""
        if self.ui.actionBorrardor.isChecked():
            self.ui.actionPuntos.setChecked(False)
            self.graphics_view.is_pencil_mode = False
        else:
            self.graphics_view.is_pencil_mode = None

    def toggle_convex_hull(self):
        """Activar o desactivar el cálculo dinámico del Convex Hull"""
        self.convex_hull_active = self.ui.actionEjecutar_Convex_Hull.isChecked()
        if self.convex_hull_active:
            self.run_convex_hull()  # Ejecutar Convex Hull inicialmente
        else:
            self.clear_convex_hull()  # Limpiar las líneas del Convex Hull cuando se desactiva

    def clear_wedges(self):
        """Borra solo las cuñas actuales de la escena, dejando los números de los puntos intactos."""
        items = self.graphics_view.scene.items()
        for item in items:
            # Borra solo líneas de las cuñas (y no los textos que muestran los números de los puntos)
            if isinstance(item, QGraphicsLineItem):
                self.graphics_view.scene.removeItem(item)

    def toggle_extend_wedges(self):
        """Alterna el estado del botón de extender cuñas y redibuja las cuñas con el nuevo estado."""
        self.extend_wedges_enabled = self.ui.actionExtender_Cu_as.isChecked()
        self.update_console(f"Extender cuñas {'activado' if self.extend_wedges_enabled else 'desactivado'}.")
        
        # Borrar cuñas anteriores
        self.clear_wedges()

        # Redibujar cuñas con el nuevo estado
        if self.wedges_data is not None:
            self.graphics_view.draw_wedges(self.wedges_data, self.alpha)

    def hide_wedges(self):
        """Ocultar todas las cuñas actualmente dibujadas en la escena."""
        if self.wedges_data is not None:
            # Borrar todas las líneas y números asociados con las cuñas
            items = self.graphics_view.scene.items()
            for item in items:
                # Identificar si el elemento es parte de las cuñas (líneas o números)
                if hasattr(item, 'is_wedge_item') and item.is_wedge_item:
                    self.graphics_view.scene.removeItem(item)

    def toggle_hide_wedges(self):
        """Alternar la visibilidad de las cuñas dibujadas."""
        if self.ui.actionBorrar_Cu_as.isChecked():
            # Ocultar cuñas (hacer invisibles los elementos dibujados)
            if hasattr(self.graphics_view, "wedge_items"):
                for item in self.graphics_view.wedge_items:
                    item.setVisible(False)
            self.update_console("Cuñas ocultadas.")
        else:
            # Mostrar cuñas (hacer visibles los elementos ya dibujados)
            if hasattr(self.graphics_view, "wedge_items"):
                for item in self.graphics_view.wedge_items:
                    item.setVisible(True)
            self.update_console("Cuñas mostradas nuevamente.")

#anteriores actualizadores de la status bar

    def update_status_bar(self, x, y):
       #Actualizar las coordenadas del mouse en la barra de estado"""
        if self.coordinates_label:
            self.coordinates_label.setText(f"Coordenadas: ({int(x)}, {int(y)})")


    def update_console(self, text):
        """Actualizar la consola con un nuevo mensaje"""
        self.ui.textEdit.append(text)
        self.ui.textEdit.setReadOnly(True)
        
    def clear_all_points(self):
        """Borrar todos los puntos, números, cuñas y líneas del algoritmo de la escena."""
    # Borrar todos los puntos y sus números
        for point_item, point_number_item in self.graphics_view.point_items:
            self.graphics_view.scene.removeItem(point_item)
            self.graphics_view.scene.removeItem(point_number_item)

        # Limpiar la lista de puntos
        self.graphics_view.point_items.clear()

        # Limpiar la matriz NumPy
        self.graphics_view.points_array = np.empty((0, 2))

        # Borrar puntos de intersección y sus números (color magenta)
        items = self.graphics_view.scene.items()
        for item in items:
            if isinstance(item, QGraphicsEllipseItem) or isinstance(item, QGraphicsTextItem):
                # Verificar si el color coincide con el magenta (intersección)
                if isinstance(item, QGraphicsEllipseItem) and item.pen().color() == QColor("#8a2be2"):
                    self.graphics_view.scene.removeItem(item)
                elif isinstance(item, QGraphicsTextItem) and item.defaultTextColor() == QColor("#FF00FF"):
                    self.graphics_view.scene.removeItem(item)
                    
        # Borrar las cuñas, incluyendo líneas y números
        if hasattr(self.graphics_view, "wedge_items"):
            for item in self.graphics_view.wedge_items:
                self.graphics_view.scene.removeItem(item)
            self.graphics_view.wedge_items.clear()  # Limpiar la lista de cuñas

        # Borrar los arcos de las cuñas
        items = self.graphics_view.scene.items()
        for item in items:
            if isinstance(item, QGraphicsPathItem):
                self.graphics_view.scene.removeItem(item)

        # Borrar las líneas del algoritmo ejecutado (Convex Hull, etc.)
        for item in items:
            if isinstance(item, QGraphicsLineItem):
                self.graphics_view.scene.removeItem(item)

        # Borrar el área achurada del Convex Hull (si existe)
        if self.graphics_view.hull_polygon_item:
            self.graphics_view.scene.removeItem(self.graphics_view.hull_polygon_item)
            self.graphics_view.hull_polygon_item = None

        # Limpiar el texto de la consola
        self.ui.textEdit.clear()  # Esto borra todo el texto escrito en el widget de consola.
        
        # Actualizar la consola
        self.update_console("Todos los objetos dibujados han sido borrados.")


    def clear_algorithm(self):
        """Borrar solo las líneas generadas por el algoritmo ejecutado (Convex Hull) pero dejar los puntos en su lugar."""
        # Obtener todos los elementos en la escena
        items = self.graphics_view.scene.items()

        # Recorrer los elementos y eliminar solo las líneas (que son del tipo QGraphicsLineItem)
        for item in items:
            if isinstance(item, QtWidgets.QGraphicsLineItem):
                self.graphics_view.scene.removeItem(item)

        # Actualizar la consola
        self.update_console("Las líneas del algoritmo ejecutado han sido borradas.")

    def clear_convex_hull(self):
        """Borrar las líneas y el área del Convex Hull"""
        # Limpiar las líneas del Convex Hull
        self.graphics_view.clear_convex_hull_lines()

        # Limpiar el área achurada del Convex Hull (el polígono)
        if self.graphics_view.hull_polygon_item:
            self.graphics_view.scene.removeItem(self.graphics_view.hull_polygon_item)
            self.graphics_view.hull_polygon_item = None

        # Actualizar la consola
        self.update_console("El Convex Hull ha sido desactivado y borrado.")


    def clear_convex_hull_lines(self):
        """Borra las líneas del Convex Hull que están actualmente dibujadas en la pizarra."""
        items = self.scene.items()
        for item in items:
            if isinstance(item, QGraphicsLineItem):
                self.scene.removeItem(item)

    def run_convex_hull(self):
        """Ejecuta el algoritmo Convex Hull usando Graham Scan y lo visualiza"""
        points_np = self.graphics_view.points_array
        
        if len(points_np) < 3:
            self.update_console("No se puede calcular el Convex Hull con menos de 3 puntos.")
            return

        # Limpiar las líneas anteriores del Convex Hull
        self.graphics_view.clear_convex_hull_lines()

        hull = graham_scan(points_np)  # Calcular el Convex Hull

        # Dibujar el área achurada del Convex Hull
        self.graphics_view.draw_convex_hull_polygon(hull)

        # Dibujar el Convex Hull en la pizarra de PyQt5
        self.graphics_view.draw_convex_hull(hull)

        # Obtener los puntos que forman el Convex Hull
        hull_points = points_np[hull]

        # Calcular el área del Convex Hull
        area = self.calcular_area_convex_hull(hull_points)

        # Mostrar el área y los puntos en la consola
        self.update_console(f"Área del Convex Hull: {area:.2f}")
        
        # Mostrar los puntos del Convex Hull en la consola
        puntos_str = ", ".join([f"({int(p[0])}, {int(-p[1])})" for p in hull_points])
        self.update_console(f"Puntos del Convex Hull: {puntos_str}")



    def calcular_area_convex_hull(self,puntos_hull):
        """Calcular el área del Convex Hull usando la fórmula del área de un polígono (Shoelace formula)."""
        n = len(puntos_hull)
        area = 0.0
        for i in range(n):
            x1, y1 = puntos_hull[i]
            x2, y2 = puntos_hull[(i + 1) % n]  # Siguiente punto, con % para volver al primer punto al final
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0

    def run_akhull(self):
        points_np = self.graphics_view.points_array  # Obtener los puntos actuales de la escena

        # Depurar el número de puntos
        self.update_console(f"Ejecutando (alpha, k)-hull con {len(points_np)} puntos.")
        
        if len(points_np) < 3:
            self.update_console("No se puede calcular el (alpha, k)-hull con menos de 3 puntos.")
            return

        # Crear instancia del algoritmo AKHull
        akhull = AKHull(self.k, self.alpha)  # Usar los valores de k y alpha configurados dinámicamente
        akhull.set_points(points_np)

        # Limpiar las líneas anteriores del (alpha, k)-hull
        self.graphics_view.clear_convex_hull_lines()

        # Calcular las líneas y arcos
        lines_and_arcs = list(akhull.compute_akhull())

        # Dibujar las líneas y arcos en la interfaz
        self.graphics_view.draw_akhull(lines_and_arcs)

          # Depuración final
        self.update_console(f"Terminó el cálculo del (alpha, k)-hull. Total de elementos dibujados: {len(lines_and_arcs)}")

#-----------wedge
    def show_wedges(self):
        """Obtiene y muestra las cuñas en la escena."""
        points_np = self.graphics_view.points_array
        if len(points_np) < 3:
            self.update_console("No se puede calcular las cuñas con menos de 3 puntos.")
            return

        wedge_calculator = WedgeCalculator(self.k, self.alpha, points_np)
        self.wedges_data = wedge_calculator.out_get_wedges_data()

        # Asignar la primera cuña como la activa por defecto
        if self.wedges_data:
            self.currentWedge = self.wedges_data[0]  # Cuña activa por defecto
            self.update_console("Primera cuña seleccionada como activa.")
        
        # Llamar a draw_wedges en GraphicsView para dibujar las cuñas
        self.graphics_view.draw_wedges(self.wedges_data, self.alpha)
        self.update_console("Cuñas mostradas en la escena.")

    def select_wedge_to_compute(self):
        if not self.wedges_data:
            self.update_console("No hay cuñas disponibles para seleccionar. Por favor, calcule las cuñas primero.")
            return

        # Crear una lista con las opciones de las cuñas disponibles
        wedge_options = [f"Cuña {i+1}" for i in range(len(self.wedges_data))]

        # Mostrar un cuadro de diálogo para seleccionar la cuña
        selected_option, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Seleccionar Cuña",
            "Seleccione la cuña para computar:",
            wedge_options,
            0,  # Selección predeterminada
            False  # No editable
        )

        if ok and selected_option:
            # Determinar el índice de la cuña seleccionada
            selected_index = wedge_options.index(selected_option)
            self.currentWedge = self.wedges_data[selected_index]
            self.update_console(f"Cuña seleccionada para computar: {selected_option}")



    def compute_wedges(self):
        """Calcula y dibuja los arcos de las cuñas en la escena."""
        points_np = self.graphics_view.points_array
        if len(points_np) < 3:
            self.update_console("No se puede calcular las cuñas con menos de 3 puntos.")
            return
        
        if self.currentWedge is None:
            self.update_console("No hay una cuña activa seleccionada.")
            return
        
        # Instanciar el WedgeCalculator con los puntos actuales
        wedge_calculator = WedgeCalculator(self.k, self.alpha, points_np)
        self.wedges_data = wedge_calculator.out_get_wedges_data(test=False)  # Generar datos de las cuñas

        # Calcular los arcos para la cuña activa
        arcs_data = wedge_calculator.out_compute_one_wedge(self.currentWedge)
    
        print("out_arcs_data:\n", arcs_data)
        
        # Dibujar los arcos de las cuñas en la escena
        self.graphics_view.draw_wedge_arcs(arcs_data, self.alpha)
        self.update_console(f"Arcos de las cuñas calculados y mostrados en la escena. Total: {len(arcs_data)}")

    def run_line_sweep(self):
        """
        Ejecuta el barrido de línea para detectar intersecciones de los arcos y visualiza los puntos resultantes.
        """
            # Verificar si hay datos de arcos disponibles
        if not hasattr(self.graphics_view, "datos_de_los_arcos") or not self.graphics_view.datos_de_los_arcos:
            self.update_console("No hay datos de arcos disponibles para procesar.")
            return

        # Obtener los datos de los arcos desde el atributo correspondiente
        arcs = self.graphics_view.datos_de_los_arcos

        # Crear una instancia de WedgeCalculator para usar su función line_sweep_for_intersections
        wedge_calculator = WedgeCalculator(self.k, self.alpha, self.graphics_view.points_array)

        # Ejecutar el barrido de línea para detectar intersecciones
        intersections = wedge_calculator.line_sweep_for_intersections(arcs)
        
        filter_point = wedge_calculator.filter_intersection_points(intersections, arcs)
        
        # Ordenar puntos filtrados
        sorted_points = wedge_calculator.sort_points_counterclockwise(filter_point)
        print(f'puntos sorteados: {sorted_points}')

        # Dibujar los puntos de intersección en la pizarra
        for intersection in sorted_points:
            intersection_point = intersection[0]
            self.graphics_view.add_intersection_point(intersection_point)

        # Mostrar en la consola cuántas intersecciones se encontraron
        self.update_console(f"Se encontraron {len(intersections)} intersecciones.")
        self.update_console(f"despues del filtrado hay {len(filter_point)} intersecciones.")


    def run_solution_area(self):
        """
        Ejecuta el cálculo del área solución y dibuja el área sombreada.
        """
        # Obtener los datos de los arcos desde el atributo correspondiente
        arcs = self.graphics_view.datos_de_los_arcos

        wedge_calculator = WedgeCalculator(self.k, self.alpha, self.graphics_view.points_array)

        # Ejecutar el barrido de línea para detectar intersecciones
        intersections = wedge_calculator.line_sweep_for_intersections(arcs)
        
        # Filtrar los puntos solucion (funciona bien en posiciones convexas)
        filter_point = wedge_calculator.filter_intersection_points(intersections, arcs)

        # Ordenar puntos filtrados
        sorted_points = wedge_calculator.sort_points_counterclockwise(filter_point)

        # Calcular el área del polígono
      #revisar   polygon_area = wedge_calculator.calculate_polygon_area(sorted_points)
  
        # Mostrar en consola el área
       #revisar  self.update_console(f"Área del polígono solución: {polygon_area:.2f} unidades cuadradas")

        grupos_de_puntos = wedge_calculator.group_intersection_points(sorted_points)

        # Dibujar el área sombreada y conectar arcos
        self.graphics_view.draw_alpha_k_area(grupos_de_puntos)




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
