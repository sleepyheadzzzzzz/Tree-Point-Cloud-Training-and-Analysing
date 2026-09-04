"""QGIS area-planning form and native vector exports (no ML in the QGIS process)."""
import json
from pathlib import Path

from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QMessageBox
from qgis.core import (QgsRasterLayer, QgsVectorLayer, QgsProject, QgsCoordinateTransform,
    QgsWkbTypes, QgsGeometry, QgsVectorFileWriter, QgsCategorizedSymbolRenderer,
    QgsRendererCategory, QgsFillSymbol, QgsColorRampShader, QgsRasterShader,
    QgsSingleBandPseudoColorRenderer)

GENERA = {3:"Acer",4:"Alnus",5:"Betula",6:"Pinus",7:"Prunus",8:"Quercus",9:"Sorbus",10:"Tilia",11:"Ulmus"}
COLORS = ["#2979b8", "#65a89f", "#9acf8c", "#638143", "#b182bb", "#edca7a", "#b96da3", "#e69a57", "#d86951"]


class PlantingForm:
    def __init__(self, panel):
        self.panel = panel
        _, form = panel.page("4. Area planting")
        self.input = QLineEdit()
        self.output = QLineEdit()
        self.boundary = QLineEdit()
        self.exclusions = QLineEdit()
        self.reliability = QLineEdit()
        self.bands = QLineEdit("3,4,5,6,7,8,9,10,11")
        self.area = QDoubleSpinBox(); self.area.setRange(0, 1e7); self.area.setValue(10); self.area.setSuffix(" m2")
        self.level = QSpinBox(); self.level.setRange(1, 7); self.level.setValue(5)
        self.loss = QSpinBox(); self.loss.setRange(0, 6); self.loss.setValue(2)
        note = QLabel("Create two alternative area allocations, not tree planting points. Highest suitability retains local winners; diversity-oriented balances genus area within suitable land. Both omit assigned patches at or below the minimum area.")
        note.setWordWrap(True); form.addRow(note)
        for title, field, kind in [("Suitability GeoTIFF (levels 1-7)", self.input, "file"),
            ("New output folder", self.output, "new"),
            ("Site boundary (optional polygons)", self.boundary, "file"),
            ("Exclusions (optional polygons)", self.exclusions, "file"),
            ("Reliability (optional, keep code 1)", self.reliability, "file")]:
            form.addRow(title, panel.path_row(field, kind))
        form.addRow("Genus band numbers", self.bands)
        order=QLabel("Band order: Acer, Alnus, Betula, Pinus, Prunus, Quercus, Sorbus, Tilia, Ulmus.")
        order.setWordWrap(True);form.addRow(order)
        form.addRow("Strict minimum connected area", self.area)
        form.addRow("Minimum suitable level", self.level)
        form.addRow("Diversity: maximum level loss", self.loss)
        button = QPushButton("Use current diagnosis period's suitability + reliability")
        button.clicked.connect(self.from_diagnosis); form.addRow(button)
        self.button = QPushButton("Generate count map + genus polygons + BOTH proposals")
        self.button.clicked.connect(self.run); form.addRow(self.button)
        note = QLabel("Count legend is always 0-9; NoData is separate. Sorbus is included only where suitable, not forced into the plan. Polygon masks use cell centres. Input alone does not exclude buildings, water, utilities or existing crowns: supply exclusions and verify the site. Default band order is specific to this study. Clip citywide rasters to your site first (maximum 2 million cells).")
        note.setWordWrap(True); form.addRow(note)

    def from_diagnosis(self):
        try:
            dock = self.panel.dock
            if not dock.meta: dock.open_package()
            mode, period = dock.mode.currentData()
            if mode != "local":
                raise ValueError("Choose a single period, not a change map")
            from .plugin import contained
            record = dock.meta["periods"][period]
            self.input.setText(str(contained(dock.root, record["suitability"])))
            self.reliability.setText(str(contained(dock.root, record["reliability"])))
        except Exception as error:
            self.panel.log.appendPlainText(str(error))

    def geometry_list(self, path, crs):
        layer = QgsVectorLayer(path, "planning mask", "ogr")
        if not layer.isValid() or not layer.crs().isValid() or layer.geometryType() != QgsWkbTypes.PolygonGeometry:
            raise ValueError("Boundary/exclusions must be polygon data with a valid CRS")
        transform = QgsCoordinateTransform(layer.crs(), crs, QgsProject.instance())
        geometries = []
        for feature in layer.getFeatures():
            geometry = QgsGeometry(feature.geometry())
            if geometry.isEmpty() or not geometry.isGeosValid():
                raise ValueError("Empty or invalid mask geometry; repair it first")
            geometry.transform(transform)
            geometries.append(json.loads(geometry.asJson(15)))
        if not geometries: raise ValueError("Mask layer is empty")
        return geometries

    def run(self):
        try:
            source = QgsRasterLayer(self.input.text(), "suitability")
            if not source.isValid(): raise ValueError("Choose a valid suitability GeoTIFF")
            bands = [int(x.strip()) for x in self.bands.text().split(",")]
            config = dict(input=self.input.text(), output=self.output.text(), bands=bands,
                min_area_m2=self.area.value(), min_level=self.level.value(),
                diversity_max_level_loss=self.loss.value(), reliability=self.reliability.text())
            for key, field in [("boundary", self.boundary), ("exclusion", self.exclusions)]:
                if field.text().strip():
                    config[key+"_geometries"] = self.geometry_list(field.text(), source.crs())
                    config[key+"_source"] = field.text()
            mapping = ", ".join(f"{name}={band}" for name, band in zip(GENERA.values(), bands))
            if len(bands) != 9: raise ValueError("Nine genus bands are required")
            message = "Confirm genus bands: "+mapping+".\nNoData is not an unsuitable observation. Without boundary/exclusions/domain data, the result is only a suitability-footprint screening. Continue?"
            if QMessageBox.question(self.panel, "Confirm area-planning inputs", message,
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return
            self.panel.start("planting", config)
        except Exception as error:
            self.panel.log.appendPlainText("Cannot plan: "+str(error))


def export_and_load(folder):
    """Write GIS-standard GPKG and SHP using QGIS, then load styled results."""
    folder = Path(folder)
    report = json.loads((folder/"planting_report.json").read_text(encoding="utf-8"))
    group = QgsProject.instance().layerTreeRoot().addGroup("Area planting | "+folder.name)
    loaded = []
    exports = {}
    for name, relative in report["vectors"].items():
        layer = QgsVectorLayer(str(folder/relative), name, "ogr")
        if not layer.isValid(): raise ValueError("Could not load "+relative)
        if layer.featureCount() == 0:
            exports[name] = "Empty: GeoJSON retained; no GIS polygons to export"
            continue
        for extension, driver in [("gpkg", "GPKG"), ("shp", "ESRI Shapefile")]:
            target = folder/(name+"."+extension)
            if target.exists(): raise FileExistsError(target)
            options = QgsVectorFileWriter.SaveVectorOptions()
            options.driverName = driver; options.fileEncoding = "UTF-8"
            result = QgsVectorFileWriter.writeAsVectorFormatV3(layer, str(target),
                QgsProject.instance().transformContext(), options)
            if result[0] != QgsVectorFileWriter.NoError:
                raise RuntimeError(str(result))
        categories = [QgsRendererCategory(name, QgsFillSymbol.createSimple(dict(color=color,
            outline_style="no")), name) for name, color in zip(GENERA.values(), COLORS)]
        layer.setRenderer(QgsCategorizedSymbolRenderer("genus", categories))
        QgsProject.instance().addMapLayer(layer, False)
        node = group.addLayer(layer)
        node.setItemVisibilityChecked(False)
        layer.saveNamedStyle(str(folder/(name+".qml")))
        exports[name] = dict(gpkg=name+".gpkg", shapefile=name+".shp", features=layer.featureCount())
        loaded.append(layer)
    count = QgsRasterLayer(str(folder/report["rasters"]["suitable_genus_count"]), "Suitable genera (0-9)")
    if not count.isValid(): raise ValueError("Count raster failed to load")
    colors = ["#f2f2f2", "#edf8e9", "#c7e9c0", "#a1d99b", "#74c476", "#41ab5d", "#238b45", "#006d2c", "#005322", "#003717"]
    ramp = QgsColorRampShader(); ramp.setColorRampType(QgsColorRampShader.Discrete)
    ramp.setColorRampItemList([QgsColorRampShader.ColorRampItem(i, QColor(c), str(i)) for i,c in enumerate(colors)])
    shader = QgsRasterShader(); shader.setRasterShaderFunction(ramp)
    count.setRenderer(QgsSingleBandPseudoColorRenderer(count.dataProvider(), 1, shader))
    QgsProject.instance().addMapLayer(count, False); group.addLayer(count)
    count.saveNamedStyle(str(folder/"suitable_genus_count.qml"))
    (folder/"GIS_EXPORT.json").write_text(json.dumps(exports, indent=2), encoding="utf-8")
    return count, loaded
