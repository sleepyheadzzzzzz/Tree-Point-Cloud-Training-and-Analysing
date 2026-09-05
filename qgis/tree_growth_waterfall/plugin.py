"""QGIS UI only. ML computation runs asynchronously in external Python."""
import html
import json
from pathlib import Path
import shutil
import tempfile

from qgis.PyQt.QtCore import Qt, QProcess, QProcessEnvironment, QSettings, QTimer
from qgis.PyQt.QtGui import QColor, QIcon, QDesktopServices
from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtSvg import QSvgWidget
from qgis.PyQt.QtWidgets import (QAction, QComboBox, QDockWidget, QFileDialog,
                                QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                                QPushButton, QScrollArea, QVBoxLayout, QWidget, QTabWidget, QMessageBox)
from qgis.core import (QgsColorRampShader, QgsCoordinateReferenceSystem,
                       QgsCoordinateTransform, QgsPointXY, QgsProject, QgsRasterLayer,
                       QgsRasterShader, QgsSingleBandPseudoColorRenderer)
from qgis.gui import QgsMapToolEmitPoint, QgsVertexMarker


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise ValueError("Path escapes package directory")
    return path


class TreeGrowthPlugin:
    def __init__(self, iface):
        self.iface, self.dock, self.action = iface, None, None

    def initGui(self):
        self.action = QAction(QIcon(str(Path(__file__).parent/"icon.svg")), "TreeSuit XAI", self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addPluginToRasterMenu("Tree Growth", self.action)

    def run(self):
        if self.dock is None:
            self.dock = WaterfallDock(self.iface)
            self.iface.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
        self.dock.show()
        self.dock.raise_()

    def unload(self):
        if self.action:
            self.iface.removePluginRasterMenu("Tree Growth", self.action)
        if self.dock:
            self.dock.shutdown()
            self.iface.removeDockWidget(self.dock)
            self.dock.deleteLater()


class WaterfallDock(QDockWidget):
    def __init__(self, iface):
        super().__init__("TreeSuit XAI | growth-suitability diagnosis", iface.mainWindow())
        self.setObjectName("TreeGrowthWaterfallDock")
        self.iface = iface
        self.root = None
        self.meta = None
        self.layer = None
        self.last_output = None
        self.trusted_model_packages=set()
        self.temp = tempfile.TemporaryDirectory(prefix="tree_growth_click_")
        self.request_id = 0
        self.request_signature=None
        self.settings = QSettings()
        body = QWidget()
        layout = QVBoxLayout(body)
        form = QFormLayout()
        self.manifest = QLineEdit()
        self.python = QLineEdit(self.settings.value("TreeGrowth/python", ""))
        self.extra_paths = QLineEdit(self.settings.value("TreeGrowth/extra_paths", ""))
        self.extra_paths.setPlaceholderText("Optional extra PYTHONPATH; normally blank")
        for title, field, callback in [
            ("Spatial package", self.manifest, self.browse_package),
            ("ML Python", self.python, self.browse_python),
        ]:
            row = QWidget()
            hbox = QHBoxLayout(row)
            hbox.setContentsMargins(0,0,0,0)
            hbox.addWidget(field)
            button = QPushButton("Browse")
            button.clicked.connect(callback)
            hbox.addWidget(button)
            form.addRow(title, row)
        form.addRow("Extra module paths", self.extra_paths)
        self.species, self.mode, self.map_kind = QComboBox(), QComboBox(), QComboBox()
        self.map_kind.addItems(["Suitability", "Continuous difference (pp)"])
        form.addRow("Species / category", self.species)
        form.addRow("Explanation", self.mode)
        form.addRow("Map", self.map_kind)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self.load_button = QPushButton("Load / update map")
        self.click_button = QPushButton("Activate click explanation")
        self.export_button = QPushButton("Save last result")
        self.load_button.clicked.connect(self.load_map)
        self.click_button.clicked.connect(self.activate_click)
        self.export_button.clicked.connect(self.save_last)
        for button in [self.load_button, self.click_button, self.export_button]:
            buttons.addWidget(button)
        layout.addLayout(buttons)
        help_button=QPushButton("User guide / installation / area planting")
        help_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://github.com/sleepyheadzzzzzz/Tree-Point-Cloud-Training-and-Analysing/blob/main/docs/QGIS_WORKBENCH.md")))
        layout.addWidget(help_button)
        self.info = QLabel("Choose a manifest.json package and an ML Python interpreter. No model is retrained.")
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        self.svg = QSvgWidget()
        self.svg.setMinimumSize(650, 570)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.svg)
        layout.addWidget(scroll)
        from .workbench import WorkbenchPanel
        self.main_tabs=QTabWidget()
        self.workbench=WorkbenchPanel(self)
        self.main_tabs.addTab(self.workbench,"Training / validation / diagnosis")
        self.main_tabs.addTab(body,"Interpretation: click a map cell")
        self.main_tabs.setCurrentIndex(1)
        self.setWidget(self.main_tabs)
        self.setMinimumWidth(710)
        self.tool = QgsMapToolEmitPoint(iface.mapCanvas())
        self.tool.canvasClicked.connect(self.clicked)
        self.marker = QgsVertexMarker(iface.mapCanvas())
        self.marker.setColor(QColor("#aa2546"))
        self.marker.setIconType(QgsVertexMarker.IconType.ICON_CROSS)
        self.marker.setIconSize(12)
        self.marker.hide()
        self.process = QProcess(self)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.timed_out)
        self.mode.currentIndexChanged.connect(self.selection_changed)
        self.species.currentIndexChanged.connect(self.selection_changed)
        self.map_kind.currentIndexChanged.connect(self.selection_changed)

    def browse_package(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open spatial package", "", "Spatial package (manifest.json)")
        if path:
            self.manifest.setText(path)
            self.open_package()

    def browse_python(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select ML environment Python executable", "", "All files (*)")
        if path:
            self.python.setText(path)

    def open_package(self):
        path = Path(self.manifest.text()).resolve()
        self.root = path.parent
        self.meta = json.loads(path.read_text(encoding="utf-8"))
        if self.meta.get("schema_version") != 1:
            raise ValueError("Unsupported spatial package")
        if self.meta["model"].get("format")=="trusted_joblib" and str(path) not in self.trusted_model_packages:
            answer=QMessageBox.question(
                self,
                "Trust this model package?",
                "This package uses a Python joblib model, which can execute code when loaded. Continue only for a package you created or otherwise trust.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer!=QMessageBox.StandardButton.Yes:
                self.meta=None
                raise ValueError("Model trust was not confirmed")
            self.trusted_model_packages.add(str(path))
        self.species.blockSignals(True)
        self.mode.blockSignals(True)
        self.species.clear()
        for code, name in sorted(self.meta["species"].items(), key=lambda item:int(item[0])):
            self.species.addItem(f"{code}  {name}", int(code))
        self.species.setCurrentIndex(min(1,self.species.count()-1))
        self.mode.clear()
        for period in self.meta["periods"]:
            self.mode.addItem(f"Local vs reference | {period}", ("local",period))
        change = self.meta["change"]
        self.mode.addItem(f"Change | {change['earlier']} to {change['later']}", ("change",change["later"]))
        self.mode.setCurrentIndex(len(self.meta["periods"])-1)
        self.species.blockSignals(False)
        self.mode.blockSignals(False)
        self.info.setText(self.meta["scope_note"] + " Select a species, load the map, then activate click explanation.")

    def selection_changed(self):
        if self.layer and self.meta:
            self.load_map(reset_extent=False)
        self.svg.renderer().load(b'<svg xmlns="http://www.w3.org/2000/svg"/>')
        self.last_output = None

    def load_map(self, checked=False, reset_extent=True):
        try:
            if not self.meta or Path(self.manifest.text()).resolve().parent != self.root:
                self.open_package()
            mode, period = self.mode.currentData()
            discrete = self.map_kind.currentIndex() == 0
            key = ("suitability_change" if discrete else "growth_change") if mode == "change" else ("suitability" if discrete else "deviation")
            record = self.meta["change"] if mode == "change" else self.meta["periods"][period]
            layer = QgsRasterLayer(str(contained(self.root, record[key])), f"{key} | {self.species.currentText()} | {period}")
            if not layer.isValid():
                raise ValueError("Raster did not load")
            shader = QgsColorRampShader()
            shader.setColorRampType(
                QgsColorRampShader.Type.Discrete if discrete
                else QgsColorRampShader.Type.Interpolated
            )
            if discrete and mode != "change":
                palette = ["#a33a36","#dd8068","#f0bb97","#eeedc5","#badd91","#72b96d","#1c743e"]
                items = [QgsColorRampShader.ColorRampItem(i+1,QColor(c),str(i+1)) for i,c in enumerate(palette)]
            elif discrete:
                items = [QgsColorRampShader.ColorRampItem(v,QColor(c),str(v)) for v,c in zip(range(-6,7),["#8b1831","#aa2e43","#c94e5d","#e17c83","#efadb0","#f4d5d6","#f4f4f4","#deeddc","#bce0b8","#8dc887","#57a962","#318346","#126334"])]
            else:
                items = [QgsColorRampShader.ColorRampItem(v,QColor(c),f"{v:+g} pp") for v,c in [(-5,"#bd3547"),(0,"#f7f7f7"),(5,"#21804d")]]
            shader.setColorRampItemList(items)
            raster_shader = QgsRasterShader()
            raster_shader.setRasterShaderFunction(shader)
            renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(),self.species.currentData(),raster_shader)
            layer.setRenderer(renderer)
            if self.layer:
                QgsProject.instance().removeMapLayer(self.layer.id())
            self.layer = layer
            QgsProject.instance().addMapLayer(layer)
            if reset_extent:
                canvas = self.iface.mapCanvas()
                extent = QgsCoordinateTransform(layer.crs(),canvas.mapSettings().destinationCrs(),QgsProject.instance()).transformBoundingBox(layer.extent())
                canvas.setExtent(extent)
            self.iface.mapCanvas().refresh()
            self.info.setText(self.meta["scope_note"] + " Click a coloured cell to explain the modelled environmental contrast.")
        except Exception as error:
            self.info.setText("Cannot load map: " + str(error))

    def activate_click(self):
        if self.meta:
            self.iface.mapCanvas().setMapTool(self.tool)

    def clicked(self, point, button=Qt.MouseButton.LeftButton):
        if button != Qt.MouseButton.LeftButton or not self.meta:
            return
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.info.setText("An explanation is running; please wait.")
            return
        executable = self.python.text().strip()
        if not Path(executable).is_file():
            self.info.setText("Choose the Python executable in an environment containing numpy, rasterio and xgboost.")
            return
        self.settings.setValue("TreeGrowth/python",executable)
        self.settings.setValue("TreeGrowth/extra_paths",self.extra_paths.text())
        worker = Path(__file__).parent/"worker/explain_spatial_cell.py"
        if not worker.exists():
            worker = Path(__file__).resolve().parents[2]/"scripts/explain_spatial_cell.py"
        crs = QgsCoordinateReferenceSystem(self.meta["crs"])
        converted = QgsCoordinateTransform(self.iface.mapCanvas().mapSettings().destinationCrs(),crs,QgsProject.instance()).transform(QgsPointXY(point))
        self.marker.setCenter(point)
        self.marker.show()
        self.request_id += 1
        self.output_stem = Path(self.temp.name)/f"click_{self.request_id}"
        mode, period = self.mode.currentData()
        self.request_signature=(str(Path(self.manifest.text()).resolve()),self.species.currentData(),mode,period)
        args = [str(worker),"--package",self.manifest.text(),"--x",str(converted.x()),"--y",str(converted.y()),
                "--species",str(self.species.currentData()),"--mode",mode,"--period",period,"--output",str(self.output_stem)]
        environment = QProcessEnvironment.systemEnvironment()
        for name in ["PYTHONPATH","PYTHONHOME","QT_PLUGIN_PATH","QGIS_PREFIX_PATH"]:
            environment.remove(name)
        if self.extra_paths.text().strip():
            environment.insert("PYTHONPATH",self.extra_paths.text().strip())
        self.process.setProcessEnvironment(environment)
        self.info.setText("Calculating exact reference-matched environmental contributions...")
        self.process.start(executable,args)
        self.timer.start(90000)

    def finished(self, code, status):
        self.timer.stop()
        current=self.mode.currentData()
        if not current or self.request_signature!=(str(Path(self.manifest.text()).resolve()),self.species.currentData(),*current):
            self.info.setText("Map settings changed while the explanation was running. Click the current map again.")
            return
        if code != 0:
            error = bytes(self.process.readAllStandardError()).decode("utf-8",errors="replace")
            self.info.setText("Explanation failed: " + error[-1800:])
            return
        try:
            result = json.loads(self.output_stem.with_suffix(".json").read_text(encoding="utf-8"))
            self.svg.load(str(self.output_stem.with_suffix(".svg")))
            self.svg.renderer().setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
            self.svg.setMinimumHeight(self.svg.renderer().defaultSize().height())
            self.last_output = self.output_stem
            if result["status"] == "ok":
                self.info.setText(f"Difference {result['delta_pp']:+.4f} pp | domain code {result['reliability_code']} | "
                                  f"additivity error {result['additivity_error_pp']:.2g} pp | " + " ".join(result["warnings"]))
            else:
                self.info.setText(result["message"])
        except Exception as error:
            self.info.setText("Could not display result: " + str(error))

    def timed_out(self):
        self.process.kill()
        self.info.setText("Explanation timed out after 90 seconds. Check the selected Python environment.")

    def process_error(self, error):
        self.timer.stop()
        self.info.setText("Could not start or run the external ML Python process: " + self.process.errorString())

    def save_last(self):
        if self.last_output is None:
            return
        folder = QFileDialog.getExistingDirectory(self,"Export the last JSON, SVG and CSV")
        if folder:
            for suffix in [".json",".svg",".csv"]:
                source = self.last_output.with_suffix(suffix)
                destination = Path(folder)/source.name
                if source.exists() and not destination.exists():
                    shutil.copy2(source,destination)
            self.info.setText("Exported available result files. Existing files were not overwritten.")

    def shutdown(self):
        self.workbench.shutdown()
        self.timer.stop()
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()
            self.process.waitForFinished(3000)
        canvas = self.iface.mapCanvas()
        if canvas.mapTool() == self.tool:
            canvas.unsetMapTool(self.tool)
        canvas.scene().removeItem(self.marker)
        self.temp.cleanup()
