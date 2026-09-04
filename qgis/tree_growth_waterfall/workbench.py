"""Desktop workflow forms. Scientific jobs run in a separate, cancellable Python."""
import html
import json
from pathlib import Path

from qgis.PyQt.QtCore import QProcess,QProcessEnvironment,QUrl
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtWidgets import (QWidget,QVBoxLayout,QFormLayout,QHBoxLayout,QLineEdit,QPushButton,
    QTabWidget,QLabel,QPlainTextEdit,QFileDialog,QMessageBox,QScrollArea,QDoubleSpinBox,QSpinBox,QComboBox)


class WorkbenchPanel(QWidget):
    def __init__(self,dock):
        super().__init__(dock);self.dock=dock;self.process=QProcess(self);self.counter=0
        layout=QVBoxLayout(self)
        note=QLabel("Use the ML Python / extra module paths in the Interpretation tab. Training uses the same projected-metre blocks for OLS, RF, XGBoost and MLP. Existing manuscript artifacts are never overwritten.")
        note.setWordWrap(True);layout.addWidget(note)
        self.tabs=QTabWidget();layout.addWidget(self.tabs)
        self.training_input=QLineEdit();self.training_output=QLineEdit();self.run_folder=QLineEdit()
        self.block=QDoubleSpinBox();self.block.setRange(10,10000);self.block.setValue(500);self.block.setSuffix(" m")
        self.crs=QLineEdit("EPSG:3879")
        self.settings=QPlainTextEdit(json.dumps({"seed":42,"split_search_iterations":2000,"bootstrap_repetitions":250,"moran_permutations":199,
            "models":{"XGB":{"n_estimators":550,"learning_rate":0.03,"max_depth":6,"min_child_weight":10,"subsample":0.85,"colsample_bytree":0.9,"reg_lambda":2,"n_jobs":4},
                      "RF":{"n_estimators":500,"max_depth":18,"min_samples_leaf":5,"max_features":0.8,"n_jobs":4},
                      "MLP":{"hidden_layer_sizes":[96,48],"learning_rate_init":0.001,"max_iter":500,"alpha":0.01,"early_stopping":False},"OLS":{}}},indent=2))
        self.settings.setMinimumHeight(180)
        train,form=self.page("1. Train / compare")
        form.addRow("Tree CSV",self.path_row(self.training_input,"file"));form.addRow("New run folder",self.path_row(self.training_output,"new"))
        form.addRow("Spatial block",self.block);form.addRow("Coordinate CRS",self.crs)
        form.addRow("Editable model / split settings (JSON)",self.settings)
        self.train_button=QPushButton("Train 70% / compare on 15% validation");form.addRow(self.train_button)
        self.train_button.clicked.connect(self.train)
        text=QLabel("Outputs: training/validation metrics, environmental VIF, frozen tree/block split and parameters. Test outcomes stay locked. New runs use positive-growth filtering without the manuscript's legacy outcome-quantile trim.")
        text.setWordWrap(True);form.addRow(text)
        validate,form=self.page("2. Validate / finalize")
        form.addRow("Completed training run",self.path_row(self.run_folder,"folder"))
        self.finalize_button=QPushButton("Refit selected model on 85% and open locked test ONCE")
        self.finalize_button.clicked.connect(self.finalize);form.addRow(self.finalize_button)
        button=QPushButton("View training / test reports");button.clicked.connect(self.reports);form.addRow(button)
        note=QLabel("Includes log-SGR, annual-% and kg-C error metrics; calibration; seven-level agreement; tree bootstrap and residual Moran's I. Do not tune after seeing test results. Different experiments are not new independent tests of the same held-out trees.")
        note.setWordWrap(True);form.addRow(note)
        diagnose,form=self.page("3. Diagnose / package")
        self.grid_input=QLineEdit();self.map_output=QLineEdit();self.model_run=QLineEdit();self.template=QLineEdit()
        self.height=QDoubleSpinBox();self.height.setRange(.1,100);self.height.setValue(10);self.height.setSuffix(" m")
        self.resolution=QDoubleSpinBox();self.resolution.setRange(.1,100);self.resolution.setValue(2)
        self.park=QComboBox();self.park.addItems(["0 Street/non-park","1 Park"]);self.park.setCurrentIndex(1)
        form.addRow("Environmental grid CSV",self.path_row(self.grid_input,"file"));form.addRow("New map-package folder",self.path_row(self.map_output,"new"))
        form.addRow("Training run (blank = frozen journal model)",self.path_row(self.model_run,"folder"))
        form.addRow("Optional template GeoTIFF",self.path_row(self.template,"file"));form.addRow("Reference height",self.height)
        form.addRow("Grid resolution (m)",self.resolution);form.addRow("Fallback park setting",self.park)
        self.diagnose_button=QPushButton("Generate suitability, deviation, growth and change rasters")
        self.diagnose_button.clicked.connect(self.diagnose);form.addRow(self.diagnose_button)
        note=QLabel("Current-model mapping needs X/Y and all eleven environmental inputs, including soil, for each period. The recovered July-28 demo uses its own eight-input archived model. Do not substitute models between packages. A template controls geometry only.")
        note.setWordWrap(True);form.addRow(note)
        self.log=QPlainTextEdit();self.log.setReadOnly(True);self.log.setMinimumHeight(150);layout.addWidget(self.log)
        buttons=QHBoxLayout();self.cancel=QPushButton("Cancel running job");self.cancel.clicked.connect(self.process.kill);buttons.addWidget(self.cancel)
        button=QPushButton("Open last output folder");button.clicked.connect(self.open_output);buttons.addWidget(button);layout.addLayout(buttons)
        self.process.readyReadStandardOutput.connect(self.read_output);self.process.readyReadStandardError.connect(self.read_error)
        self.process.finished.connect(self.finished);self.process.errorOccurred.connect(self.failed_start)
        self.last_folder=None;self.action=None

    def page(self,title):
        body=QWidget();form=QFormLayout(body)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(body);self.tabs.addTab(scroll,title)
        return body,form

    def path_row(self,field,kind):
        widget=QWidget();layout=QHBoxLayout(widget);layout.setContentsMargins(0,0,0,0);layout.addWidget(field)
        button=QPushButton("Browse");layout.addWidget(button)
        def browse():
            if kind=="file": result=QFileDialog.getOpenFileName(self,"Choose input file",field.text(),"All files (*)")[0]
            else:
                result=QFileDialog.getExistingDirectory(self,"Choose parent folder" if kind=="new" else "Choose folder",field.text())
                if result and kind=="new": result=str(Path(result)/("new_training_run" if field is self.training_output else "new_spatial_package"))
            if result: field.setText(result)
        button.clicked.connect(browse);return widget

    def train(self):
        try:
            config=json.loads(self.settings.toPlainText())
            config.update(input=self.training_input.text(),output=self.training_output.text(),block_size_m=self.block.value(),crs=self.crs.text())
            self.start("train",config)
        except Exception as e: self.log.appendPlainText("Cannot start: "+str(e))

    def finalize(self):
        if QMessageBox.question(self,"Open locked test?","This permanently records test access for this run. Finalize the validation-selected model now?",QMessageBox.Yes|QMessageBox.No,QMessageBox.No)!=QMessageBox.Yes:
            return
        self.start("finalize",{"run":self.run_folder.text()})

    def diagnose(self):
        self.start("diagnose",dict(input=self.grid_input.text(),output=self.map_output.text(),training_run=self.model_run.text(),
            template_raster=self.template.text(),height=self.height.value(),park_context=self.park.currentIndex(),resolution=self.resolution.value(),crs=self.crs.text()))

    def start(self,action,config):
        if self.process.state()!=QProcess.NotRunning:
            self.log.appendPlainText("Wait for or cancel the current job.");return
        executable=self.dock.python.text().strip()
        if not Path(executable).is_file(): self.log.appendPlainText("Choose ML Python in the Interpretation tab first.");return
        if action!="finalize" and (not config.get("output") or Path(config["output"]).exists()):
            self.log.appendPlainText("Enter a NEW output directory; existing directories are protected.");return
        worker=Path(__file__).parent/"worker/tree_growth_workbench.py"
        if not worker.exists(): worker=Path(__file__).resolve().parents[2]/"scripts/tree_growth_workbench.py"
        self.counter+=1;path=Path(self.dock.temp.name)/f"workflow_{self.counter}.json"
        path.write_text(json.dumps(config,indent=2),encoding="utf-8")
        args=["-u",str(worker),action,"--run",config["run"]] if action=="finalize" else ["-u",str(worker),action,"--config",str(path)]
        environment=QProcessEnvironment.systemEnvironment()
        for key in ["PYTHONPATH","PYTHONHOME","QT_PLUGIN_PATH","QGIS_PREFIX_PATH"]: environment.remove(key)
        if self.dock.extra_paths.text().strip(): environment.insert("PYTHONPATH",self.dock.extra_paths.text().strip())
        environment.insert("MPLBACKEND","Agg");self.process.setProcessEnvironment(environment)
        self.action=action;self.current_config=config;self.last_folder=Path(config.get("output") or config["run"])
        self.log.clear();self.log.appendPlainText("Starting "+action+". No existing manuscript model or map will be overwritten.")
        self.process.start(executable,args)

    def read_output(self): self.log.appendPlainText(bytes(self.process.readAllStandardOutput()).decode("utf-8",errors="replace").rstrip())
    def read_error(self): self.log.appendPlainText(bytes(self.process.readAllStandardError()).decode("utf-8",errors="replace").rstrip())
    def failed_start(self,error): self.log.appendPlainText("Job process error: "+self.process.errorString())

    def finished(self,code,status):
        self.read_output();self.read_error()
        self.log.appendPlainText("Completed." if code==0 else f"Stopped/failed (exit {code}). Partial output is retained for inspection; use a new output folder to retry.")
        if self.last_folder and self.last_folder.is_dir():
            with (self.last_folder/"QGIS_JOB_LOG.txt").open("a",encoding="utf-8") as f: f.write(self.log.toPlainText()+"\n")
        if code==0 and self.action=="train":
            self.run_folder.setText(str(self.last_folder));self.model_run.setText(str(self.last_folder));self.tabs.setCurrentIndex(1)
        if code==0 and self.action=="diagnose":
            self.dock.trusted_model_packages.add(str((self.last_folder/"manifest.json").resolve()))
            self.dock.manifest.setText(str(self.last_folder/"manifest.json"));self.dock.open_package();self.dock.load_map();self.dock.activate_click()
            self.dock.main_tabs.setCurrentIndex(1)

    def reports(self):
        folder=Path(self.run_folder.text())
        self.log.clear()
        for name in ["training_report.json","tables/candidate_metrics.csv","tables/training_environment_vif.csv","finalized/validation_report.json"]:
            path=folder/name
            if path.is_file(): self.log.appendPlainText(name+"\n"+path.read_text(encoding="utf-8"))
        self.last_folder=folder

    def open_output(self):
        if self.last_folder and self.last_folder.exists(): QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_folder)))

    def shutdown(self):
        if self.process.state()!=QProcess.NotRunning:
            self.process.kill();self.process.waitForFinished(3000)
