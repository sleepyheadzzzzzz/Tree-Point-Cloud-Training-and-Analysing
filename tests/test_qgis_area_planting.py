"""Run with QGIS Python: real dock/job/vector round trip on synthetic data.

Uses an isolated application, not the user's current QGIS project/profile.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

qroot=Path(sys.executable).resolve().parents[1]
dll_dirs=[qroot/"bin",qroot/"apps/Qt5/bin",qroot/"apps/qgis-ltr/bin"]
dll_handles=[os.add_dll_directory(str(p)) for p in dll_dirs if p.exists()] if hasattr(os,"add_dll_directory") else []
os.environ["QT_QPA_PLATFORM"]="offscreen"
from qgis.PyQt.QtCore import Qt, QSettings, QProcess
from qgis.PyQt.QtWidgets import QMainWindow, QMessageBox
from qgis.PyQt.QtGui import QFontDatabase, QFont
from qgis.core import QgsApplication, QgsProject, QgsCoordinateReferenceSystem, QgsVectorLayer, QgsRasterLayer, Qgis
from qgis.gui import QgsMapCanvas

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--plugin-dir",required=True)
p.add_argument("--python",required=True)
p.add_argument("--extra-paths",default="")
p.add_argument("--synthetic-raster",required=True)
p.add_argument("--output",required=True)
p.add_argument("--site-output",help="Optional already generated private site output to export/inspect locally")
args=p.parse_args()
out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
settings=tempfile.TemporaryDirectory()
QSettings.setDefaultFormat(QSettings.IniFormat)
QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,settings.name)
app=QgsApplication([],True);app.initQgis()
font=Path(os.environ.get("WINDIR",""))/"Fonts/arial.ttf"
if font.exists(): QFontDatabase.addApplicationFont(str(font));app.setFont(QFont("Arial",10))
sys.path.insert(0,args.plugin_dir)
from tree_growth_waterfall.plugin import TreeGrowthPlugin
from tree_growth_waterfall.planting import export_and_load


class Interface:
    def __init__(self):
        self.window=QMainWindow();self.canvas=QgsMapCanvas();self.window.setCentralWidget(self.canvas)
        self.window.resize(1900,1150);self.menu=[]
        self.canvas.setCanvasColor(Qt.white)
        self.canvas.setDestinationCrs(QgsCoordinateReferenceSystem("EPSG:3879"))
    def mainWindow(self): return self.window
    def mapCanvas(self): return self.canvas
    def addDockWidget(self,area,dock): self.window.addDockWidget(area,dock)
    def removeDockWidget(self,dock): self.window.removeDockWidget(dock)
    def addPluginToRasterMenu(self,name,action): self.menu.append(action)
    def removePluginRasterMenu(self,name,action): self.menu.remove(action)


iface=Interface();plugin=TreeGrowthPlugin(iface);plugin.initGui();plugin.run()
assert len(iface.menu)==1
dock=plugin.dock;panel=dock.workbench
dock.python.setText(args.python);dock.extra_paths.setText(args.extra_paths)
dock.main_tabs.setCurrentIndex(0);panel.tabs.setCurrentIndex(3)
panel.planting.input.setText(args.synthetic_raster)
panel.planting.output.setText(str(out/"synthetic_plan"))
iface.window.show()
# In this isolated automated test only, acknowledge the band-order confirmation.
question=QMessageBox.question
QMessageBox.question=lambda *a,**k:QMessageBox.Yes
panel.planting.button.click()
QMessageBox.question=question
deadline=time.monotonic()+120
while panel.process.state()!=QProcess.NotRunning and time.monotonic()<deadline:
    app.processEvents();time.sleep(.02)
app.processEvents()
assert panel.process.state()==QProcess.NotRunning,"Planning timed out"
assert (out/"synthetic_plan/GIS_EXPORT.json").exists(),panel.log.toPlainText()
results={"qgis_version":Qgis.QGIS_VERSION,"synthetic_job_button":"passed"}


def audit(folder):
    exports=json.loads((folder/"GIS_EXPORT.json").read_text())
    report=json.loads((folder/"planting_report.json").read_text())
    details={}
    for name,record in exports.items():
        if not isinstance(record,dict): continue
        for kind in ["gpkg","shapefile"]:
            layer=QgsVectorLayer(str(folder/record[kind]),name,"ogr")
            assert layer.isValid()
            assert layer.featureCount()==record["features"]
            assert layer.crs().authid()==report["crs"]
            total=0
            for feature in layer.getFeatures():
                assert feature.geometry().isGeosValid(),(name,feature.id())
                assert feature["area_m2"]>10
                assert abs(feature.geometry().area()-feature["area_m2"])<.002
                total+=feature["area_m2"]
            details[name+"_"+kind]={"features":layer.featureCount(),"area_m2":total}
    return details


results["synthetic_exports"]=audit(out/"synthetic_plan")
if args.site_output:
    folder=Path(args.site_output)
    QgsProject.instance().clear()
    if not (folder/"GIS_EXPORT.json").exists():
        count,layers=export_and_load(folder)
    else:
        layers=[]
        group=QgsProject.instance().layerTreeRoot().addGroup("Area planting | site")
        for name in ["species_suitable_areas","highest_suitability","diversity_oriented"]:
            layer=QgsVectorLayer(str(folder/(name+".gpkg")),name,"ogr")
            layer.loadNamedStyle(str(folder/(name+".qml")))
            QgsProject.instance().addMapLayer(layer,False)
            group.addLayer(layer).setItemVisibilityChecked(False);layers.append(layer)
        count=QgsRasterLayer(str(folder/"suitable_genus_count.tif"),"Suitable genera (0-9)")
        count.loadNamedStyle(str(folder/"suitable_genus_count.qml"))
        QgsProject.instance().addMapLayer(count,False);group.addLayer(count)
    results["site_exports"]=audit(folder)
    panel.planting.input.setText(json.loads((folder/"planting_report.json").read_text())["configuration"]["input"])
    panel.planting.output.setText(str(folder.parent/"new_area_plan"))
    panel.log.setPlainText("PRIVATE SITE RESULTS (not the synthetic test):\n"+json.dumps(json.loads((folder/"planting_report.json").read_text())["summaries"],indent=2))
    iface.canvas.setLayers([count]);iface.canvas.setExtent(count.extent());iface.canvas.refresh()
    deadline=time.monotonic()+3
    while time.monotonic()<deadline: app.processEvents();time.sleep(.02)
    iface.window.grab().save(str(out/"area_planting_QGIS.png"))
    QgsProject.instance().setFileName(str(folder/"area_planting.qgz"))
    assert QgsProject.instance().write()
    for layer in layers:
        if layer.name()=="diversity_oriented":
            iface.canvas.setLayers([layer]);iface.canvas.refresh()
            deadline=time.monotonic()+2
            while time.monotonic()<deadline: app.processEvents();time.sleep(.02)
            iface.canvas.saveAsImage(str(out/"diversity_map.png"))
(out/"QGIS_AUDIT.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
print(json.dumps(results,indent=2))
plugin.unload();assert not iface.menu
QgsProject.instance().clear()
# Avoid third-party GDAL/Qt shutdown-order issues in an isolated headless process.
sys.stdout.flush();os._exit(0)
